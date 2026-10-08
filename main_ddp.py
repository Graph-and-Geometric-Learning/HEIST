from tracemalloc import start
import warnings
warnings.filterwarnings('ignore')

import torch
import random
import os
os.environ["OMP_NUM_THREADS"] = "2"
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
import logging
from utils.dataloader import create_dataloader, create_dataloader_ddp
from model.model import GraphEncoder, GIN_decoder
from model.loss import (contrastive_loss_cell, mae_loss_cell, niche_cut_loss, niche_balance_loss,
                        niche_ortho_loss)
import torch.optim as optim
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import time
import numpy as np
from argparse import ArgumentParser
import gc
from sklearn.model_selection import train_test_split
from glob import glob
import socket
from datetime import timedelta
import wandb

torch.autograd.set_detect_anomaly(os.environ.get('HEIST_DEBUG_ANOMALY','0')=='1')

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  
        return s.getsockname()[1] 

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    # Was hardcoded, which makes two concurrent jobs landing on the SAME node collide on rendezvous.
    # Arm A and Arm B run at the same time, so let the launcher pass a distinct port.
    os.environ["MASTER_PORT"] = os.environ.get("HEIST_MASTER_PORT", "29500")
    os.environ["NCCL_IB_DISABLE"] = "1"
    dist.init_process_group("nccl",  timeout=timedelta(minutes=180), rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, pyg_nn.TransformerConv):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def validate(rank, world_size, model, decoder, val_idx, all_files, args):# -> Any:
    device = torch.device(f"cuda:{rank}")
    model.eval()
    decoder.eval()

    total_val_loss = torch.tensor(0.0, device=device)
    nb = 0

    with torch.no_grad():
        for graph_idx in tqdm(val_idx):
            graphs = torch.load(all_files[graph_idx], weights_only=False)
            try:
                # NOT create_dataloader_ddp: its DistributedSampler shards by rank, so rank-0-only
                # validation would score just 1/world_size of each file.
                dataloader = create_dataloader(graphs, args.batch_size, permute=False)
            except Exception as e:
                logging.error("Dataloader creation failed.")
                logging.error("Exception: %s", str(e))
                continue
            for high_level_subgraph, low_level_batch, batch_idx in dataloader:
                high_level_subgraph = high_level_subgraph.to(device)
                low_level_batch = low_level_batch.to(device)
                low_level_batch.batch_idx = batch_idx.to(device)

                high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.2).long().to(device)
                high_true_v, low_true_v = high_level_subgraph.X, low_level_batch.X   # BEFORE forward masks X
                present = torch.where(low_level_batch.X.T[0])[0]   # .T[0]: X.T is [1,N]; where(2-D)[0] was all zeros
                if len(present) == 0:
                    continue
                masked = present[torch.randint(0, len(present), (max(int(len(present)*0.2),1), 1))]
                low_mask = torch.ones_like(low_level_batch.X).long().to(device)
                low_mask[masked] = 0
                high_emb, low_emb, niche_S = model.module(high_level_subgraph, low_level_batch,
                                                          high_mask, low_mask, return_niche=True)

                contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 10)

                _high_emb = high_emb * high_mask
                _low_emb = low_emb * low_mask
                decoded_high, decoded_low, alpha_sigmoid = decoder.module(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                # Targets must be the PRE-mask values: forward() rebinds low_level_graphs.X = X*mask,
                # so scoring against low_level_batch.X here measured 'predict zeros' and rewarded a
                # degenerate model during checkpoint selection.
                if((1 - high_mask).sum()):
                    recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_true_v.float()*(1-high_mask), reduction='sum')/high_mask.sum() + F.mse_loss(decoded_low*(1-low_mask), low_true_v.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                else:
                    recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_true_v.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                                
                #orthogonal_loss = 0.1 * (F.normalize(_high_emb.T) @ F.normalize(_high_emb) - torch.eye(_high_emb.shape[1]).to(device)).square().mean() \
                #               + 0.1 * (F.normalize(_low_emb.T) @ F.normalize(_low_emb) - torch.eye(_low_emb.shape[1]).to(device)).square().mean()

                loss = alpha_sigmoid * contrastive_loss + (1 - alpha_sigmoid) * recon_loss #+ orthogonal_loss
                # Mirror the niche terms from training. Checkpoint selection is driven by this
                # number, so omitting them would select for a model that ignores the niche head.
                if niche_S is not None:
                    loss = loss + args.w_niche_cut * niche_cut_loss(niche_S, high_level_subgraph.edge_index) \
                                + args.w_niche_ortho * niche_ortho_loss(niche_S) \
                                + args.w_niche_balance * niche_balance_loss(niche_S)
                # loss = contrastive_loss
                if torch.isnan(loss) or not torch.isfinite(loss):
                    print(f"Rank {rank}: NaN loss encountered, skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                total_val_loss += loss
                nb += 1

    model.train(); decoder.train()      # eval() was never undone -> dropout stayed off for good
    if nb == 0:
        # Every batch was skipped -- the model is degenerate (typically all-NaN weights). Returning
        # 0.0 here (the old `max(nb,1)` behaviour) made a dead model score BETTER than any healthy
        # one and win checkpoint selection: job 23872253 saved a 421/421-NaN checkpoint as "best"
        # with val 0.0. inf makes it unselectable.
        print(f"Rank {rank}: validation produced no finite batches -> val loss inf")
        return float('inf')
    return (total_val_loss / nb).item()   # mean, so it is comparable across runs

def train(rank, world_size, args):
    """Main training loop for DDP."""
    setup(rank, world_size)
    model_path = args.save_path

    wandb_run = None
    if rank == 0 and args.wandb:
        # Never let logging kill training. `wandb` in this venv is an EMPTY namespace package
        # (wandb.__file__ is None, no .init), so this call raised AttributeError on rank 0, which
        # tore down the TCPStore and made every other rank die in DDP() with an unrelated-looking
        # NCCL rendezvous error. Cost one job and a confusing traceback; not again.
        try:
            wandb_run = wandb.init(project="HEIST", name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"[warn] wandb disabled ({type(e).__name__}: {e}); continuing without it", flush=True)
            wandb_run = None

    try:
        device = torch.device(f"cuda:{rank}")

        # Load dataset
        all_files = sorted(glob(args.data_dir + "*/*"))
        train_idx, val_idx = train_test_split(np.arange(len(all_files)), test_size=0.2, random_state = 42)

        # Shard in ONE place only. Previously ranks split FILES here AND partitions inside each
        # file (DistributedSampler), so each rank saw ~1/world_size^2 of the cells and the union
        # covered only ~1/world_size of the corpus per epoch. Unequal step counts also made DDP
        # match allreduces across different steps.
        train_idx_for_rank = train_idx
        model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim,
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe,
                            marker_embedding=args.marker_embedding,
                            num_markers=args.num_markers,
                            niche_attention=args.niche_attention,
                            num_niches=args.num_niches,
                            niche_heads=args.niche_heads,
                            niche_detach=args.niche_detach,
                            niche_feat=args.niche_feat,
                            rank_pe_fixed=args.rank_pe_fixed).to(device)
        model.apply(initialize_weights)
        best_val_loss = float('inf')
        start_epoch = 0
        # Resume: without this a 48h wall or a REQUEUE loses everything, AND best_val_loss resets
        # to inf so the restarted run's first (random-init) epoch overwrites the good checkpoint.
        if args.resume and os.path.exists(args.resume):
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            best_val_loss = ck.get('best_val_loss', float('inf'))
            start_epoch = ck.get('epoch', -1) + 1
            print(f'Rank {rank}: resumed {args.resume} @ epoch {start_epoch} best={best_val_loss:.4f}')

        summary(model)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        decoder = GIN_decoder(args.output_dim, args.output_dim, nonneg_recon=args.nonneg_recon).to(device)
        decoder.apply(initialize_weights)
        decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        optimizer = optim.AdamW(list(model.parameters())+list(decoder.parameters()), 
                                lr=args.lr, weight_decay=args.wd)
        if args.resume and os.path.exists(args.resume):
            _ck = torch.load(args.resume, map_location=device, weights_only=False)
            decoder.module.load_state_dict(_ck['decoder_state_dict'])
            optimizer.load_state_dict(_ck['optimizer_state_dict'])
        for epoch in range(start_epoch, args.num_epochs):
            start_time = time.time()
            total_loss = 0
            nonfinite_steps = 0
            total_steps = 0
            niche_cut_sum, niche_ortho_sum, niche_ent_sum, niche_sharp_sum, niche_n = 0.0, 0.0, 0.0, 0.0, 0
            for graph_idx in tqdm(train_idx_for_rank, desc=f"Rank {rank} | Epoch {epoch}"):
                graphs = torch.load(all_files[graph_idx], weights_only=False)
                try:
                    dataloader = create_dataloader_ddp(graphs, args.batch_size, rank, world_size)
                except Exception as e:
                    logging.error("Dataloader creation failed.")
                    logging.error("Exception: %s", str(e))
                    continue

                for high_level_subgraph, low_level_batch, batch_idx in (dataloader):
                    optimizer.zero_grad()
                    high_level_subgraph = high_level_subgraph.to(device)
                    low_level_batch = low_level_batch.to(device)
                    low_level_batch.batch_idx = batch_idx.to(device)
                    high_true = high_level_subgraph.X
                    low_true = low_level_batch.X
                    high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.2).long().to(device)
                    present = torch.where(low_level_batch.X.T[0])[0]
                    masked = present[torch.randint(0, len(present), (int(len(present)*0.2), 1))]
                    low_mask = torch.ones_like(low_level_batch.X).long().to(device)
                    low_mask[masked] = 0
                    # low_mask = 1 - torch.bernoulli(torch.ones(low_level_batch.num_nodes, 1)*0.2).long().to(device)

                    high_emb, low_emb, niche_S = model(high_level_subgraph, low_level_batch,
                                                       high_mask, low_mask, return_niche=True)
                    contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 16)
                    _high_emb = high_emb * high_mask
                    _low_emb = low_emb * low_mask
                    decoded_high, decoded_low, alpha_sigmoid = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                    if((1 - high_mask).sum()):
                        recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_true.float()*(1-high_mask), reduction='sum')/high_mask.sum() + F.mse_loss(decoded_low*(1-low_mask), low_true.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                    else:
                        recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_true.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                    
                    contrastive_component = args.w_contrastive * contrastive_loss
                    recon_component = args.w_recon * recon_loss
                    orthogonal_loss = 0.1 * (F.normalize(_high_emb.T) @ F.normalize(_high_emb) - torch.eye(_high_emb.shape[1]).to(device)).square().mean() \
                            + 0.1 * (F.normalize(_low_emb.T) @ F.normalize(_low_emb) - torch.eye(_low_emb.shape[1]).to(device)).square().mean()

                    loss = contrastive_component + recon_component + orthogonal_loss

                    if niche_S is not None:
                        cut = niche_cut_loss(niche_S, high_level_subgraph.edge_index)
                        ortho = niche_ortho_loss(niche_S)
                        bal = niche_balance_loss(niche_S)
                        loss = loss + args.w_niche_cut * cut + args.w_niche_ortho * ortho \
                                    + args.w_niche_balance * bal
                        niche_cut_sum += cut.item()
                        niche_ortho_sum += ortho.item()
                        niche_ent_sum += -bal.item()
                        niche_sharp_sum += niche_S.max(-1).values.mean().item()
                        niche_n += 1

                    if torch.isnan(loss) or not torch.isfinite(loss):
                        print(f"Rank {rank}: NaN loss encountered, skipping")
                        print(f"Contra: {contrastive_loss.item():.4f}, Recon: {recon_loss.item():.4f}")#, Ortho: {orthogonal_loss.item():.4f}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue                       
                    loss.backward()
                    total_steps += 1
                    if args.clip_grad > 0:
                        gn_m = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                        gn_d = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad)
                        if not (torch.isfinite(gn_m) and torch.isfinite(gn_d)):
                            optimizer.zero_grad(set_to_none=True)
                            nonfinite_steps += 1
                            continue
                    optimizer.step()
                    total_loss += loss.item()
                    del high_level_subgraph, low_level_batch, loss, high_emb, low_emb, contrastive_loss, high_mask, low_mask, _high_emb, _low_emb, recon_loss, niche_S
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()

            end_time = time.time()
            print(f"Rank {rank} - Epoch: {epoch + 1}, Loss: {total_loss}, Time = {(end_time-start_time)//3600} hours")
            if nonfinite_steps:
                # Rate, not just count: a burst the guard absorbs looks very different from a model
                # that has stopped training. Watch this trend across epochs.
                print(f"Rank {rank} - Epoch: {epoch + 1}, skipped {nonfinite_steps}/{total_steps} "
                      f"non-finite-grad steps ({nonfinite_steps / max(total_steps, 1):.1%})")
            # Fail loudly instead of no-oping. Once the weights are NaN every batch hits the skip path,
            # so training silently does nothing: job 23872253 ran 23 h that way after dying in epoch 1.
            _bad = [n for n, p in model.module.named_parameters() if not torch.isfinite(p).all()]
            if _bad:
                raise RuntimeError(
                    f"Rank {rank}: model weights went non-finite in epoch {epoch + 1} "
                    f"({len(_bad)} tensors, e.g. {_bad[:3]}). Aborting rather than burning walltime.")
            if niche_n:
                print(f"Rank {rank} - Epoch: {epoch + 1}, niche_cut={niche_cut_sum/niche_n:.4f}, "
                      f"niche_ortho={niche_ortho_sum/niche_n:.4f}, "
                      f"niche_usage_entropy={niche_ent_sum/niche_n:.4f}, "
                      f"niche_sharpness={niche_sharp_sum/niche_n:.4f} "
                      f"(sharpness ~ 1/K={1.0/args.num_niches:.4f} means the head learned nothing)")
            
            # dist.barrier()
            if rank == 0:  # Save only from rank 0
                val_loss = validate(rank, world_size, model, decoder, val_idx, all_files, args)
                print(f"[Validation] Epoch {epoch+1} | Rank {rank} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

                if wandb_run is not None:
                    _log = {"epoch": epoch + 1, "train_loss": total_loss, "val_loss": val_loss}
                    if niche_n:
                        _log["niche_cut"] = niche_cut_sum / niche_n
                        _log["niche_ortho"] = niche_ortho_sum / niche_n
                        _log["niche_usage_entropy"] = niche_ent_sum / niche_n
                        _log["niche_sharpness"] = niche_sharp_sum / niche_n
                    wandb_run.log(_log, step=epoch + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'decoder_state_dict': decoder.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'args': args,
                        'epoch': epoch,
                        'best_val_loss':best_val_loss
                    }, args.save_path)
                    print(f"✅ New best model saved at Epoch {epoch+1} with Val Loss {val_loss:.4f}")
                _last = args.save_path.replace('.pth', '_last.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                            'decoder_state_dict': decoder.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'args': args, 'best_val_loss': best_val_loss}, _last + '.tmp')
                os.replace(_last + '.tmp', _last)
                    
            # dist.barrier()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup()

if __name__ == "__main__":
    parser = ArgumentParser(description="HEIST")
    parser.add_argument('--data_dir', type=str, default='data/pretraining/', help="Root with one subfolder of .pt chunks per source")
    parser.add_argument('--pe_dim', type=int, default=128, help="Positional encoding dim")
    parser.add_argument('--init_dim', type=int, default=128, help="Input projection dim")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dim of the graph layers")
    parser.add_argument('--output_dim', type=int, default=128, help="Embedding dim")
    parser.add_argument('--blending', action='store_true', help="Unused; kept for old launch scripts")
    parser.add_argument('--pe', action='store_true', help="Use positional encodings")
    parser.add_argument('--cross_message_passing', action='store_true', help="Cell-gene cross message passing")
    parser.add_argument('--num_layers', type=int, default=10, help="Number of graph layers")
    parser.add_argument('--num_heads', type=int, default=8, help="Transformer heads")
    parser.add_argument('--batch_size', type=int, default=50, help="Cells per METIS partition")
    parser.add_argument('--graph_idx', type=int, default=0, help="Unused")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=3e-3, help="Weight decay")
    parser.add_argument('--clip_grad', type=float, default=0.0, help="Max grad norm (0 = off)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index")
    parser.add_argument('--save_path', type=str, default='saved_models/HEIST.pth', help="Best checkpoint path")
    parser.add_argument('--rank_pe_fixed', action='store_true', help="Standard-frequency gene rank PE")
    parser.add_argument('--marker_embedding', action='store_true', help="Learned per-gene embedding")
    parser.add_argument('--num_markers', type=int, default=None, help="Gene vocab size for --marker_embedding")
    parser.add_argument('--nonneg_recon', action='store_true', help="abs() the gene reconstruction")
    parser.add_argument('--w_contrastive', type=float, default=0.1, help="Contrastive loss weight")
    parser.add_argument('--w_recon', type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument('--niche_attention', action='store_true', help="Enable the niche head")
    parser.add_argument('--num_niches', type=int, default=16, help="Number of niches")
    parser.add_argument('--niche_heads', type=int, default=4, help="Neighbourhood bandwidths")
    parser.add_argument('--niche_feat', type=str, default='low', choices=['low', 'high', 'concat', 'markers'],
                        help="Niche input feature ('markers' transfers across regions)")
    parser.add_argument('--niche_detach', action='store_true', help="Detach the niche head from the backbone")
    parser.add_argument('--w_niche_cut', type=float, default=0.1, help="MinCut coherence weight")
    parser.add_argument('--w_niche_ortho', type=float, default=0.5, help="MinCut orthogonality weight (keep > 0)")
    parser.add_argument('--w_niche_balance', type=float, default=0.5, help="Anti-collapse weight (keep > 0)")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--wandb', action='store_true', help="wandb logging (rank 0)")
    parser.add_argument('--run_name', type=str, default=None, help="wandb run name")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)