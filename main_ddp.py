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
from utils.dataloader import create_dataloader_ddp
from model.model import GraphEncoder, GIN_decoder
from model.loss import contrastive_loss_cell, mae_loss_cell
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

torch.autograd.set_detect_anomaly(True)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  
        return s.getsockname()[1] 

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # Set a free port for communication
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

    with torch.no_grad():
        for graph_idx in tqdm(val_idx):
            graphs = torch.load(all_files[graph_idx], weights_only=False)
            try:
                dataloader = create_dataloader_ddp(graphs, args.batch_size, rank, world_size)
            except Exception as e:
                logging.error("Dataloader creation failed.")
                logging.error("Exception: %s", str(e))
                continue
            for high_level_subgraph, low_level_batch, batch_idx in dataloader:
                high_level_subgraph = high_level_subgraph.to(device)
                low_level_batch = low_level_batch.to(device)
                low_level_batch.batch_idx = batch_idx.to(device)

                high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.2).long().to(device)
                present = torch.where(low_level_batch.X.T)[0]
                masked = present[torch.randint(0, len(present), (int(len(present)*0.2), 1))]
                low_mask = torch.ones_like(low_level_batch.X).long().to(device)
                low_mask[masked] = 0
                high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)

                contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 10)

                _high_emb = high_emb * high_mask
                _low_emb = low_emb * low_mask
                decoded_high, decoded_low, alpha_sigmoid = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                if((1 - high_mask).sum()):
                    recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_level_subgraph.X.float()*(1-high_mask), reduction='sum')/high_mask.sum() + F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                else:
                    recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                                
                #orthogonal_loss = 0.1 * (F.normalize(_high_emb.T) @ F.normalize(_high_emb) - torch.eye(_high_emb.shape[1]).to(device)).square().mean() \
                #               + 0.1 * (F.normalize(_low_emb.T) @ F.normalize(_low_emb) - torch.eye(_low_emb.shape[1]).to(device)).square().mean()

                loss = alpha_sigmoid * contrastive_loss + (1 - alpha_sigmoid) * recon_loss #+ orthogonal_loss
                # loss = contrastive_loss
                if torch.isnan(loss) or not torch.isfinite(loss):
                    print(f"Rank {rank}: NaN loss encountered, skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                total_val_loss += loss

    return total_val_loss.item()

def train(rank, world_size, args):
    """Main training loop for DDP."""
    setup(rank, world_size)
    model_path = f"saved_models/final_model_with_custom_layer_full.pth"

    try:
        device = torch.device(f"cuda:{rank}")

        # Load dataset
        all_files = sorted(glob(args.data_dir + "*/*"))
        train_idx, val_idx = train_test_split(np.arange(len(all_files)), test_size=0.2, random_state = 42)

        train_idx_for_rank = train_idx[rank::world_size]#[121:]
        model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.blending).to(device)
        model.apply(initialize_weights)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_loss = float('inf')
        # best_val_loss= checkpoint['best_val_loss']

        summary(model)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        decoder = GIN_decoder(args.output_dim, args.output_dim).to(device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.apply(initialize_weights)
        decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        optimizer = optim.AdamW(list(model.parameters())+list(decoder.parameters()), 
                                lr=args.lr, weight_decay=args.wd)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for epoch in range(args.num_epochs):
            start_time = time.time()
            total_loss = 0
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

                    high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)
                    contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 16)
                    _high_emb = high_emb * high_mask
                    _low_emb = low_emb * low_mask
                    decoded_high, decoded_low, alpha_sigmoid = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                    if((1 - high_mask).sum()):
                        recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_true.float()*(1-high_mask), reduction='sum')/high_mask.sum() + F.mse_loss(decoded_low*(1-low_mask), low_true.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                    else:
                        recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_true.float()*(1-low_mask), reduction='sum')/low_mask.sum()
                    
                    contrastive_component = alpha_sigmoid * contrastive_loss
                    recon_component = (1 - alpha_sigmoid) * recon_loss
                    orthogonal_loss = 0.1 * (F.normalize(_high_emb.T) @ F.normalize(_high_emb) - torch.eye(_high_emb.shape[1]).to(device)).square().mean() \
                            + 0.1 * (F.normalize(_low_emb.T) @ F.normalize(_low_emb) - torch.eye(_low_emb.shape[1]).to(device)).square().mean()

                    loss = contrastive_component + recon_component + orthogonal_loss
                    if torch.isnan(loss) or not torch.isfinite(loss):
                        print(f"Rank {rank}: NaN loss encountered, skipping")
                        print(f"Contra: {contrastive_loss.item():.4f}, Recon: {recon_loss.item():.4f}")#, Ortho: {orthogonal_loss.item():.4f}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue                       
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    # logging.error(f"Rank {rank}: Loss = {loss.item()}, Total loss = {total_loss}")
                    del high_level_subgraph, low_level_batch, loss, high_emb, low_emb, contrastive_loss, high_mask, low_mask, _high_emb, _low_emb, recon_loss
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()

            end_time = time.time()
            print(f"Rank {rank} - Epoch: {epoch + 1}, Loss: {total_loss}, Time = {(end_time-start_time)//3600} hours")
            
            # dist.barrier()
            if rank == 0:  # Save only from rank 0
                val_loss = validate(rank, world_size, model, decoder, val_idx, all_files, args)
                print(f"[Validation] Epoch {epoch+1} | Rank {rank} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = f"saved_models/HEIST.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'decoder_state_dict': decoder.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'args': args,
                        'epoch': epoch,
                        'best_val_loss':best_val_loss
                    }, model_path)
                    print(f"✅ New best model saved at Epoch {epoch+1} with Val Loss {val_loss:.4f}")
                    
            # dist.barrier()
    finally:
        cleanup()

if __name__ == "__main__":
    parser = ArgumentParser(description="HEIST")
    parser.add_argument('--data_dir', type=str, default = 'data/pretraining/', help="Directory where the raw data is stored")
    parser.add_argument('--pe_dim', type=int, default= 128, help="Dimension of the positional encodings")
    parser.add_argument('--init_dim', type=int, default= 128, help="Hidden dim for the MLP")
    parser.add_argument('--hidden_dim', type=int, default= 128, help="Hidden dim for the MLP")
    parser.add_argument('--output_dim', type=int, default= 128, help="Output dim for the MLP")
    parser.add_argument('--blending', action='store_true')
    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--cross_message_passing', action='store_true')
    parser.add_argument('--num_layers', type=int, default= 10, help="Number of MLP layers")
    parser.add_argument('--num_heads', type=int, default= 8, help="Number of transformer heads")
    parser.add_argument('--batch_size', type=int, default= 50, help="Batch size")
    parser.add_argument('--graph_idx', type=int, default= 0, help="Batch size")
    parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
    parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
    parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
    parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
