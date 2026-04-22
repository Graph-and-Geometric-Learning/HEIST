import gc
import logging
import os
import random
import time
import warnings
import wandb
from argparse import ArgumentParser
from datetime import timedelta
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from tqdm import tqdm

from model.loss import contrastive_loss_cell
from model.model import GIN_decoder, GraphEncoder
from utils.dataloader import create_dataloader_ddp


warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "2"
torch.autograd.set_detect_anomaly(False)  # Disable in production - saves memory

# =============================================================================
# Constants
# =============================================================================
MASK_RATIO = 0.2
NUM_NEGATIVES_TRAIN = 16
NUM_NEGATIVES_VAL = 10
ORTHOGONAL_WEIGHT = 0.1
NCCL_TIMEOUT_MINUTES = 180


# =============================================================================
# Setup & Initialization
# =============================================================================
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_IB_DISABLE"] = "1"
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=NCCL_TIMEOUT_MINUTES),
        rank=rank,
        world_size=world_size
    )
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


# =============================================================================
# Masking & Loss Helpers
# =============================================================================
def create_masks(high_level_subgraph, low_level_batch, device):
    high_mask = 1 - torch.bernoulli(
        torch.ones(high_level_subgraph.num_nodes, 1) * MASK_RATIO
    ).long().to(device)

    present = torch.where(low_level_batch.X.T[0])[0]
    num_to_mask = int(len(present) * MASK_RATIO)
    masked_indices = present[torch.randint(0, len(present), (num_to_mask, 1))]
    low_mask = torch.ones_like(low_level_batch.X).long().to(device)
    low_mask[masked_indices] = 0

    return high_mask, low_mask


def compute_reconstruction_loss(decoded_high, decoded_low, high_true, low_true,
                                high_mask, low_mask):
    inv_high_mask = 1 - high_mask
    inv_low_mask = 1 - low_mask

    low_recon = F.mse_loss(
        decoded_low * inv_low_mask,
        low_true.float() * inv_low_mask,
        reduction='mean'
    )

    if inv_high_mask.sum() > 0:
        high_recon = F.mse_loss(
            decoded_high * inv_high_mask,
            high_true.float() * inv_high_mask,
            reduction='mean'
        )
        return high_recon + low_recon

    return low_recon#.clamp(recon_loss, max=1e3)


def compute_orthogonal_loss(high_emb, low_emb, device):
    """Compute orthogonality regularization loss."""
    high_norm = F.normalize(high_emb.T)
    low_norm = F.normalize(low_emb.T)

    high_ortho = (high_norm @ high_norm.T - torch.eye(high_emb.shape[1], device=device)).square().mean()
    low_ortho = (low_norm @ low_norm.T - torch.eye(low_emb.shape[1], device=device)).square().mean()

    return ORTHOGONAL_WEIGHT * (high_ortho + low_ortho)


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def check_loss_valid(loss, rank):
    if torch.isnan(loss) or not torch.isfinite(loss):
        print(f"Rank {rank}: NaN/Inf loss encountered, skipping batch")
        clear_memory()
        return False
    return True


def load_checkpoint(checkpoint_path, model, decoder, optimizer, device, rank):
    """Load checkpoint and return the starting epoch and best validation loss."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.module.load_state_dict(checkpoint['model_state_dict'])
    decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    if rank == 0:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        print(f"Resuming from epoch {start_epoch + 1} with best val loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


# =============================================================================
# Validation
# =============================================================================
def validate(rank, world_size, model, decoder, val_idx, all_files, args):
    device = torch.device(f"cuda:{rank}")
    model.eval()
    decoder.eval()
    total_val_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for graph_idx in tqdm(val_idx, desc=f"Validating"):
            graphs = torch.load(all_files[graph_idx], weights_only=False)

            try:
                dataloader = create_dataloader_ddp(graphs, args.batch_size, rank, world_size)
            except Exception as e:
                logging.error("Dataloader creation failed: %s", str(e))
                continue

            for high_level_subgraph, low_level_batch, batch_idx in dataloader:
                high_level_subgraph = high_level_subgraph.to(device)
                low_level_batch = low_level_batch.to(device)
                low_level_batch.batch_idx = batch_idx.to(device)

                high_mask, low_mask = create_masks(high_level_subgraph, low_level_batch, device)
                high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)

                contrastive_loss = contrastive_loss_cell(
                    low_level_batch.cell_type, high_emb, low_level_batch, low_emb,
                    NUM_NEGATIVES_VAL
                )

                masked_high_emb = high_emb * high_mask
                masked_low_emb = low_emb * low_mask
                decoded_high, decoded_low, alpha = decoder(
                    masked_high_emb, high_level_subgraph, masked_low_emb, low_level_batch
                )

                recon_loss = compute_reconstruction_loss(
                    decoded_high, decoded_low,
                    high_level_subgraph.X, low_level_batch.X,
                    high_mask, low_mask
                )

                loss = alpha * contrastive_loss + (1 - alpha) * recon_loss

                if not check_loss_valid(loss, rank):
                    continue

                total_val_loss += loss

    return total_val_loss.item()


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(rank, model, decoder, optimizer, train_idx, all_files, args, device, epoch):
    model.train()
    decoder.train()
    total_loss = 0.0

    for graph_idx in tqdm(train_idx, desc=f"Rank {rank}"):
        graphs = torch.load(all_files[graph_idx], weights_only=False)

        try:
            dataloader = create_dataloader_ddp(graphs, args.batch_size, rank, args.world_size)
        except Exception as e:
            logging.error("Dataloader creation failed: %s", str(e))
            continue

        for high_level_subgraph, low_level_batch, batch_idx in dataloader:
            optimizer.zero_grad()

            high_level_subgraph = high_level_subgraph.to(device)
            low_level_batch = low_level_batch.to(device)
            low_level_batch.batch_idx = batch_idx.to(device)

            high_true = high_level_subgraph.X
            low_true = low_level_batch.X

            high_mask, low_mask = create_masks(high_level_subgraph, low_level_batch, device)
            high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)

            contrastive_loss = contrastive_loss_cell(
                low_level_batch.cell_type, high_emb, low_level_batch, low_emb,
                NUM_NEGATIVES_TRAIN
            )

            masked_high_emb = high_emb * high_mask
            masked_low_emb = low_emb * low_mask
            decoded_high, decoded_low, alpha = decoder(
                masked_high_emb, high_level_subgraph, masked_low_emb, low_level_batch
            )

            recon_loss = 1e-6 * compute_reconstruction_loss(
                decoded_high, decoded_low, high_true, low_true, high_mask, low_mask
            )
            orthogonal_loss = compute_orthogonal_loss(masked_high_emb, masked_low_emb, device)

            loss = alpha * contrastive_loss + (1-alpha) * recon_loss + orthogonal_loss

            if not check_loss_valid(loss, rank):
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Per-step wandb logging (rank 0 only)
            if args.wandb and rank == 0:
                wandb.log({
                    'epoch': epoch + 1,
                    'step_loss': loss.item(),
                    'contrastive_loss': contrastive_loss.item(),
                    'reconstruction_loss': recon_loss.item(),
                    'orthogonal_loss': orthogonal_loss.item(),
                    'alpha': alpha.item(),
                })

        clear_memory()

    return total_loss


def save_checkpoint(model, decoder, optimizer, epoch, val_loss, args):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'decoder_state_dict': decoder.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': val_loss,
        'args': args,
    }
    torch.save(checkpoint, args.save_path)
    print(f"New best model saved at Epoch {epoch + 1} with Val Loss {val_loss:.4f}")


def train(rank, world_size, args):
    """Main training loop for DDP."""
    setup(rank, world_size)
    args.world_size = world_size

    try:
        device = torch.device(f"cuda:{rank}")

        # Load and split dataset
        all_files = sorted(glob(args.data_dir + "*/*"))
        train_idx, val_idx = train_test_split(
            np.arange(len(all_files)), test_size=0.2, random_state=42
        )
        train_idx_for_rank = train_idx[rank::world_size]

        # Initialize model
        model = GraphEncoder(
            args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim,
            args.num_layers, args.num_heads, args.cross_message_passing, args.pe
        ).to(device)
        model.apply(initialize_weights)

        if rank == 0:
            summary(model)

        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Initialize decoder
        decoder = GIN_decoder(args.output_dim, args.output_dim).to(device)
        decoder.apply(initialize_weights)
        decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Initialize optimizer
        optimizer = optim.AdamW(
            list(model.parameters()) + list(decoder.parameters()),
            lr=args.lr,
            weight_decay=args.wd
        )

        best_val_loss = float('inf')
        start_epoch = 0

        # Load checkpoint if provided
        if args.checkpoint is not None:
            start_epoch, best_val_loss = load_checkpoint(
                args.checkpoint, model, decoder, optimizer, device, rank
            )

        # Initialize wandb on rank 0
        if args.wandb and rank == 0:
            wandb.init(
                project="HEIST",
                name=args.run_name,
                config=vars(args)
            )

        # Training loop
        for epoch in range(start_epoch, args.num_epochs):
            start_time = time.time()

            total_loss = train_one_epoch(
                rank, model, decoder, optimizer, train_idx_for_rank[:2], all_files, args, device, epoch
            )

            elapsed = time.time() - start_time
            print(f"Rank {rank} - Epoch {epoch + 1}: Loss={total_loss:.4f}, Time={elapsed / 3600:.2f}h")

            # Synchronize all ranks before validation
            dist.barrier()

            # Validation and checkpointing (rank 0 only)
            if rank == 0:
                val_loss = validate(rank, world_size, model, decoder, val_idx, all_files, args)
                print(f"[Validation] Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

                # Log to wandb
                if args.wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': total_loss,
                        'val_loss': val_loss,
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, decoder, optimizer, epoch, best_val_loss, args)

            # Synchronize all ranks after validation
            dist.barrier()

        # Finish wandb run
        if args.wandb and rank == 0:
            wandb.finish()

    finally:
        cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="HEIST - Hierarchical Graph Neural Network Training")

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/pretraining/',
                        help="Directory containing training data")
    parser.add_argument('--save_path', type=str, default='saved_models/HEIST.pth',
                        help="Path to save model checkpoint")

    # Model architecture
    parser.add_argument('--pe_dim', type=int, default=128,
                        help="Positional encoding dimension")
    parser.add_argument('--init_dim', type=int, default=128,
                        help="Initial hidden dimension")
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help="Hidden layer dimension")
    parser.add_argument('--output_dim', type=int, default=128,
                        help="Output embedding dimension")
    parser.add_argument('--num_layers', type=int, default=10,
                        help="Number of GNN layers")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--pe', action='store_true',
                        help="Use positional encodings")
    parser.add_argument('--cross_message_passing', action='store_true',
                        help="Enable cross-level message passing")

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=50,
                        help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--wd', type=float, default=3e-3,
                        help="Weight decay")
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help="Enable wandb logging")
    parser.add_argument('--run_name', type=str, default=None,
                        help="Wandb run name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
