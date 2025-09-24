import warnings
warnings.filterwarnings('ignore')

import torch
import time
from utils.dataloader import create_dataloader
from model.model import GraphEncoder, GIN_decoder
import torch.optim as optim
from torch_geometric.data import DataLoader as DataLoader_PyG
import numpy as np
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
import json
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torchinfo import summary
import sys
from glob import glob
from argparse import ArgumentParser
import lovely_tensors as lt
lt.monkey_patch()
parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, default = 'melanoma', help="Directory where the raw data is stored")
parser.add_argument('--model_name', type=str, default = 'HEIST', help="Model name")
parser.add_argument('--pe_dim', type=int, default= 128, help="Dimension of the positional encodings")
parser.add_argument('--init_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default= 128, help="Output dim for the MLP")
parser.add_argument('--blending', action='store_true')
parser.add_argument('--fine_tune', action='store_true')
parser.add_argument('--anchor_pe', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--cross_message_passing', action='store_true')
parser.add_argument('--num_layers', type=int, default= 10, help="Number of MLP layers")
parser.add_argument('--num_heads', type=int, default= 8, help="Number of transformer heads")
parser.add_argument('--batch_size', type=int, default= 512, help="Batch size")
parser.add_argument('--graph_idx', type=int, default= 0, help="Batch size")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
args = parser.parse_args()




def PearsonCorr1d(y_true, y_pred):
    assert len(y_true.shape) == 1
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c) / torch.sqrt(torch.sum(y_true_c * y_true_c)) 
                         / torch.sqrt(torch.sum(y_pred_c * y_pred_c)))
    return pearson

if __name__ == '__main__':
    if(args.data_name=='melanoma'):
        graphs = torch.load("data/melanoma_preprocessed/R10C2.pt", weights_only = False)
        mask = [12, 3, 4, 8]
    else:
        graphs = torch.load("SCGFM/data/placenta/02021424.pt", weights_only = False)
        mask = [186, 124, 8, 100]
    print("================================")
    print("Gene imputation (Zero-shot)")
    print("================================")
    model_path = f"saved_models/{args.model_name}.pth"
    checkpoint = torch.load(model_path)
    fine_tune = args.fine_tune
    args = checkpoint['args']
    args.fine_tune = fine_tune

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    args.batch_size = 1000
    model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.anchor_pe, args.blending).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder = GIN_decoder(args.output_dim, args.output_dim).to(args.device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model = model.eval()
    decoder = decoder.eval()
    criterion = torch.nn.L1Loss()
    model.eval()
    corr_over_runs = []
    dataloader = create_dataloader(graphs, args.batch_size, False)
    for run in range(1):
        true = []
        pred = []
        with torch.no_grad():
            for high_level_subgraph, low_level_batch, batch_idx in (dataloader):
                # import pdb; pdb.set_trace()
                high_level_subgraph = high_level_subgraph.to(args.device)
                low_level_batch = low_level_batch.to(args.device)
                low_level_batch.batch_idx = batch_idx.to(args.device)
                cells, genes = high_level_subgraph.num_nodes, low_level_batch.num_nodes//high_level_subgraph.num_nodes
                low_true = low_level_batch.X#.view(cells, genes)


                present = torch.where(low_level_batch.X.T[0])[0]
                masked = present[torch.randint(0, len(present), (int(len(present)*0.1), 1))]
                mask = torch.ones_like(low_level_batch.X).long().to(args.device)
                mask[masked] = 0
                        
                high_emb, low_emb = model.encode(high_level_subgraph, low_level_batch)
                _, imputed_gene, _ = decoder(high_emb, high_level_subgraph, low_emb, low_level_batch)

                true.append((low_true*(1 - mask)).T[0])
                pred.append((imputed_gene*(1 - mask)).T[0])
        end = time.time()
        corr_over_runs.append(PearsonCorr1d(torch.log1p(torch.cat(true)), torch.log1p(torch.cat(pred))).item())
        print(f"{corr_over_runs[-1]:.3f} $\\pm$ 0.00")