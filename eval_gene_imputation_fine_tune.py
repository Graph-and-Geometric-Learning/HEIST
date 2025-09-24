import warnings
warnings.filterwarnings('ignore')

from eval_mlp import train
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
parser.add_argument('--num_epochs', type=int, default= 100, help="Number of epochs")
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
        graphs = torch.load("data/placenta/02021424.pt", weights_only = False)
        mask = [186, 124, 8, 100]
    print("================================")
    print("Gene imputation (Fine-tuned)")
    print("================================")
    model_path = f"saved_models/final_model_with_custom_layer_full.pth"
    checkpoint = torch.load(model_path)
    fine_tune = args.fine_tune
    args = checkpoint['args']
    args.fine_tune = fine_tune

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    args.batch_size = 1000
    args.num_epochs = 150
    model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.anchor_pe, args.blending).to(args.device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Optional, but recommended if encoder includes BatchNorm/Dropout

    # Optimizer ONLY for decoder
    summary(model)
    print(f"Number of cells: {graphs[0].num_nodes}, Number of genes: {graphs[1].num_nodes}")
    criterion = torch.nn.MSELoss()
    
    cell_indices = np.arange(graphs[0].num_nodes)
    train_cells, temp_cells = train_test_split(cell_indices, test_size=0.2, random_state=42)
    val_cells, test_cells = train_test_split(temp_cells, test_size=0.5, random_state=42)

    train_mask = torch.zeros(graphs[0].num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(graphs[0].num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(graphs[0].num_nodes, dtype=torch.bool)
    train_mask[train_cells] = True
    val_mask[val_cells] = True
    test_mask[test_cells] = True

    graphs[0].train_mask = train_mask
    graphs[0].val_mask = val_mask
    graphs[0].test_mask = test_mask

    print("Extracting representations:")
    model.eval()
    corr_over_runs = []
    dataloader = create_dataloader(graphs, args.batch_size, False)
    for run in range(5):
        decoder = GIN_decoder(args.output_dim, args.output_dim).to(args.device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.wd)

        best_corr = -1
        best_test_corr = -1
        for epoch in range(args.num_epochs):
            corr_val = []
            corr_test = []
            for high_level_subgraph, low_level_batch, batch_idx in (dataloader):
                train_cells = high_level_subgraph.train_mask.nonzero(as_tuple=True)[0]
                val_cells = high_level_subgraph.val_mask.nonzero(as_tuple=True)[0]
                test_cells = high_level_subgraph.test_mask.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    high_level_subgraph = high_level_subgraph.to(args.device)
                    
                    low_level_batch = low_level_batch.to(args.device)
                    low_level_batch.batch_idx = batch_idx.to(args.device)
                    
                    cells, genes = high_level_subgraph.num_nodes, low_level_batch.num_nodes//high_level_subgraph.num_nodes

                    low_level_batch.X = low_level_batch.X.view(cells, genes)
                    low_true = low_level_batch.X[:, mask]
                    low_level_batch.X[:, mask] = 0

                    low_level_batch_val = low_level_batch.clone()
                    low_level_batch_test = low_level_batch.clone()
                    
                    
                    low_level_batch.X[val_cells] = 0
                    low_level_batch.X[test_cells] = 0
                    
                    low_level_batch_val.X[train_cells] = 0
                    low_level_batch_val.X[test_cells] = 0
                                
                    low_level_batch_test.X[train_cells] = 0
                    low_level_batch_test.X[val_cells] = 0
                    
                    low_level_batch.X = low_level_batch.X.ravel().unsqueeze(1)
                    low_level_batch_test.X = low_level_batch_test.X.ravel().unsqueeze(1)
                    low_level_batch_val.X = low_level_batch_val.X.ravel().unsqueeze(1)
                    

                    high_emb_train, low_emb_train = model.encode(high_level_subgraph, low_level_batch)
                    high_emb_val, low_emb_val = model.encode(high_level_subgraph, low_level_batch_val)
                    high_emb_test, low_emb_test = model.encode(high_level_subgraph, low_level_batch_test)
                
                optimizer.zero_grad()
                _, imputed_gene, _ = decoder(high_emb_train, high_level_subgraph, low_emb_train, low_level_batch)
                preds = imputed_gene.view(cells, genes)[:, mask]
                loss = F.mse_loss(preds[train_cells].float(), low_true[train_cells].float())
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.eval()
                    _, imputed_gene, _ = decoder(high_emb_val, high_level_subgraph, low_emb_val, low_level_batch_val)
                    preds = imputed_gene.view(cells, genes)[:, mask]
                    # corr = PearsonCorr1d((preds[val_cells].ravel()), low_true[val_cells].ravel()).item()
                    corr = [PearsonCorr1d(preds[i], low_true[i]).item() for i in val_cells]
                    corr: float = np.array(corr)
                    corr = corr[~np.isnan(corr)].mean()
                    corr_val.append(corr)
                    # corr: float = sum(corr)/len(corr)
            corr_val = sum(corr_val)/len(corr_val)
            for high_level_subgraph, low_level_batch, batch_idx in (dataloader):
                train_cells = high_level_subgraph.train_mask.nonzero(as_tuple=True)[0]
                val_cells = high_level_subgraph.val_mask.nonzero(as_tuple=True)[0]
                test_cells = high_level_subgraph.test_mask.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    high_level_subgraph = high_level_subgraph.to(args.device)
                    
                    low_level_batch = low_level_batch.to(args.device)
                    low_level_batch.batch_idx = batch_idx.to(args.device)
                    
                    cells, genes = high_level_subgraph.num_nodes, low_level_batch.num_nodes//high_level_subgraph.num_nodes

                    low_level_batch.X = low_level_batch.X.view(cells, genes)
                    low_true = low_level_batch.X[:, mask]
                    low_level_batch.X[:, mask] = 0

                    low_level_batch_val = low_level_batch.clone()
                    low_level_batch_test = low_level_batch.clone()
                    
                    
                    low_level_batch.X[val_cells] = 0
                    low_level_batch.X[test_cells] = 0
                    
                    low_level_batch_val.X[train_cells] = 0
                    low_level_batch_val.X[test_cells] = 0
                                
                    low_level_batch_test.X[train_cells] = 0
                    low_level_batch_test.X[val_cells] = 0
                    
                    low_level_batch.X = low_level_batch.X.ravel().unsqueeze(1)
                    low_level_batch_test.X = low_level_batch_test.X.ravel().unsqueeze(1)
                    low_level_batch_val.X = low_level_batch_val.X.ravel().unsqueeze(1)
                    

                    high_emb_train, low_emb_train = model.encode(high_level_subgraph, low_level_batch)
                    high_emb_val, low_emb_val = model.encode(high_level_subgraph, low_level_batch_val)
                    high_emb_test, low_emb_test = model.encode(high_level_subgraph, low_level_batch_test)
                if corr_val > best_corr:
                    best_corr = corr_val
                    _, imputed_gene, _ = decoder(high_emb_test, high_level_subgraph, low_emb_test, low_level_batch_test)
                    preds = imputed_gene.view(cells, genes)[:, mask]
                    corr = [PearsonCorr1d(preds[i], low_true[i]).item() for i in test_cells]
                    corr: float = np.array(corr)
                    corr = corr[~np.isnan(corr)].mean()
                    corr_test.append(corr)
                    # corr = PearsonCorr1d((preds[test_cells].ravel()), low_true[test_cells].ravel()).item()
            if(len(corr_test)):
                best_test_corr = sum(corr_test)/len(corr_test)
            print(f"Epoch {epoch+1} - Best val corr: {best_corr:.4f} | Best test corr: {best_test_corr:.4f}")
        corr_over_runs.append(best_test_corr)
    corr_over_runs = np.array(corr_over_runs)
    print(f"Mean: {corr_over_runs.mean():.3f} $\\pm$ {corr_over_runs.std():.3f}")