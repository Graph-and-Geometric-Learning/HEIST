import warnings
warnings.filterwarnings('ignore')

import torch
from utils.dataloader import create_dataloader
from model.model import GraphEncoder
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
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torchinfo import summary
import sys
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, default = 'dfci', help="Directory where the raw data is stored")
parser.add_argument('--pe_dim', type=int, default= 32, help="Dimension of the positional encodings")
parser.add_argument('--init_dim', type=int, default= 256, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default= 512, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default= 512, help="Output dim for the MLP")
parser.add_argument('--model_name', type=str, default = 'model_6M_attention_sinusoidal_pe_cross_attention_blending_orthogonal_sea_only')
parser.add_argument('--wavelet_blending', action='store_true')
parser.add_argument('--anchor_pe', action='store_true')
parser.add_argument('--num_layers', type=int, default= 1, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default= 350, help="Batch size")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 10000, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

INPUT_DIM_HIGH = 2
INPUT_DIM_LOW = 1

if __name__ == '__main__':
    print("Loading graphs")
    if(args.data_name == 'sea'):
        graphs_list = [torch.load(file, weights_only = False) for file in tqdm(glob("data/pretraining/sea_preprocessed/*"))]
    elif(args.data_name in ['dfci', 'upmc', 'charville']):
        graphs_list = [torch.load(file, weights_only = False) for file in tqdm(glob(f"data/space-gm/{args.data_name}/*"))]
    elif(args.data_name == 'lung'):
        graphs_list = [torch.load(file, weights_only = False) for file in tqdm(glob("data/merfish_lung_preprocessed/*"))]
    elif(args.data_name == 'melanoma'):
        graphs_list = [torch.load(file, weights_only = False) for file in tqdm(glob("data/melanoma_lung_preprocessed/*"))]
    elif(args.data_name == 'placenta'):
        graphs_list = [torch.load(file, weights_only = False) for file in tqdm(glob("data/placenta_preprocessed/*"))]
    
    model_path = f"saved_models/{args.model_name}.pth"
    model_name = args.model_name
    checkpoint = torch.load(model_path)
    
    args = checkpoint['args']
    args.model_name = model_name
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.anchor_pe, args.blending).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.eval()
    summary(model)
    args.batch_size = 10000
    
    print("Extracting representations:")
    labels = []
    model.eval()
    _high_level_graphs = []
    for i in tqdm(range(len(graphs_list))):
        graphs = graphs_list[i]
        dataloader = create_dataloader(graphs, args.batch_size, False)
        graph_embeddings = torch.zeros((graphs[0].num_nodes, 2*args.output_dim)).to(args.device)
        _high_level_graph = graphs[0].to(args.device)
        with torch.no_grad():
            for high_level_subgraph, low_level_batch, batch_idx in dataloader:
                high_level_subgraph = high_level_subgraph.to(args.device)
                low_level_batch = low_level_batch.to(args.device)
                low_level_batch.batch_idx = batch_idx.to(args.device)
                
                high_emb, low_emb = model.encode(high_level_subgraph, low_level_batch)
                high_emb, low_emb = model.encode(high_level_subgraph, low_level_batch, args.pe_dim)
                graph_embeddings[batch_idx] = torch.cat([high_emb, global_mean_pool(low_emb, low_level_batch.batch)], dim=1)
            _high_level_graph.X = graph_embeddings #F.normalize(graph_embeddings)
            _high_level_graph.y = labels
            _high_level_graphs.append(_high_level_graph)
    torch.save(_high_level_graphs, f"data/{args.data_name}_graphs_{'_'.join(args.model_name.split('_')[1:])}.pt")