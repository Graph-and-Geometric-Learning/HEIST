import warnings
warnings.filterwarnings('ignore')

from utils.dataloader import CustomDataset
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from model.model import MLP, GIN
from model.loss import AUCPRHingeLoss,aucpr_hinge_loss
import torch.optim as optim
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoader_PyG
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
from glob import glob
import json
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool
from argparse import ArgumentParser

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--init_dim', type=int, default=256, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default=64, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=1, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--model_name', type=str)


def initialize_weights(layer):
    # Handle linear layers with Xavier initialization
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    # Handle Conv2d layers (if applicable)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def eval(model, loader):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == y).float()
            total += len(y)
    return ((total_correct * 100) / total).item()

def eval_roc_auc(model, loader):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            true.append(y)
            pred.append(preds)
    true = torch.cat(true).cpu().detach().numpy()
    pred = torch.cat(pred).cpu().detach().numpy()
    return roc_auc_score(true, pred,average='micro')

def train(model, train_loader, val_loader, test_loader):
    best_test_acc = 0
    best_val_acc = eval(model, val_loader)
    # loss_fn = AUCPRHingeLoss()
    loss_fn = CrossEntropyLoss()
    # loss_fn = aucpr_hinge_loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # loss_fn = AUCMLoss(margin=0.1)
    # opt = PESG(model.parameters(), loss_fn=loss_fn, lr = args.lr, weight_decay = args.wd)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=1000, verbose=True)

    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            model.train()
            for X,y in train_loader:
                opt.zero_grad()
                logits = model(X)
                loss = loss_fn(logits, y) #+ loss_fn2(logits, y)
                loss.backward()
                opt.step()
            scheduler.step(eval(model, val_loader))

            train_acc = eval(model, train_loader)
            val_acc = eval(model, val_loader)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = eval(model, test_loader)
            tq.set_description("Loss = %.4f, Train acc = %.4f, Val acc = %.4f, Best val acc = %.4f, Best acc = %.4f" % (loss.item(), train_acc, val_acc, best_val_acc, best_test_acc))
    return best_test_acc

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
INPUT_DIM_HIGH = 2
INPUT_DIM_LOW = 1

if __name__ == '__main__':

        print("================================")
        print("Tissue classification Melanoma Response")
        print("================================")
        _high_level_graphs = torch.load(f"data/melanoma_graphs_{'_'.join(args.model_name.split('_')[1:])}.pt")
        indices = []
        for i in range(len(_high_level_graphs)):
            if hasattr(_high_level_graphs[i], "y") and not math.isnan(getattr(_high_level_graphs[i], "y")):
                if(torch.any(torch.isnan(_high_level_graphs[i].X))):
                    print(i)
                    continue
                indices.append(i)
        _high_level_graphs = [_high_level_graphs[i] for i in indices]
        labels = torch.LongTensor([getattr(i, 'y') for i in _high_level_graphs]).to(args.device)
        print(torch.bincount(labels))
        embeddings = []
        for graph in _high_level_graphs:
            embeddings.append(graph.X.mean(0).tolist())  # Pooling for MLP input
        embeddings = torch.FloatTensor(embeddings).to(args.device)
        # embeddings[:, embeddings.shape[1]//2:] = embeddings[:, embeddings.shape[1]//2:]/30
        class_weights = torch.tensor([(1 - labels.float().mean()), labels.float().mean()]).to(args.device)
        
        roc_scores = []
        for fold in range(5):
                model = MLP(embeddings.shape[1], args.hidden_dim, 2, args.num_layers).to(args.device)
                train_idx, test_idx = train_test_split(np.arange(embeddings.shape[0]), test_size = 0.2, stratify= labels.cpu().numpy())
                val_idx, test_idx = train_test_split(test_idx, test_size = 0.5, stratify= labels[test_idx].cpu().numpy())
                train_idx = torch.LongTensor(train_idx).to(args.device)
                val_idx = torch.LongTensor(val_idx).to(args.device)
                test_idx = torch.LongTensor(test_idx).to(args.device)
                train_loader = DataLoader(CustomDataset(embeddings[train_idx], labels[train_idx]), batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(CustomDataset(embeddings[val_idx], labels[val_idx]), batch_size=args.batch_size, shuffle=False)#, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
                test_loader = DataLoader(CustomDataset(embeddings[test_idx], labels[test_idx]), batch_size=args.batch_size, shuffle=False)#x, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])

                best_acc = train(model, train_loader, val_loader, test_loader)
                roc_scores.append(best_acc)
        roc_scores = np.array(roc_scores)
        print(f"Mean:{roc_scores.mean()}, Std:{roc_scores.std()}.")
        print(f"Max:{roc_scores.max()}.")