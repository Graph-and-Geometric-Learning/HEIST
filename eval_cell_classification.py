import warnings
warnings.filterwarnings('ignore')
import sys

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

model_name = sys.argv[1]
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    model = MLP(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    best_val_f1 = 0
    for epoch in (range(1500)):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(val_pred, y_val, average='macro')
            if(val_f1 > best_val_f1):
                best_val_f1 = val_f1
                best_model = model
    return best_val_f1, best_model

if __name__ == '__main__':
    saved_file = f"data/sea_graphs_{'_'.join(model_name.split('_')[1:])}.pt"
    print("================================")
    print("Cell type annotation")
    print("================================")
    graphs = torch.load(saved_file)
    accuracies, f1_scores = [], []

    for g in tqdm(graphs):
        X_all = g.X.cpu().numpy()#.append(g.X.cpu().numpy())
        X_all = X_all[:, X_all.shape[1]//2:]
        y_all = g.cell_type.cpu().numpy()

        for run in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=run)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=run)

            y_val_pred, model = train_mlp(X_train, y_train, X_val, y_val, input_dim=X_all.shape[1], num_classes=len(np.unique(y_all)), device="cuda" if torch.cuda.is_available() else "cpu")
            # val_acc = accuracy_score(y_val, y_val_pred)

            X_test_tensor = torch.from_numpy(X_test).float().to("cuda" if torch.cuda.is_available() else "cpu")
            y_test_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average="macro")
            accuracies.append(acc)
            f1_scores.append(f1)

    print(f"\nFinal Mean ± Std Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Final Mean ± Std F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
