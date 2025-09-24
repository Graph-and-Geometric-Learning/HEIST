import warnings
warnings.filterwarnings('ignore')
import sys

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score

model_name = sys.argv[1]

if __name__ == '__main__':
    files = [f"data/merfish_lung_{'_'.join(model_name.split('_')[1:])}.pt",
             f"data/dfci_graphs_{'_'.join(model_name.split('_')[1:])}.pt",
             f"data/charville_graphs_{'_'.join(model_name.split('_')[1:])}.pt",
             f"data/upmc_graphs_{'_'.join(model_name.split('_')[1:])}.pt",
             f"data/sea_graphs_{'_'.join(model_name.split('_')[1:])}.pt"
             ]
    for saved_file in files:
        print(saved_file)
        print("Loading graphs")
        _high_level_graphs = torch.load(saved_file)
        NMIs = []
        for i in range(len(_high_level_graphs)):
            X = _high_level_graphs[i].X.cpu().numpy()
            y_true = _high_level_graphs[i].cell_type.cpu().numpy()
            for k in range(5):
                kmeans = KMeans(n_clusters=len(np.unique(y_true)))
                y_pred = kmeans.fit_predict(X)
                NMIs.append(accuracy_score(y_true, y_pred))
        NMIs = np.array(NMIs)
        print(f"Max: {NMIs.max()}, Mean: {NMIs.mean()}, Std: {NMIs.std()}")
        del(_high_level_graphs)
        torch.cuda.empty_cache()
