import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

class WeightedSumConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)

class GraphWaveletTransform(nn.Module):
    def __init__(self, edge_index, edge_weight, X, J, device):
        super().__init__()
        self.device = device
        # We'll store the graph
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)
        self.X_init = X.to(device)  # node features

        self.conv = WeightedSumConv()
        self.J = J
        self.num_feats = self.X_init.size(1)

        self.max_scale = 2 ** (J - 1)

    def diffuse(self, x=None):
        if x is None:
            x = self.X_init

        x_curr = x
        out_list = []
        for step in range(1, self.max_scale + 1):
            x_curr = self.conv(x_curr, self.edge_index, self.edge_weight)
            if (step & (step - 1)) == 0:
                out_list.append(x_curr)
        return out_list

    def first_order_feature(self, diff_list):
        F1 = torch.cat([torch.abs(diff_list[i-1]- diff_list[i]) for i in range(1, len(diff_list))],1)
        return F1

    def second_order_feature(self, diff_list):
        U = torch.cat(diff_list, dim=1)
        U_diff_list = self.diffuse(U)

        results = []
        for j in range(self.J):
            col_start = j * self.num_feats
            col_end   = (j + 1) * self.num_feats
            for j_prime in range(j+1, self.J):
                block_jp   = U_diff_list[j_prime][:, col_start:col_end]
                block_jp_1 = U_diff_list[j_prime-1][:, col_start:col_end]
                results.append(torch.abs(block_jp - block_jp_1))
        return torch.cat(results, dim=1)

    def generate_timepoint_features(self, batch = None):
        diff_list = self.diffuse()   
        #F0 = diff_list[-1]
        F1 = self.first_order_feature(diff_list)
        F2 = self.second_order_feature(diff_list)
        feats = torch.cat([F1, F2], dim=1)

        return feats