import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class TemporalGNN_a3t(nn.Module):
    def __init__(self, node_features, in_periods, out_periods, batch_size):
        super(TemporalGNN_a3t, self).__init__()

        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(
            in_channels=node_features, out_channels=in_periods, periods=in_periods, batch_size=batch_size
        )
        # Equals single-shot prediction
        self.linear = nn.Linear(in_periods, out_periods)

    def forward(self, x, edge_index):
        # x: (B, N, F, T)
        h = self.tgnn(x, edge_index)  # h: (B, N, out_C (T))
        h += x[:, :, 0, :]  # residual connection
        h = F.relu(h)
        h = self.linear(h)
        return h


class TemporalGNN_a3t_vanilla(TemporalGNN_a3t):
    def forward(self, x, edge_index):
        # x: (B, N, F, T)
        h = self.tgnn(x, edge_index)  # h: (B, N, out_C (T))
        h = F.relu(h)
        h = self.linear(h)
        return h


class GRU(nn.Module):
    def __init__(self, node_features, in_periods, out_periods, batch_size):
        super(GRU, self).__init__()

        self.gru = nn.GRU(node_features, 1, batch_first=True)
        self.linear = nn.Linear(in_periods, out_periods)

    def forward(self, x, _):
        B, N, Fe, T = x.shape
        # x: (B, N, Fe, T) -> (B*N, T, Fe)
        x = x.view(-1, Fe, T)
        x = x.permute(0, 2, 1)
        h = self.gru(x)[0]  # h: (B*N, out_C (T), 1)
        h = F.relu(h.squeeze())   # h: (B*N, out_C (T))
        h = self.linear(h)  # h: (B*N, pre T)
        h = h.view(B, N, -1)  # h: (B, N, pre T)
        return h
