import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set


class MPNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, hidden_channels=64):
        super().__init__()
        self.lin0 = torch.nn.Linear(in_channels, hidden_channels)

        nn = Sequential(Linear(edge_dim, 128), ReLU(), Linear(
            128, hidden_channels * hidden_channels))
        self.conv = NNConv(hidden_channels, hidden_channels, nn, aggr='mean')
        self.gru = GRU(hidden_channels, hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
