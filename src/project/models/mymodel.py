from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, GRUCell, Linear
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import softmax


class GATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0, negative_slope: float = 0.01,
                 contain_edge_attr: bool = False):
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        self.dropout = Dropout(p=dropout)
        if contain_edge_attr:
            self.align = Linear(3 * in_channels, 1)
        else:
            self.align = Linear(2 * in_channels, 1)
        self.attend = Linear(in_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.align.weight)
        glorot(self.attend.weight)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.elu(out)

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int],
                edge_attr: Optional[Tensor] = None) -> Tensor:
        if edge_attr is not None:
            align_score = F.leaky_relu(self.align(
                self.dropout(torch.cat([x_i, x_j, edge_attr], dim=-1))))
        else:
            align_score = F.leaky_relu(self.align(
                self.dropout(torch.cat([x_i, x_j], dim=-1))))
        attention_weight = softmax(align_score, index, ptr, size_i)
        neighbor_feature = self.attend(self.dropout(x_j))
        return neighbor_feature * attention_weight


class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super(GATEConv, self).__init__(aggr='add', node_dim=0)
        self.dropout = Dropout(p=dropout)
        self.align = Linear(2 * out_channels, 1)
        self.attend = Linear(out_channels, out_channels)
        self.neighbor_linear = Linear(in_channels + edge_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.align.weight)
        glorot(self.attend.weight)
        glorot(self.neighbor_linear.weight)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.elu(out)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x_j = self.neighbor_linear(torch.cat([x_j, edge_attr], dim=-1))
        x_j = F.leaky_relu(x_j)
        align_score = F.leaky_relu(self.align(
            self.dropout(torch.cat([x_i, x_j], dim=-1))))
        attention_weight = softmax(align_score, index, ptr, size_i)
        neighbor_feature = self.attend(self.dropout(x_j))
        return neighbor_feature * attention_weight


class AttentiveFP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super(AttentiveFP, self).__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = Dropout(p=dropout)

        self.lin1 = Linear(in_channels, hidden_channels)

        conv = GATEConv(in_channels, hidden_channels, edge_dim, dropout)
        gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, negative_slope=0.01,
                                contain_edge_attr=False)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        if num_layers >= num_timesteps:
            self.cached_index = [-i-1 for i in range(num_timesteps)]
            self.cached_index = self.cached_index[::-1]
        else:
            self.cached_index = [-i-1 for i in range(num_layers)]
            self.cached_index = self.cached_index[::-1]
            for _ in range(num_timesteps-num_layers):
                self.cached_index.append(-1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, raw, edge_index, edge_attr, batch):
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(raw))
        cached = []
        h = self.atom_convs[0]((raw, x), edge_index, edge_attr)
        h = self.dropout(h)
        x = self.atom_grus[0](h, x).relu_()
        cached.append(x)

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = conv(x, edge_index)
            h = self.dropout(h)
            x = gru(h, x).relu_()
            cached.append(x)

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for i in range(self.num_timesteps):
            h = self.mol_conv(
                ((x + cached[self.cached_index[i]]) / 2, out), edge_index)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = self.lin2(self.dropout(out))
        return out
