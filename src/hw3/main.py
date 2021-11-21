import argparse
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
from torch import Tensor
from torch_geometric.data import Data, dataset
from torch_geometric.nn import ChebConv, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.models.basic_gnn import GAT, GCN, GIN, GraphSAGE
from torch_geometric.utils import to_undirected

from utils import (load_attribute_from_file, load_edges_from_file,
                   load_labels_from_file, load_nodes_from_file)


class GNNClassifier(torch.nn.Module):
    def __init__(self, model_name: str, num_features: int, num_classes: int,
                 hidden_dim: int = 128):
        super(GNNClassifier, self).__init__()

        if model_name == 'gcn':
            GNN_MODEL = GCN
        elif model_name == 'gat':
            GNN_MODEL = GAT
        elif model_name == 'sage':
            GNN_MODEL = GraphSAGE
        else:
            GNN_MODEL = GIN

        self.gnn = GNN_MODEL(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=2,
            dropout=0.5,
        )
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        # self.head = torch.nn.Sequential(
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(hidden_dim, 64),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(64, num_classes),
        # )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embedding = self.gnn(x, edge_index)
        logits = self.head(embedding)
        return logits

    def fetch_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        return self.gnn(x, edge_index)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(
        model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits = model(data)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def plot_points(embeddings, model_name, dataset_name):
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700', '#000000'
    ]
    model.eval()
    z = TSNE(n_components=2).fit_transform(embeddings.detach().cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(data.y.unique().size(0)):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')

    title = " %s %s " % (dataset_name.capitalize(), model_name.upper())
    plt.title(title, fontdict={'fontsize': 40})
    plt.savefig('%s_%s.png' % (dataset_name, model_name), dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage', 'gin'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--early_stop', type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name: str = args.dataset
    model_name: str = args.gnn

    print((" %s %s " % (dataset_name.capitalize(),
          model_name.upper())).center(52, '='))

    labels = load_labels_from_file('data/%s/labels' % dataset_name)
    train_nodes = load_nodes_from_file('data/%s/train_nodes' % dataset_name)
    val_nodes = load_nodes_from_file('data/%s/val_nodes' % dataset_name)
    test_nodes = load_nodes_from_file('data/%s/test_nodes' % dataset_name)
    edges = load_edges_from_file('data/%s/edges' % dataset_name)
    attributes = load_attribute_from_file('data/%s/attributes' % dataset_name)

    labels = Tensor(labels).long()[:, 1]
    train_nodes = Tensor(train_nodes).long()
    val_nodes = Tensor(val_nodes).long()
    test_nodes = Tensor(test_nodes).long()
    edges = Tensor(edges).long().T
    attributes = Tensor(attributes)

    train_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    train_mask[train_nodes] = True
    val_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    val_mask[val_nodes] = True
    test_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    test_mask[test_nodes] = True

    num_features = attributes.shape[1]
    num_classes = labels.max() + 1

    data = Data(x=attributes, edge_index=to_undirected(edges), y=labels)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GNNClassifier(model_name, num_features, num_classes, args.hidden)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = test_acc = 0
    early_stop_count = 0
    for epoch in range(1, args.epochs + 1):
        train(model, data, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            early_stop_count = 0
        else:
            early_stop_count += 1
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        if epoch % args.verbose == 0:
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        if early_stop_count >= args.early_stop:
            print('Early stopping! Current Epoch: {:03d}'.format(epoch))
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
            break

    plot_points(model.fetch_embedding(data), model_name, dataset_name)
