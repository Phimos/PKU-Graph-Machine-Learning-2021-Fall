import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch import Tensor
from torch.utils.data.dataset import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

from utils import (load_attribute_from_file, load_edges_from_file,
                   load_labels_from_file, load_nodes_from_file)

# Borrow code from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py


class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(in_features, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        return x


def train_test_mlp(train_x, train_y, valid_x, valid_y, test_x, test_y, **kwargs):
    hidden_size = kwargs.get('hidden_size', 64)
    early_stop = kwargs.get('early_stop', 10)
    max_epochs = kwargs.get('max_epochs', 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = train_x.shape[1]
    num_classes = train_y.max().item() + 1

    model = MLP(n_features, hidden_size, num_classes)
    model.to(device)

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_accuracy = 0
    stop_epoch = 0

    for epoch in range(1, max_epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        accuracy = correct / total

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            stop_epoch = epoch

        # early stopping
        if epoch - stop_epoch >= early_stop:
            break

        print('Epoch: {:03d}, Valid Accuracy: {:.4f}'.format(
            epoch, accuracy))

    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    print('Test Accuracy: {:.4f}'.format(correct / total))


def main(args):
    dataset_name = args.dataset
    feature_mode = args.feature

    gnn_model = args.gnn
    embedding_dim = args.embedding_dim
    walk_length = args.walk_length
    context_size = args.context_size
    walks_per_node = args.walks_per_node
    walks_per_node = args.walks_per_node
    num_negative_samples = args.num_negative_samples
    p = 1 if gnn_model == 'deepwalk' else args.p
    q = 1 if gnn_model == 'deepwalk' else args.q

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

    data = Data(x=attributes, edge_index=to_undirected(edges), y=labels)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                     context_size=context_size, walks_per_node=walks_per_node,
                     num_negative_samples=num_negative_samples, p=p, q=q, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for _ in range(1, 101):
        train()

    model.eval()
    embed = model()
    embed = embed.detach().cpu()

    if feature_mode == 'attributes':
        features = attributes
    elif feature_mode == 'embeddings':
        features = embed
    else:
        features = torch.cat([attributes, embed], dim=1)

    mlp_params = {
        'hidden_size': args.hidden_dim,
        'early_stop': args.early_stop,
    }

    train_test_mlp(features[train_mask], labels[train_mask], features[val_mask],
                   labels[val_mask], features[test_mask], labels[test_mask], **mlp_params)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Node2Vec')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature', type=str, default='attributes',
                        choices=['attributes', 'embeddings', 'concat'])

    gnn_group = parser.add_argument_group('GNN')
    gnn_group.add_argument('--gnn', type=str, default='node2vec',
                           choices=['deepwalk', 'node2vec'])
    gnn_group.add_argument('--embedding_dim', type=int, default=128)
    gnn_group.add_argument('--walk_length', type=int, default=20)
    gnn_group.add_argument('--context_size', type=int, default=10)
    gnn_group.add_argument('--walks_per_node', type=int, default=10)
    gnn_group.add_argument('--num_negative_samples', type=int, default=1)
    gnn_group.add_argument('--p', type=float, default=1)
    gnn_group.add_argument('--q', type=float, default=1)

    mlp_group = parser.add_argument_group('MLP')
    mlp_group.add_argument('--hidden_dim', type=int, default=64)
    mlp_group.add_argument('--early_stop', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    main(args)
