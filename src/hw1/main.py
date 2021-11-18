import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from models import ProbabilisticRelationalClassifier, accuracy
from utils import (load_edges_from_file, load_labels_from_file,
                   load_nodes_from_file)

# Load data
edge_index = torch.Tensor(
    load_edges_from_file('dataset/Cora/edges')).long().T
train_nodes = torch.Tensor(
    load_nodes_from_file('dataset/Cora/train_nodes')).long()
test_nodes = torch.Tensor(load_nodes_from_file(
    'dataset/Cora/test_nodes')).long()
labels = torch.Tensor(load_labels_from_file('dataset/Cora/labels'))
labels = labels[:, 1].long()

num_classes = labels.max().item() + 1
one_hot_labels = F.one_hot(labels, num_classes).float()

x = one_hot_labels
x[test_nodes] = 1. / num_classes

data = Data(x=x, edge_index=to_undirected(edge_index))

print(f'Number of classes: {num_classes}')
print(f'Number of train nodes: {train_nodes.size(0)}')
print(f'Number of test nodes: {test_nodes.size(0)}')

# Create model
model = ProbabilisticRelationalClassifier()

# Train model
history = []
for i in range(10):
    data.x = model(data.x, data.edge_index)
    data.x[train_nodes] = one_hot_labels[train_nodes]
    history.append({
        'epoch': i+1,
        'accuracy': accuracy(data.x[test_nodes], labels[test_nodes])
    })
    print(i, accuracy(data.x[test_nodes], labels[test_nodes]), accuracy(
        data.x[train_nodes], labels[train_nodes]))

# Plot results
plt.plot([h['epoch'] for h in history], [h['accuracy'] for h in history])
plt.savefig('accuracy.png', dpi=300)
plt.close()
