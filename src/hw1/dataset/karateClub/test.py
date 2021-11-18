import torch
import torch_geometric
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
graph = dataset[0]
print(graph)