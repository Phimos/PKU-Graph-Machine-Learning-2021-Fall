from typing import Optional

from torch import Tensor
from torch_geometric.nn import MessagePassing


class ProbabilisticRelationalClassifier(MessagePassing):
    """This class implements a probabilistic relational classifier.
    """

    def __init__(self, aggr: Optional[str] = "mean", flow: str = "source_to_target", node_dim: int = -2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)

    def message(self, x_j: Tensor) -> Tensor:
        return super().message(x_j)

    def forward(self, x: Tensor, edge_index) -> Tensor:
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)


def accuracy(pred: Tensor, target: Tensor) -> float:
    """Computes the accuracy of a prediction.
    """
    pred = pred.argmax(dim=1)
    return (pred == target).float().mean().item()
