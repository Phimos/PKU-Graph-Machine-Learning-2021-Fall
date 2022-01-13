from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchmetrics import AUROC, MeanSquaredError, Metric
from torchmetrics.utilities.data import dim_zero_cat


class RMSE(MeanSquaredError):
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn,
                         squared=False)

    def update(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        if mask is None:
            super().update(pred.detach().cpu(), target.detach().cpu())
        else:
            super().update(pred[mask].detach().cpu(), target[mask].detach().cpu())


class ROCAUC(Metric):
    def __init__(self,
                 num_targets: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 pos_label: Optional[int] = None,
                 average: Optional[str] = "macro",
                 multi_classes: Optional[str] = "ovr",
                 max_fpr: Optional[float] = None,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,):
        assert num_classes > 1
        assert multi_classes in ["ovr", "ovo"]
        self.num_targets = num_targets if num_targets is not None else 1
        self.num_classes = num_classes if num_classes is not None else 1
        self.multi_classes = multi_classes
        self.max_fpr = max_fpr
        self.calculators = [AUROC(num_classes=num_classes,
                                  pos_label=pos_label,
                                  average=average,
                                  max_fpr=max_fpr,
                                  compute_on_step=compute_on_step,
                                  dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group,
                                  dist_sync_fn=dist_sync_fn)
                            for _ in range(self.num_targets)]

    def reset(self):
        for calc in self.calculators:
            calc.reset()

    def update(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        pred = pred.reshape(-1, self.num_targets, self.num_classes)
        target = target.reshape(-1, self.num_targets)
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        else:
            mask = mask.reshape(-1, self.num_targets)

        for i in range(self.num_targets):
            if mask[:, i].any():
                self.calculators[i].update(
                    pred[:, i][mask[:, i]], target[:, i][mask[:, i]])

    def _compute_ovo_rocauc(self, calculator):
        preds = dim_zero_cat(calculator.preds).cpu()
        target = dim_zero_cat(calculator.target).cpu().numpy()
        classes = sorted(set(target))
        preds = F.softmax(preds[:, classes], dim=-1).numpy()
        target = np.array([classes.index(t) for t in target])
        return roc_auc_score(target, preds, max_fpr=self.max_fpr, multi_class='ovo')

    def compute(self):
        if self.multi_classes == "ovr":
            return np.mean([calc.compute().item() for calc in self.calculators])
        else:
            return np.mean([self._compute_ovo_rocauc(calc) for calc in self.calculators])
