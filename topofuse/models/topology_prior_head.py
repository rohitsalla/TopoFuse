"""
Topology Prior Head  (paper §4.6)
=================================
At training the projection uses the oracle target derived from ground-truth
labels.  At inference the prior head H_pi predicts a compact topology proxy
directly from the (GAP-pooled) fused features, removing oracle dependence:

    Pi_hat*_c = H_pi( GAP(F_fused) )

H_pi is a three-layer MLP with hidden dimensions [256, 128] and ReLU.  It
predicts, per class:
    * estimated Betti numbers (beta0_hat, beta2_hat)  -- softplus, >= 0
    * persistence budgets at the six thresholds {0.03,0.05,0.08,0.10,0.15,0.20}
      for each of the two dimensions d in {0, 2}.

Trained jointly via l1 regression against ground-truth topology statistics
(L_prior).  The learned-vs-oracle gap (paper Table 2/3) is attributable to
prediction error here, not to the projection mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from ..topo.pseudo_diagram import BUDGET_THRESHOLDS

N_BUDGETS = len(BUDGET_THRESHOLDS)   # 6
N_DIMS = 2                            # d in {0, 2}


class TopologyPriorHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.C = num_classes
        # per class: beta0, beta2, then N_DIMS x N_BUDGETS budget entries
        self.per_class = 2 + N_DIMS * N_BUDGETS
        out = num_classes * self.per_class

        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out)

    def forward(self, gap: torch.Tensor) -> Dict[str, torch.Tensor]:
        """gap: (B, in_dim).  Returns dict with beta0,(B,C); beta2,(B,C);
        budgets,(B,C,2,6)."""
        B = gap.shape[0]
        out = self.head(self.mlp(gap)).view(B, self.C, self.per_class)
        beta0 = F.softplus(out[:, :, 0])                       # >= 0
        beta2 = F.softplus(out[:, :, 1])                       # >= 0
        budgets = torch.sigmoid(
            out[:, :, 2:].view(B, self.C, N_DIMS, N_BUDGETS))  # (B,C,2,6) in (0,1)
        return {"beta0": beta0, "beta2": beta2, "budgets": budgets}
