"""
Topology losses  (paper §4.7)
=============================
L_topo  : Wasserstein topology loss applied to the *pre-projection*
          probabilities, biasing the encoder toward topologically feasible
          outputs before the projection layer corrects them.  Following [21],
          it matches the prediction's persistence diagram to the ground-truth
          diagram (per class, per dimension d in {0,2}) and penalises the
          Wasserstein-2 transport cost.  Linearly warmed up over 5,000 steps.

              L_topo = sum_c d_W( PH(Downs(P_hat_c)), PH(Downs(Y_c)) )

L_prior : l1 regression of the prior head against ground-truth topology
          statistics (beta0, beta2 and the six persistence budgets).

The diagrams are computed with the exact CubicalRipser backend (topo/ph.py) and
the cost is differentiable through the critical-cell rule [4].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..topo.ph import differentiable_diagram
from ..topo.matching import wasserstein_cost

DIMS = (0, 2)


class WassersteinTopoLoss(nn.Module):
    """Differentiable PH Wasserstein-2 loss with 5k-step linear warmup."""

    def __init__(self, downsample_s: int = 2, delta: float = 0.05,
                 warmup_steps: int = 5000):
        super().__init__()
        self.s = downsample_s
        self.delta = delta
        self.warmup = warmup_steps
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

    @property
    def weight(self) -> float:
        return float(min(1.0, self._step.item() / max(1, self.warmup)))

    def _down(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, self.s, self.s) if self.s > 1 else x

    def forward(self, prob_pre: torch.Tensor, target: torch.Tensor,
                advance: bool = True) -> torch.Tensor:
        """prob_pre: (B,C,D,H,W); target: (B,D,H,W) int or (B,C,D,H,W) one-hot."""
        if advance and self.training:
            self._step += 1
        B, C = prob_pre.shape[:2]
        if target.dim() == prob_pre.dim() - 1:
            y = F.one_hot(target.long(), C).permute(0, 4, 1, 2, 3).float()
        else:
            y = target.float()

        pd = self._down(prob_pre)
        yd = self._down(y)
        total = prob_pre.new_zeros(())
        for b in range(B):
            for c in range(C):
                dgm_p = differentiable_diagram(pd[b, c], maxdim=2, delta=self.delta)
                dgm_y = differentiable_diagram(yd[b, c].detach(), maxdim=2,
                                               delta=self.delta)
                for dim in DIMS:
                    total = total + wasserstein_cost(dgm_p, dgm_y, dim,
                                                     device=prob_pre.device)
        total = total / max(1, B * C)
        return self.weight * total


class PriorRegressionLoss(nn.Module):
    """l1 regression for the topology prior head (beta0, beta2, budgets)."""

    def forward(self, pred: dict, gt: dict) -> torch.Tensor:
        loss = F.l1_loss(pred["beta0"], gt["beta0"].float()) + \
               F.l1_loss(pred["beta2"], gt["beta2"].float())
        if "budgets" in gt and gt["budgets"] is not None:
            loss = loss + F.l1_loss(pred["budgets"], gt["budgets"].float())
        return loss
