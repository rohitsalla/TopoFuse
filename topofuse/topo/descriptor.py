"""
Topology descriptor  (paper §4.4)
=================================
Computes the compact descriptor ``t = psi(P_bar)`` from the preliminary averaged
prediction.  Per the paper it has exactly ``d_t = 6C`` scalars:

    per class c, per dimension d in {0, 2}:
        * feature counts at persistence thresholds {0.05, 0.15}   -> 2 values
        * total persistence mass for that dimension               -> 1 value
    => 3 values x 2 dims = 6 per class  => 6C total.

Gradients flow through PH at this stage via the critical-cell rule [4]: counts
are made differentiable with a soft (sigmoid) surrogate over feature lifetimes,
and persistence mass is the (differentiable) sum of lifetimes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..topo.ph import differentiable_diagram

COUNT_THRESHOLDS = (0.05, 0.15)
DESC_DIMS = (0, 2)


class TopologyDescriptor(nn.Module):
    """Differentiable 6C topology descriptor from a probability volume."""

    def __init__(self, num_classes: int, tau: float = 0.02, delta: float = 0.05,
                 use_ph: bool = True):
        super().__init__()
        self.C = num_classes
        self.tau = tau                 # softness of the count surrogate
        self.delta = delta
        self.use_ph = use_ph           # may be disabled for speed in unit tests
        self.out_dim = 6 * num_classes

    def _soft_count(self, lifetimes: torch.Tensor, thr: float) -> torch.Tensor:
        if lifetimes.numel() == 0:
            return lifetimes.new_zeros(())
        return torch.sigmoid((lifetimes - thr) / self.tau).sum()

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        """prob: (B, C, D, H, W) probabilities. Returns (B, 6C)."""
        B, C, D, H, W = prob.shape
        out = prob.new_zeros(B, self.out_dim)
        if not self.use_ph:
            # cheap surrogate (no PH) -- only for environments without cripser
            for c in range(C):
                p = prob[:, c]
                base = c * 6
                for di, _ in enumerate(DESC_DIMS):
                    for ti, thr in enumerate(COUNT_THRESHOLDS):
                        act = torch.sigmoid((p - thr) / 0.02)
                        out[:, base + di * 3 + ti] = act.mean(dim=[1, 2, 3])
                    out[:, base + di * 3 + 2] = (p * (1 - p)).mean(dim=[1, 2, 3])
            return out

        for b in range(B):
            for c in range(C):
                dgm = differentiable_diagram(prob[b, c], maxdim=2, delta=self.delta)
                base = c * 6
                for di, dim in enumerate(DESC_DIMS):
                    bt = dgm.birth_t.get(dim, prob.new_zeros(0))
                    dt = dgm.death_t.get(dim, prob.new_zeros(0))
                    life = (bt - dt).abs() if bt.numel() else prob.new_zeros(0)
                    # 2 soft counts at the two thresholds
                    out[b, base + di * 3 + 0] = self._soft_count(life, COUNT_THRESHOLDS[0])
                    out[b, base + di * 3 + 1] = self._soft_count(life, COUNT_THRESHOLDS[1])
                    # total persistence mass for this dimension
                    out[b, base + di * 3 + 2] = life.sum() if life.numel() else \
                        prob.new_zeros(())
        return out
