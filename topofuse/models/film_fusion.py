"""
Topology-Conditioned FiLM Fusion  (paper §4.4)
==============================================
Fuses the three planar logit volumes with learned, topology-aware weights.

  1. Topology descriptor   t = psi(P_bar) in R^{6C}  (topo/descriptor.py)
     extracted from the preliminary averaged prediction P_bar = sigmoid(mean_p Z^p).
  2. A 3-D CNN  G(.)  (three 3^3-kernel layers, 64 channels) processes the
     stacked planar logits {Z^(q)} and produces per-voxel, per-plane weight logits.
  3. FiLM modulation:
        alpha^(p)(x) = softmax_p( gamma(t) ⊙ G(x;{Z^(q)}) + beta(t) )
        Z(x)         = sum_p alpha^(p)(x) · Z^(p)(x)
     where gamma and beta are two-layer MLPs of the descriptor t.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..topo.descriptor import TopologyDescriptor

PLANES = ["xy", "xz", "yz"]


class FiLMFusion(nn.Module):
    def __init__(self, num_classes: int, film_hidden: int = 128, cnn_ch: int = 64,
                 desc_tau: float = 0.02, delta: float = 0.05, use_ph_desc: bool = True):
        super().__init__()
        self.C = num_classes
        self.P = len(PLANES)
        self.desc = TopologyDescriptor(num_classes, tau=desc_tau, delta=delta,
                                       use_ph=use_ph_desc)
        desc_dim = self.desc.out_dim                       # 6C

        # G(.) : three-layer 3^3 3-D CNN, 64 channels -> per-plane-per-class logits
        in_ch = self.P * num_classes
        self.G = nn.Sequential(
            nn.Conv3d(in_ch, cnn_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(cnn_ch, cnn_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(cnn_ch, in_ch, 3, padding=1),
        )

        def mlp(out):
            return nn.Sequential(
                nn.Linear(desc_dim, film_hidden), nn.ReLU(inplace=True),
                nn.Linear(film_hidden, out),
            )
        self.gamma = mlp(in_ch)
        self.beta = mlp(in_ch)

    def forward(self, planar_logits: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """planar_logits[p]: (B,C,D,H,W).  Returns (Z_fused (B,C,D,H,W), t (B,6C))."""
        P, C = self.P, self.C
        Z_planes = torch.stack([planar_logits[p] for p in PLANES], dim=1)   # (B,P,C,D,H,W)
        B = Z_planes.shape[0]
        D, H, W = Z_planes.shape[-3:]

        # preliminary average prediction -> topology descriptor t
        Z_bar = Z_planes.mean(dim=1)                       # (B,C,D,H,W)
        P_bar = torch.sigmoid(Z_bar)
        t = self.desc(P_bar)                               # (B, 6C)

        # G over stacked planar logits
        Z_stack = Z_planes.reshape(B, P * C, D, H, W)
        g = self.G(Z_stack)                                # (B, P*C, D,H,W)

        gam = self.gamma(t).view(B, P * C, 1, 1, 1)
        bet = self.beta(t).view(B, P * C, 1, 1, 1)
        mod = (gam * g + bet).view(B, P, C, D, H, W)

        alpha = F.softmax(mod, dim=1)                      # softmax over planes
        Z_fused = (alpha * Z_planes).sum(dim=1)            # (B,C,D,H,W)
        return Z_fused, t


class MeanFusion(nn.Module):
    """Ablation fusion (§5 ablations, "w/o FiLM"): plain mean of planar logits,
    no topology conditioning. Returns a zero descriptor for interface parity."""

    def __init__(self, num_classes: int, **_ignored):
        super().__init__()
        self.C = num_classes
        self.out_dim = 6 * num_classes

    def forward(self, planar_logits):
        Z = torch.stack([planar_logits[p] for p in PLANES], dim=1).mean(dim=1)
        t = Z.new_zeros(Z.shape[0], self.out_dim)
        return Z, t
