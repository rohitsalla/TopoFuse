"""
TopoFuse  (paper Fig. 1, §4.2)
=============================
Full four-stage pipeline:

  (I)   Tri-planar SAM encoder      -> axis-specific planar logits + GAP feature
  (II)  Topology-conditioned FiLM   -> fused logits Z, topology descriptor t
  (III) Differentiable projection   -> sparse PH-guided critical-voxel edits +
                                        repair certificate C
  (IV)  Topology prior head H_pi     -> predicts the correction target at inference

Training uses the oracle target (from labels) inside the projection; the prior
head is trained jointly via L_prior so the model is self-contained at inference.

Total loss (§4.7):
    L = L_DiceCE(P'_hat, Y) + lambda_t L_topo(P_hat, Y)
        + lambda_pi L_prior + lambda_a L_DiceCE(P_hat, Y)
with lambda_t=0.1, lambda_pi=0.05, lambda_a=0.5.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .triplanar_encoder import TriPlanarEncoder
from .film_fusion import FiLMFusion, MeanFusion
from .topology_prior_head import TopologyPriorHead
from ..losses.topo_projection import TopologyProjection, oracle_targets_from_labels
from ..losses.seg_losses import DiceCELoss
from ..losses.topo_losses import WassersteinTopoLoss, PriorRegressionLoss


class TopoFuse(nn.Module):
    def __init__(self, num_classes: int = 3, feature_dim: int = 256,
                 film_hidden: int = 128, prior_hidden=None,
                 T_max: int = 5, epsilon: float = 0.05, downsample_s: int = 2,
                 delta: float = 0.05, slice_size: int = 256,
                 sam_checkpoint: str = None,
                 lambda_topo: float = 0.1, lambda_prior: float = 0.05,
                 lambda_aux: float = 0.5, warmup_steps: int = 5000,
                 use_ph_desc: bool = True, project_in_train: bool = True,
                 use_film: bool = True, project_enabled: bool = True):
        super().__init__()
        if prior_hidden is None:
            prior_hidden = [256, 128]
        self.C = num_classes
        self.downsample_s = downsample_s
        self.delta = delta
        self.lambda_topo = lambda_topo
        self.lambda_prior = lambda_prior
        self.lambda_aux = lambda_aux
        self.project_in_train = project_in_train
        self.project_enabled = project_enabled    # ablation: w/o ProjT (§5)

        self.encoder = TriPlanarEncoder(
            num_classes, feature_dim, slice_size=slice_size,
            sam_checkpoint=sam_checkpoint)
        if use_film:
            self.fusion = FiLMFusion(num_classes, film_hidden=film_hidden,
                                     delta=delta, use_ph_desc=use_ph_desc)
        else:
            self.fusion = MeanFusion(num_classes)     # ablation: w/o FiLM (§5)
        self.projection = TopologyProjection(num_classes, T_max, epsilon,
                                             downsample_s, delta)
        self.prior_head = TopologyPriorHead(feature_dim, num_classes, prior_hidden)

        # losses (instantiated once so the topo-loss warmup counter persists)
        self.seg_loss = DiceCELoss()
        self.topo_loss = WassersteinTopoLoss(downsample_s, delta, warmup_steps)
        self.prior_loss = PriorRegressionLoss()

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, volume: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        planar_logits, fused_feat, gap = self.encoder(volume)
        Z_fused, topo_desc = self.fusion(planar_logits)
        prior_pred = self.prior_head(gap)

        if not self.project_enabled:                       # ablation: w/o ProjT
            Z_post, certs = Z_fused, []
        elif self.training and self.project_in_train and labels is not None:
            targets = oracle_targets_from_labels(
                labels, self.C, self.downsample_s, self.delta)
            Z_post, certs = self.projection(Z_fused, target_diagrams=targets)
        else:
            Z_post, certs = self.projection(Z_fused, prior_pred=prior_pred)

        return {
            "logits_pre": Z_fused,
            "logits_post": Z_post,
            "prob_pre": torch.sigmoid(Z_fused),
            "prob_post": torch.sigmoid(Z_post),
            "prior_pred": prior_pred,
            "topo_desc": topo_desc,
            "certificates": certs,
            "gap": gap,
        }

    # ── loss (§4.7) ──────────────────────────────────────────────────────────
    def compute_loss(self, outputs: Dict, labels: torch.Tensor,
                     gt_topology: Optional[Dict] = None) -> Dict:
        L_main = self.seg_loss(outputs["prob_post"], labels)
        L_aux = self.seg_loss(outputs["prob_pre"], labels)
        L_topo = self.topo_loss(outputs["prob_pre"], labels)
        if gt_topology is not None:
            L_prior = self.prior_loss(outputs["prior_pred"], gt_topology)
        else:
            L_prior = labels.new_zeros((), dtype=torch.float32)

        total = (L_main
                 + self.lambda_topo * L_topo
                 + self.lambda_prior * L_prior
                 + self.lambda_aux * L_aux)
        return {"total": total, "main": L_main, "aux": L_aux,
                "topo": L_topo, "prior": L_prior}
