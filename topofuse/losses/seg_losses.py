"""
Segmentation losses  (paper §4.7)
=================================
L_DiceCE = soft Dice + per-class binary cross-entropy, applied per-class via the
one-vs-rest sigmoid convention used throughout TopoFuse.  Used twice in the total
loss: on the post-projection probabilities P'_hat (main term) and on the
pre-projection probabilities P_hat (auxiliary term, weight lambda_a=0.5).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_onehot(target: torch.Tensor, C: int, pred_dim: int) -> torch.Tensor:
    """(B,D,H,W) int -> (B,C,D,H,W) float one-hot, or pass through if already one-hot."""
    if target.dim() == pred_dim - 1:
        oh = F.one_hot(target.long(), C).permute(0, 4, 1, 2, 3).float()
        return oh
    return target.float()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        B, C = prob.shape[:2]
        pf = prob.reshape(B, C, -1)
        yf = onehot.reshape(B, C, -1)
        inter = (pf * yf).sum(-1)
        denom = pf.sum(-1) + yf.sum(-1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Dice + (per-class) BCE on probabilities, via the one-vs-rest sigmoid."""

    def __init__(self, ce_w: float = 1.0, dice_w: float = 1.0):
        super().__init__()
        self.ce_w, self.dice_w = ce_w, dice_w
        self.dice = DiceLoss()

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = prob.shape[1]
        y = _to_onehot(target, C, prob.dim())
        eps = 1e-7
        p = prob.clamp(eps, 1 - eps)
        logits = torch.log(p / (1 - p))                    # invert sigmoid
        ce = F.binary_cross_entropy_with_logits(logits, y)
        return self.ce_w * ce + self.dice_w * self.dice(prob, y)
