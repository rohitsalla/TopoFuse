"""
Pseudo-diagram target construction  (paper §4.6 + supplement §A.1)
=================================================================
At inference the oracle persistence diagram is unavailable, so the projection
target is built from the prior head's predicted Betti counts (beta0, beta2) and
persistence budgets.  We construct a typed pseudo-diagram per class & dimension
exactly as described in §A.1:

  (1) Feature instantiation:  instantiate ``round(beta_d)`` target pairs.
  (2) Birth assignment:  births are anchored to the *current* diagram's
      unmatched critical birth values (preserving the present filtration
      ordering so edits stay local).  If fewer than beta_d unmatched births
      exist, the remaining births are placed at the lowest-``Downs(P_c)``
      voxels (diagonal-feasible, minimally disruptive) and the fallback is
      flagged in the certificate.
  (3) Death assignment:  deaths are placed by the predicted budget bins so each
      retained feature's lifetime exceeds its predicted threshold.

Features in the current diagram exceeding the predicted count become *spurious*;
shortfalls become *missing* -- this falls out of the matching in matching.py.

The six budget thresholds are {0.03, 0.05, 0.08, 0.10, 0.15, 0.20} (paper §4.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .ph import Diagram, Feature

BUDGET_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)


@dataclass
class TargetDiagram:
    """A typed pseudo-diagram target consumed by the bottleneck matching."""
    points_by_dim: dict          # {dim: (k,2) ndarray of (birth, death)}
    fallback_used: bool = False

    def points(self, dim: int) -> np.ndarray:
        return self.points_by_dim.get(dim, np.zeros((0, 2), np.float64))


def _budget_to_lifetime(budget_vec: np.ndarray) -> float:
    """Map a 6-d predicted budget vector to a single representative lifetime.

    Each entry is the predicted lower-bound persistence at the matching
    threshold; we take the largest active threshold so the instantiated
    feature's lifetime exceeds the predicted budget (§A.1 step 3).
    """
    b = np.asarray(budget_vec, dtype=np.float64).reshape(-1)
    thr = np.asarray(BUDGET_THRESHOLDS, dtype=np.float64)[: b.shape[0]]
    active = thr[b > 0.5]
    if active.size:
        return float(active.max())
    # default: just above the persistence-pruning delta
    return float(thr.min())


def build_pseudo_diagram(current: Diagram,
                         beta0: int, beta2: int,
                         budgets0: np.ndarray, budgets2: np.ndarray,
                         prob_down: np.ndarray,
                         delta: float = 0.05) -> TargetDiagram:
    """Construct the inference-time pseudo-diagram target for one class.

    Args:
        current:   current persistence diagram of the (downsampled) prediction.
        beta0/2:   predicted Betti counts (rounded, non-negative ints).
        budgets0/2:predicted 6-d persistence budgets per dimension.
        prob_down: the downsampled probability field (numpy, D,H,W) -- used to
                   place fallback births at the lowest-probability voxels.
        delta:     persistence-pruning threshold (used for lifetime floor).
    """
    points_by_dim = {}
    fallback = False

    for dim, beta, budgets in ((0, beta0, budgets0), (2, beta2, budgets2)):
        k = int(max(0, round(float(beta))))
        if k == 0:
            points_by_dim[dim] = np.zeros((0, 2), np.float64)
            continue

        # (2) Birth assignment -- anchor to current unmatched critical births.
        cur_feats = sorted(current.features(dim),
                           key=lambda f: f.lifetime, reverse=True)
        births: List[float] = [f.birth for f in cur_feats[:k]]

        # fallback: place remaining births at lowest-prob voxels (diagonal-feasible)
        if len(births) < k:
            need = k - len(births)
            flat = np.argsort(prob_down.ravel())            # lowest prob first
            # lowest Downs(P_c) voxels -> their prob values become the births
            extra = prob_down.ravel()[flat[:need]].tolist()
            births.extend(float(v) for v in extra)
            fallback = True

        # (3) Death assignment -- budget bins => lifetime > predicted threshold.
        life = _budget_to_lifetime(budgets)
        life = max(life, delta * 1.5)                       # ensure > pruning delta
        pts = []
        for b in births:
            d = max(0.0, b - life)                          # death below birth (prob units)
            pts.append((b, d))
        points_by_dim[dim] = np.asarray(pts, dtype=np.float64)

    return TargetDiagram(points_by_dim=points_by_dim, fallback_used=fallback)


def oracle_diagram_from_label(label_np: np.ndarray, class_id: int,
                              downsample_s: int = 2, delta: float = 0.05):
    """Build the oracle target diagram from a ground-truth label volume.

    Used during training (paper §4.5: ``Pi*_c`` is the oracle PH diagram derived
    from ground-truth labels).  Returns (points_by_dim) on the s=2 grid.
    """
    from skimage.transform import downscale_local_mean
    from .ph import compute_ph_raw, _INF

    mask = (label_np == class_id).astype(np.float64)
    if downsample_s > 1:
        mask = downscale_local_mean(
            mask, (downsample_s, downsample_s, downsample_s)) > 0.5
        mask = mask.astype(np.float64)
    rows = compute_ph_raw(-mask, maxdim=2)
    pts = {}
    for dim in (0, 2):
        d_rows = rows[rows[:, 0] == dim]
        feats = []
        for r in d_rows:
            b = -float(r[1])
            d = 0.0 if r[2] >= _INF else -float(r[2])
            if abs(b - d) > delta:
                feats.append((b, d))
        pts[dim] = np.asarray(feats, dtype=np.float64) if feats \
            else np.zeros((0, 2), np.float64)
    return TargetDiagram(points_by_dim=pts, fallback_used=False)
