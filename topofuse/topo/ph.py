"""
Persistent Homology backend (CubicalRipser)
===========================================
This module implements the persistent-homology primitives used everywhere in
TopoFuse.  It follows the paper's §3 (Background) and §4.5 conventions exactly:

  * Sublevel-set filtration of a cubical scalar field (CubicalRipser [12]).
  * We track *foreground* connected components (d=0) and enclosed voids (d=2)
    by filtering ``-prob`` (sublevel of ``-prob`` == superlevel of ``prob``),
    so a feature's persistence is expressed directly in probability units.
  * Each birth/death pair is anchored to its **critical voxels** -- the exact
    cells where the feature appears/disappears.  CubicalRipser returns these
    coordinates, which is what the differentiable projection (§4.5) edits.
  * Differentiability follows Gabrielsson et al. [4]: the gradient of any
    function of (birth, death) w.r.t. the field is non-zero only at the critical
    voxels.  We realise this by *gathering* the prediction at the critical-voxel
    indices, so the returned birth/death values are differentiable PyTorch
    tensors whose gradient flows to exactly those voxels.

CubicalRipser output (``cripser.computePH``) is an ``(N, 9)`` array with columns:
    [ dim, birth, death, b_x, b_y, b_z, d_x, d_y, d_z ]
where (b_x,b_y,b_z) is the birth critical voxel and (d_x,d_y,d_z) the death one.

Persistence threshold ``delta`` (paper: 0.05) prunes near-diagonal noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cripser  # CubicalRipser python bindings  (pip install cripser)
    _HAS_CRIPSER = True
except Exception:                                            # pragma: no cover
    _HAS_CRIPSER = False

# CubicalRipser encodes essential (never-dying) features with death == +inf.
_INF = 1.79769313e308
DIMS = (0, 2)            # paper enforces / tracks d in {0, 2}; d=1 excluded (§4.5)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Feature:
    """A single persistence-diagram point with anchored critical voxels."""
    dim: int
    birth: float                       # birth value in prob units
    death: float                       # death value in prob units (0.0 if essential)
    birth_idx: Tuple[int, int, int]    # birth critical voxel (z, y, x) on PH grid
    death_idx: Tuple[int, int, int]    # death critical voxel (z, y, x) on PH grid
    essential: bool = False

    @property
    def lifetime(self) -> float:
        return abs(self.birth - self.death)

    def as_point(self) -> Tuple[float, float]:
        """(birth, death) coordinate in the persistence plane (prob units)."""
        return (self.birth, self.death)


@dataclass
class Diagram:
    """Persistence diagram for a single class, split by homological dimension."""
    by_dim: dict = field(default_factory=lambda: {d: [] for d in DIMS})
    # Differentiable birth/death tensors aligned with the per-dim feature lists.
    birth_t: dict = field(default_factory=dict)
    death_t: dict = field(default_factory=dict)

    def features(self, dim: int) -> List[Feature]:
        return self.by_dim.get(dim, [])

    def betti(self, dim: int, persistence_thresh: float = 0.0) -> int:
        return sum(1 for f in self.features(dim) if f.lifetime > persistence_thresh)

    def points(self, dim: int) -> np.ndarray:
        feats = self.features(dim)
        if not feats:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([f.as_point() for f in feats], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Raw PH on a numpy field
# ─────────────────────────────────────────────────────────────────────────────
def compute_ph_raw(field_np: np.ndarray, maxdim: int = 2) -> np.ndarray:
    """Run CubicalRipser on a 3-D scalar field (sublevel filtration).

    Returns the raw (N, 9) array.  Caller is responsible for the sign convention
    of ``field_np`` (we feed ``-prob`` for foreground tracking).
    """
    if not _HAS_CRIPSER:
        raise RuntimeError(
            "CubicalRipser ('cripser') is not installed.  Install it with "
            "`pip install cripser`.  It is required for the exact PH projection."
        )
    arr = np.ascontiguousarray(field_np.astype(np.float64))
    return cripser.computePH(arr, maxdim=maxdim)


def _rows_to_features(rows: np.ndarray, dim: int, prob_shape: Tuple[int, int, int],
                      delta: float) -> List[Feature]:
    """Convert CubicalRipser rows (on the ``-prob`` field) into prob-space Features."""
    feats: List[Feature] = []
    Dz, Dy, Dx = prob_shape
    for r in rows:
        b_neg, d_neg = float(r[1]), float(r[2])
        essential = d_neg >= _INF
        # field = -prob  =>  prob = -field.   Birth at high prob (low -prob).
        birth = -b_neg
        death = 0.0 if essential else -d_neg
        bz, by, bx = int(r[3]), int(r[4]), int(r[5])
        if essential:
            dz_, dy_, dx_ = bz, by, bx          # no finite death cell
        else:
            dz_, dy_, dx_ = int(r[6]), int(r[7]), int(r[8])
        # clamp indices into the PH grid
        clamp = lambda v, hi: max(0, min(int(v), hi - 1))
        bidx = (clamp(bz, Dz), clamp(by, Dy), clamp(bx, Dx))
        didx = (clamp(dz_, Dz), clamp(dy_, Dy), clamp(dx_, Dx))
        f = Feature(dim, birth, death, bidx, didx, essential)
        if f.lifetime > delta:                   # persistence-threshold pruning
            feats.append(f)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable diagram from a probability tensor (the main entry point)
# ─────────────────────────────────────────────────────────────────────────────
def differentiable_diagram(prob: torch.Tensor, maxdim: int = 2,
                           delta: float = 0.05) -> Diagram:
    """Compute a persistence diagram whose birth/death values are differentiable.

    Args:
        prob:   3-D probability tensor (D, H, W), values in [0, 1] (e.g. on the
                downsampled s=2 grid).  ``requires_grad`` may be True.
        maxdim: maximum homological dimension to compute (2).
        delta:  persistence threshold (paper: 0.05).

    Returns:
        ``Diagram`` with ``by_dim`` Feature lists and, for each dim, differentiable
        ``birth_t``/``death_t`` tensors gathered from ``prob`` at the critical
        voxels.  This realises the critical-cell gradient rule of [4].
    """
    assert prob.dim() == 3, f"expected (D,H,W), got {tuple(prob.shape)}"
    prob_np = prob.detach().float().cpu().numpy()
    field_np = -prob_np                          # sublevel(-prob) == superlevel(prob)
    rows = compute_ph_raw(field_np, maxdim=maxdim)

    dgm = Diagram(by_dim={d: [] for d in range(maxdim + 1)})
    for d in range(maxdim + 1):
        d_rows = rows[rows[:, 0] == d]
        feats = _rows_to_features(d_rows, d, prob.shape, delta)
        dgm.by_dim[d] = feats

        if feats:
            b_idx = torch.tensor([f.birth_idx for f in feats], device=prob.device,
                                 dtype=torch.long)
            d_idx = torch.tensor([f.death_idx for f in feats], device=prob.device,
                                 dtype=torch.long)
            # Differentiable gather: birth/death = prob at the critical voxels.
            birth_t = prob[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]]
            # essential features die at prob floor 0 (constant, no grad needed)
            ess = torch.tensor([f.essential for f in feats], device=prob.device)
            death_gather = prob[d_idx[:, 0], d_idx[:, 1], d_idx[:, 2]]
            death_t = torch.where(ess, torch.zeros_like(death_gather), death_gather)
        else:
            birth_t = prob.new_zeros(0)
            death_t = prob.new_zeros(0)
        dgm.birth_t[d] = birth_t
        dgm.death_t[d] = death_t
    # keep only requested DIMS in the convenience accessor while preserving tensors
    return dgm


def betti_numbers(prob_np: np.ndarray, delta: float = 0.05,
                  maxdim: int = 2) -> dict:
    """Exact Betti numbers (per dim) of a probability field via PH (no gradients)."""
    field_np = -prob_np.astype(np.float64)
    rows = compute_ph_raw(field_np, maxdim=maxdim)
    out = {}
    for d in range(maxdim + 1):
        d_rows = rows[rows[:, 0] == d]
        life = np.abs(d_rows[:, 1] - np.where(d_rows[:, 2] >= _INF,
                                              0.0, d_rows[:, 2]))
        out[d] = int((life > delta).sum())
    return out
