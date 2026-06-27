"""
Persistence-diagram matching
============================
Implements the matching operations the projection (§4.5) and the topology loss
(§4.7) depend on:

  * ``bottleneck`` distance d_B (paper uses GUDHI; falls back to a Hungarian
    L_inf bottleneck if GUDHI is unavailable).
  * ``optimal_matching`` -- the assignment that realises the diagram distance,
    augmented with the diagonal so points may be matched to their projection.
  * ``unmatched_features`` -- splits the current diagram against a target into
    **spurious** (present now, absent in target -> remove) and **missing**
    (present in target, absent now -> create), exactly as in §4.5.
  * ``wasserstein_cost`` -- a *differentiable* Wasserstein-2 matching cost used
    by the pre-projection topology loss L_topo (§4.7).

The paper deliberately uses the **bottleneck** distance inside the projection
(it targets the single worst-case unmatched feature per step, aligning with
sparse repair), and a **Wasserstein** loss outside it for smooth gradients.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .ph import Diagram, Feature

try:
    import gudhi
    _HAS_GUDHI = True
except Exception:                                            # pragma: no cover
    _HAS_GUDHI = False


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck distance
# ─────────────────────────────────────────────────────────────────────────────
def _diag_proj(p: np.ndarray) -> np.ndarray:
    """Project a point onto the diagonal: ((b+d)/2, (b+d)/2)."""
    m = (p[0] + p[1]) / 2.0
    return np.array([m, m])


def _canonical(points: np.ndarray) -> np.ndarray:
    """Orient points above the diagonal (birth <= death) so persistence is
    positive.  TopoFuse uses a superlevel filtration, which yields points below
    the diagonal (high-prob birth, low-prob death); GUDHI's bottleneck assumes
    the standard above-diagonal convention, so we canonicalise before calling
    it.  Persistence |b-d| and distance-to-diagonal are preserved."""
    if len(points) == 0:
        return points
    p = np.asarray(points, np.float64).reshape(-1, 2)
    lo = np.minimum(p[:, 0], p[:, 1])
    hi = np.maximum(p[:, 0], p[:, 1])
    return np.stack([lo, hi], axis=1)


def bottleneck(points_pred: np.ndarray, points_tgt: np.ndarray) -> float:
    """Bottleneck distance d_B between two diagrams given as (N,2) arrays."""
    if _HAS_GUDHI:
        A = _canonical(points_pred).tolist() if len(points_pred) else []
        B = _canonical(points_tgt).tolist() if len(points_tgt) else []
        return float(gudhi.bottleneck_distance(A, B))
    # ---- Hungarian L_inf bottleneck fallback (orientation-invariant) ----------
    cost, _, _ = _augmented_cost(points_pred, points_tgt, order=np.inf)
    if cost.size == 0:
        return 0.0
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max())


# ─────────────────────────────────────────────────────────────────────────────
# Optimal matching with diagonal augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augmented_cost(P: np.ndarray, Q: np.ndarray, order=2.0):
    """Build the (n+m) x (n+m) augmented cost matrix used for optimal matching.

    Rows  = pred points (0..n-1)  then diagonal slots for target (n..n+m-1)
    Cols  = target points (0..m-1) then diagonal slots for pred  (m..m+n-1)
    """
    n, m = len(P), len(Q)
    size = n + m
    INF = 1e9
    cost = np.full((size, size), 0.0, dtype=np.float64)

    def dist(a, b):
        if order == np.inf:
            return np.max(np.abs(a - b))
        return np.linalg.norm(a - b, ord=order)

    # pred i <-> target j
    for i in range(n):
        for j in range(m):
            cost[i, j] = dist(P[i], Q[j])
    # pred i <-> its diagonal projection (cols m..m+n-1)
    for i in range(n):
        dp = dist(P[i], _diag_proj(P[i]))
        for j in range(m, m + n):
            cost[i, j] = dp if (j - m) == i else INF
    # target j diagonal (rows n..n+m-1) <-> target j
    for r in range(n, n + m):
        jj = r - n
        dq = dist(Q[jj], _diag_proj(Q[jj]))
        cost[r, jj] = dq
        for j in range(m):
            if j != jj:
                cost[r, j] = INF
    # diagonal <-> diagonal: free
    for r in range(n, n + m):
        for j in range(m, m + n):
            cost[r, j] = 0.0
    return cost, n, m


def optimal_matching(P: np.ndarray, Q: np.ndarray, order=2.0):
    """Return (row_ind, col_ind, cost_matrix, n, m) of the optimal assignment."""
    cost, n, m = _augmented_cost(P, Q, order=order)
    if cost.size == 0:
        return np.array([], int), np.array([], int), cost, n, m
    r, c = linear_sum_assignment(cost)
    return r, c, cost, n, m


# ─────────────────────────────────────────────────────────────────────────────
# Spurious / missing feature identification (§4.5)
# ─────────────────────────────────────────────────────────────────────────────
def unmatched_features(dgm: Diagram, target_points: np.ndarray, dim: int,
                       order=2.0) -> Tuple[List[Feature], List[Tuple[float, float]]]:
    """Split ``dgm`` (current) vs ``target_points`` into spurious & missing.

    Returns:
        spurious : list of current Features matched to the diagonal
                   (present now, absent in target -> must be removed).
        missing  : list of target (birth, death) points matched to the diagonal
                   (present in target, absent now -> must be created).
    """
    feats = dgm.features(dim)
    P = dgm.points(dim)
    Q = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    n, m = len(P), len(Q)

    if n == 0 and m == 0:
        return [], []
    if m == 0:                                   # everything is spurious
        return list(feats), []
    if n == 0:                                   # everything is missing
        return [], [tuple(q) for q in Q]

    r, c, cost, n, m = optimal_matching(P, Q, order=order)
    spurious, missing = [], []
    for ri, ci in zip(r, c):
        if ri < n and ci >= m:                   # pred i -> diagonal => spurious
            spurious.append(feats[ri])
        elif ri >= n and ci < m:                 # target j -> diagonal => missing
            missing.append(tuple(Q[ci]))
    return spurious, missing


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Wasserstein-2 cost for L_topo (§4.7)
# ─────────────────────────────────────────────────────────────────────────────
def wasserstein_cost(dgm_pred: Diagram, dgm_tgt: Diagram, dim: int,
                     device=None) -> torch.Tensor:
    """Differentiable Wasserstein-2 matching cost between two diagrams.

    The optimal assignment is computed on detached coordinates (combinatorial,
    non-differentiable), then the squared transport cost is *re-expressed* using
    the differentiable birth/death tensors so gradients flow to the critical
    voxels (critical-cell rule, [4]).
    """
    device = device or (dgm_pred.birth_t[dim].device
                        if dim in dgm_pred.birth_t and dgm_pred.birth_t[dim].numel()
                        else (dgm_tgt.birth_t[dim].device
                              if dim in dgm_tgt.birth_t and dgm_tgt.birth_t[dim].numel()
                              else torch.device("cpu")))
    P = dgm_pred.points(dim)
    Q = dgm_tgt.points(dim)
    n, m = len(P), len(Q)
    if n == 0 and m == 0:
        return torch.zeros((), device=device)

    r, c, cost, n, m = optimal_matching(P, Q, order=2.0)

    bp = dgm_pred.birth_t.get(dim, torch.zeros(0, device=device))
    dp = dgm_pred.death_t.get(dim, torch.zeros(0, device=device))
    bq = dgm_tgt.birth_t.get(dim, torch.zeros(0, device=device))
    dq = dgm_tgt.death_t.get(dim, torch.zeros(0, device=device))

    total = torch.zeros((), device=device)
    for ri, ci in zip(r, c):
        if ri < n and ci < m:                    # matched pred<->target
            pr = torch.stack([bp[ri], dp[ri]])
            tg = torch.stack([bq[ci], dq[ci]])
            total = total + ((pr - tg) ** 2).sum()
        elif ri < n and ci >= m:                 # pred -> diagonal
            mid = (bp[ri] + dp[ri]) / 2.0
            proj = torch.stack([mid, mid])
            pr = torch.stack([bp[ri], dp[ri]])
            total = total + ((pr - proj) ** 2).sum()
        elif ri >= n and ci < m:                 # target -> diagonal (no pred grad)
            mid = (bq[ci] + dq[ci]) / 2.0
            proj = torch.stack([mid, mid])
            tg = torch.stack([bq[ci], dq[ci]])
            total = total + ((tg - proj) ** 2).sum()
    return total
