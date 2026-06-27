"""
Evaluation metrics  (paper §5.1)
================================
Voxel quality   : Dice, IoU, Normalised Surface Dice (NSD, 2-voxel tolerance).
Topology        : BE_d = E[|beta_hat_d - beta*_d|] for d in {0, 2}, and
                  Betti Matching Error (BME) via diagram matching.
Certificate     : edit sparsity, convergence rate, iteration count.

Topology metrics follow the BettiMatching protocol (Stucki et al. 2023):
binarised predictions at s=2, threshold 0.5, 26-connectivity for the
foreground.  Betti numbers are computed exactly (PH on the binary mask via
CubicalRipser) rather than estimated from probability lifetimes.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.transform import downscale_local_mean

from ..topo.ph import compute_ph_raw, _INF
from ..topo.matching import bottleneck


# ── voxel metrics ───────────────────────────────────────────────────────────
def dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + smooth) / (union + smooth))


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def nsd(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """Normalised Surface Dice at `tol`-voxel tolerance."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    sp, sg = _surface(pred), _surface(gt)
    if sp.sum() == 0 and sg.sum() == 0:
        return 1.0
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    dt_g = distance_transform_edt(~sg)
    dt_p = distance_transform_edt(~sp)
    p_ok = (dt_g[sp] <= tol).sum()
    g_ok = (dt_p[sg] <= tol).sum()
    return float((p_ok + g_ok) / (sp.sum() + sg.sum()))


# ── exact Betti numbers of a binary mask (PH, CubicalRipser) ────────────────
def betti_binary(mask: np.ndarray) -> Dict[int, int]:
    """Exact Betti numbers b0 (components) and b2 (cavities) of a 3-D binary
    mask via persistent homology on the binary field.  Essential features of
    the superlevel filtration are the topology of the mask itself."""
    mask = mask.astype(np.float64)
    if mask.sum() == 0:
        return {0: 0, 2: 0}
    rows = compute_ph_raw(-mask, maxdim=2)          # superlevel of mask
    out = {0: 0, 2: 0}
    for d in (0, 2):
        d_rows = rows[rows[:, 0] == d]
        # essential (death at +inf) features = true Betti number of the mask
        ess = np.sum(d_rows[:, 2] >= _INF)
        # for d=0 there is always >=1 essential comp per nonempty mask; PH on the
        # binary field returns exactly the count of components / cavities.
        out[d] = int(ess)
    return out


def _binarise_downsample(prob: np.ndarray, thr: float = 0.5, s: int = 2):
    b = (prob >= thr)
    if s > 1:
        b = downscale_local_mean(b.astype(np.float64), (s,) * 3) >= 0.5
    return b.astype(bool)


def betti_errors(prob_pred: np.ndarray, gt_mask: np.ndarray,
                 thr: float = 0.5, s: int = 2) -> Dict[str, float]:
    """BE_0 and BE_2 between a predicted prob map and a GT binary mask, both
    binarised + downsampled to s=2 (26-connectivity foreground)."""
    bp = _binarise_downsample(prob_pred, thr, s)
    bg = _binarise_downsample(gt_mask.astype(np.float64), 0.5, s)
    betti_p = betti_binary(bp)
    betti_g = betti_binary(bg)
    return {
        "BE0": float(abs(betti_p[0] - betti_g[0])),
        "BE2": float(abs(betti_p[2] - betti_g[2])),
        "beta0_pred": betti_p[0], "beta0_gt": betti_g[0],
        "beta2_pred": betti_p[2], "beta2_gt": betti_g[2],
    }


# ── Betti Matching Error (diagram-level) ────────────────────────────────────
def _diagram(prob: np.ndarray, s: int, delta: float, dim: int) -> np.ndarray:
    field = -prob.astype(np.float64)
    if s > 1:
        field = downscale_local_mean(field, (s,) * 3)
    rows = compute_ph_raw(field, maxdim=2)
    d_rows = rows[rows[:, 0] == dim]
    pts = []
    for r in d_rows:
        b = -float(r[1])
        d = 0.0 if r[2] >= _INF else -float(r[2])
        if abs(b - d) > delta:
            pts.append((min(b, d), max(b, d)))
    return np.asarray(pts, np.float64) if pts else np.zeros((0, 2))


def bme(prob_pred: np.ndarray, gt_mask: np.ndarray,
        s: int = 2, delta: float = 0.05) -> float:
    """Betti Matching Error: mean bottleneck distance between pred and GT
    persistence diagrams over d in {0, 2} (proxy for the induced matching of
    Stucki et al. 2023)."""
    gt = gt_mask.astype(np.float64)
    tot = 0.0
    for dim in (0, 2):
        dp = _diagram(prob_pred, s, delta, dim)
        dg = _diagram(gt, s, delta, dim)
        tot += bottleneck(dp, dg)
    return float(tot / 2.0)


# ── certificate aggregation ─────────────────────────────────────────────────
def certificate_stats(certs: List) -> Dict[str, float]:
    """Aggregate repair certificates C = (I, V_C, Delta_C) across a batch/set."""
    if not certs:
        return {"conv_rate": float("nan"), "mean_sparsity": float("nan"),
                "mean_iters": float("nan"), "fallback_rate": float("nan")}
    conv, spars, iters, fb = [], [], [], []
    for c in certs:
        conv.append(float(getattr(c, "converged", False)))
        spars.append(float(getattr(c, "spatial_sparsity", 0.0)))
        iters.append(float(getattr(c, "iterations", 0)))
        fb.append(float(getattr(c, "fallback_used", False)))
    return {
        "conv_rate": float(np.mean(conv)),
        "mean_sparsity": float(np.mean(spars)),
        "median_sparsity": float(np.median(spars)),
        "mean_iters": float(np.mean(iters)),
        "fallback_rate": float(np.mean(fb)),
    }


# ── top-level per-volume metric bundle ──────────────────────────────────────
def compute_metrics(prob_post: np.ndarray, label: np.ndarray,
                    num_classes: int, prob_pre: Optional[np.ndarray] = None,
                    thr: float = 0.5, s: int = 2, delta: float = 0.05,
                    nsd_tol: float = 2.0) -> Dict[str, float]:
    """Per-volume metrics, averaged over foreground classes.

    prob_post : (C, D, H, W) post-projection probabilities (one-vs-rest sigmoid)
    label     : (D, H, W) int GT labels in [0, C-1]
    """
    fg = range(1, num_classes) if num_classes > 1 else range(num_classes)
    D, I, N, B0, B2, M = [], [], [], [], [], []
    for c in fg:
        pc = prob_post[c]
        gc = (label == c)
        hard = pc >= thr
        D.append(dice(hard, gc))
        I.append(iou(hard, gc))
        N.append(nsd(hard, gc, nsd_tol))
        be = betti_errors(pc, gc, thr, s)
        B0.append(be["BE0"]); B2.append(be["BE2"])
        M.append(bme(pc, gc, s, delta))
    out = {
        "Dice": float(np.mean(D)), "IoU": float(np.mean(I)),
        "NSD": float(np.mean(N)),  "BE0": float(np.mean(B0)),
        "BE2": float(np.mean(B2)), "BME": float(np.mean(M)),
    }
    if prob_pre is not None:
        pre_d = [dice(prob_pre[c] >= thr, label == c) for c in fg]
        out["Dice_pre"] = float(np.mean(pre_d))
    return out


def aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean ± std over a list of per-volume metric dicts."""
    if not records:
        return {}
    keys = records[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records if k in r], float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out
