"""
SYN Synthetic Cryo-ET Benchmark Generator
==========================================
Paper spec:
  - 2000 volumes, 64^3 resolution
  - n0 ~ Unif(1,5) connected components (d=0)
  - n2 ~ Unif(0,2) enclosed voids (d=2)
  - Missing-wedge filter in Fourier space
  - Poisson noise at SNR in {0.05, 0.10, 0.30}
  - Train/Val/Test split: 1400 / 200 / 400
  - Ground-truth labels with exact topology known by construction
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path
from scipy.ndimage import (
    binary_fill_holes, label as ndlabel,
    gaussian_filter, distance_transform_edt
)
from skimage.morphology import ball, binary_dilation, binary_erosion
try:
    from tqdm import tqdm
except ImportError:                      # graceful fallback if tqdm absent
    def tqdm(total=None, desc=None, **k):
        class _N:
            def update(self, n=1): pass
            def close(self): pass
        if desc:
            print(desc + " ...")
        return _N()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def make_ellipsoid(shape, center, radii):
    """Binary ellipsoid mask."""
    zz, yy, xx = np.ogrid[
        :shape[0], :shape[1], :shape[2]
    ]
    dist = (
        ((zz - center[0]) / radii[0]) ** 2 +
        ((yy - center[1]) / radii[1]) ** 2 +
        ((xx - center[2]) / radii[2]) ** 2
    )
    return dist <= 1.0


def make_hollow_ellipsoid(shape, center, radii, shell_thickness=3):
    """
    Hollow ellipsoid = outer ellipsoid minus inner ellipsoid.
    Creates a closed membrane-like surface enclosing a void (d=2 feature).
    """
    outer = make_ellipsoid(shape, center, radii)
    inner_radii = np.maximum(radii - shell_thickness, 1.0)
    inner = make_ellipsoid(shape, center, inner_radii)
    return outer & ~inner, inner  # shell mask, void mask


def place_components(rng, shape, n0, n2, margin=8, min_radius=5, max_radius=14):
    """
    Place n0 solid components and n2 hollow components (vesicles).
    Returns:
        label_mask  : uint8 foreground mask
        void_mask   : uint8 interior void mask (inside hollow shells)
        topology    : dict with true beta0, beta2
        placement   : list of dicts for each object
    """
    vol = np.zeros(shape, dtype=np.uint8)
    void_vol = np.zeros(shape, dtype=np.uint8)
    placement = []
    placed_centers = []
    placed_radii = []

    def no_overlap(c, r):
        for pc, pr in zip(placed_centers, placed_radii):
            if np.linalg.norm(np.array(c) - np.array(pc)) < (r + pr + 3):
                return False
        return True

    total = n0 + n2

    # ── Solid components (contribute to β0, no β2) ──────────────────────────
    for i in range(n0):
        for _ in range(200):  # placement attempts
            r = rng.integers(min_radius, max_radius + 1)
            radii = rng.uniform(0.7, 1.3, size=3) * r
            radii = np.clip(radii, 3, max_radius)
            c = [
                rng.integers(margin + int(radii[k]), shape[k] - margin - int(radii[k]))
                for k in range(3)
            ]
            if no_overlap(c, max(radii)):
                mask = make_ellipsoid(shape, c, radii)
                vol |= mask.astype(np.uint8)
                placed_centers.append(c)
                placed_radii.append(max(radii))
                placement.append({"type": "solid", "center": c, "radii": radii.tolist()})
                break

    # ── Hollow components (contribute to β0 AND β2) ─────────────────────────
    for i in range(n2):
        for _ in range(200):
            r = rng.integers(min_radius + 3, max_radius + 1)
            radii = rng.uniform(0.8, 1.2, size=3) * r
            radii = np.clip(radii, 5, max_radius)
            shell_t = rng.integers(2, 5)
            c = [
                rng.integers(margin + int(radii[k]), shape[k] - margin - int(radii[k]))
                for k in range(3)
            ]
            if no_overlap(c, max(radii)):
                shell, void = make_hollow_ellipsoid(shape, c, radii, shell_t)
                vol |= shell.astype(np.uint8)
                void_vol |= void.astype(np.uint8)
                placed_centers.append(c)
                placed_radii.append(max(radii))
                placement.append({
                    "type": "hollow",
                    "center": c,
                    "radii": radii.tolist(),
                    "shell_thickness": int(shell_t)
                })
                break

    # Verify actual topology using connected components
    labeled, actual_n0 = ndlabel(vol)
    labeled_void, actual_n2 = ndlabel(void_vol)

    return vol, void_vol, {
        "beta0": int(actual_n0),
        "beta2": int(actual_n2),
        "n0_requested": n0,
        "n2_requested": n2,
    }, placement


# ─────────────────────────────────────────────────────────────────────────────
# Missing-wedge filter
# ─────────────────────────────────────────────────────────────────────────────

def missing_wedge_mask(shape, tilt_angle_deg=60.0):
    """
    Binary mask in Fourier space implementing the missing wedge.
    Standard cryo-ET tilt range: ±tilt_angle_deg around Y axis.
    Frequencies outside the tilt range are zeroed.

    shape : (D, H, W) — Z is the axial (beam) direction
    """
    D, H, W = shape
    kz = np.fft.fftfreq(D)[:, None, None]
    kx = np.fft.fftfreq(W)[None, None, :]
    # Missing wedge: |kz| > |kx| * tan(tilt_angle) → missing
    tilt_rad = np.deg2rad(tilt_angle_deg)
    # Accessible region: |kz / kx| <= tan(tilt_angle)  (with kx != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(kz) / (np.abs(kx) + 1e-12)
    mask = ratio <= np.tan(tilt_rad)
    mask[:, :, 0] = 1  # kx=0 plane: always accessible (DC)
    return mask.astype(np.float32)


def apply_missing_wedge(volume, tilt_angle_deg=60.0):
    """Apply missing-wedge filter to a float volume."""
    F = np.fft.fftn(volume)
    mw = missing_wedge_mask(volume.shape, tilt_angle_deg)
    F_filtered = F * mw
    return np.real(np.fft.ifftn(F_filtered))


# ─────────────────────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────────────────────

def add_poisson_noise(volume, snr):
    """
    Add Poisson noise to achieve target SNR.
    SNR = signal_mean / noise_std  (for Poisson: noise_std = sqrt(mean))
    We scale the signal so that sqrt(scale * signal_mean) / (scale * signal_mean)
    gives the desired SNR, then add Poisson-distributed counts.

    For Poisson: SNR = sqrt(lambda) where lambda is mean photon count.
    So lambda = SNR^2. We normalise signal to [0,1], scale to lambda,
    sample Poisson, then normalise back.
    """
    sig = volume - volume.min()
    sig_max = sig.max()
    if sig_max < 1e-8:
        sig_max = 1.0
    sig = sig / sig_max  # [0, 1]

    # Background photon level
    lambda_bg = (1.0 / snr) ** 2
    lambda_signal = lambda_bg * sig

    noisy = np.random.poisson(lambda_signal + lambda_bg).astype(np.float32)
    # Normalise back to [0,1] range
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min() + 1e-8)
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# Single volume generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_volume(rng, shape=(64, 64, 64), snr=0.10, tilt_angle=60.0):
    """
    Generate one synthetic cryo-ET volume with ground-truth mask and topology.

    Returns:
        noisy_vol   : float32 (D,H,W)  — corrupted input
        label_mask  : uint8  (D,H,W)  — binary GT segmentation
        topo        : dict  — true beta0, beta2
        placement   : list — object metadata
    """
    n0 = int(rng.integers(1, 6))   # Unif(1,5)
    n2 = int(rng.integers(0, 3))   # Unif(0,2)

    label_mask, void_mask, topo, placement = place_components(
        rng, shape, n0, n2
    )

    # ── Build clean signal ───────────────────────────────────────────────────
    # Foreground = 1.0, background = 0.0, smooth edges slightly
    clean = label_mask.astype(np.float32)
    clean = gaussian_filter(clean, sigma=0.8)

    # ── Missing-wedge degradation ────────────────────────────────────────────
    degraded = apply_missing_wedge(clean, tilt_angle_deg=tilt_angle)

    # ── Poisson noise ─────────────────────────────────────────────────────────
    noisy = add_poisson_noise(degraded, snr=snr)

    return noisy.astype(np.float32), label_mask.astype(np.uint8), topo, placement


# ─────────────────────────────────────────────────────────────────────────────
# Exact GT topology (matches topofuse.data.dataset.compute_gt_topology)
# kept torch-free so the generator depends only on numpy/scipy/skimage/cripser
# ─────────────────────────────────────────────────────────────────────────────
BUDGET_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
_INF = 1.7976931348623157e308


def exact_topology(label_np, num_classes, downsample_s=2, delta=0.05,
                   thresholds=BUDGET_THRESHOLDS):
    """Per-class (beta0, beta2, budgets[C,2,T]) computed by PH on the label."""
    import cripser
    from skimage.transform import downscale_local_mean
    T = len(thresholds)
    beta0 = np.zeros(num_classes, np.float32)
    beta2 = np.zeros(num_classes, np.float32)
    budgets = np.zeros((num_classes, 2, T), np.float32)
    for c in range(num_classes):
        mask = (label_np == c).astype(np.float64)
        if downsample_s > 1:
            mask = (downscale_local_mean(mask, (downsample_s,) * 3) > 0.5).astype(np.float64)
        if mask.sum() == 0:
            continue
        rows = cripser.computePH(-mask, maxdim=2)
        for j, dim in enumerate((0, 2)):
            d_rows = rows[rows[:, 0] == dim]
            lifes = []
            for r in d_rows:
                b = -float(r[1]); dth = 0.0 if r[2] >= _INF else -float(r[2])
                lifes.append(abs(b - dth))
            lifes = np.asarray(lifes, np.float64)
            n = int((lifes > delta).sum())
            (beta0 if dim == 0 else beta2)[c] = n
            for t, thr in enumerate(thresholds):
                budgets[c, j, t] = float((lifes > thr).sum())
    return beta0, beta2, budgets


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation  →  manifest.json + metadata.csv + volumes/ + labels/
# (exactly the layout topofuse.data.SynDataset auto-detects)
# ─────────────────────────────────────────────────────────────────────────────
SPLITS = {"train": 1400, "val": 200, "test": 400}   # = 2000 volumes (paper §5.1)
SNRS   = [0.05, 0.10, 0.30]                          # paper spec
SHAPE  = (64, 64, 64)
TILT   = 60.0                                        # ±60° standard cryo-ET


def generate_dataset(out_dir: Path, seed: int = 42, num_classes: int = 2,
                     compute_budgets: bool = True):
    """Generate SYN.  One geometry per volume; one file (and one manifest record
    + CSV row) per (volume × SNR).  Foreground membrane = class 1, bg = class 0."""
    import csv
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    manifest, csv_rows = [], []
    bud_cols = [f"budget_d{d}_t{t}" for d in (0, 2) for t in range(len(BUDGET_THRESHOLDS))]
    total = sum(SPLITS.values())
    pbar = tqdm(total=total * len(SNRS), desc="Generating SYN")

    for split, n_vols in SPLITS.items():
        for idx in range(n_vols):
            n0 = int(rng.integers(1, 6))     # Unif(1,5)
            n2 = int(rng.integers(0, 3))     # Unif(0,2)
            label_mask, void_mask, topo, placement = place_components(rng, SHAPE, n0, n2)
            # foreground membrane -> class 1 (matches one-vs-rest channel layout)
            label = label_mask.astype(np.uint8)   # already {0,1}
            clean = gaussian_filter(label.astype(np.float32), sigma=0.8)
            degraded = apply_missing_wedge(clean, tilt_angle_deg=TILT)

            base = f"{split}_{idx:04d}"
            lab_rel = f"labels/{base}.npy"
            np.save(out_dir / lab_rel, label)

            if compute_budgets:
                b0, b2, budgets = exact_topology(label, num_classes)
            else:
                b0 = np.zeros(num_classes); b0[1] = topo["beta0"]
                b2 = np.zeros(num_classes); b2[1] = topo["beta2"]
                budgets = np.zeros((num_classes, 2, len(BUDGET_THRESHOLDS)))

            for snr in SNRS:
                noisy = add_poisson_noise(degraded.copy(), snr=snr).astype(np.float32)
                tag = str(snr).replace(".", "p")
                vid = f"{base}_snr{tag}"
                vol_rel = f"volumes/{vid}.npy"
                np.save(out_dir / vol_rel, noisy)
                manifest.append({"id": vid, "volume": vol_rel, "label": lab_rel,
                                 "split": split, "snr": snr})
                row = {"id": vid, "split": split, "snr": snr,
                       "beta0": int(b0[1]), "beta2": int(b2[1])}
                for d_i, d in enumerate((0, 2)):
                    for t in range(len(BUDGET_THRESHOLDS)):
                        # store the foreground (class-1) budget
                        row[f"budget_d{d}_t{t}"] = float(budgets[1, d_i, t])
                csv_rows.append(row)
                pbar.update(1)
    pbar.close()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    fieldnames = ["id", "split", "snr", "beta0", "beta2"] + bud_cols
    with open(out_dir / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    print(f"\nSYN written to {out_dir}")
    print(f"  volumes : {len(manifest)} files ({total} geometries × {len(SNRS)} SNR)")
    print(f"  manifest.json : {len(manifest)} records")
    print(f"  metadata.csv  : {len(csv_rows)} rows, columns = {fieldnames}")
    b0s = [r['beta0'] for r in csv_rows]; b2s = [r['beta2'] for r in csv_rows]
    print(f"  beta0 {np.mean(b0s):.2f}±{np.std(b0s):.2f}   "
          f"beta2 {np.mean(b2s):.2f}±{np.std(b2s):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the SYN cryo-ET benchmark")
    ap.add_argument("--out_dir", type=str, default="./SYN_dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=2,
                    help="incl. background; SYN membrane = class 1")
    ap.add_argument("--no_budgets", action="store_true",
                    help="skip exact PH budget columns (faster)")
    args = ap.parse_args()
    generate_dataset(Path(args.out_dir), seed=args.seed,
                     num_classes=args.num_classes,
                     compute_budgets=not args.no_budgets)
