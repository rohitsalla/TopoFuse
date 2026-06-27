"""
TopoFuse data loading  (paper §5.1)
===================================
Two dataset classes:

  * ``SynDataset``    : the SYN synthetic benchmark (2,000 volumes @ 64^3),
                        the format the user ships separately as a zip.
  * ``CryoETDataset`` : real benchmarks EMD-0506 / EMPIAR-10499 / EMPIAR-10045
                        (same on-disk contract, different crop size / no SNR).

────────────────────────────────────────────────────────────────────────────
EXPECTED SYN ZIP LAYOUT   (this is the contract the loader auto-detects)
────────────────────────────────────────────────────────────────────────────
    <data_root>/
        manifest.json            # index of every volume (see below)
        metadata.csv             # per-volume topology ground truth + SNR + split
        volumes/                 # the raw tomograms
            syn_00000.npy        # float32, shape (D, H, W)  (D=H=W=64 for SYN)
            syn_00001.npy
            ...
        labels/                  # integer label volumes (same shape)
            syn_00000.npy        # int, values in [0, C-1]  (0 = background)
            ...

manifest.json   — EITHER a flat list of records:
    [
      {"id": "syn_00000",
       "volume": "volumes/syn_00000.npy",
       "label":  "labels/syn_00000.npy",
       "split":  "train",
       "snr":    0.10},
      ...
    ]
                — OR a split-keyed dict:
    {"train": [ {...}, ... ], "val": [ ... ], "test": [ ... ]}

  Path fields may be relative to <data_root> or absolute.  Field NAMES are
  auto-detected from a list of aliases (see _VOL_KEYS / _LAB_KEYS / ...), so
  "vol"/"volume"/"image"/"path" all work, etc.  .npy and .mrc are both read.

metadata.csv    — one row per volume, indexed by id.  Columns are matched by
  alias (case-insensitive).  Recognised:
      id        : id | name | volume_id | filename
      split     : split | subset | fold
      snr       : snr | noise
      beta0     : beta0 | b0 | n0 | num_components | components
      beta2     : beta2 | b2 | n2 | num_voids | voids | cavities
      budgets   : budget_d0_t0 ... budget_d0_t5, budget_d2_t0 ... budget_d2_t5
                  (optional; 2 dims x 6 thresholds = 12 columns).  If absent,
                  budgets are derived exactly from the label volume via PH and
                  cached to <data_root>/.topo_cache/<id>.npz.

Ground-truth topology (beta0, beta2, budgets) is what the prior head H_pi is
regressed against (L_prior, §4.6) and what the oracle projection target is
built from at train time.  If the CSV omits beta0/beta2 they are likewise
computed exactly from the label volume.

If your zip differs, point ``--data-root`` at it and run
``python scripts/inspect_syn.py --data-root <path>`` to see exactly what the
loader detected; then either rename columns or extend the alias lists below.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter

from ..topo.pseudo_diagram import BUDGET_THRESHOLDS


# ── column / field alias tables (lower-cased, stripped) ─────────────────────
_ID_KEYS    = ("id", "name", "volume_id", "vol_id", "filename", "file")
_VOL_KEYS   = ("volume", "vol", "image", "img", "tomogram", "tomo", "data", "path")
_LAB_KEYS   = ("label", "labels", "mask", "gt", "seg", "segmentation", "label_path")
_SPLIT_KEYS = ("split", "subset", "fold", "partition")
_SNR_KEYS   = ("snr", "noise", "noise_level")
_B0_KEYS    = ("beta0", "b0", "n0", "num_components", "components", "ncomp")
_B2_KEYS    = ("beta2", "b2", "n2", "num_voids", "voids", "cavities", "nvoid")


def _first(d: dict, keys) -> Optional[str]:
    """Return the value of the first matching key (case-insensitive)."""
    low = {str(k).strip().lower(): k for k in d.keys()}
    for k in keys:
        if k in low:
            return d[low[k]]
    return None


def _load_volume(path: Path) -> np.ndarray:
    """Read a .npy or .mrc volume as float32 (D, H, W)."""
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf in (".mrc", ".rec", ".map"):
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as m:
            arr = np.asarray(m.data)
    elif suf == ".npz":
        arr = np.load(path)["arr_0"]
    else:
        raise ValueError(f"Unsupported volume format: {path}")
    return np.ascontiguousarray(arr)


# ── exact GT topology from a label volume (cached) ──────────────────────────
def compute_gt_topology(label_np: np.ndarray, num_classes: int,
                        downsample_s: int = 2, delta: float = 0.05,
                        thresholds=BUDGET_THRESHOLDS) -> Dict[str, np.ndarray]:
    """Per-class (beta0, beta2, budgets) computed exactly from a label volume.

    * beta0 / beta2 : Betti numbers of the (downsampled) binary class mask.
    * budgets[c,j,t]: normalised count of dim-(0 or 2) features whose lifetime
                      exceeds threshold t  (the lifetime-budget target H_pi
                      regresses against, §4.6).
    """
    from skimage.transform import downscale_local_mean
    from ..topo.ph import compute_ph_raw, _INF

    nthr = len(thresholds)
    beta0 = np.zeros(num_classes, np.float32)
    beta2 = np.zeros(num_classes, np.float32)
    budgets = np.zeros((num_classes, 2, nthr), np.float32)

    for c in range(num_classes):
        mask = (label_np == c).astype(np.float64)
        if downsample_s > 1:
            mask = (downscale_local_mean(
                mask, (downsample_s,) * 3) > 0.5).astype(np.float64)
        if mask.sum() == 0:
            continue
        rows = compute_ph_raw(-mask, maxdim=2)
        for j, dim in enumerate((0, 2)):
            d_rows = rows[rows[:, 0] == dim]
            lifes = []
            for r in d_rows:
                b = -float(r[1])
                dth = 0.0 if r[2] >= _INF else -float(r[2])
                lifes.append(abs(b - dth))
            lifes = np.asarray(lifes, np.float64)
            n = int((lifes > delta).sum())
            if dim == 0:
                beta0[c] = n
            else:
                beta2[c] = n
            for t, thr in enumerate(thresholds):
                budgets[c, j, t] = float((lifes > thr).sum())
    return {"beta0": beta0, "beta2": beta2, "budgets": budgets}


# ── augmentation ────────────────────────────────────────────────────────────
def _random_flip(vol, lab, rng):
    for ax in (0, 1, 2):
        if rng.random() < 0.5:
            vol = np.flip(vol, ax)
            lab = np.flip(lab, ax)
    return np.ascontiguousarray(vol), np.ascontiguousarray(lab)


def _elastic(vol, lab, rng, sigma=10.0, alpha=12.0):
    """Elastic deformation (sigma=10 per paper).  Labels use nearest-order=0."""
    shape = vol.shape
    dz = gaussian_filter((rng.random(shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((rng.random(shape) * 2 - 1), sigma, mode="constant") * alpha
    dx = gaussian_filter((rng.random(shape) * 2 - 1), sigma, mode="constant") * alpha
    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                          np.arange(shape[2]), indexing="ij")
    idx = (z + dz, y + dy, x + dx)
    vol = map_coordinates(vol, idx, order=1, mode="reflect").reshape(shape)
    lab = map_coordinates(lab, idx, order=0, mode="reflect").reshape(shape)
    return vol.astype(np.float32), lab.astype(np.int64)


def _intensity_jitter(vol, rng, scale=0.1, shift=0.1):
    return (vol * (1 + (rng.random() * 2 - 1) * scale)
            + (rng.random() * 2 - 1) * shift).astype(np.float32)


def _crop_or_pad(vol, lab, crop, rng, random_crop):
    """Random (train) or centred (eval) crop to `crop`^3, padding small volumes."""
    if crop is None:
        return vol, lab
    out_v = np.zeros((crop, crop, crop), np.float32)
    out_l = np.zeros((crop, crop, crop), np.int64)
    starts, csz = [], []
    for dim in range(3):
        s = vol.shape[dim]
        if s >= crop:
            off = rng.integers(0, s - crop + 1) if random_crop else (s - crop) // 2
            starts.append((off, 0, crop))
        else:
            pad = (crop - s) // 2
            starts.append((0, pad, s))
        csz.append(crop)
    (sz, dz, lz), (sy, dy, ly), (sx, dx, lx) = starts
    out_v[dz:dz+lz, dy:dy+ly, dx:dx+lx] = vol[sz:sz+lz, sy:sy+ly, sx:sx+lx]
    out_l[dz:dz+lz, dy:dy+ly, dx:dx+lx] = lab[sz:sz+lz, sy:sy+ly, sx:sx+lx]
    return out_v, out_l


# ── base dataset ────────────────────────────────────────────────────────────
class _ManifestDataset(Dataset):
    """Shared loader for the manifest.json + metadata.csv contract."""

    def __init__(self, root: str, split: str = "train", crop: Optional[int] = 128,
                 num_classes: int = 3, downsample_s: int = 2, delta: float = 0.05,
                 augment: bool = True, snr: Optional[float] = None,
                 normalize: bool = True, cache_topology: bool = True,
                 seed: int = 0):
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.C = num_classes
        self.s = downsample_s
        self.delta = delta
        self.augment = augment and split == "train"
        self.snr_filter = snr
        self.normalize = normalize
        self.cache_topology = cache_topology
        self.rng = np.random.default_rng(seed)

        self.records = self._read_manifest()
        self.meta = self._read_metadata()
        self._filter_split_snr()
        if cache_topology:
            (self.root / ".topo_cache").mkdir(exist_ok=True)

        if len(self.records) == 0:
            raise RuntimeError(
                f"No records for split='{split}'"
                + (f", snr={snr}" if snr is not None else "")
                + f" under {self.root}. Run scripts/inspect_syn.py to debug.")

    # -- manifest / metadata parsing ----------------------------------------
    def _read_manifest(self) -> List[dict]:
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"manifest.json not found at {mpath}")
        data = json.loads(mpath.read_text())
        if isinstance(data, dict):
            # split-keyed dict, or a wrapper {"volumes":[...]} / {"records":[...]}
            for key in ("volumes", "records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if self.split in data:
                return data[self.split]
            # flatten all splits, keep their split tag
            flat = []
            for k, v in data.items():
                if isinstance(v, list):
                    for r in v:
                        r = dict(r)
                        r.setdefault("split", k)
                        flat.append(r)
            return flat
        return list(data)

    def _read_metadata(self) -> Dict[str, dict]:
        cpath = self.root / "metadata.csv"
        if not cpath.exists():
            return {}
        out = {}
        with open(cpath, newline="") as f:
            for row in csv.DictReader(f):
                vid = _first(row, _ID_KEYS)
                if vid is None:
                    continue
                out[str(vid).strip()] = row
        return out

    def _rec_id(self, rec: dict) -> str:
        rid = _first(rec, _ID_KEYS)
        if rid is None:
            v = _first(rec, _VOL_KEYS) or ""
            rid = Path(str(v)).stem
        return str(rid).strip()

    def _rec_split(self, rec: dict) -> Optional[str]:
        s = _first(rec, _SPLIT_KEYS)
        if s is None:
            row = self.meta.get(self._rec_id(rec), {})
            s = _first(row, _SPLIT_KEYS)
        return str(s).strip().lower() if s is not None else None

    def _rec_snr(self, rec: dict) -> Optional[float]:
        s = _first(rec, _SNR_KEYS)
        if s is None:
            row = self.meta.get(self._rec_id(rec), {})
            s = _first(row, _SNR_KEYS)
        try:
            return float(s) if s is not None and str(s) != "" else None
        except ValueError:
            return None

    def _filter_split_snr(self):
        keep = []
        for rec in self.records:
            rs = self._rec_split(rec)
            if rs is not None and rs != self.split.lower():
                continue
            if self.snr_filter is not None:
                rsnr = self._rec_snr(rec)
                if rsnr is None or abs(rsnr - self.snr_filter) > 1e-6:
                    continue
            keep.append(rec)
        self.records = keep

    def _resolve(self, rel: str) -> Path:
        p = Path(str(rel))
        return p if p.is_absolute() else (self.root / p)

    # -- gt topology (csv fast-path, else exact-from-label, cached) ----------
    def _gt_topology(self, rid: str, label_np: np.ndarray) -> Dict[str, np.ndarray]:
        row = self.meta.get(rid, {})
        b0 = _first(row, _B0_KEYS)
        b2 = _first(row, _B2_KEYS)
        # budget columns budget_d{0,2}_t{0..5}
        bud = np.zeros((self.C, 2, len(BUDGET_THRESHOLDS)), np.float32)
        has_bud = False
        low = {str(k).strip().lower(): k for k in row.keys()}
        for j, dim in enumerate((0, 2)):
            for t in range(len(BUDGET_THRESHOLDS)):
                col = f"budget_d{dim}_t{t}"
                if col in low:
                    has_bud = True
                    try:
                        bud[:, j, t] = float(row[low[col]])
                    except ValueError:
                        pass

        if b0 is not None and b2 is not None and has_bud:
            beta0 = np.full(self.C, float(b0), np.float32)
            beta2 = np.full(self.C, float(b2), np.float32)
            # CSV usually reports global n0/n2 -> attribute to foreground class 1
            if self.C > 1:
                beta0[:] = 0; beta0[min(1, self.C-1)] = float(b0)
                beta2[:] = 0; beta2[min(1, self.C-1)] = float(b2)
            return {"beta0": beta0, "beta2": beta2, "budgets": bud}

        # else compute exactly from label, with on-disk cache
        if self.cache_topology:
            cp = self.root / ".topo_cache" / f"{rid}.npz"
            if cp.exists():
                z = np.load(cp)
                return {"beta0": z["beta0"], "beta2": z["beta2"],
                        "budgets": z["budgets"]}
        topo = compute_gt_topology(label_np, self.C, self.s, self.delta)
        if self.cache_topology:
            np.savez(self.root / ".topo_cache" / f"{rid}.npz", **topo)
        return topo

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        rid = self._rec_id(rec)
        vol_p = self._resolve(_first(rec, _VOL_KEYS))
        lab_rel = _first(rec, _LAB_KEYS)
        vol = _load_volume(vol_p).astype(np.float32)
        if lab_rel is not None:
            lab = _load_volume(self._resolve(lab_rel)).astype(np.int64)
        else:
            lab = np.zeros_like(vol, np.int64)

        if self.normalize:
            mu, sd = float(vol.mean()), float(vol.std()) + 1e-6
            vol = (vol - mu) / sd

        # gt topology from the *un-cropped* label (matches volume-level metric)
        topo = self._gt_topology(rid, lab)

        vol, lab = _crop_or_pad(vol, lab, self.crop, self.rng,
                                random_crop=self.augment)
        if self.augment:
            vol, lab = _random_flip(vol, lab, self.rng)
            if self.rng.random() < 0.5:
                vol, lab = _elastic(vol, lab, self.rng)
            vol = _intensity_jitter(vol, self.rng)

        return {
            "volume": torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(0),
            "label":  torch.from_numpy(np.ascontiguousarray(lab)).long(),
            "gt_topology": {
                "beta0":   torch.from_numpy(topo["beta0"]).float(),
                "beta2":   torch.from_numpy(topo["beta2"]).float(),
                "budgets": torch.from_numpy(topo["budgets"]).float(),
            },
            "id": rid,
            "snr": self._rec_snr(rec) if self._rec_snr(rec) is not None else -1.0,
        }


def collate(batch: List[dict]) -> dict:
    """Stack a list of __getitem__ dicts into a batch."""
    return {
        "volume": torch.stack([b["volume"] for b in batch]),
        "label":  torch.stack([b["label"] for b in batch]),
        "gt_topology": {
            "beta0":   torch.stack([b["gt_topology"]["beta0"] for b in batch]),
            "beta2":   torch.stack([b["gt_topology"]["beta2"] for b in batch]),
            "budgets": torch.stack([b["gt_topology"]["budgets"] for b in batch]),
        },
        "id":  [b["id"] for b in batch],
        "snr": torch.tensor([b["snr"] for b in batch], dtype=torch.float32),
    }


# ── public dataset classes ──────────────────────────────────────────────────
class SynDataset(_ManifestDataset):
    """SYN synthetic benchmark (64^3).  Default crop=64 (full volume)."""
    def __init__(self, root, split="train", crop=64, num_classes=3,
                 downsample_s=2, delta=0.05, augment=True, snr=None,
                 normalize=True, cache_topology=True, seed=0):
        super().__init__(root, split, crop, num_classes, downsample_s, delta,
                         augment, snr, normalize, cache_topology, seed)


class CryoETDataset(_ManifestDataset):
    """Real cryo-ET benchmarks.  Default crop=128 (paper §5)."""
    def __init__(self, root, split="train", crop=128, num_classes=3,
                 downsample_s=2, delta=0.05, augment=True, snr=None,
                 normalize=True, cache_topology=True, seed=0):
        super().__init__(root, split, crop, num_classes, downsample_s, delta,
                         augment, snr, normalize, cache_topology, seed)
