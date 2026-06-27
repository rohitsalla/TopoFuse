#!/usr/bin/env python3
"""
EMPIAR / EMD ingestion -> TopoFuse loader contract
==================================================
Converts a downloaded real cryo-ET entry (EMPIAR-10499 membranes,
EMPIAR-10045 ribosomes, EMD-0506, ...) into the manifest.json + metadata.csv +
volumes/ + labels/ layout that ``topofuse.data.CryoETDataset`` reads.

Tomograms are large, so by default the original .mrc volumes are *referenced in
place* (absolute paths in the manifest) rather than copied; only label volumes
are written (compact .npy). The dataset's 128^3 random crop / sliding-window
inference then operates on the full tomograms directly.

Three annotation modes (pick with --ann-type):

  coords   : a coordinate list per tomogram (.star / .csv / .coords / .tsv) with
             columns for (z, y, x) particle centres.  Spheres of --radius voxels
             of class --class-id are painted into the label volume.  Use for
             particle-pick datasets (e.g. ribosomes, EMPIAR-10045).
  mask     : an existing dense label/mask volume per tomogram (.mrc/.npy).
             Values are mapped to [0, C-1] (binarised by default, or kept if
             already integer-classed).  Use for membrane segmentations
             (e.g. EMPIAR-10499).
  instance : an instance-segmentation volume per tomogram; every non-zero
             instance is mapped to the same foreground class --class-id
             (or kept per-class with --keep-instance-classes).

Pairing tomograms <-> annotations is by shared stem, or by an explicit
--pairs CSV with columns: tomogram, annotation[, split, snr].

Run scripts/inspect_syn.py --data-root <out> afterwards to verify, then train
with:  bash scripts/run_train.sh <out> cryoet 0
"""
from __future__ import annotations
import argparse, csv, json, sys, os
from pathlib import Path

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topofuse.data.dataset import _load_volume, compute_gt_topology


# ── annotation readers ──────────────────────────────────────────────────────
def read_coords(path: Path, cols=("z", "y", "x")):
    """Read (N,3) z,y,x coordinates from .star/.csv/.tsv/.coords.

    Auto-detects delimiter and a header.  For .star files the first numeric
    block is parsed; columns are taken in the given order (default z,y,x) unless
    a header names rlnCoordinate{X,Y,Z}."""
    text = path.read_text().splitlines()
    suf = path.suffix.lower()
    rows = []
    if suf == ".star":
        # find data rows: lines with >=3 numeric fields, skipping loop_/_rln headers
        colmap = {}
        idx = 0
        for ln in text:
            s = ln.strip()
            if s.startswith("_rln"):
                name = s.split()[0]
                if "#" in s:
                    idx = int(s.split("#")[1]) - 1
                if "CoordinateX" in name: colmap["x"] = idx
                if "CoordinateY" in name: colmap["y"] = idx
                if "CoordinateZ" in name: colmap["z"] = idx
                idx += 1
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue
                if colmap:
                    rows.append((vals[colmap.get("z", 2)],
                                 vals[colmap.get("y", 1)],
                                 vals[colmap.get("x", 0)]))
                else:
                    rows.append((vals[0], vals[1], vals[2]))
        return np.asarray(rows, np.float64).reshape(-1, 3)

    delim = "\t" if (suf == ".tsv" or "\t" in (text[0] if text else "")) else ","
    if suf in (".coords", ".txt") and "," not in (text[0] if text else ""):
        delim = None  # whitespace
    reader = csv.reader(text, delimiter=delim) if delim else (
        ln.split() for ln in text)
    header = None
    for i, parts in enumerate(reader):
        if not parts:
            continue
        if i == 0 and any(c.isalpha() for c in "".join(parts)):
            header = [p.strip().lower() for p in parts]
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if header:
            gi = lambda k, d: header.index(k) if k in header else d
            rows.append((vals[gi("z", 0)], vals[gi("y", 1)], vals[gi("x", 2)]))
        else:
            rows.append((vals[0], vals[1], vals[2]))
    return np.asarray(rows, np.float64).reshape(-1, 3)


def paint_spheres(shape, coords_zyx, radius, class_id=1):
    """Paint solid spheres of `class_id` at each (z,y,x) centre."""
    lab = np.zeros(shape, np.int64)
    r = int(np.ceil(radius))
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (zz**2 + yy**2 + xx**2) <= radius**2
    D, H, W = shape
    for (z, y, x) in coords_zyx:
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(0, z - r), min(D, z + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        bz0, by0, bx0 = z0 - (z - r), y0 - (y - r), x0 - (x - r)
        sub = ball[bz0:bz0 + (z1 - z0), by0:by0 + (y1 - y0), bx0:bx0 + (x1 - x0)]
        lab[z0:z1, y0:y1, x0:x1][sub] = class_id
    return lab


def mask_to_label(mask_np, num_classes, keep_classes=False, class_id=1):
    m = np.asarray(mask_np)
    if keep_classes:
        out = np.clip(np.round(m).astype(np.int64), 0, num_classes - 1)
    else:
        out = np.where(m > 0, class_id, 0).astype(np.int64)
    return out


# ── pairing ─────────────────────────────────────────────────────────────────
def build_pairs(args):
    if args.pairs:
        pairs = []
        with open(args.pairs, newline="") as f:
            for row in csv.DictReader(f):
                low = {k.strip().lower(): v for k, v in row.items()}
                pairs.append({
                    "tomogram": low.get("tomogram") or low.get("volume"),
                    "annotation": low.get("annotation") or low.get("label"),
                    "split": low.get("split", args.default_split),
                    "snr": low.get("snr", ""),
                })
        return pairs
    # pair by shared stem
    tomos = {p.stem: p for ext in ("*.mrc", "*.rec", "*.map", "*.npy")
             for p in Path(args.tomograms).glob(ext)}
    anns = {p.stem.replace("_label", "").replace("_mask", "").replace("_coords", ""): p
            for p in Path(args.annotations).glob("*") if p.is_file()}
    pairs = []
    for stem, tp in sorted(tomos.items()):
        ap = anns.get(stem)
        if ap is None:
            print(f"  ! no annotation found for {stem}, skipping")
            continue
        pairs.append({"tomogram": str(tp), "annotation": str(ap),
                      "split": args.default_split, "snr": ""})
    return pairs


def assign_splits(pairs, ratios=(0.7, 0.1, 0.2), seed=0):
    """Assign train/val/test if not already set."""
    unsplit = [p for p in pairs if not p.get("split")]
    if not unsplit:
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(unsplit))
    n = len(unsplit)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    for j, i in enumerate(idx):
        unsplit[i]["split"] = ("train" if j < n_tr else
                               "val" if j < n_tr + n_va else "test")
    return pairs


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tomograms", required=True, help="dir of .mrc/.rec/.npy tomograms")
    ap.add_argument("--annotations", help="dir of annotation files (coords or masks)")
    ap.add_argument("--out", required=True, help="output data root")
    ap.add_argument("--ann-type", choices=["coords", "mask", "instance"], required=True)
    ap.add_argument("--num-classes", type=int, default=2, help="incl. background")
    ap.add_argument("--class-id", type=int, default=1, help="foreground class for coords/instance")
    ap.add_argument("--radius", type=float, default=8.0, help="sphere radius (coords mode)")
    ap.add_argument("--keep-instance-classes", action="store_true")
    ap.add_argument("--copy-volumes", action="store_true",
                    help="copy tomograms into out/volumes (default: reference in place)")
    ap.add_argument("--pairs", help="optional CSV: tomogram,annotation[,split,snr]")
    ap.add_argument("--default-split", default="", help="train|val|test (blank = auto 70/10/20)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.copy_volumes:
        (out / "volumes").mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args)
    if not pairs:
        sys.exit("No tomogram/annotation pairs found.")
    pairs = assign_splits(pairs, seed=args.seed)
    print(f"ingesting {len(pairs)} tomograms ({args.ann_type} annotations)")

    manifest, rows = [], []
    bud_thr = 6
    for i, pr in enumerate(pairs):
        tp = Path(pr["tomogram"])
        vol = _load_volume(tp).astype(np.float32)
        shape = vol.shape
        vid = tp.stem

        if args.ann_type == "coords":
            coords = read_coords(Path(pr["annotation"]))
            lab = paint_spheres(shape, coords, args.radius, args.class_id)
        else:
            mask = _load_volume(Path(pr["annotation"]))
            lab = mask_to_label(mask, args.num_classes,
                                keep_classes=(args.ann_type == "instance"
                                              and args.keep_instance_classes)
                                             or args.ann_type == "mask"
                                             and args.keep_instance_classes,
                                class_id=args.class_id)
            if lab.shape != shape:
                sys.exit(f"  ! shape mismatch {vid}: vol {shape} vs label {lab.shape}")

        lab_rel = f"labels/{vid}.npy"
        np.save(out / lab_rel, lab)

        if args.copy_volumes:
            vol_rel = f"volumes/{vid}.npy"
            np.save(out / vol_rel, vol)
            vol_field = vol_rel
        else:
            vol_field = str(tp.resolve())     # reference large .mrc in place

        topo = compute_gt_topology(lab, args.num_classes)
        manifest.append({"id": vid, "volume": vol_field, "label": lab_rel,
                         "split": pr["split"], "snr": pr.get("snr", "")})
        row = {"id": vid, "split": pr["split"], "snr": pr.get("snr", ""),
               "beta0": int(topo["beta0"][args.class_id]
                            if args.class_id < args.num_classes else topo["beta0"][1]),
               "beta2": int(topo["beta2"][min(args.class_id, args.num_classes - 1)])}
        cid = min(args.class_id, args.num_classes - 1)
        for d_i, d in enumerate((0, 2)):
            for t in range(bud_thr):
                row[f"budget_d{d}_t{t}"] = float(topo["budgets"][cid, d_i, t])
        rows.append(row)
        print(f"  [{i+1}/{len(pairs)}] {vid}: shape={shape} "
              f"beta0={row['beta0']} beta2={row['beta2']} split={pr['split']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cols = ["id", "split", "snr", "beta0", "beta2"] + \
           [f"budget_d{d}_t{t}" for d in (0, 2) for t in range(bud_thr)]
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    splits = {}
    for m in manifest:
        splits[m["split"]] = splits.get(m["split"], 0) + 1
    print(f"\nwrote {out}/manifest.json ({len(manifest)} records), metadata.csv")
    print(f"splits: {splits}")
    print(f"verify with:  python scripts/inspect_syn.py --data-root {out} "
          f"--num-classes {args.num_classes}")


if __name__ == "__main__":
    main()
