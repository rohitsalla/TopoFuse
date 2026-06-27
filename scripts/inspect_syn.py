#!/usr/bin/env python3
"""
Inspect a SYN (or real) data root and report exactly what the loader detects.

Run this FIRST against your separately-shipped SYN zip to confirm the
manifest.json / metadata.csv schema is understood before training:

    python scripts/inspect_syn.py --data-root /path/to/SYN

It prints: number of records, split breakdown, detected path/column fields,
SNR levels, a couple of sample records, and verifies that the first volume +
label actually load and report a topology.
"""
import sys, os, json, csv, argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from topofuse.data.dataset import (
    _first, _load_volume, _ID_KEYS, _VOL_KEYS, _LAB_KEYS,
    _SPLIT_KEYS, _SNR_KEYS, _B0_KEYS, _B2_KEYS, compute_gt_topology,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--num-classes", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.data_root)

    print(f"== inspecting {root} ==")
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"!! manifest.json MISSING at {mpath}"); sys.exit(1)
    data = json.loads(mpath.read_text())

    if isinstance(data, dict):
        print(f"manifest.json: dict with keys {list(data.keys())}")
        recs = []
        for k, v in data.items():
            if isinstance(v, list):
                for r in v:
                    r = dict(r); r.setdefault("split", k); recs.append(r)
        if not recs and any(isinstance(v, list) for v in data.values()):
            pass
    else:
        recs = list(data)
        print(f"manifest.json: flat list of {len(recs)} records")

    if not recs:
        print("!! no records parsed"); sys.exit(1)

    r0 = recs[0]
    print(f"\nsample record keys : {list(r0.keys())}")
    print(f"  detected id     -> {_first(r0, _ID_KEYS)}")
    print(f"  detected volume -> {_first(r0, _VOL_KEYS)}")
    print(f"  detected label  -> {_first(r0, _LAB_KEYS)}")
    print(f"  detected split  -> {_first(r0, _SPLIT_KEYS)}")
    print(f"  detected snr    -> {_first(r0, _SNR_KEYS)}")

    splits = {}
    snrs = set()
    for r in recs:
        s = _first(r, _SPLIT_KEYS) or "??"
        splits[s] = splits.get(s, 0) + 1
        v = _first(r, _SNR_KEYS)
        if v not in (None, ""):
            snrs.add(float(v))
    print(f"\nsplit breakdown : {splits}")
    print(f"snr levels      : {sorted(snrs) if snrs else 'none in manifest'}")

    cpath = root / "metadata.csv"
    if cpath.exists():
        with open(cpath, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"\nmetadata.csv: {len(rows)} rows, columns = {list(rows[0].keys())}")
        m0 = rows[0]
        print(f"  detected beta0 -> {_first(m0, _B0_KEYS)}")
        print(f"  detected beta2 -> {_first(m0, _B2_KEYS)}")
        bud = [c for c in m0 if str(c).lower().startswith("budget_")]
        print(f"  budget columns -> {bud if bud else 'none (will compute from labels)'}")
    else:
        print("\nmetadata.csv: NOT present (beta/budgets will be computed from labels)")

    # try to load the first volume + label
    def resolve(rel):
        p = Path(str(rel));  return p if p.is_absolute() else root / p
    try:
        vol = _load_volume(resolve(_first(r0, _VOL_KEYS)))
        print(f"\nfirst volume loads: shape={vol.shape} dtype={vol.dtype} "
              f"range=[{vol.min():.3f},{vol.max():.3f}]")
        lab_rel = _first(r0, _LAB_KEYS)
        if lab_rel:
            lab = _load_volume(resolve(lab_rel)).astype(np.int64)
            print(f"first label loads : shape={lab.shape} unique={np.unique(lab)[:8]}")
            topo = compute_gt_topology(lab, args.num_classes)
            print(f"computed topology : beta0={topo['beta0']} beta2={topo['beta2']}")
    except Exception as e:
        print(f"\n!! failed to load first volume/label: {e}")

    print("\nOK — loader should accept this root. Use it as --data-root.")


if __name__ == "__main__":
    main()
