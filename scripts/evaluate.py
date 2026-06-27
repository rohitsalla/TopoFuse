#!/usr/bin/env python3
"""
TopoFuse evaluation  (paper §5)
===============================
Sliding-window inference (128^3 crops, 64 stride; full-volume for 64^3 SYN),
full metric bundle per volume (Dice, IoU, NSD, BE0, BE2, BME), repair
certificate statistics (convergence rate, edit sparsity), and a per-SNR
breakdown for SYN including the exact topology-recovery rate (fraction of
volumes with BE0 == 0 after binarisation, paper §5.5).

Writes per_volume.csv and summary.json to the output dir.

Usage:
    python scripts/evaluate.py --config configs/topofuse_default.yaml \
        --data-root /path/to/SYN --dataset syn --ckpt runs/topofuse/seed0/best.pt
"""
import sys, os, argparse, json, csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    yaml = None

from topofuse.models.topofuse import TopoFuse
from topofuse.data.dataset import SynDataset, CryoETDataset, collate
from topofuse.evaluation.metrics import compute_metrics, certificate_stats
from topofuse.utils.common import sliding_window_inference


def load_cfg(path):
    if path is None:
        return {}
    txt = Path(path).read_text()
    return yaml.safe_load(txt) if yaml is not None else json.loads(txt)


def build_model(cfg, ckpt, device):
    model = TopoFuse(
        num_classes=cfg["num_classes"], feature_dim=cfg.get("feature_dim", 256),
        T_max=cfg.get("T_max", 5), epsilon=cfg.get("epsilon", 0.05),
        downsample_s=cfg["downsample_s"], delta=cfg["delta"],
        slice_size=cfg.get("slice_size", 256),
        sam_checkpoint=cfg.get("sam_checkpoint", None),
        use_ph_desc=cfg.get("use_ph_desc", True),
        use_film=cfg.get("use_film", True),
        project_enabled=cfg.get("project_enabled", True),
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/topofuse_default.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="syn", choices=["syn", "cryoet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/topofuse/eval")
    ap.add_argument("--T", type=int, default=None, help="override projection iters")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg.setdefault("num_classes", 2)
    cfg.setdefault("downsample_s", 2)
    cfg.setdefault("delta", 0.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, args.ckpt, device)
    if args.T is not None:
        model.projection.T_max = args.T

    is_syn = args.dataset == "syn"
    crop = cfg.get("crop_syn", 64) if is_syn else cfg.get("crop", 128)
    stride = crop if is_syn else cfg.get("stride", 64)
    DS = SynDataset if is_syn else CryoETDataset
    ds = DS(root=args.data_root, split=args.split, crop=None,
            num_classes=cfg["num_classes"], downsample_s=cfg["downsample_s"],
            delta=cfg["delta"], augment=False)
    ld = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    print(f"evaluating {len(ds)} volumes ({args.split})")

    rows, all_certs = [], []
    for i, batch in enumerate(ld):
        vol = batch["volume"].to(device)
        lab = batch["label"][0].numpy()
        snr = float(batch["snr"][0])
        prob, certs = sliding_window_inference(
            model, vol, crop=crop, stride=stride,
            num_classes=cfg["num_classes"], device=device)
        prob = prob.cpu().numpy()
        m = compute_metrics(prob, lab, cfg["num_classes"],
                            s=cfg["downsample_s"], delta=cfg["delta"])
        m["id"] = batch["id"][0]; m["snr"] = snr
        m["BE0_zero"] = 1.0 if m["BE0"] == 0 else 0.0   # exact recovery
        cs = certificate_stats(certs)
        m["converged"] = cs["conv_rate"]; m["sparsity"] = cs["mean_sparsity"]
        rows.append(m); all_certs.extend(certs)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ds)}  running Dice="
                  f"{np.mean([r['Dice'] for r in rows]):.3f}")

    # per-volume CSV
    keys = ["id", "snr", "Dice", "IoU", "NSD", "BE0", "BE2", "BME",
            "BE0_zero", "converged", "sparsity"]
    with open(out_dir / "per_volume.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # overall + per-SNR summary
    def summarise(subset):
        if not subset:
            return {}
        return {k: float(np.mean([r[k] for r in subset]))
                for k in ["Dice", "IoU", "NSD", "BE0", "BE2", "BME",
                          "BE0_zero", "converged", "sparsity"]}

    summary = {"overall": summarise(rows), "n": len(rows)}
    summary["overall"]["sparsity_median"] = float(
        np.median([r["sparsity"] for r in rows]))
    if is_syn:
        summary["per_snr"] = {}
        for s in sorted({r["snr"] for r in rows}):
            sub = [r for r in rows if r["snr"] == s]
            summary["per_snr"][f"{s:.2f}"] = {
                **summarise(sub),
                "recovery_rate": float(np.mean([r["BE0_zero"] for r in sub])),
                "n": len(sub)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n== summary ==")
    o = summary["overall"]
    print(f"Dice={o['Dice']:.3f} IoU={o['IoU']:.3f} NSD={o['NSD']:.3f} "
          f"BE0={o['BE0']:.3f} BE2={o['BE2']:.3f} BME={o['BME']:.3f}")
    print(f"conv={o['converged']:.3f} median_sparsity={o['sparsity_median']:.4f}")
    if is_syn:
        for s, v in summary["per_snr"].items():
            print(f"  SNR {s}: Dice={v['Dice']:.3f} BE0={v['BE0']:.3f} "
                  f"recovery(BE0=0)={v['recovery_rate']*100:.1f}%  (n={v['n']})")
    print(f"\nwritten: {out_dir/'per_volume.csv'}, {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
