#!/usr/bin/env python3
"""
Publication figures from REAL run outputs (no synthetic/placeholder data).

Reads what scripts/train.py and scripts/evaluate.py write under runs/:
  - train_log.jsonl   -> training-curve figure (loss terms vs step)
  - per_volume.csv    -> Dice-vs-BE0 operating-point scatter (per run/ablation)
  - summary.json      -> threshold/seed sensitivity (mean ± std)

Every figure is built only from files that exist; if a run hasn't been
evaluated yet, it is skipped (and noted), so figures never contain made-up
numbers.

Usage:
  python scripts/make_figures.py --runs runs/ --out figures/
"""
import argparse, json, glob, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def training_curves(runs, out):
    logs = sorted(glob.glob(os.path.join(runs, "**", "train_log.jsonl"), recursive=True))
    if not logs:
        print("  (no train_log.jsonl found — skipping training curves)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for lg in logs:
        steps, total = [], []
        for ln in Path(lg).read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "step" in r and "total" in r:
                steps.append(r["step"]); total.append(r["total"])
        if steps:
            ax.plot(steps, total, label=os.path.relpath(os.path.dirname(lg), runs))
    ax.set_xlabel("step"); ax.set_ylabel("total loss"); ax.legend(fontsize=7)
    ax.set_title("Training curves"); fig.tight_layout()
    fig.savefig(Path(out) / "training_curves.pdf"); plt.close(fig)
    print(f"  wrote training_curves.pdf ({len(logs)} runs)")


def dice_vs_be0(runs, out):
    import csv
    csvs = sorted(glob.glob(os.path.join(runs, "**", "per_volume.csv"), recursive=True))
    if not csvs:
        print("  (no per_volume.csv found — skipping Dice-vs-BE0 frontier)")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for cp in csvs:
        dice, be0 = [], []
        with open(cp) as f:
            for row in csv.DictReader(f):
                try:
                    dice.append(float(row.get("Dice", row.get("dice"))))
                    be0.append(float(row.get("BE0", row.get("be0"))))
                except (TypeError, ValueError):
                    continue
        if dice:
            ax.scatter(be0, dice, s=10, alpha=0.6,
                       label=os.path.relpath(os.path.dirname(cp), runs))
    ax.set_xlabel(r"Betti-0 error  BE$_0$"); ax.set_ylabel("Dice")
    ax.set_title("Dice vs. topology error"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(Path(out) / "dice_vs_be0.pdf"); plt.close(fig)
    print(f"  wrote dice_vs_be0.pdf ({len(csvs)} runs)")


def sensitivity(runs, out):
    """If runs are named *_t<value> (a hyperparameter sweep), plot mean metric vs value."""
    summaries = sorted(glob.glob(os.path.join(runs, "**", "summary.json"), recursive=True))
    pts = []
    for sp in summaries:
        name = os.path.basename(os.path.dirname(os.path.dirname(sp)))
        if "_t" in name:
            try:
                val = float(name.rsplit("_t", 1)[1])
            except ValueError:
                continue
            o = json.loads(Path(sp).read_text()).get("overall", {})
            if "BME" in o:
                pts.append((val, o["BME"]))
    if len(pts) < 2:
        print("  (need >=2 sweep runs named *_t<value> — skipping sensitivity)")
        return
    pts.sort()
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("threshold"); ax.set_ylabel("BME"); ax.set_title("Threshold sensitivity")
    fig.tight_layout(); fig.savefig(Path(out) / "threshold_sensitivity.pdf"); plt.close(fig)
    print(f"  wrote threshold_sensitivity.pdf ({len(pts)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"building figures from {args.runs} -> {args.out}")
    training_curves(args.runs, args.out)
    dice_vs_be0(args.runs, args.out)
    sensitivity(args.runs, args.out)
    print("done.")


if __name__ == "__main__":
    main()
