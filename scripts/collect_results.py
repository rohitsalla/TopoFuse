#!/usr/bin/env python3
"""
Collect evaluation results into a single RESULTS.md table.

Scans a runs directory for `summary.json` files produced by scripts/evaluate.py
and appends/refreshes a Markdown table so your numbers are preserved in one
place across seeds, datasets, and SNR levels.

Usage:
    python scripts/collect_results.py --runs runs/ --out RESULTS.md
"""
import argparse, json, glob, os, datetime
from pathlib import Path


def fmt(x, p=3):
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()

    rows = []
    for sp in glob.glob(os.path.join(args.runs, "**", "summary.json"), recursive=True):
        try:
            s = json.loads(Path(sp).read_text())
        except Exception:
            continue
        run = os.path.relpath(os.path.dirname(sp), args.runs)
        o = s.get("overall", {})
        rows.append({
            "run": run, "n": s.get("n", "—"), "snr": "all",
            "Dice": o.get("Dice"), "IoU": o.get("IoU"), "NSD": o.get("NSD"),
            "BE0": o.get("BE0"), "BE2": o.get("BE2"), "BME": o.get("BME"),
            "recovery": None,
            "conv": o.get("converged"),
            "sparsity": o.get("sparsity_median", o.get("sparsity")),
        })
        for snr, v in (s.get("per_snr") or {}).items():
            rows.append({
                "run": run, "n": v.get("n", "—"), "snr": snr,
                "Dice": v.get("Dice"), "IoU": v.get("IoU"), "NSD": v.get("NSD"),
                "BE0": v.get("BE0"), "BE2": v.get("BE2"), "BME": v.get("BME"),
                "recovery": v.get("recovery_rate"),
                "conv": v.get("converged"), "sparsity": v.get("sparsity"),
            })

    rows.sort(key=lambda r: (r["run"], r["snr"]))
    hdr = ("| run | SNR | n | Dice | IoU | NSD | BE0 | BE2 | BME | "
           "recovery(BE0=0) | conv | sparsity |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines = [
        "# TopoFuse — Results",
        "",
        f"_Auto-generated from `{args.runs}/**/summary.json` on "
        f"{datetime.date.today().isoformat()}._",
        "",
        hdr, sep,
    ]
    for r in rows:
        rec = "—" if r["recovery"] is None else f"{100*float(r['recovery']):.1f}%"
        lines.append(
            f"| {r['run']} | {r['snr']} | {r['n']} | {fmt(r['Dice'])} | "
            f"{fmt(r['IoU'])} | {fmt(r['NSD'])} | {fmt(r['BE0'])} | "
            f"{fmt(r['BE2'])} | {fmt(r['BME'])} | {rec} | {fmt(r['conv'],2)} | "
            f"{fmt(r['sparsity'],4)} |")
    if not rows:
        lines.append("| _(no summary.json found yet — run scripts/evaluate.py)_ "
                     "| | | | | | | | | | | |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(rows)} rows from {args.runs}")


if __name__ == "__main__":
    main()
