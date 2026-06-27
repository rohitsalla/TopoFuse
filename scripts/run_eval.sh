#!/usr/bin/env bash
# Evaluate a trained TopoFuse checkpoint.
# Example: ./scripts/run_eval.sh /data/SYN syn runs/topofuse/seed0/best.pt
set -e
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${1:-./SYN_dataset}"
DATASET="${2:-syn}"
CKPT="${3:-runs/topofuse/seed0/best.pt}"
CONFIG="${CONFIG:-configs/topofuse_default.yaml}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-runs/topofuse/eval}"

${PYTHON} scripts/evaluate.py \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --split "${SPLIT}" \
  --out "${OUT}"
