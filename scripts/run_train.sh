#!/usr/bin/env bash
# Train TopoFuse. Pass the data root as $1 (defaults to ./SYN_dataset).
# Example: ./scripts/run_train.sh /data/SYN syn 0
set -e
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${1:-./SYN_dataset}"
DATASET="${2:-syn}"          # syn | cryoet
SEED="${3:-0}"
CONFIG="${CONFIG:-configs/topofuse_default.yaml}"
OUT="${OUT:-runs/topofuse}"

echo "== sanity-checking the data root first =="
${PYTHON} scripts/inspect_syn.py --data-root "${DATA_ROOT}" || true

echo "== training (dataset=${DATASET} seed=${SEED}) =="
${PYTHON} scripts/train.py \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --out "${OUT}"
