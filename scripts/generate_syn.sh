#!/usr/bin/env bash
# Generate the SYN synthetic cryo-ET benchmark (2000 volumes @ 64^3, 3 SNRs).
# Emits manifest.json + metadata.csv + volumes/ + labels/ (the loader contract).
set -e
PYTHON="${PYTHON:-python3}"
OUT="${1:-./SYN_dataset}"
SEED="${SEED:-42}"
echo "== generating SYN -> ${OUT} (seed=${SEED}) =="
${PYTHON} data/generate_syn.py --out_dir "${OUT}" --seed "${SEED}"
echo "== inspecting generated dataset =="
${PYTHON} scripts/inspect_syn.py --data-root "${OUT}"
