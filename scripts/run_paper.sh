#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Reproduce the full TopoFuse experimental grid and populate RESULTS.md.
#
# Runs SYN + each configured real dataset over SEEDS seeds, evaluates every
# checkpoint, then aggregates with collect_results.py. ALL numbers come from
# these runs — nothing is pre-filled.
#
# Configure the dataset roots below (point them at your SYN + ingested EMPIAR
# data), set SEEDS, then:  bash scripts/run_paper.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/topofuse_default.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4}"          # paper averages 5 seeds
OUT_ROOT="${OUT_ROOT:-runs}"

# name:dataset_kind:data_root   (edit these to your paths; comment out any you lack)
DATASETS=(
  "syn:syn:${SYN_ROOT:-./SYN_dataset}"
  "empiar10499:cryoet:${EMPIAR10499_ROOT:-./data/empiar10499}"
  "empiar10045:cryoet:${EMPIAR10045_ROOT:-./data/empiar10045}"
  "emd0506:cryoet:${EMD0506_ROOT:-./data/emd0506}"
)

for entry in "${DATASETS[@]}"; do
  name="${entry%%:*}"; rest="${entry#*:}"; kind="${rest%%:*}"; root="${rest#*:}"
  if [ ! -e "${root}/manifest.json" ]; then
    echo "== skipping ${name}: no manifest.json at ${root} =="
    continue
  fi
  echo "############ ${name}  (${kind})  ${root} ############"
  ${PYTHON} scripts/inspect_syn.py --data-root "${root}" || true
  for seed in ${SEEDS}; do
    run="${OUT_ROOT}/${name}/seed${seed}"
    echo "==== train ${name} seed ${seed} ===="
    ${PYTHON} scripts/train.py --config "${CONFIG}" --data-root "${root}" \
        --dataset "${kind}" --seed "${seed}" --out "${OUT_ROOT}/${name}"
    echo "==== eval ${name} seed ${seed} ===="
    ${PYTHON} scripts/evaluate.py --config "${CONFIG}" --data-root "${root}" \
        --dataset "${kind}" --split test \
        --ckpt "${run}/best.pt" --out "${run}/eval"
  done
done

echo "######## aggregating results ########"
${PYTHON} scripts/collect_results.py --runs "${OUT_ROOT}" --out RESULTS.md
echo "Done. See RESULTS.md (means over seeds = group rows by dataset)."
