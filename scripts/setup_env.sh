#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TopoFuse environment setup
# Installs PH backend (CubicalRipser + GUDHI), SAM, and I/O deps, then
# optionally downloads the SAM ViT-B checkpoint used by the tri-planar encoder.
# ─────────────────────────────────────────────────────────────────────────────
set -e

PYTHON="${PYTHON:-python3}"
PIP="${PYTHON} -m pip"

echo "== installing Python dependencies =="
${PIP} install --upgrade pip
${PIP} install -r requirements.txt

echo "== verifying PH backend =="
${PYTHON} - << 'PY'
import cripser, gudhi, numpy as np
x = np.zeros((8,8,8)); x[2:6,2:6,2:6] = 1.0
r = cripser.computePH(-x, maxdim=2)
print("cripser OK  rows:", r.shape, "| gudhi", gudhi.__version__)
PY

# ── SAM ViT-B checkpoint (optional but recommended) ──────────────────────────
SAM_DIR="${SAM_DIR:-checkpoints}"
SAM_CKPT="${SAM_DIR}/sam_vit_b_01ec64.pth"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
if [ "${DOWNLOAD_SAM:-1}" = "1" ]; then
  mkdir -p "${SAM_DIR}"
  if [ ! -f "${SAM_CKPT}" ]; then
    echo "== downloading SAM ViT-B checkpoint =="
    if command -v wget >/dev/null; then wget -O "${SAM_CKPT}" "${SAM_URL}";
    else curl -L -o "${SAM_CKPT}" "${SAM_URL}"; fi
  else
    echo "SAM checkpoint already present: ${SAM_CKPT}"
  fi
  echo "Set sam_checkpoint: ${SAM_CKPT} in configs/topofuse_default.yaml"
else
  echo "Skipping SAM download (DOWNLOAD_SAM=0). Encoder will use fallback stem."
fi

echo "== done. activate this env and run scripts/run_train.sh =="
