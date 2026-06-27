# TopoFuse

**Topology-Aware Tri-Planar Fusion for Cryo-Electron Tomography Segmentation**
<br>Official implementation · ECCV 2026 (paper ID 4700)

TopoFuse segments cryo-ET volumes while *guaranteeing* the topology of the
output. A tri-planar SAM encoder produces an initial prediction, a
topology-conditioned FiLM module fuses the three views, and a **differentiable
projection driven by exact persistent homology** repairs the segmentation to a
target topology — emitting a per-volume *repair certificate* that records exactly
which voxels were edited.

This repository is a clean, runnable skeleton: it reproduces every experiment in
the paper and is meant to be read, run, and modified. It ships **no precomputed
results** — every number is produced by your own runs.

---

## Table of contents

- [Highlights](#highlights)
- [Method at a glance](#method-at-a-glance)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation & metrics](#evaluation--metrics)
- [Reproducing the full paper](#reproducing-the-full-paper)
- [Ablations](#ablations)
- [Figures](#figures)
- [Configuration reference](#configuration-reference)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Extending TopoFuse](#extending-topofuse)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation) · [License](#license)

---

## Highlights

- **Exact persistent homology**, not an approximation. Cubical PH (CubicalRipser)
  with critical-cell coordinates; GUDHI for the bottleneck distance. Births/deaths
  are gathered from the prediction at critical voxels, so the diagram is
  differentiable (critical-cell rule).
- **Real SAM ViT-B encoder** — tri-planar `{xy, xz, yz}` slice encoding, first 8
  blocks frozen / last 4 + decoder fine-tuned, with a clearly-warned fallback stem
  when no checkpoint is provided.
- **Topology projection with certificates** — iterative sparse edits with a
  backtracking line search; each volume gets a certificate `C = (I, V_C, Δ_C)`
  (iterations, edited voxels, spatial sparsity) and an honest convergence flag.
- **Self-contained at inference** — a topology prior head predicts Betti numbers
  and six-threshold persistence budgets, removing the oracle dependence.
- **Reproducible by design** — fixed seeds, config embedded in every checkpoint,
  per-dataset and per-ablation configs, a one-command paper grid, and a results
  aggregator. Tested (`pytest`) and pip-installable.

## Method at a glance

```
volume (B,1,D,H,W)
        │
        ▼
┌───────────────────────┐   §4.3  tri-planar SAM ViT-B encoder
│ TriPlanarEncoder      │ ──────────────────────────────► planar logits {xy,xz,yz}
└───────────────────────┘                                  + GAP feature
        │
        ▼
┌───────────────────────┐   §4.4  topology-conditioned FiLM
│ FiLMFusion            │   α^(p) = softmax_p(γ(t)⊙G + β(t))
└───────────────────────┘ ──────────────────────────────► fused logits  Z
        │
        ▼
┌───────────────────────┐   §4.5  exact-PH critical-cell sparse edits
│ TopologyProjection    │   argmin‖Z'−Z‖²  s.t.  d_B(Dgm(Z'), Π*) ≤ ε
└───────────────────────┘ ──────────────────────────────► Z'  + certificate C
        ▲
        │ target Π*
┌───────────────────────┐   §4.6  prior head (β0,β2 + 6-threshold budgets)
│ TopologyPriorHead     │   oracle target in training; predicted at inference
└───────────────────────┘

loss (§4.7):  L = DiceCE(P') + 0.1·W_PH(P) + 0.05·L_prior + 0.5·DiceCE(P)
```

---

## Installation

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

```bash
git clone <your-repo-url> topofuse && cd topofuse

pip install -e .            # core + exact-PH backend (CubicalRipser, GUDHI)
pip install -e '.[sam]'     # also install segment-anything for the real encoder
```

Or use the helper, which installs dependencies, verifies the PH backend, and
downloads the SAM ViT-B checkpoint to `checkpoints/`:

```bash
bash scripts/setup_env.sh
# DOWNLOAD_SAM=0 bash scripts/setup_env.sh   # skip the checkpoint download
```

Verify the install:

```bash
python -m pytest -q          # 5 tests: exact PH, projection edits, train step
```

> Without `cripser`/`gudhi` the package still imports, but exact PH is required
> for paper-faithful results — install them (they are in `requirements.txt`).
> The SAM checkpoint (`checkpoints/sam_vit_b_01ec64.pth`) is required for real
> numbers; without it a fallback stem runs and prints a warning.

---

## Quick start

```bash
# 1. generate the synthetic benchmark (SYN: 2000 vols @ 64³, SNR {0.05,0.10,0.30})
bash scripts/generate_syn.sh ./SYN_dataset

# 2. train one model
bash scripts/run_train.sh ./SYN_dataset syn 0      # args: DATA_ROOT  DATASET  SEED

# 3. evaluate it
bash scripts/run_eval.sh ./SYN_dataset syn runs/topofuse/seed0/best.pt
```

The wrappers call the Python entry points directly, which you can use for full
control:

```bash
python scripts/train.py    --config configs/syn.yaml \
                           --data-root ./SYN_dataset --dataset syn --seed 0 \
                           --out runs/topofuse
python scripts/evaluate.py --config configs/syn.yaml \
                           --data-root ./SYN_dataset --dataset syn --split test \
                           --ckpt runs/topofuse/seed0/best.pt \
                           --out runs/topofuse/seed0/eval
```

`--dataset` is `syn` for the synthetic benchmark or `cryoet` for ingested real
data. Configs live in `configs/`: `syn`, `empiar10499`, `empiar10045`, `emd0506`,
and `topofuse_default`.

---

## Data preparation

### Synthetic (SYN)

```bash
python data/generate_syn.py --out_dir ./SYN_dataset --seed 42
python scripts/inspect_syn.py --data-root ./SYN_dataset    # verify
```

### Real datasets (EMPIAR / EMD)

Real entries ship as raw `.mrc` tomograms plus separate annotations.
`scripts/ingest_empiar.py` converts them into the loader contract, handling three
annotation forms:

```bash
# particle picks  (e.g. ribosomes, EMPIAR-10045): paint spheres at coordinates
python scripts/ingest_empiar.py --ann-type coords \
    --tomograms /data/10045/tomos --annotations /data/10045/coords \
    --out ./data/empiar10045 --num-classes 2 --class-id 1 --radius 8

# dense masks     (e.g. membranes, EMPIAR-10499)
python scripts/ingest_empiar.py --ann-type mask \
    --tomograms /data/10499/tomos --annotations /data/10499/masks \
    --out ./data/empiar10499 --num-classes 2

# instance volumes
python scripts/ingest_empiar.py --ann-type instance \
    --tomograms ... --annotations ... --out ./data/emd0506 --num-classes 2

python scripts/inspect_syn.py --data-root ./data/empiar10045 --num-classes 2
```

Large tomograms are referenced in place; only label volumes are written.
Coordinate files in `.star`/`.csv`/`.tsv`/`.coords` are auto-parsed. Pairing is by
shared filename stem, or an explicit `--pairs` CSV. Splits default to 70/10/20.

### Loader contract

```
<root>/
  manifest.json     # [{id, volume, label, split, snr}, ...]
  metadata.csv      # per-volume beta0, beta2, budget_d{0,2}_t{0..5}, snr, split
  volumes/*.npy     # float32 (D,H,W);  .mrc/.rec supported
  labels/*.npy      # int (D,H,W), values in [0, C-1]  (0 = background)
```

If `metadata.csv` lacks the topology columns they are computed exactly from the
labels via PH and cached under `<root>/.topo_cache/`.

---

## Training

```bash
python scripts/train.py --config <cfg> --data-root <path> --dataset {syn,cryoet} \
                        --seed <int> --out <dir> [--steps N] [--batch-size N] \
                        [--no-amp] [--workers N]
```

AdamW, cosine schedule, AMP, gradient clipping, multi-GPU via `DataParallel`.
Validation Dice selects `best.pt`. The four-term loss (§4.7) and the projection's
oracle target (from labels in training) are handled internally.

## Evaluation & metrics

```bash
python scripts/evaluate.py --config <cfg> --data-root <path> --dataset {syn,cryoet} \
                           --ckpt <best.pt> --split test --out <dir>
```

Reports, per volume and aggregated: **Dice, IoU, NSD, BE₀, BE₂, BME** (exact PH),
certificate stats (sparsity, convergence rate, iterations), and the per-SNR exact
topology-recovery rate (fraction with `BE₀ = 0`). Writes `per_volume.csv` and
`summary.json`.

---

## Reproducing the full paper

Configure dataset roots and run the whole grid (all datasets × 5 seeds), then
aggregate:

```bash
SYN_ROOT=./SYN_dataset \
EMPIAR10499_ROOT=./data/empiar10499 \
EMPIAR10045_ROOT=./data/empiar10045 \
EMD0506_ROOT=./data/emd0506 \
SEEDS="0 1 2 3 4" \
bash scripts/run_paper.sh

python scripts/collect_results.py --runs runs/ --out RESULTS.md
```

Datasets without a `manifest.json` are skipped, so a subset works. `RESULTS.md` is
generated from `runs/**/summary.json` — average per-dataset rows across seeds for
the paper tables. Real-set runs require the SAM checkpoint (set `sam_checkpoint`
in the config or run `setup_env.sh`).

## Ablations

Each ablation is a self-contained config in `configs/ablations/` (one component
toggled against the SYN base):

| Config | Toggle | Effect |
|---|---|---|
| `no_projection.yaml`    | `project_enabled: false` | drop the topology projection (ProjT) |
| `no_film.yaml`          | `use_film: false`        | mean fusion instead of FiLM |
| `no_ph_descriptor.yaml` | `use_ph_desc: false`     | disable the PH descriptor in fusion |
| `no_topo_loss.yaml`     | `lambda_topo: 0.0`       | drop the Wasserstein PH loss |
| `no_prior_loss.yaml`    | `lambda_prior: 0.0`      | drop the prior-head regression |
| `no_aux_loss.yaml`      | `lambda_aux: 0.0`        | drop the auxiliary pre-projection loss |

```bash
python scripts/train.py --config configs/ablations/no_projection.yaml \
                        --data-root ./SYN_dataset --dataset syn --seed 0 \
                        --out runs/ablate_no_projection
```

Compose your own by editing the toggles. The "TopoFuse w/o topology" baseline is
`no_projection` + `no_topo_loss`. Hyperparameter sensitivity (`T_max`, `epsilon`,
`delta`, `downsample_s`) is config-driven; name sweep runs `*_t<value>` for the
sensitivity figure.

## Figures

```bash
python scripts/make_figures.py --runs runs/ --out figures/
# training_curves.pdf, dice_vs_be0.pdf, threshold_sensitivity.pdf
```

Built strictly from real run outputs (`train_log.jsonl`, `per_volume.csv`,
`summary.json`); unevaluated runs are skipped, so figures never contain
placeholder numbers.

---

## Configuration reference

All knobs live in YAML; override any on the command line.

| Key | Default | Meaning |
|---|---|---|
| `num_classes` | 2 | classes incl. background |
| `feature_dim` | 256 | SAM ViT-B output channels |
| `slice_size` | 256 | slice resize before the encoder (§4.3) |
| `sam_checkpoint` | `null` | path to `sam_vit_b_01ec64.pth`; `null` → fallback stem |
| `T_max` / `epsilon` | 5 / 0.05 | projection iteration cap / bottleneck tolerance |
| `downsample_s` / `delta` | 2 / 0.05 | PH grid factor / persistence pruning |
| `lambda_topo` / `lambda_prior` / `lambda_aux` | 0.1 / 0.05 / 0.5 | loss weights (§4.7) |
| `warmup_steps` | 5000 | linear warmup of the topology loss |
| `steps` / `batch_size` / `lr` / `wd` | 200000 / 2 / 1e-4 / 1e-4 | optimisation |
| `crop` / `crop_syn` / `stride` | 128 / 64 / 64 | train crop (real/SYN) / inference stride |
| `use_film` / `use_ph_desc` | true / true | fusion ablation toggles |
| `project_in_train` / `project_enabled` | true / true | projection ablation toggles |

## Outputs

```
runs/<name>/
  seed0/
    best.pt          # best-by-val-Dice (state dict + config + val metrics)
    last.pt          # most recent checkpoint
    train_log.jsonl  # per-step losses, lr, it/s
    val_log.jsonl    # per-validation metrics
    eval/
      per_volume.csv # one row per test volume
      summary.json   # overall + per-SNR means, exact-recovery rate
RESULTS.md           # aggregated across all runs (collect_results.py)
figures/*.pdf        # make_figures.py
```

Each checkpoint embeds its config, so it is self-describing and re-evaluable.

## Project structure

```
topofuse/
  topo/         ph, matching, pseudo_diagram, descriptor   # exact-PH engine
  models/       triplanar_encoder, film_fusion, topology_prior_head, topofuse
  losses/       topo_projection (core), seg_losses, topo_losses
  data/         dataset (manifest + metadata loader)
  evaluation/   metrics (Dice, IoU, NSD, BE0, BE2, BME, certificate stats)
  utils/        seeding, param count, sliding-window inference
scripts/        train, evaluate, inspect_syn, ingest_empiar, collect_results,
                make_figures, run_{train,eval,paper}.sh, generate_syn.sh, setup_env.sh
data/           generate_syn.py
configs/        topofuse_default, syn, empiar10499, empiar10045, emd0506,
                ablations/*.yaml
tests/          test_smoke.py, test_ph.py
pyproject.toml  requirements.txt  LICENSE  CITATION.cff
```

## Extending TopoFuse

The code is organised so each component is swappable:

- **New dataset** — produce the loader contract (manifest + metadata + volumes +
  labels), or extend `scripts/ingest_empiar.py` with a reader for your annotation
  format. No model changes needed; train with `--dataset cryoet`.
- **New baseline** — implement a `nn.Module` returning per-class logits and drop it
  into `scripts/train.py` in place of `TopoFuse`; reuse the loss/metrics as-is. The
  internal "no-topology" baseline is the `no_projection` + `no_topo_loss` config.
- **New loss term** — add it in `topofuse/losses/` and wire a weight into
  `TopoFuse.compute_loss`.
- **Swap the encoder** — replace `TriPlanarEncoder` (e.g. a different SAM variant or
  a 3-D backbone); it only needs to return `(planar_logits, fused_feat, gap)`.
- **Change the topology target** — the projection accepts oracle diagrams (training)
  or prior-head budgets (inference); both are built in `topofuse/topo/pseudo_diagram.py`.

## Testing

```bash
python -m pytest -q
```

Covers exact Betti numbers, the bottleneck orientation fix (superlevel points must
not collapse to zero), real critical-voxel edits in the projection, and a full
train step (4-term loss + backward) plus the inference path.

## Troubleshooting

- **"No SAM checkpoint provided — using the fallback stem."** Expected without a
  checkpoint; set `sam_checkpoint` for real results.
- **`cripser`/`gudhi` import errors.** `pip install -r requirements.txt`; both are
  needed for exact PH.
- **Topology counts look wrong after ingestion.** Check the `beta0`/`beta2` printed
  by `ingest_empiar.py` against expectations and adjust `--radius` / `--num-classes`
  / `--class-id`.
- **Slow PH.** It runs on the `downsample_s` grid (default 2). Increase it to trade
  fidelity for speed.

## Citation

```bibtex
@inproceedings{topofuse2026,
  title     = {TopoFuse: Topology-Aware Tri-Planar Fusion for Cryo-Electron
               Tomography Segmentation},
  author    = {Salla, Rohit and others},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

See `CITATION.cff`. Please complete the author list and add the DOI once the
proceedings are final.

## License

MIT — see [`LICENSE`](LICENSE). You are free to use, modify, and distribute this
code; please cite the paper if you build on it.

## Acknowledgements

Built on [Segment Anything](https://github.com/facebookresearch/segment-anything),
[CubicalRipser](https://github.com/shizuo-kaji/CubicalRipser_3dim), and
[GUDHI](https://gudhi.inria.fr/).
# TopoFuse
# TopoFuse
