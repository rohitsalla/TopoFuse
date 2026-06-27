#!/usr/bin/env python3
"""
TopoFuse training  (paper §5.1)
===============================
200k steps, AdamW (lr=1e-4), 128^3 crops (64^3 for SYN), flips + elastic
(sigma=10) + intensity jitter, mixed precision, gradient clip 1.0, 5 seeds,
DataParallel across 4×A100.

During training the projection consumes the ORACLE target derived from the
batch labels (§4.5); the prior head H_pi is trained jointly via L_prior so the
model is self-contained at inference.

Total loss (§4.7):
    L = L_DiceCE(P'_hat,Y) + 0.1 L_topo(P_hat,Y) + 0.05 L_prior
        + 0.5 L_DiceCE(P_hat,Y)         (L_topo warmed up over 5000 steps)

Usage:
    python scripts/train.py --config configs/topofuse_default.yaml \
        --data-root /path/to/SYN --dataset syn --seed 0
"""
import sys, os, argparse, random, time, json
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
from topofuse.evaluation.metrics import compute_metrics, certificate_stats, aggregate
from topofuse.utils.common import set_seed, count_params


def load_cfg(path):
    if path is None:
        return {}
    txt = Path(path).read_text()
    if yaml is not None:
        return yaml.safe_load(txt)
    return json.loads(txt)   # fall back to JSON-compatible YAML


def build_dataset(name, root, split, cfg, seed, augment):
    kw = dict(root=root, split=split, num_classes=cfg["num_classes"],
              downsample_s=cfg["downsample_s"], delta=cfg["delta"],
              augment=augment, seed=seed)
    if name == "syn":
        return SynDataset(crop=cfg.get("crop_syn", 64), **kw)
    return CryoETDataset(crop=cfg.get("crop", 128), **kw)


def move_topology(gt, device):
    return {k: v.to(device) for k, v in gt.items()}


@torch.no_grad()
def validate(model, loader, device, num_classes, s, delta):
    model.eval()
    recs, certs = [], []
    for batch in loader:
        vol = batch["volume"].to(device)
        lab = batch["label"].to(device)
        out = model(vol)                       # inference path -> prior target
        pp = out["prob_post"].detach().cpu().numpy()
        pre = out["prob_pre"].detach().cpu().numpy()
        labn = lab.cpu().numpy()
        for b in range(pp.shape[0]):
            recs.append(compute_metrics(pp[b], labn[b], num_classes,
                                        prob_pre=pre[b], s=s, delta=delta))
        certs.extend(out["certificates"])
    model.train()
    agg = aggregate(recs)
    agg.update({f"cert_{k}": v for k, v in certificate_stats(certs).items()})
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/topofuse_default.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="syn", choices=["syn", "cryoet"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/topofuse")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg.setdefault("num_classes", 2)
    cfg.setdefault("downsample_s", 2)
    cfg.setdefault("delta", 0.05)
    steps = args.steps or cfg.get("steps", 200_000)
    bs = args.batch_size or cfg.get("batch_size", 2)
    lr = cfg.get("lr", 1e-4)
    val_every = cfg.get("val_every", 2000)
    log_every = cfg.get("log_every", 50)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────
    train_ds = build_dataset(args.dataset, args.data_root, "train", cfg, args.seed, True)
    val_ds   = build_dataset(args.dataset, args.data_root, "val",   cfg, args.seed, False)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=args.workers, collate_fn=collate)
    print(f"train={len(train_ds)}  val={len(val_ds)}  bs={bs}  steps={steps}")

    # ── model ───────────────────────────────────────────────────────────────
    model = TopoFuse(
        num_classes=cfg["num_classes"], feature_dim=cfg.get("feature_dim", 256),
        T_max=cfg.get("T_max", 5), epsilon=cfg.get("epsilon", 0.05),
        downsample_s=cfg["downsample_s"], delta=cfg["delta"],
        slice_size=cfg.get("slice_size", 256),
        sam_checkpoint=cfg.get("sam_checkpoint", None),
        lambda_topo=cfg.get("lambda_topo", 0.1),
        lambda_prior=cfg.get("lambda_prior", 0.05),
        lambda_aux=cfg.get("lambda_aux", 0.5),
        warmup_steps=cfg.get("warmup_steps", 5000),
        use_ph_desc=cfg.get("use_ph_desc", True),
        use_film=cfg.get("use_film", True),
        project_in_train=cfg.get("project_in_train", True),
        project_enabled=cfg.get("project_enabled", True),
    ).to(device)
    print(f"params: {count_params(model)['total_M']:.1f}M")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel over {torch.cuda.device_count()} GPUs")

    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── train loop ──────────────────────────────────────────────────────────
    best_dice, step, t0 = -1.0, 0, time.time()
    log_path = out_dir / "train_log.jsonl"
    train_iter = iter(train_ld)
    model.train()
    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ld); batch = next(train_iter)

        vol = batch["volume"].to(device)
        lab = batch["label"].to(device)
        gt_topo = move_topology(batch["gt_topology"], device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = core(vol, labels=lab)        # oracle projection in train
            losses = core.compute_loss(out, lab, gt_topology=gt_topo)
            loss = losses["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(opt); scaler.update(); sched.step()
        step += 1

        if step % log_every == 0:
            rate = step / (time.time() - t0)
            msg = {"step": step, "lr": sched.get_last_lr()[0],
                   **{k: float(v.detach()) for k, v in losses.items()}, "it_s": rate}
            print(f"[{step:>7}/{steps}] loss={loss.item():.4f} "
                  f"main={losses['main'].item():.4f} topo={losses['topo'].item():.4f} "
                  f"prior={losses['prior'].item():.4f} ({rate:.2f} it/s)")
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")

        if step % val_every == 0 or step == steps:
            vm = validate(core, val_ld, device, cfg["num_classes"],
                          cfg["downsample_s"], cfg["delta"])
            dice = vm.get("Dice_mean", -1)
            print(f"  >> val Dice={dice:.3f}  BE0={vm.get('BE0_mean',-1):.3f}  "
                  f"conv={vm.get('cert_conv_rate',-1):.3f}  "
                  f"spars={vm.get('cert_mean_sparsity',-1):.4f}")
            with open(out_dir / "val_log.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **vm}) + "\n")
            if dice > best_dice:
                best_dice = dice
                torch.save({"model": core.state_dict(), "step": step,
                            "cfg": cfg, "val": vm}, out_dir / "best.pt")
                print(f"  >> new best (Dice={dice:.3f}) saved")
        if step % cfg.get("ckpt_every", 10000) == 0:
            torch.save({"model": core.state_dict(), "step": step, "cfg": cfg},
                       out_dir / "last.pt")

    torch.save({"model": core.state_dict(), "step": step, "cfg": cfg},
               out_dir / "last.pt")
    print(f"done. best val Dice={best_dice:.3f}. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
