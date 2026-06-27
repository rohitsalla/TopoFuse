"""Common utilities: seeding, sliding-window inference, parameter counting."""
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": train / 1e6}


@torch.no_grad()
def sliding_window_inference(model, volume, crop: int = 128, stride: int = 64,
                             num_classes: int = 3, device="cpu"):
    """Overlap-tiled inference (paper §5: 128^3 crops, 64-voxel stride).

    Returns (prob_post (C,D,H,W), certificates list) accumulated over tiles.
    """
    model.eval()
    B, _, D, H, W = volume.shape
    assert B == 1, "sliding window expects batch size 1"
    acc = torch.zeros(num_classes, D, H, W, device=device)
    cnt = torch.zeros(1, D, H, W, device=device)
    certs = []

    zs = list(range(0, max(D - crop, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - crop, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - crop, 0) + 1, stride)) or [0]
    if zs[-1] != max(D - crop, 0): zs.append(max(D - crop, 0))
    if ys[-1] != max(H - crop, 0): ys.append(max(H - crop, 0))
    if xs[-1] != max(W - crop, 0): xs.append(max(W - crop, 0))

    for z in zs:
        for y in ys:
            for x in xs:
                ze, ye, xe = min(z + crop, D), min(y + crop, H), min(x + crop, W)
                tile = volume[:, :, z:ze, y:ye, x:xe].to(device)
                out = model(tile)
                acc[:, z:ze, y:ye, x:xe] += out["prob_post"][0]
                cnt[:, z:ze, y:ye, x:xe] += 1
                certs.extend(out["certificates"])
    prob = acc / cnt.clamp(min=1)
    return prob, certs
