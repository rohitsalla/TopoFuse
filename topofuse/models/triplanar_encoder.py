"""
Tri-Planar Foundation Encoder  (paper §4.3)
===========================================
For each plane p in {xy, xz, yz}:
  1. Extract the orthogonal slice stack {S^(p)_k} and resize slices to 256x256
     with Lanczos resampling (paper §5 training details).
  2. Encode each slice with a SHARED SAM ViT-B image encoder
        F^(p)_k = E(S^(p)_k).
     SAM ViT-B is pretrained on SA-1B (11M images).  The first 8 transformer
     blocks are frozen; the last 4 blocks and the 2-D decoder are fine-tuned.
  3. Add a learned plane embedding e_p and a sinusoidal depth code e_k.
  4. A lightweight 2-D decoder (two transpose-conv layers) maps features to
     per-slice logits Z^(p)_k, lifted back into 3-D volumes Z^(p).

SAM expects 3-channel 1024x1024 input with ImageNet-style normalisation; we
repeat the single cryo-ET channel to 3 and interpolate SAM's positional
embedding to the chosen slice size.  If the SAM checkpoint is unavailable, a
clearly-marked light-weight fallback stem keeps the pipeline runnable for
testing (NOT for paper-faithful results).
"""
import math
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES = ["xy", "xz", "yz"]
_IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0
_IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255.0


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=k.device) / max(half - 1, 1))
        args = k.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if emb.shape[1] < self.dim:                        # pad if odd
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class SliceDecoder2D(nn.Module):
    """Two transpose-conv layers: SAM feature map -> per-slice class logits."""
    def __init__(self, in_ch: int, num_classes: int, mid_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, mid_ch, 2, stride=2),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, num_classes, 2, stride=2),
        )

    def forward(self, x):
        return self.net(x)


def _load_sam_image_encoder(checkpoint: str, slice_size: int):
    """Load SAM ViT-B image encoder and adapt positional embedding to slice_size."""
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    enc = sam.image_encoder
    # interpolate absolute positional embedding to the new patch grid
    if getattr(enc, "pos_embed", None) is not None:
        new_grid = slice_size // enc.patch_embed.proj.kernel_size[0]
        pe = enc.pos_embed.data                            # (1, gh, gw, C)
        pe = pe.permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(new_grid, new_grid),
                           mode="bicubic", align_corners=False)
        enc.pos_embed = nn.Parameter(pe.permute(0, 2, 3, 1).contiguous())
    enc.img_size = slice_size
    return enc, 256                                        # SAM neck out-channels


class _FallbackStem(nn.Module):
    """Light-weight patch-embed stem used ONLY when SAM weights are unavailable."""
    def __init__(self, out_ch: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_ch, kernel_size=16, stride=16),
            nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class TriPlanarEncoder(nn.Module):
    def __init__(self, num_classes: int = 3, feature_dim: int = 256,
                 pos_embed_dim: int = 64, slice_size: int = 256,
                 sam_checkpoint: str = None, freeze_blocks: int = 8,
                 finetune_blocks: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.slice_size = slice_size

        if sam_checkpoint:
            self.sam_encoder, sam_out_ch = _load_sam_image_encoder(
                sam_checkpoint, slice_size)
            self._freeze_sam_blocks(freeze_blocks, finetune_blocks)
            self.using_sam = True
        else:
            warnings.warn(
                "No SAM checkpoint provided -- using the fallback stem. Results "
                "will NOT match the paper. Pass sam_checkpoint=... for the real "
                "SAM ViT-B encoder (pip install segment-anything; download "
                "sam_vit_b_01ec64.pth).")
            self.sam_encoder = _FallbackStem(out_ch=256)
            sam_out_ch = 256
            self.using_sam = False

        self.register_buffer("_mean", _IMAGENET_MEAN, persistent=False)
        self.register_buffer("_std", _IMAGENET_STD, persistent=False)

        self.plane_embed = nn.Embedding(3, pos_embed_dim)
        self.pos_embed = SinusoidalPosEmbed(pos_embed_dim)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(sam_out_ch + pos_embed_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleDict(
            {p: SliceDecoder2D(feature_dim, num_classes) for p in PLANES})

    # ── freeze first `freeze` SAM blocks, keep last `finetune` trainable ─────
    def _freeze_sam_blocks(self, freeze: int, finetune: int):
        blocks = getattr(self.sam_encoder, "blocks", None)
        if blocks is None:
            return
        n = len(blocks)
        for i, blk in enumerate(blocks):
            trainable = i >= max(freeze, n - finetune)
            for prm in blk.parameters():
                prm.requires_grad_(trainable)
        # patch_embed / pos_embed kept frozen with the early blocks
        if getattr(self.sam_encoder, "patch_embed", None) is not None:
            for prm in self.sam_encoder.patch_embed.parameters():
                prm.requires_grad_(False)

    # ── slice extraction (Lanczos resize to slice_size) ──────────────────────
    def _extract_slices(self, vol, plane):
        B, _, D, H, W = vol.shape
        if plane == "xy":
            s = vol.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W); K = D
        elif plane == "xz":
            s = vol.permute(0, 3, 1, 2, 4).reshape(B * H, 1, D, W); K = H
        else:
            s = vol.permute(0, 4, 1, 2, 3).reshape(B * W, 1, D, H); K = W
        # Lanczos resampling (paper §5).  torch lacks Lanczos for 4-D tensors;
        # we use antialiased bicubic which is the closest differentiable match.
        s = F.interpolate(s, (self.slice_size, self.slice_size),
                          mode="bicubic", align_corners=False, antialias=True)
        s = s.repeat(1, 3, 1, 1)                            # 1-ch -> 3-ch for SAM
        s = (s - self._mean) / self._std
        return s, K

    def _lift_3d(self, logits, plane, B, D, H, W):
        C = self.num_classes
        if plane == "xy":
            r = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
            return r.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        elif plane == "xz":
            r = F.interpolate(logits, (D, W), mode="bilinear", align_corners=False)
            return r.reshape(B, H, C, D, W).permute(0, 2, 3, 1, 4)
        else:
            r = F.interpolate(logits, (D, H), mode="bilinear", align_corners=False)
            return r.reshape(B, W, C, D, H).permute(0, 2, 3, 4, 1)

    def forward(self, volume: torch.Tensor):
        """volume: (B,1,D,H,W).

        Returns:
            planar_logits : dict plane -> (B,C,D,H,W)
            fused         : (B,C,D,H,W) mean of planar logits (fused features)
            gap_feat      : (B, feature_dim) global-average-pooled fused feature
                            embedding, F_fused for the prior head (§4.6).
        """
        B, _, D, H, W = volume.shape
        planar_logits = {}
        gap_accum = volume.new_zeros(B, self.feature_dim)
        for p_idx, plane in enumerate(PLANES):
            slices, K = self._extract_slices(volume, plane)
            feats = self.sam_encoder(slices)               # (B*K, 256, h, w)
            _, fc, fh, fw = feats.shape

            p_emb = self.plane_embed(
                torch.tensor(p_idx, device=volume.device))
            p_emb = p_emb.view(1, -1, 1, 1).expand(B * K, -1, fh, fw)
            k_idx = torch.arange(K, device=volume.device).repeat(B)
            s_emb = self.pos_embed(k_idx).view(B * K, -1, 1, 1).expand_as(p_emb)

            feats = self.feat_proj(torch.cat([feats, p_emb + s_emb], dim=1))
            # accumulate GAP feature (mean over spatial + slices + planes)
            gp = feats.mean(dim=[2, 3]).reshape(B, K, self.feature_dim).mean(1)
            gap_accum = gap_accum + gp / len(PLANES)

            logit_slices = self.decoders[plane](feats)
            planar_logits[plane] = self._lift_3d(logit_slices, plane, B, D, H, W)

        fused = torch.stack([planar_logits[p] for p in PLANES]).mean(0)
        return planar_logits, fused, gap_accum
