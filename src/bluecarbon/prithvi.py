"""Prithvi (NASA / IBM geospatial foundation model) fine-tuned for habitat segmentation.

Prithvi-EO-1.0-100M is a ViT-Base masked autoencoder pre-trained on years of NASA Harmonized Landsat Sentinel-2
imagery over the US. Its encoder takes six bands (blue, green, red, narrow NIR, SWIR1, SWIR2) in 16 x 16 patches.
Here the pre-trained encoder reads those six Sentinel-2 bands, a small convolutional branch reads every input
(indices, elevation, radar, ...) at full resolution, and a light decoder merges the two back up to 10 m pixels.

Weights: Hugging Face ibm-nasa-geospatial/Prithvi-EO-1.0-100M (Apache-2.0), downloaded once in Colab. Free; no
Google Cloud service is involved.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HF_REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
HF_FILES = ("Prithvi_EO_V1_100M.pt", "Prithvi_100M.pt")
PRITHVI_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]  # HLS blue, green, red, narrow NIR, SWIR1, SWIR2
EMBED, DEPTH, HEADS, PATCH = 768, 12, 12, 16


def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1).astype(np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_pos_embed(dim: int, gh: int, gw: int) -> torch.Tensor:
    """Prithvi's fixed 3-D sin-cos position code for one time step: width 6/16, height 6/16, time 4/16 of dim."""
    wd, td = dim // 16 * 6, dim // 16 * 4
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    emb = np.concatenate([_sincos_1d(wd, gx), _sincos_1d(wd, gy), _sincos_1d(td, np.zeros(gh * gw))], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)


class PrithviSeg(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, band_idx: list[int], pretrained: bool = True):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.register_buffer("band_idx", torch.as_tensor(band_idx, dtype=torch.long), persistent=False)
        self.patch_embed = nn.Conv2d(len(band_idx), EMBED, PATCH, PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED))
        self.blocks = nn.ModuleList([Block(EMBED, HEADS, 4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
                                     for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED)
        # every input at full / half / quarter resolution
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.c2 = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True))
        self.c4 = nn.Sequential(nn.Conv2d(96, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        # decoder: ViT tokens (1/16) -> 1/4 -> 1/2 -> 1/1
        self.proj = nn.Sequential(nn.Conv2d(EMBED * 2, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))

        def fuse(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

        self.d4, self.d2, self.d1 = fuse(256 + 128, 128), fuse(128 + 96, 96), fuse(96 + 64, 64)
        self.head = nn.Conv2d(64, n_classes, 1)
        self.pretrained_loaded = False
        if pretrained:
            self.load_pretrained()

    def load_pretrained(self) -> None:
        from huggingface_hub import hf_hub_download

        last = None
        for name in HF_FILES:
            try:
                path = hf_hub_download(HF_REPO, name)
                break
            except Exception as e:  # file was renamed between model versions
                last = e
        else:
            raise RuntimeError(f"could not download Prithvi weights from {HF_REPO}: {last}")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        own = {}
        for k, v in sd.items():
            k = k.removeprefix("model.").removeprefix("encoder.")
            if k == "patch_embed.proj.weight":
                own["patch_embed.weight"] = v.reshape(v.shape[0], v.shape[1], *v.shape[-2:])  # Conv3d (t=1) -> 2d
            elif k == "patch_embed.proj.bias":
                own["patch_embed.bias"] = v
            elif k == "cls_token" or k.startswith(("blocks.", "norm.")):
                own[k] = v
        missing = [k for k in self.state_dict() if k.startswith(("blocks.", "patch_embed", "norm.", "cls_token"))
                   and k not in own]
        if missing:
            raise RuntimeError(f"Prithvi checkpoint is missing {len(missing)} encoder weights, e.g. {missing[:3]}")
        self.load_state_dict(own, strict=False)
        self.pretrained_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        ph, pw = (-H) % PATCH, (-W) % PATCH
        xs = F.pad(x, (0, pw, 0, ph), mode="replicate") if ph or pw else x
        t = self.patch_embed(xs.index_select(1, self.band_idx))
        gh, gw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2) + sincos_pos_embed(EMBED, gh, gw).to(t)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], 1)
        mid = None
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i == DEPTH // 2 - 1:
                mid = t
        t = self.norm(t)
        tok = torch.cat([mid, t], 2)[:, 1:].transpose(1, 2).reshape(B, EMBED * 2, gh, gw)
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f4 = self.c4(f2)
        y = F.interpolate(self.proj(tok), size=f4.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d4(torch.cat([y, f4], 1))
        y = self.d2(torch.cat([F.interpolate(y, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        y = self.d1(torch.cat([F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        return self.head(y)


def build_prithvi(in_channels: int, n_classes: int, features: list[str], pretrained: bool) -> PrithviSeg:
    missing = [b for b in PRITHVI_BANDS if b not in features]
    if missing:
        raise ValueError(f"Prithvi needs Sentinel-2 bands {missing}")
    return PrithviSeg(in_channels, n_classes, [features.index(b) for b in PRITHVI_BANDS], pretrained)
