"""Whole-scene inference: overlapping tiles blended with a smooth window + flip TTA.

v1 predicted non-overlapping tiles, which leaves visible seams at every tile edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

from .features import Normalizer, compute_features, valid_mask
from .schema import IGNORE_INDEX, N_CLASSES


def _window(size: int) -> np.ndarray:
    w = np.hanning(size + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


@torch.no_grad()
def predict_array(model: nn.Module, norm: Normalizer, bands: np.ndarray, tile: int = 256, overlap: int = 64,
                  tta: bool = True, batch: int = 8, device=None, progress=None,
                  anc: np.ndarray | None = None, names: list[str] | None = None,
                  class_bias: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(C,H,W) raw S2 bands -> (class map uint8 with 255 = nodata, confidence float32 0..1).

    `names`: the model's feature list (from its checkpoint). `class_bias`: per-class log-probability
    offsets tuned on validation data (e.g. how readily to call seagrass)."""
    device = device or next(model.parameters()).device
    model.eval()
    feats = norm(compute_features(bands, anc=anc, names=names))
    C, H, W = feats.shape
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:
        feats = np.pad(feats, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    Hp, Wp = feats.shape[1:]
    step = tile - overlap
    ys = list(range(0, max(Hp - tile, 0) + 1, step))
    xs = list(range(0, max(Wp - tile, 0) + 1, step))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)
    coords = [(y, x) for y in ys for x in xs]

    acc = np.zeros((N_CLASSES, Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    win = _window(tile)
    for i in range(0, len(coords), batch):
        cs = coords[i : i + batch]
        xb = torch.from_numpy(np.stack([feats[:, y : y + tile, x : x + tile] for y, x in cs])).to(device)
        prob = model(xb).float().softmax(1)
        if tta:
            prob = prob + model(xb.flip(-1)).float().softmax(1).flip(-1)
            prob = prob + model(xb.flip(-2)).float().softmax(1).flip(-2)
            prob = prob / 3
        prob = prob.cpu().numpy()
        for (y, x), p in zip(cs, prob, strict=True):
            acc[: p.shape[0], y : y + tile, x : x + tile] += p * win
            wsum[y : y + tile, x : x + tile] += win
        if progress:
            progress(min(1.0, (i + batch) / len(coords)))
    prob = (acc / wsum)[:, :H, :W]
    if class_bias is not None and any(class_bias):
        prob = prob * np.exp(np.asarray(class_bias, np.float32)[: prob.shape[0], None, None])
        prob = prob / np.maximum(prob.sum(0, keepdims=True), 1e-12)
    cls = prob.argmax(0).astype(np.uint8)
    conf = prob.max(0)
    cls[~valid_mask(bands)] = IGNORE_INDEX
    return cls, conf


def predict_raster(model: nn.Module, norm: Normalizer, image_path: str | Path, out_path: str | Path,
                   tile: int = 256, overlap: int = 64, tta: bool = True, progress=None,
                   names: list[str] | None = None, class_bias: list[float] | None = None) -> Path:
    with rasterio.open(image_path) as src:
        bands = src.read()
        prof = src.profile.copy()
    cls, conf = predict_array(model, norm, bands, tile, overlap, tta, progress=progress, names=names,
                              class_bias=class_bias)
    prof.update(count=2, dtype="uint8", nodata=IGNORE_INDEX, compress="deflate")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(cls, 1)
        dst.write(np.round(conf * 100).astype(np.uint8), 2)
        dst.descriptions = ("class", "confidence_pct")
    return out_path
