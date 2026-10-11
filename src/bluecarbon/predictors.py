"""One interface over both model families (U-Net checkpoints and spectral LightGBM models)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Predictor:
    kind: str
    meta: dict
    needs_ancillary: bool = False

    def predict(self, bands: np.ndarray, tile: int = 256, overlap: int = 64, tta: bool = True,
                progress=None, anc: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _check_anc(self, anc):
        if self.needs_ancillary and anc is None:
            raise ValueError("This model needs the ancillary layers (elevation, tidal probability); "
                             "pass anc= or put ancillary.tif next to the image")
        return anc if self.needs_ancillary else None


class TorchPredictor(Predictor):
    kind = "unet"

    def __init__(self, path: str | Path, device: str = "auto"):
        from .features import N_FEATURES
        from .model import load_checkpoint, resolve_device

        self.model, self.norm, ck = load_checkpoint(path, resolve_device(device))
        self.features = list(ck["features"])
        self.class_bias = ck.get("class_bias")
        self.needs_ancillary = len(ck["features"]) > N_FEATURES
        self.meta = {"arch": ck["arch"], "encoder": ck["encoder"], "kind": self.kind,
                     "metrics": ck.get("metrics", {}), "extra": ck.get("extra", {})}

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None, anc=None):
        from .predict import predict_array

        return predict_array(self.model, self.norm, bands, tile, overlap, tta, progress=progress,
                             anc=self._check_anc(anc), names=self.features, class_bias=self.class_bias)


class SpectralPredictor(Predictor):
    kind = "spectral-lgbm"

    def __init__(self, path: str | Path):
        from .spectral import SpectralModel

        self.model = SpectralModel.load(path)
        self.features = list(self.model.features)
        self.needs_ancillary = self.model.uses_ancillary
        self.meta = {**self.model.info, "metrics": self.model.metrics, "extra": self.model.extra}

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None, anc=None):
        out = self.model.predict(bands, anc=self._check_anc(anc))
        if progress:
            progress(1.0)
        return out


def is_spectral(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix == ".json":
        return True
    with open(p, "rb") as f:
        return f.read(1) == b"{"


class _WithSeagrass(Predictor):
    """The main model, with the seagrass specialist's second opinion on water / seagrass pixels."""

    def __init__(self, base: Predictor, specialist):
        self.base, self.specialist = base, specialist
        self.kind, self.meta, self.needs_ancillary = base.kind, base.meta, base.needs_ancillary
        self.features = getattr(base, "features", [])
        self.meta = {**base.meta, "seagrass_specialist": specialist.metrics}

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None, anc=None):
        cls, conf = self.base.predict(bands, tile, overlap, tta, progress=progress, anc=anc)
        fused = self.specialist.apply(cls, bands, anc[:8] if anc is not None else None)
        return fused, conf


def load_predictor(path: str | Path, device: str = "auto", seagrass: bool = True) -> Predictor:
    base = SpectralPredictor(path) if is_spectral(path) else TorchPredictor(path, device)
    if seagrass:
        from .seagrass import SeagrassSpecialist, specialist_path_for

        sp = specialist_path_for(path)
        if sp.exists():
            s = SeagrassSpecialist.load(sp)
            if s.enabled:
                return _WithSeagrass(base, s)
    return base


def model_card(path: str | Path) -> dict:
    """Metadata for display without loading weights into a network."""
    if is_spectral(path):
        d = json.loads(Path(path).read_text())
        return {"arch": "LightGBM", "encoder": "spectral-context", "kind": d["kind"], "metrics": d.get("metrics", {}),
                "extra": d.get("extra", {})}
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {"arch": ck["arch"], "encoder": ck["encoder"], "kind": "unet", "metrics": ck.get("metrics", {}),
            "extra": ck.get("extra", {})}


def ancillary_path_for(image_path: str | Path) -> Path | None:
    """ancillary.tif lives next to image.tif (or t0_image.tif / t1_image.tif)."""
    p = Path(image_path)
    for cand in (p.with_name(p.name.replace("image", "ancillary")), p.parent / "ancillary.tif"):
        if cand.exists() and cand != p:
            return cand
    return None


def radar_path_for(image_path: str | Path) -> Path | None:
    """radar.tif (Sentinel-1) next to image.tif, image_2020.tif -> radar_2020.tif, t0_image -> t0_radar."""
    p = Path(image_path)
    cand = p.with_name(p.name.replace("image", "radar"))
    return cand if cand.exists() and cand != p else None


def read_ancillary(image_path: str | Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Context stack for a model: ancillary.tif bands, then Sentinel-1 radar bands when radar.tif exists."""
    import rasterio

    a = ancillary_path_for(image_path)
    if a is None:
        return None
    with rasterio.open(a) as ds:
        arr = ds.read()
    if arr.shape[1:] != tuple(shape):
        raise ValueError(f"{a} is {arr.shape[1:]} but the image is {shape}")
    r = radar_path_for(image_path)
    if r is not None:
        with rasterio.open(r) as ds:
            rad = ds.read()
        if rad.shape[1:] == arr.shape[1:]:
            arr = np.concatenate([arr.astype(np.int16), rad.astype(np.int16)])
    return arr
