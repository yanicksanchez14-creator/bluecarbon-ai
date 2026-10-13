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


class EnsemblePredictor(Predictor):
    """The U-Net and the LightGBM model blended per pixel: p = w_c * p_unet + (1 - w_c) * p_lgbm for each class c,
    with the weights w_c tuned on validation tiles (`bluecarbon ensemble`). Defined by a small
    <name>.ensemble.json next to both models: {"unet": "model.pt", "spectral": "spectral.json", "weights": [...]}."""

    kind = "ensemble"

    def __init__(self, path: str | Path | None = None, device: str = "auto", *, unet=None, lgbm=None, weights=None):
        d = json.loads(Path(path).read_text()) if path else {}
        base = Path(path).parent if path else Path(".")
        self.unet = unet or TorchPredictor(base / d["unet"], device)
        self.lgbm = lgbm or SpectralPredictor(base / d["spectral"])
        self.weights = np.asarray(d["weights"] if weights is None else weights, np.float32)
        self.needs_ancillary = self.unet.needs_ancillary or self.lgbm.needs_ancillary
        self.features = sorted(set(self.unet.features) | set(self.lgbm.features))
        self.meta = {**self.unet.meta, "kind": self.kind, "arch": f"{self.unet.meta['arch']} + LightGBM",
                     "metrics": {**self.unet.meta.get("metrics", {}), **d.get("metrics", {})},
                     "ensemble": {k: v for k, v in d.items() if k != "metrics"} | {"weights": self.weights.tolist()}}

    def probs(self, bands, tile=256, overlap=64, tta=True, progress=None, anc=None):
        """(U-Net probabilities, LightGBM probabilities), each (N_CLASSES, H, W)."""
        from .predict import predict_array

        u = self.unet
        _, _, pu = predict_array(u.model, u.norm, bands, tile, overlap, tta, progress=progress,
                                 anc=u._check_anc(anc), names=u.features, class_bias=u.class_bias, return_prob=True)
        ps = self.lgbm.model.predict_proba(bands, anc=self.lgbm._check_anc(anc))
        return pu, ps

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None, anc=None):
        from .features import valid_mask
        from .schema import IGNORE_INDEX

        prob = blend(*self.probs(bands, tile, overlap, tta, progress, anc), self.weights)
        cls = prob.argmax(0).astype(np.uint8)
        cls[~valid_mask(bands)] = IGNORE_INDEX
        return cls, prob.max(0).astype(np.float32)


def blend(pu: np.ndarray, ps: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = min(pu.shape[0], ps.shape[0], len(w))
    return w[:n, None, None] * pu[:n] + (1 - w[:n, None, None]) * ps[:n]


def is_ensemble(path: str | Path) -> bool:
    return str(path).endswith(".ensemble.json")


def is_spectral(path: str | Path) -> bool:
    from .model import resolve_model_file

    p = resolve_model_file(path)
    if is_ensemble(p):
        return False
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
    if is_ensemble(path):
        base = EnsemblePredictor(path, device)
    else:
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
    if is_ensemble(path):
        d = json.loads(Path(path).read_text())
        card = model_card(Path(path).parent / d["unet"])
        return {**card, "kind": "ensemble", "arch": f"{card['arch']} + LightGBM",
                "ensemble": {k: v for k, v in d.items() if k != "metrics"},
                "metrics": {**card.get("metrics", {}), **d.get("metrics", {})}}
    if is_spectral(path):
        d = json.loads(Path(path).read_text())
        return {"arch": "LightGBM", "encoder": "spectral-context", "kind": d["kind"], "metrics": d.get("metrics", {}),
                "extra": d.get("extra", {})}
    import torch

    from .model import resolve_model_file

    ck = torch.load(resolve_model_file(path), map_location="cpu", weights_only=False)
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


def lband_path_for(image_path: str | Path) -> Path | None:
    """lband.tif (ALOS PALSAR L-band) next to image.tif, image_2020.tif -> lband_2020.tif."""
    p = Path(image_path)
    cand = p.with_name(p.name.replace("image", "lband"))
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
            lb = lband_path_for(image_path)
            if lb is not None:
                with rasterio.open(lb) as ds:
                    lband = ds.read()
                if lband.shape[1:] == arr.shape[1:]:
                    arr = np.concatenate([arr, lband.astype(np.int16)])
    return arr
