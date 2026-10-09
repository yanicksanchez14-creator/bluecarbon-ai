"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ImageryCfg(BaseModel):
    collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    cloud_score_collection: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    cloud_score_band: str = "cs_cdf"
    clear_threshold: float = 0.6
    scale_m: int = 10
    tile_px: int = 512


class LabelsCfg(BaseModel):
    worldcover: str = "ESA/WorldCover/v200"
    intertidal: str = "UQ/murray/Intertidal/v1_1/global_intertidal"
    reef_habitat: str = "ACA/reef_habitat/v2_0"
    seagrass_value: int = 14
    marsh_max_elev_m: float = 5
    tidal_wetland: str = "JCU/Murray/GIC/global_tidal_wetland_change/2019"
    tidal_wetland_band: str = "twprobability_end"
    tidal_wetland_min_prob: float = 50
    seagrass_vectors: list[str] = []
    seagrass_surveys: list[dict] = []
    bathymetry: str = "projects/sat-io/open-datasets/gebco/gebco_grid"
    seagrass_max_depth_m: float | None = None
    wetland_map: str = "projects/sat-io/open-datasets/GWL_FCS30"
    wetland_year: int = 2021
    dem: str = "NASA/NASADEM_HGT/001"
    overrides: str | None = None
    boundary_ignore_px: int = 1


class ChipsCfg(BaseModel):
    size: int = 256
    stride: int = 256
    min_labeled_frac: float = 0.2
    block_km: float = 5
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42


class ModelCfg(BaseModel):
    arch: str = "Unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    use_ancillary: bool = True


class TrainCfg(BaseModel):
    epochs: int = 60
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    class_weighting: str = "sqrt_inverse"
    amp: bool = True
    num_workers: int = 2
    patience: int = 12
    rare_oversample: float = 4.0
    device: str = "auto"


class PredictCfg(BaseModel):
    tile: int = 256
    overlap: int = 64
    tta: bool = True


class CarbonClassCfg(BaseModel):
    soil: tuple[float, float, float]
    biomass: tuple[float, float, float]
    accumulation: tuple[float, float, float]


class CarbonCfg(BaseModel):
    co2_per_c: float = 3.667
    classes: dict[str, CarbonClassCfg] = Field(
        default_factory=lambda: {
            "mangrove": CarbonClassCfg(soil=(436, 471, 510), biomass=(25, 42, 80), accumulation=(1.3, 1.62, 2.0)),
            "saltmarsh": CarbonClassCfg(soil=(254, 255, 297), biomass=(2, 5, 10), accumulation=(0.7, 0.91, 1.1)),
            "seagrass": CarbonClassCfg(soil=(84, 108, 139), biomass=(0.5, 1, 2), accumulation=(0.2, 0.43, 0.7)),
        }
    )
    price_usd_per_tco2e: tuple[float, float, float] = (15, 25, 40)
    monte_carlo: int = 5000


class Config(BaseModel):
    project: str = "bluecarbon-ai"
    workdir: str = "runs/default"
    imagery: ImageryCfg = ImageryCfg()
    labels: LabelsCfg = LabelsCfg()
    chips: ChipsCfg = ChipsCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    predict: PredictCfg = PredictCfg()
    carbon: CarbonCfg = CarbonCfg()

    @property
    def work(self) -> Path:
        return Path(self.workdir)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    data: dict = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config.model_validate(data)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
