"""Turn a prediction raster into area + carbon numbers (JSON and Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from .carbon import carbon_report, change_report, class_areas_ha, class_pixel_counts, pixel_area_ha
from .config import CarbonCfg
from .metrics import error_adjusted_area
from .schema import BLUE_CARBON_KEYS, CLASSES


def read_classes(path: str | Path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform, ds.crs


def scene_report(pred_path: str | Path, cfg: CarbonCfg, test_confusion: list | None = None,
                 biomass_stats: dict | None = None) -> dict:
    from .priors import apply_to_classes, apply_to_confusion, raster_center_lonlat

    cls, transform, crs = read_classes(pred_path)
    lon, lat = raster_center_lonlat(transform, crs, *cls.shape)
    cls = apply_to_classes(cls, lat)
    areas = class_areas_ha(cls, transform, crs)
    rep = {"areas_ha": areas}
    sd, basis = None, areas
    if test_confusion is not None:
        # Maps are biased (e.g. over-predicting a rare class). Correct areas with the model's held-out
        # confusion matrix and use the corrected areas for carbon (Olofsson et al. 2014 good practice).
        px_ha = float(np.mean(pixel_area_ha(transform, crs, cls.shape[0])))
        adj = error_adjusted_area(apply_to_confusion(test_confusion, lat), class_pixel_counts(cls), px_ha)
        rep["error_adjusted_areas_ha"] = adj
        basis = {k: v["adjusted_ha"] for k, v in adj.items()}
        sd = {k: v["ci95_ha"] / 1.96 for k, v in adj.items()}
    rep["carbon_area_basis"] = "error_adjusted" if sd else "mapped"
    rep["carbon"] = carbon_report(basis, cfg, sd, lat=lat, lon=lon, biomass_stats=biomass_stats)
    return rep


def _areas_with_priors(path):
    from .priors import apply_to_classes, raster_center_lat

    cls, transform, crs = read_classes(path)
    return class_areas_ha(apply_to_classes(cls, raster_center_lat(transform, crs, *cls.shape)), transform, crs)


def change_scene_report(t0_path, t1_path, cfg: CarbonCfg) -> dict:
    from .priors import raster_center_lonlat

    a0 = _areas_with_priors(t0_path)
    a1 = _areas_with_priors(t1_path)
    cls, transform, crs = read_classes(t0_path)
    lon, lat = raster_center_lonlat(transform, crs, *cls.shape)
    return {"areas_t0_ha": a0, "areas_t1_ha": a1, "change": change_report(a0, a1, cfg, lat=lat, lon=lon)}


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"


def to_markdown(rep: dict, title: str = "BlueCarbon-AI report") -> str:
    lines = [f"# {title}", "", "| Habitat | Area (ha) |", "|---|---:|"]
    for c in CLASSES:
        lines.append(f"| {c.name} | {_fmt(rep['areas_ha'][c.key])} |")
    cb = rep["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    lines += [
        "",
        f"**Blue carbon habitat:** {_fmt(cb['blue_carbon_area_ha'])} ha  ",
        f"**Carbon stock:** {_fmt(s['mean'])} tCO2e (90% interval {_fmt(s['p05'])}-{_fmt(s['p95'])})  ",
        f"**Sequestration:** {_fmt(q['mean'])} tCO2e/yr (90% interval {_fmt(q['p05'])}-{_fmt(q['p95'])})  ",
        "",
        f"_Method: {cb['method']}. Crediting still requires measurements at the project site._",
    ]
    return "\n".join(lines)


def write_report(rep: dict, out_dir: str | Path, stem: str = "report") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / f"{stem}.json", out / f"{stem}.md"
    j.write_text(json.dumps(rep, indent=2))
    if "carbon" in rep:
        m.write_text(to_markdown(rep))
    return j, m


__all__ = ["scene_report", "change_scene_report", "to_markdown", "write_report", "BLUE_CARBON_KEYS"]
