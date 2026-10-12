"""Kelp canopy: FAI background removal, fraction and calibration against a synthetic Kelpwatch."""

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from bluecarbon.kelp import calibrate, canopy_ha, kelp_fraction, our_cells, site_series


def _scene(seed, kelp_cols=(30, 40), cover=0.6):
    rng = np.random.default_rng(seed)
    land = np.zeros((90, 90), bool)
    land[:, :10] = True
    fai = rng.normal(0.0, 0.0008, land.shape).astype(np.float32)   # ocean background
    a, b = kelp_cols
    fai[20:70, a:b] += 0.004 + cover * (0.08 - 0.004)              # canopy: fraction `cover` of the end member
    truth = np.zeros(land.shape)
    truth[20:70, a:b] = cover
    return fai, land, truth


def test_fraction_and_area():
    fai, land, truth = _scene(0)
    f = kelp_fraction(fai, land, 20.0)
    assert abs(canopy_ha(f, 20.0) - truth.sum() * 400 / 1e4) / (truth.sum() * 400 / 1e4) < 0.1
    assert np.nansum(f[:, :10]) == 0                                 # never on land


def test_calibration_recovers_and_scores_held_out():
    samples = []
    for i, (site, cover) in enumerate([("a", 0.5), ("a", 0.7), ("b", 0.4), ("b", 0.8), ("t", 0.6)]):
        fai, land, truth = _scene(i, cover=cover)
        kw = our_cells(truth, 20.0)                                   # "Kelpwatch" = the truth, in m2 per cell
        samples.append({"site": site, "year": 2020, "quarter": i % 4 + 1, "fai": fai, "land": land, "px_m": 20.0,
                        "kw_cells": kw, "tr": from_origin(0, 0, 20, 20), "crs": CRS.from_epsg(32611)})
    cal = calibrate(samples, {"t"})
    assert cal["threshold"] <= 0.006 and 0.06 <= cal["endmember"] <= 0.10
    assert cal["held_out"]["cells_m2"]["corr"] > 0.9
    s = site_series([x for x in samples if x["site"] == "t"], cal)
    assert abs(s[0]["canopy_ha"] - s[0]["kelpwatch_ha"]) / s[0]["kelpwatch_ha"] < 0.15


def test_read_kelpwatch_layout(tmp_path):
    import pytest

    xr = pytest.importorskip("xarray")
    from bluecarbon.kelp import kelpwatch_cells, read_kelpwatch

    ds = xr.Dataset({"area": (("time", "station"), np.array([[900.0, 0.0, 450.0], [0.0, 900.0, 900.0]])),
                     "latitude": ("station", np.array([32.70, 32.71, 33.9])),
                     "longitude": ("station", np.array([-117.27, -117.26, -117.0])),
                     "year": ("time", np.array([2020, 2020])), "quarter": ("time", np.array([2, 3]))})
    p = tmp_path / "kw.nc"
    ds.to_netcdf(p)
    kw = read_kelpwatch(p, [-117.30, 32.64, -117.23, 32.74])
    assert kw["area_m2"].shape == (2, 2) and list(kw["quarter"]) == [2, 3]
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    cells = kelpwatch_cells(kw, 1, from_origin(474000, 3621000, 20, 20), CRS.from_epsg(32611), (600, 600))
    assert cells.sum() == 900.0


def test_surf_zone_foam_is_masked_by_calibration():
    samples = []
    for i, (site, cover) in enumerate([("a", 0.5), ("a", 0.7), ("b", 0.4), ("b", 0.8), ("t", 0.6)]):
        fai, land, truth = _scene(i, cover=cover)
        fai[:, 10:12] += 0.05                                         # breaking waves along the shore (40 m wide)
        samples.append({"site": site, "year": 2020, "quarter": i % 4 + 1, "fai": fai, "land": land, "px_m": 20.0,
                        "kw_cells": our_cells(truth, 20.0), "tr": from_origin(0, 0, 20, 20),
                        "crs": CRS.from_epsg(32611)})
    cal = calibrate(samples, {"t"})
    assert cal["shore_m"] >= 40
    assert abs(cal["held_out"]["site_quarter_ha"].get("bias", 0)) < 2


def test_zone_fits_and_every_bed_scored_out_of_fit():
    from bluecarbon.kelp import params_for

    samples = []
    for i, (site, cover, em) in enumerate([("g1", .5, .08), ("g2", .7, .08), ("g3", .6, .08),
                                           ("b1", .5, .16), ("b2", .7, .16), ("b3", .6, .16)]):
        rng = np.random.default_rng(i)
        land = np.zeros((90, 90), bool)
        land[:, :10] = True
        fai = rng.normal(0.0, 0.0008, land.shape).astype(np.float32)
        fai[20:70, 30:40] += 0.004 + cover * (em - 0.004)        # bull kelp (b*) is brighter per unit cover here
        truth = np.zeros(land.shape)
        truth[20:70, 30:40] = cover
        samples.append({"site": site, "year": 2020, "quarter": 1, "fai": fai, "land": land, "px_m": 20.0,
                        "kw_cells": our_cells(truth, 20.0), "tr": from_origin(0, 0, 20, 20), "crs": CRS.from_epsg(32611)})
    zones = {s: ("giant" if s[0] == "g" else "bull") for s in ("g1", "g2", "g3", "b1", "b2", "b3")}
    cal = calibrate(samples, set(), zones, log=lambda *_: None)
    assert set(cal["zones"]) == {"giant", "bull"}
    assert params_for(cal, "b1")[1] > params_for(cal, "g1")[1]          # bull zone gets its own end member
    assert set(cal["per_site"]) == set(zones)                             # every bed has an out-of-fit score
    assert all(v["how"] == "left out of its own fit" for v in cal["per_site"].values())
