"""Measured soil carbon: core stock arithmetic, carbonate filter, and local lookup with IPCC fallback."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bluecarbon.carbon import carbon_report, local_soil, soil_sources
from bluecarbon.config import load_config


def _build_script():
    spec = importlib.util.spec_from_file_location("bsc", Path(__file__).parents[1] / "scripts" / "build_soil_carbon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_core_stock_and_carbonate_filter():
    pd = pytest.importorskip("pandas")  # only the offline build script needs pandas
    bsc = _build_script()
    cores = pd.DataFrame({"study_id": ["A", "B"], "core_id": ["a1", "b1"], "latitude": [10, 10], "longitude": [20, 20],
                          "habitat": ["mangrove", "seagrass"], "country": ["X", "X"]})
    # 0-50 cm, bulk density 0.5 g/cm3, 10% organic carbon -> 0.05 g C/cm3 -> 0.05 x 100 cm x 100 = 500 t C/ha
    depth = pd.DataFrame({"study_id": ["A", "B"], "core_id": ["a1", "b1"], "depth_min": [0, 0], "depth_max": [50, 50],
                          "dry_bulk_density": [0.5, 0.5], "fraction_carbon": [0.10, 0.10],
                          "fraction_organic_matter": [np.nan, 0.05]})
    methods = pd.DataFrame({"study_id": ["A", "B"], "fraction_carbon_type": ["organic carbon", "total carbon"],
                            "carbonates_removed": [True, False], "carbonate_removal_method": ["acid fumigation", "none specified"]})
    out = bsc.core_stocks(cores, depth, methods).set_index("study_id")
    assert np.isclose(out.loc["A", "soc_1m_tC_ha"], 500)
    # B reports TOTAL carbon -> its 10% is ignored; organic C from 5% organic matter = 0.02 + 0.000625
    assert np.isclose(out.loc["B", "soc_1m_tC_ha"], 0.5 * (0.40 * 0.05 + 0.25 * 0.05**2) * 100 * 100, atol=0.2)


def test_local_soil_and_fallback():
    cfg = load_config(None).carbon
    tampa = local_soil(27.7, -82.6)
    assert "mangrove" in tampa and tampa["mangrove"]["n_cores"] >= 8
    lo, mid, hi = tampa["mangrove"]["soil"]
    assert lo <= mid <= hi
    nowhere = soil_sources(cfg, -60.0, -150.0)  # open Southern Ocean: no cores -> IPCC for every habitat
    assert all(v["source"] == "ipcc" for v in nowhere.values())
    rep = carbon_report({"mangrove": 100.0}, cfg, lat=27.7, lon=-82.6)
    assert rep["classes"]["mangrove"]["soil"]["source"] == "measured" and "Coastal Carbon" in rep["method"]


def test_measured_mangrove_biomass():
    """NASA biomass replaces the IPCC default for mangrove only, with IPCC factors and +/-30% map error."""
    from bluecarbon.carbon import AGB_MAP_REL_ERROR, carbon_report, climate_zone, measured_biomass
    from bluecarbon.config import CarbonCfg

    cfg = CarbonCfg()
    assert climate_zone(27.7, 1300) == "subtropical" and climate_zone(10, 800) == "tropical dry"
    assert climate_zone(10, 2000) == climate_zone(10, None) == "tropical wet"
    stats = {"agb_mean": 150.0, "agb_count": 5000, "precip_mm": 2200}
    b = measured_biomass(stats, 9.0, cfg)
    lo, mid, hi = b["mangrove"]["biomass"]
    assert b["mangrove"]["source"] == "measured" and b["saltmarsh"]["source"] == "ipcc"
    assert abs(mid - 150 * 1.49 * 0.451) < 1e-6 and lo < mid < hi
    assert abs(lo / (150 * (1 - AGB_MAP_REL_ERROR) * 1.47 * 0.429) - 1) < 1e-6
    # too few measured pixels -> IPCC
    assert measured_biomass({**stats, "agb_count": 10}, 9.0, cfg)["mangrove"]["source"] == "ipcc"
    rep = carbon_report({"mangrove": 100.0}, cfg, lat=9.0, lon=-80.0, biomass_stats=stats)
    assert rep["classes"]["mangrove"]["biomass"]["source"] == "measured" and "Simard" in rep["method"]
