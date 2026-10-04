"""Build data/soil_carbon_cores.csv from the Smithsonian Coastal Carbon Library.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/Smithsonian/CCN-Data-Library ccn
    git -C ccn sparse-checkout set data/CCN_synthesis
    python scripts/build_soil_carbon.py ccn/data/CCN_synthesis

For every soil core it computes the soil organic carbon stock to 1 m (t C/ha), the same quantity as
the IPCC Tier 1 "soil carbon to 1 m" default, so measured local values can replace the global ones:

  carbon density (g C/cm3) = dry bulk density x fraction organic carbon
  stock to 1 m (t C/ha)    = mean carbon density over the sampled 0-100 cm x 100 cm x 100

Measured fraction carbon is used only when the study's methods say it is organic carbon (or that
carbonates were removed / the soil is low in carbonate). Many studies report TOTAL carbon, which in
carbonate seafloor (Florida, the Bahamas, the Red Sea) counts limestone as carbon and inflates stocks
several-fold. For those, and for studies without method records, organic carbon is estimated from
organic matter (loss on ignition) with the Craft et al. (1991) relation OC = 0.40 OM + 0.25 OM^2
(fractions); if there is no organic matter value the sample is dropped. Cores with less than 30 cm of
usable samples are dropped, as are values outside 5-2000 t C/ha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HABITATS = {"marsh": "saltmarsh", "mangrove": "mangrove", "seagrass": "seagrass"}
MIN_SAMPLED_CM = 30


ORGANIC_OK = {"acid fumigation", "direct acid treatment", "low carbonate soil", "total carbon difference after LOI"}


def organic_carbon_studies(methods: pd.DataFrame) -> set[str]:
    """Studies whose `fraction_carbon` is organic carbon (not total carbon including carbonate)."""
    m = methods
    ok = (m["fraction_carbon_type"].eq("organic carbon") | m["carbonates_removed"].astype(str).eq("True")
          | m["carbonate_removal_method"].isin(ORGANIC_OK))
    bad = m["fraction_carbon_type"].eq("total carbon") & ~m["carbonate_removal_method"].isin(ORGANIC_OK) \
        & ~m["carbonates_removed"].astype(str).eq("True")
    good = set(m.loc[ok, "study_id"])
    return good - set(m.loc[bad, "study_id"])


def core_stocks(cores: pd.DataFrame, depth: pd.DataFrame, methods: pd.DataFrame | None = None) -> pd.DataFrame:
    d = depth[["study_id", "core_id", "depth_min", "depth_max", "dry_bulk_density",
               "fraction_carbon", "fraction_organic_matter"]].copy()
    for c in d.columns[2:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    om = d["fraction_organic_matter"].clip(0, 1)
    fc = d["fraction_carbon"].where(d["fraction_carbon"].between(0, 0.6), np.nan)
    if methods is not None:
        fc = fc.where(d["study_id"].isin(organic_carbon_studies(methods)), np.nan)
    d["fc"] = fc.fillna(0.40 * om + 0.25 * om**2)
    d["bd"] = d["dry_bulk_density"].where(d["dry_bulk_density"].between(0.01, 3.0))
    d = d.dropna(subset=["depth_min", "depth_max", "fc", "bd"])
    d = d[(d.depth_max > d.depth_min) & (d.depth_min < 100)]
    d["top"], d["bot"] = d.depth_min.clip(0, 100), d.depth_max.clip(0, 100)
    d["thick"] = d.bot - d.top
    d = d[d.thick > 0]
    d["cd_x_t"] = d.bd * d.fc * d.thick
    g = d.groupby(["study_id", "core_id"]).agg(sampled_cm=("thick", "sum"), cdt=("cd_x_t", "sum")).reset_index()
    g = g[g.sampled_cm >= MIN_SAMPLED_CM]
    g["soc_1m_tC_ha"] = g.cdt / g.sampled_cm * 100 * 100
    keep = cores[["study_id", "core_id", "latitude", "longitude", "habitat", "country"]].copy()
    keep["habitat"] = keep["habitat"].map(HABITATS)
    out = g.merge(keep, on=["study_id", "core_id"]).dropna(subset=["latitude", "longitude", "habitat"])
    out = out[out.soc_1m_tC_ha.between(5, 2000)]
    out["latitude"], out["longitude"] = out.latitude.round(4), out.longitude.round(4)
    out["soc_1m_tC_ha"] = out.soc_1m_tC_ha.round(1)
    out["sampled_cm"] = out.sampled_cm.round(0).astype(int)
    return out[["habitat", "latitude", "longitude", "soc_1m_tC_ha", "sampled_cm", "country", "study_id"]]


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "ccn/data/CCN_synthesis")
    cores = pd.read_csv(src / "CCN_cores.csv", low_memory=False)
    depth = pd.read_csv(src / "CCN_depthseries.csv", low_memory=False)
    methods = pd.read_csv(src / "CCN_methods.csv", low_memory=False)
    out = core_stocks(cores, depth, methods).sort_values(["habitat", "latitude", "longitude"])
    dst = Path(__file__).resolve().parents[1] / "data" / "soil_carbon_cores.csv"
    dst.parent.mkdir(exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"{len(out):,} cores -> {dst}")
    print(out.groupby("habitat").soc_1m_tC_ha.describe(percentiles=[0.1, 0.5, 0.9]).round(0))


if __name__ == "__main__":
    main()
