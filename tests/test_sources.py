from bluecarbon.sources import number_sources


def test_every_number_labelled():
    cls = {"mangrove": {"area_ha": 120, "soil": {"source": "measured", "n_cores": 42, "n_studies": 3, "radius_km": 100},
                        "biomass": {"source": "measured", "agb_mg_ha": 18.5}},
           "seagrass": {"area_ha": 50, "soil": {"source": "ipcc"}, "biomass": {"source": "ipcc"}},
           "saltmarsh": {"area_ha": 0.2}}
    rows = number_sources(cls, lidar={"footprints": 300, "gedi_agb": 20}, has_sea_level=True)
    kinds = {r["number"]: r["kind"] for r in rows}
    assert kinds["Mangrove soil carbon"] == "measured" and kinds["Seagrass soil carbon"] == "ipcc"
    assert kinds["Habitat areas (mangrove, seagrass)"] == "mapped"
    assert "GEDI" in next(r["source"] for r in rows if r["number"] == "Mangrove biomass")
    assert number_sources(cls, lang="es")[0]["label"] == "Mapeado"
