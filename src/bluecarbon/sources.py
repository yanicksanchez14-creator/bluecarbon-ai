"""Where every published number comes from: one label per number (measured, mapped, IPCC default, computed,
illustrative) and the named source. Used by the website's site pages, the PDF site screens and the live tool, so
the three always say the same thing."""

from __future__ import annotations

from .schema import BLUE_CARBON_KEYS

LABELS = {
    "en": {"measured": "Measured", "mapped": "Mapped", "ipcc": "IPCC default", "computed": "Computed",
           "illustrative": "Illustrative"},
    "es": {"measured": "Medido", "mapped": "Mapeado", "ipcc": "Valor del IPCC", "computed": "Calculado",
           "illustrative": "Ilustrativo"},
}
NAMES = {"en": {"mangrove": "Mangrove", "saltmarsh": "Salt marsh", "seagrass": "Seagrass"},
         "es": {"mangrove": "Manglar", "saltmarsh": "Marisma", "seagrass": "Pasto marino"}}
T = {
    "en": dict(
        area="Habitat areas ({h})", area_src="BlueCarbon-AI habitat map ({model}), corrected for the model's known errors and "
                                  "capped at what its accuracy supports",
        soil="{h} soil carbon", soil_m="{n} soil cores from {s} studies within {r} km (Smithsonian Coastal Carbon "
                                       "Library), 5-95% range", soil_i="IPCC 2013 Wetlands Supplement, Tier 1 range",
        bio="{h} biomass", bio_m="NASA mangrove canopy-height biomass map, {agb:.0f} t/ha above ground, ±30% "
                                 "(Simard et al. 2019)", bio_i="IPCC 2013 Wetlands Supplement, Tier 1 range",
        gedi=" Checked against {n} GEDI lidar footprints ({g:.0f} t/ha).",
        burial="Carbon absorbed per hectare ({h})", burial_src="IPCC 2013 Wetlands Supplement, Tier 1 burial rates",
        totals="Carbon stored and absorbed (totals)", totals_src="5,000 Monte Carlo draws over every input above; "
                                                                 "90% interval shown",
        value="Credit potential and value", value_src="Price and deduction assumptions; a registry sets the real "
                                                      "deductions (Verra VM0033)",
        slr="Room to move inland (sea-level rise)", slr_src="NASADEM elevation and ESA WorldCover 2021 (not built up)",
    ),
    "es": dict(
        area="Áreas de hábitat ({h})", area_src="Mapa de hábitats de BlueCarbon-AI ({model}), corregido por los errores conocidos "
                                     "del modelo y limitado a lo que su exactitud respalda",
        soil="Carbono del suelo, {h}", soil_m="{n} núcleos de suelo de {s} estudios a menos de {r} km (Smithsonian "
                                              "Coastal Carbon Library), rango 5-95%",
        soil_i="Suplemento de Humedales del IPCC 2013, rango Nivel 1",
        bio="Biomasa, {h}", bio_m="Mapa de biomasa por altura de dosel de la NASA, {agb:.0f} t/ha aérea, ±30% "
                                  "(Simard et al. 2019)", bio_i="Suplemento de Humedales del IPCC 2013, rango Nivel 1",
        gedi=" Comparado con {n} mediciones lidar GEDI ({g:.0f} t/ha).",
        burial="Carbono absorbido por hectárea ({h})", burial_src="Suplemento de Humedales del IPCC 2013, tasas de "
                                                                 "entierro Nivel 1",
        totals="Carbono almacenado y absorbido (totales)", totals_src="5.000 simulaciones Monte Carlo sobre todos "
                                                                      "los datos anteriores; intervalo del 90%",
        value="Potencial y valor de créditos", value_src="Supuestos de precio y deducciones; el registro fija las "
                                                         "deducciones reales (Verra VM0033)",
        slr="Espacio para migrar tierra adentro", slr_src="Elevación NASADEM y ESA WorldCover 2021 (sin urbanizar)",
    ),
}


def number_sources(carbon_classes: dict, lang: str = "en", model: str = "model v2", lidar: dict | None = None,
                   has_sea_level: bool = False, min_area_ha: float = 1.0) -> list[dict]:
    """[{number, kind, label, source}] for one site. carbon_classes = the carbon report's per-habitat entries
    (with area_ha, soil{source,n_cores,n_studies,radius_km}, biomass{source,agb_mg_ha})."""
    t, lab, nm = T[lang], LABELS[lang], NAMES[lang]
    rows = []

    def add(number, kind, source):
        rows.append({"number": number, "kind": kind, "label": lab[kind], "source": source})

    present = [k for k in BLUE_CARBON_KEYS if (carbon_classes.get(k) or {}).get("area_ha", 0) >= min_area_ha]
    hl = ", ".join(nm[k].lower() for k in present)
    if present:
        add(t["area"].format(h=hl), "mapped", t["area_src"].format(model=model))
    for k in present:
        s = carbon_classes[k].get("soil") or {}
        if s.get("source") == "measured":
            add(t["soil"].format(h=nm[k].lower() if lang == "es" else nm[k]), "measured",
                t["soil_m"].format(n=s.get("n_cores") or s.get("cores"), s=s.get("n_studies") or s.get("studies"),
                                   r=s.get("radius_km")))
        else:
            add(t["soil"].format(h=nm[k].lower() if lang == "es" else nm[k]), "ipcc", t["soil_i"])
        b = carbon_classes[k].get("biomass") or {}
        if b.get("source") == "measured":
            src = t["bio_m"].format(agb=b.get("agb_mg_ha") or 0)
            if lidar and lidar.get("footprints"):
                src += t["gedi"].format(n=lidar["footprints"], g=lidar.get("gedi_agb") or 0)
            add(t["bio"].format(h=nm[k].lower() if lang == "es" else nm[k]), "measured", src)
        else:
            add(t["bio"].format(h=nm[k].lower() if lang == "es" else nm[k]), "ipcc", t["bio_i"])
    if present:
        add(t["burial"].format(h=hl), "ipcc", t["burial_src"])
        add(t["totals"], "computed", t["totals_src"])
        add(t["value"], "illustrative", t["value_src"])
    if has_sea_level:
        add(t["slr"], "mapped", t["slr_src"])
    return rows
