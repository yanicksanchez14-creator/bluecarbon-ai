"""Turn every fetched site into a demo page for the web app (run in Colab after training).

    python scripts/export_all_sites.py            # uses runs/default/model/best.*

Training sites are labelled as such on their page, because maps of places the model trained on
look better than they would on a new coastline. Held-out sites are the honest showcase.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

WORK = Path("runs/default")

TITLES = {
    "mission_bay_ca": ("Mission Bay, San Diego", "California, USA"),
    "south_sd_bay_ca": ("South San Diego Bay", "California, USA"),
    "south_sf_bay_ca": ("South San Francisco Bay", "California, USA"),
    "ten_thousand_isl_fl": ("Ten Thousand Islands", "Florida, USA"),
    "florida_bay_fl": ("Florida Bay, Everglades", "Florida, USA"),
    "andros_bahamas": ("Andros Island", "The Bahamas"),
    "laguna_terminos_mx": ("Laguna de Términos", "Campeche, Mexico"),
    "sundarbans_bd": ("Sundarbans", "Bangladesh"),
    "wadden_sea_de": ("Wadden Sea", "Schleswig-Holstein, Germany"),
    "moreton_bay_au": ("Moreton Bay", "Queensland, Australia"),
    "tampa_bay_fl": ("Tampa Bay", "Florida, USA"),
    "florida_keys_fl": ("Lower Florida Keys", "Florida, USA"),
    "belize_lagoon_bz": ("Belize Barrier Reef lagoon", "Belize"),
    "exuma_bahamas": ("Exuma Cays", "The Bahamas"),
    "hinchinbrook_au": ("Hinchinbrook Island", "Queensland, Australia"),
    "chwaka_bay_tz": ("Chwaka Bay", "Zanzibar, Tanzania"),
    "safaga_eg": ("Safaga", "Red Sea, Egypt"),
    "sapelo_ga": ("Sapelo Island", "Georgia, USA"),
    "blackwater_md": ("Blackwater, Chesapeake Bay", "Maryland, USA"),
    "venice_lagoon_it": ("Venice Lagoon", "Veneto, Italy"),
    "yellow_river_cn": ("Yellow River Delta", "Shandong, China"),
    "shark_bay_au": ("Shark Bay", "Western Australia"),
    "charlotte_harbor_fl": ("Charlotte Harbor", "Florida, USA"),
    "cedar_key_fl": ("Cedar Key, Big Bend", "Florida, USA"),
    "puerto_morelos_mx": ("Puerto Morelos", "Quintana Roo, Mexico"),
    "marawah_ae": ("Marawah", "Abu Dhabi, UAE"),
    "bazaruto_mz": ("Bazaruto Archipelago", "Mozambique"),
    "coron_ph": ("Coron, Calamian Islands", "Palawan, Philippines"),
    "jobos_bay_pr": ("Jobos Bay", "Puerto Rico"),
    "al_wajh_sa": ("Al Wajh", "Red Sea, Saudi Arabia"),
    "st_joseph_bay_fl": ("St. Joseph Bay", "Florida, USA"),
    "indian_river_fl": ("Indian River Lagoon", "Florida, USA"),
    "biscayne_bay_fl": ("Biscayne Bay", "Florida, USA"),
    "delaware_bay_nj": ("Delaware Bay", "New Jersey, USA"),
    "morecambe_bay_uk": ("Morecambe Bay", "England, UK"),
    "mont_st_michel_fr": ("Bay of Mont-Saint-Michel", "Normandy, France"),
    "chongming_cn": ("Chongming Dongtan", "Shanghai, China"),
    "bahia_blanca_ar": ("Bahia Blanca", "Buenos Aires, Argentina"),
    "westernport_au": ("Western Port", "Victoria, Australia"),
    "knysna_za": ("Knysna Estuary", "Western Cape, South Africa"),
    "ca_mau_vn": ("Ca Mau", "Mekong Delta, Vietnam"),
    "zambezi_delta_mz": ("Zambezi Delta", "Mozambique"),
    "plum_island_ma": ("Plum Island, Great Marsh", "Massachusetts, USA"),
    "shoalwater_bay_au": ("Shoalwater Bay", "Queensland, Australia"),
    "lower_laguna_madre_tx": ("Lower Laguna Madre", "Texas, USA"),
    "redfish_bay_tx": ("Redfish Bay, Corpus Christi", "Texas, USA"),
    "mobjack_bay_va": ("Mobjack Bay, Chesapeake Bay", "Virginia, USA"),
    "tangier_sound_md": ("Smith Island, Tangier Sound", "Maryland, USA"),
    "south_bay_va": ("South Bay, Virginia Coast Reserve", "Virginia, USA"),
    "buzzards_bay_ma": ("Buzzards Bay", "Massachusetts, USA"),
    "duxbury_bay_ma": ("Duxbury and Plymouth Bays", "Massachusetts, USA"),
    "humboldt_bay_ca": ("Humboldt Bay", "California, USA"),
    "elkhorn_slough_ca": ("Elkhorn Slough", "California, USA"),
    "willapa_bay_wa": ("Willapa Bay", "Washington, USA"),
    "barnegat_bay_nj": ("Barnegat Bay", "New Jersey, USA"),
    "cape_romain_sc": ("Cape Romain", "South Carolina, USA"),
    "great_bay_nh": ("Great Bay Estuary", "New Hampshire, USA"),
    "rookery_bay_fl": ("Rookery Bay, Naples", "Florida, USA"),
    "port_stephens_au": ("Port Stephens", "New South Wales, Australia"),
    "jervis_bay_au": ("Jervis Bay", "New South Wales, Australia"),
    "botany_bay_au": ("Botany Bay", "New South Wales, Australia"),
    "wallis_lake_au": ("Wallis Lake", "New South Wales, Australia"),
    "great_sandy_strait_au": ("Great Sandy Strait", "Queensland, Australia"),
    "exmouth_gulf_au": ("Exmouth Gulf", "Western Australia"),
    "the_wash_uk": ("The Wash", "England, UK"),
    "chichester_harbour_uk": ("Chichester Harbour", "England, UK"),
    "blackwater_essex_uk": ("Blackwater Estuary, Essex", "England, UK"),
    "gazi_bay_ke": ("Gazi Bay", "Kenya"),
    "rufiji_delta_tz": ("Rufiji Delta", "Tanzania"),
    "matang_my": ("Matang Mangrove Forest Reserve", "Perak, Malaysia"),
    "mahakam_delta_id": ("Mahakam Delta", "East Kalimantan, Indonesia"),
    "bintuni_bay_id": ("Bintuni Bay", "West Papua, Indonesia"),
    "braganca_br": ("Bragança Peninsula", "Pará, Brazil"),
    "guayas_ec": ("Gulf of Guayaquil", "Guayas, Ecuador"),
    "gulf_of_fonseca_hn": ("Gulf of Fonseca", "Honduras"),
    "niger_delta_ng": ("Niger Delta", "Rivers State, Nigeria"),
}


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    model = WORK / "model" / (WORK / "model" / "best.txt").read_text().split()[1]
    cfg = yaml.safe_load(Path("configs/sites.yaml").read_text())
    for s in cfg["sites"]:
        d = WORK / "sites" / s["name"]
        if not (d / "image.tif").exists():
            print(f"skip {s['name']} (not downloaded)")
            continue
        title, region = TITLES.get(s["name"], (s["name"], ""))
        held_out = s.get("role") == "test"
        note = s.get("note", "").removeprefix("Held-out: ")
        desc = (f"{note}. Sentinel-2 {cfg['year']} composite. "
                + ("This area was held out of training, so the map shows how the AI does on a coastline it has "
                   "never seen." if held_out else
                   "This area was part of the training data, so the map looks better than it would on a new coastline."))
        run("bluecarbon", "predict", str(d / "image.tif"), "-m", str(model), "--out", str(d / "pred.tif"))
        run("bluecarbon", "export-demo", str(d), "--name", s["name"], "--title", title, "-m", str(model),
            "--description", desc)
        meta_p = Path("demo_data") / s["name"] / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta.update(region=region, period=f"Full year {cfg['year']}", held_out=held_out)
        meta_p.write_text(json.dumps(meta, indent=2))
    print("done:", sorted(p.name for p in Path("demo_data").iterdir()))


if __name__ == "__main__":
    sys.exit(main())
