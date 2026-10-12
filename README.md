# BlueCarbon-AI

**Mapping the coastal ecosystems that store carbon (mangroves, salt marshes and seagrass meadows) from
satellite imagery with machine learning, and estimating how much carbon they hold.**

[**Website →**](https://yanicksanchez14-creator.github.io/bluecarbon-ai/) &nbsp; [**Live analysis tool →**](https://bluecarbon-ai.streamlit.app)

[![CI](https://github.com/yanicksanchez14-creator/bluecarbon-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/yanicksanchez14-creator/bluecarbon-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Held-out coastlines: Sentinel-2 imagery (top) and the BlueCarbon-AI habitat map (bottom)](docs/img/hero.png)

## The problem

Coastal wetlands ("blue carbon" ecosystems) capture carbon up to several times faster per hectare than
land forests and keep it locked in their soils for centuries. They are also disappearing, and
conservation groups, governments and carbon-credit projects need to know **where they are, how large
they are, and how much carbon is at stake**. Mapping them by hand from imagery is slow and doesn't
scale.

## What BlueCarbon-AI does

1. **Pulls free satellite imagery.** It builds a cloud-free Sentinel-2 composite for any coastline and
   season through Google Earth Engine.
2. **Maps habitats automatically.** A trained model labels every 10 × 10 m pixel as open water,
   mangrove, salt marsh, seagrass, tidal flat or other land.
3. **Estimates carbon with honest uncertainty.** Habitat areas are corrected for the model's known
   error rates, then converted to stored carbon and yearly uptake using IPCC reference values, reported
   as a likely range rather than a single number.
4. **Serves it in a web app.** Anyone can explore mapped sites, switch between satellite, false-color and
   habitat views, and read a plain-language summary of the results.

## Highlights

- **End-to-end geospatial ML pipeline.** Earth Engine ingestion, Cloud Score+ cloud masking, seasonal
  median composites, and tiled download of any area size in the correct map projection. One CLI and one
  YAML config drive every stage.
- **Trained on published scientific maps.** Reference labels are fused from ESA WorldCover, the GWL_FCS30
  global wetland map, the Murray et al. global tidal-flat maps and the Allen Coral Atlas across **76 coastal
  sites on six continents**.
- **Evaluation designed not to cheat.** Training tiles never overlap, whole 5 km blocks go to one data
  split, and entire estuaries are held out, so scores reflect performance on coastlines the model has
  never seen.
- **Two model families, the best one wins.** A gradient-boosted (LightGBM) spectral model with
  water-column features, and a U-Net deep neural network (ResNet-34 encoder). The pipeline trains both
  and keeps whichever maps blue carbon habitats more accurately.
- **Carbon from real soil measurements.** Soil carbon comes from 2,515 measured soil cores (133 studies, 31 countries) in the
  Smithsonian Coastal Carbon Library: each site uses the cores of each habitat within 100–300 km, and falls back
  to IPCC Tier 1 values only where none exist. Total-carbon measurements that count limestone carbonate are
  filtered out. Areas are bias-corrected (Olofsson et al., 2014) and carbon is estimated with 5,000-draw Monte
  Carlo sampling.
- **Production engineering.** Installable Python package, typed configuration, self-describing model
  files, an automated test suite that runs the full pipeline offline, and GitHub Actions CI.

<!-- results:start -->
## Results

Scores are from areas the model **never saw during training**: held-out 5 km blocks and three entire estuaries. The selected model is the **U-Net (resnet34)**, which beat the alternative on the blue carbon habitats (◆).

| Habitat | IoU | F1 | Test pixels |
|---|---:|---:|---:|
| Open water | 0.86 | 0.93 | 22,217,159 |
| Mangrove ◆ | 0.96 | 0.98 | 3,978,161 |
| Salt marsh ◆ | 0.88 | 0.94 | 1,308,344 |
| Seagrass ◆ | 0.46 | 0.63 | 3,334,596 |
| Tidal flat / bare | 0.74 | 0.85 | 2,801,983 |
| Other land | 0.93 | 0.97 | 15,203,443 |
| Freshwater wetland | 0.55 | 0.71 | 882,771 |
| **Mean** | **0.77** | **0.86** | |

IoU (intersection over union) measures how well the predicted map overlaps the reference map, where 1.0 is a perfect match. Overall pixel accuracy (92%) is not the headline metric, because it is dominated by open water.

![Held-out sites: satellite image (top) and BlueCarbon-AI habitat map (bottom)](docs/img/hero.png)

The live app includes 78 mapped sites, including a 2018 → 2024 change analysis of Mission Bay.
<!-- results:end -->

## How it works

```
Sentinel-2 imagery ──► cloud masking + seasonal composite ──► 10 bands + 9 spectral indices
                                                                        │
ESA WorldCover · tidal-flat maps · Allen Coral Atlas ──► fused labels ──┤
                                                                        ▼
                                             LightGBM spectral model  /  U-Net  (best one kept)
                                                                        │
                                   habitat map ──► error-corrected areas ──► carbon stored & absorbed per year
```

The full write-up, with data sources, model details, the carbon method and limitations, is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Tech stack

**Python** · PyTorch · segmentation-models-pytorch · LightGBM · Google Earth Engine API · Rasterio / GDAL ·
NumPy / SciPy · Streamlit · Folium / Leaflet · Pydantic · Typer · pytest · GitHub Actions

## Run it locally

```bash
git clone https://github.com/yanicksanchez14-creator/bluecarbon-ai.git && cd bluecarbon-ai
pip install -e ".[gee,app,dev]"

streamlit run app/streamlit_app.py      # the web app, using the bundled demo sites
pytest -q                               # offline end-to-end tests
```

The full pipeline (download, label, train, map) runs with `bluecarbon fetch`, `chips`, `train` and `scene`.
It needs a Google Earth Engine project, and there's a ready-made GPU notebook in
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Project structure

```
src/bluecarbon/
  gee.py         Earth Engine imagery, fused reference labels, tiled download
  features.py    spectral bands and indices, normalization
  spectral.py    LightGBM spectral-context model
  model.py       U-Net models and checkpoints
  train.py       deep-learning training loop
  predict.py     whole-scene inference
  tiling.py      leakage-free spatial train / validation / test split
  metrics.py     accuracy metrics and error-corrected area estimation
  carbon.py      carbon stock and sequestration with Monte Carlo uncertainty
  cli.py         `bluecarbon` command-line tool
app/             Streamlit web app
configs/         pipeline settings and the 76 study sites
tests/           automated tests
docs/            methodology and figures
```

## Limitations

Carbon figures use measured soil cores near each site where available (IPCC global averages elsewhere) and are suited to screening and prioritizing sites,
not to issuing carbon credits, which requires field measurements. Seagrass is the hardest habitat to see
from space because it grows underwater, and tides change what is visible in intertidal areas.

## Data credits

Sentinel-2 (ESA Copernicus) · Cloud Score+ (Google) · ESA WorldCover 2021 · GWL_FCS30 (Zhang et al.)
· Murray et al. global tidal flats · FWC Florida seagrass · Seamap Australia (Moreton Bay seagrass) · Allen Coral Atlas · NASADEM · Smithsonian Coastal Carbon Library (soil cores) · IPCC 2013 Wetlands Supplement.

---

**Yanick Sanchez** · Independent project, 2025–2026 · [MIT License](LICENSE)
