# Run of Oct 12 2026 (code 74e5ea6, labels v11): what to update after it

The run's `results.zip` (also in My Drive / bluecarbon-ai-results) contains everything below. Install only on a
PUBLISH verdict in `compare.txt`.

| New in this run | Where it shows up in results | Website update after ingest |
|---|---|---|
| ALOS PALSAR-2 L-band radar input | model's feature list (`metrics.json`, model card) | Accuracy page model card: add L-band to "Architecture" inputs; methodology inputs list |
| Stumpf relative-depth input (PSDB_CLEAR) | model's feature list | same as above |
| Seagrass learned only where the seafloor is visible | `seafloor_mask.json` (share skipped per training site) | Accuracy page "Limits": replace murky-water text with the rule and the per-site share |
| 19 new sites (95 total) | `demo_data/<site>/`, `label_summary.json` | Site maps: new pages appear via ingest; check titles/regions; home page site count and map |
| 7 new answer keys (NJ, NH, EMODnet Europe, Torres Strait, Wales, Scotland, Ireland) | label sources in each `meta.json` | Methodology "Training data" section: list them with citations; per-site source line |
| 2 new held-out estuaries (Brisbane Water, Ribble) | `held_out_check.txt`, `compare.txt`, `metrics.json` test_per_site | Accuracy page: 7 estuaries in tables, score cards, summary text; "five estuaries" wording everywhere -> seven |
| Meta 1 m canopy height + GEDI height-to-biomass fit | `site_biomass.json` (meta_h_mean, hb_*, height_agb_mean) | Site pages already render it (explore.js lidar note); check it reads well; error table row for biomass |
| 200-point spot-check per held-out site | `spotcheck.csv` | Do the by-eye check (fill `by_eye`), then publish agreement % on the Accuracy page |
| U-Net + LightGBM blend (on only if it beats both on validation) | `ensemble_metrics.json`, blend line in `held_out_check.txt`, extra columns in `compare.txt`; `best.txt` says `ensemble` if it won | Accuracy "Architecture" row switches automatically (model_report `model_kind`); methodology already describes it |
| 3 more networks: ConvNeXt U-Net, SegFormer, Prithvi | `scores.json` + leaderboard in `held_out_check.txt`; `<name>_metrics.json`; columns in `compare.txt` | Architecture row names the winner automatically. If Prithvi wins, its file is ~180 MB (over GitHub's 100 MB limit): attach it to a GitHub release and set MODEL_URL |
| Model version / date | `code_version.txt`, model card | Footer version and Accuracy "This model" row (automatic from model_report.json) |

Also after ingest: refresh_carbon (area cap), rebuild site, Lighthouse check, and update the product plan doc
("Where we are now").
