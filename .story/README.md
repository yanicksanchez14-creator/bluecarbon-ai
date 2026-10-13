# BlueCarbon-AI: the full story

This is the project's internal history: where it started, every training run in Google Colab, what each one
taught us, and every major change, in order. It is not linked from the website or the main README. Keep it up to
date after every run (see "How to keep this file going" at the end).

Owner: Yanick Sanchez. Built with Claude (AI-assisted development) from September 29, 2026.

---

## 0. Where it started: last year's version (2025)

The first BlueCarbon-AI was built in 2025: a Streamlit app using Google Earth Engine, Sentinel-2 imagery and QGIS
hand-drawn habitat labels of Mission Bay, San Diego, with a custom segmentation model. It reported **94.23%
validation accuracy**.

When it was rechecked in September 2026, that number did not hold up:

- **The test data overlapped the training data.** Tiles from the same place were used both to train and to test,
  so the model was graded on answers it had already seen. The test set was also very small.
- **One bay only.** It had only ever seen Mission Bay, so nothing said it would work on any other coast.
- **Accuracy is the wrong score here.** Most pixels are water or land, so a model can score over 90% while missing
  most of the mangrove, salt marsh and seagrass, which are the only habitats that matter for blue carbon.
- **Fragile files.** The model's normalisation statistics lived in a separate file, so the model could silently
  run with the wrong ones.

Decision: rebuild from scratch, test only on places the model never saw, and report the honest number even when
it is worse.

## 1. The rebuild (Sep 29, 2026): "v2"

First commit `0a8ae7c`, "BlueCarbon-AI v2: ground-up rebuild":

- Earth Engine pipeline: Sentinel-2 images and reference labels fused from published global maps.
- **Leakage-free spatial split**: tiles are split by area, not at random, and whole estuaries are held out.
- U-Net segmentation model (ResNet-34 encoder) with self-describing checkpoints (weights, inputs, normalisation
  and class list in one file, fixing the v1 stats-file failure).
- Scores as **IoU per habitat** (overlap with the reference map, 1.0 = perfect), not overall accuracy.
- Error-adjusted areas (Olofsson et al.) and IPCC Tier 1 carbon with Monte Carlo uncertainty.
- Streamlit app, Colab training notebook, tests and continuous integration on GitHub.
- The old single-bay pilot model was removed once a multi-site model existed.

## 2. Every Colab training run, in order

IoU = overlap with the reference map on held-out areas (1.0 is perfect). "Held-out" = estuaries never used in
training.

| # | Date | Code | What was new | Result | Published? |
|---|---|---|---|---|---|
| 1 | Sep 29 | `a5d7786` | U-Net on 11 coastal sites; mangrove latitude rule | held-out mangrove 0.90, mIoU 0.61; salt marsh 0.13, seagrass 0.00 | Yes, 12 site pages |
| 2 | Sep 29 | `2fc0faa` | LightGBM "spectral" model with water-column ratios and 30/90 m context; `train --kind both` keeps the better model | salt marsh 0.13 -> 0.72, seagrass 0.00 -> 0.52 (on its validation) | Experiment |
| 3 | Sep 29-30 | `226e044` | Elevation, tidal-wetland probability and latitude as inputs; 6 Allen Coral Atlas seagrass sites; rare-habitat oversampling | Pooled score looked good, but the model **used latitude as a shortcut** ("no mangroves north of ~25 degrees") and missed Tampa Bay and Moreton Bay mangroves | **No**. Led to the per-estuary check (`4050701`) |
| 4 | Sep 30 | `acd6fbe` | Latitude removed as input; clear-water image for seagrass; GWL_FCS30 salt marsh labels; 19 training sites | held-out mangrove 0.94, salt marsh 0.81, seagrass 0.45 (was 0.90 / 0.35 / 0.00); Tampa and Moreton mangrove 0.86 | Yes, 23 pages |
| 5 | Sep 30 | `b976900` | Official seagrass surveys (FWC Florida, Moreton Bay 2015) as answer keys | unseen Tampa Bay seagrass 0.89; but murky water over-called as seagrass | Yes, 25 pages |
| 6 | Oct 1-8 | `e4dd0cd` -> `084cd02` | 44 sites on six continents, 5 held-out estuaries; site data kept in Google Drive so a disconnect is resumable; Earth Engine timeouts + retries; 2,515 measured soil cores (Smithsonian Coastal Carbon Library) | water mistaken for seagrass 10.8% -> 3.3%; mIoU 0.73 -> 0.75; salt marsh 0.84 (Plum Island 0.93); mangrove 0.93; seagrass conservative 0.37 | Yes, 45 pages |
| 7 | Oct 9-10 | `300f3cc` | Twice the training data (2020 + 2022 images), 80 epochs, Seamap labels (v8), seagrass bias tuning; GEBCO depth dropped (read ~18 m in 1-3 m Florida Bay) | Tampa mangrove 0.91, Plum Island salt marsh 0.91, Tampa seagrass 0.62, Moreton seagrass 0.12 (previously ~0) | Yes. First like-for-like **compare** tool added after it (`5c3e99d`) |
| 8 | Oct 10-11 | `642934e` -> `1acff23` | **71 training sites** with official survey answer keys (NWI, Environment Agency, NSW, VIMS, Texas, MassDEP, California); labels v10; resumable U-Net training | held-out mean blue carbon IoU **0.50 vs 0.43** live; verdict **PUBLISH**; pooled test mIoU 0.77 (mangrove 0.96, salt marsh 0.88, seagrass 0.46) | **Yes. This is the model live today** (76 site pages) |
| K1 | Oct 11-12 | `6de5969` -> `8ec6bdb` | Kelp canopy (beta), California, Sentinel-2 Floating Algae Index, calibrated against KelpWatch | First version over-estimated by ~32 ha per bed; fixed with a surf-zone mask, patchy-quarter filter and site-total fit: held-out r2 **0.75**, corr 0.90, +/-13 ha | Yes, 7 beds |
| K2 | Oct 12 | `e485dab` -> `9723b23` | Kelp expansion: 11 beds Baja to Washington, giant vs bull kelp zones, Landsat record 1984-2015, sea temperature | The new joint calibration scored **worse** on held-out beds (r2 0.41 vs 0.75) | Partly: new data published, old proven calibration kept, new beds marked preliminary |
| 9 | Oct 12-13 | `cd6d006` | 95 sites, labels v11, ALOS L-band radar, Stumpf water depth, seagrass learned only where the seafloor is visible, Meta 1 m tree height, 7 held-out estuaries, 200-point spot-check | held-out mean **0.59 -> 0.54**; Moreton mangrove mapped 2,033 ha vs 1,035; compare crashed | **No (REVIEW)**. See below |
| 10 | Oct 13 | `82d0263` | Fixes from run 9 + **five competing models** (U-Net, LightGBM, ConvNeXt U-Net, SegFormer, NASA/IBM Prithvi) + blend of the best deep model with LightGBM | (running) | (pending) |

### Run 9 in detail: why it got worse

- The 19 new sites (including held-out Brisbane Water and Ribble) were downloaded **before** their imagery existed,
  so they never got Sentinel-1 or L-band radar. The safety guard then dropped those inputs for every site, so the
  model silently trained **without radar at all**.
- The new "seafloor visible" rule skipped up to 90% of some sites' seagrass labels, too aggressive.
- The comparison crashed on the sites missing radar.
- Fixed in `cf3ea39`: fetch order, missing-layer re-fetch, comparison skips a site a model can't run, seafloor
  rule capped at 25% of a tile, Landsat history empty-year bug.
- It also showed the two models win in different places (LightGBM 0.81 vs U-Net 0.07 on Mission Bay seagrass;
  U-Net 0.78 vs 0.30 on Tampa Bay seagrass), which led to the blend and the five-model comparison in run 10.

## 3. Lessons that became permanent rules

1. **Check every held-out estuary separately.** A good average hid the latitude shortcut (run 3). Every run now
   prints a per-estuary table and area warnings (`held_out_check.txt`).
2. **Compare like for like before publishing.** The live and new models are graded on the same estuaries and
   the same current labels; PUBLISH only if the mean blue carbon IoU is at least the live one's and no habitat at
   any estuary drops by more than 0.05.
3. **Never let a run silently lose an input.** Preflight checks every dataset and a real download before a long
   run; retrain refuses to start if preflight failed.
4. **Earth Engine only, no paid Google Cloud services** (owner's rule, Oct 9). Enforced by
   `tests/test_no_paid_services.py`.
5. **Show "-" instead of a score for habitats too small to score** (under 25 ha of reference).
6. **Conservative carbon:** blue carbon areas are capped at mapped x precision / recall, so carbon is never
   counted on habitat the map didn't find (Laguna seagrass 5,969 -> 2,502 ha).
7. **Every number says where it came from** (measured, mapped, IPCC default, computed, illustrative).

## 4. Everything else that was built (by date)

- **Sep 29**: app redesign for non-experts, false-colour view, freshwater class, demo pages for every site.
- **Sep 30**: seagrass survey labels fetched automatically; the notebook always runs the latest code and records
  its version in the results.
- **Oct 4**: measured soil carbon from 2,515 cores instead of IPCC defaults where available.
- **Oct 9**: Mexico site screen (Laguna de Terminos 2019 -> 2025 change, credits calculator, English/Spanish PDF).
- **Oct 10**: public website on GitHub Pages (the Streamlit app became only the live analysis tool); per-site PDF
  reports; measured mangrove biomass (NASA, Simard 2019); sea-level-rise "room to move inland" screen;
  per-pixel confidence layer; Sentinel-1 radar input; habitat history 1985 to today from Landsat; seagrass
  specialist second opinion; "error on every number" table; approved product plan pinned in CLAUDE.md.
- **Oct 11**: Accuracy page (model card, held-out scores, comparison, mapped vs reference area, known
  weaknesses); live tool fixes (map projection on Streamlit Cloud, boxes across the date line); GEDI lidar and
  canopy height as an independent biomass check; area-correction cap.
- **Oct 12**: kelp page rebuilt as an explorer (quarterly timeline from 1984, satellite pictures, KelpWatch
  comparison, highlight mode); source label on every number; Global Mangrove Watch change check; live-tool
  sources CSV; mobile and page-weight checks; live tool opens on Rookery Bay, Florida; confidence colours changed
  to magenta/violet so doubt can't be mistaken for murky water.
- **Oct 13**: run 9 diagnosis and fixes; U-Net + LightGBM blend; three more models (ConvNeXt U-Net, SegFormer,
  Prithvi) with a validation leaderboard.

## 5. Where things stand (Oct 13, 2026)

- **Live model:** run 8 (71 training + 5 held-out sites, labels v10), installed Oct 11.
- **Live kelp:** run K1 calibration (California), with the K2 expansion data marked preliminary.
- **Running:** run 10 (five models, 95 sites, 7 held-out estuaries).
- **Known weak spots:** seagrass in murky or deep water; Plum Island false seagrass; Mission Bay salt marsh edges.
- **Later (product plan):** Sanderman 2018 mangrove soil carbon (owner downloads the map); 2019 + 2023 imagery
  and wet/dry seasons; hard-example mining; tide-relative elevation; kelp in Tasmania, Chile and South Africa;
  Puget Sound eelgrass; Netherlands and China marsh maps. Paid upgrades only with the owner's decision.

## How to keep this file going

After every Colab run: add a row to the table in section 2 (date, code version from `code_version.txt`, what was
new, the held-out result from `compare.txt`, published or not and why), and add anything learned to section 3.
