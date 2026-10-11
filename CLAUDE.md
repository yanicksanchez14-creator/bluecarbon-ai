# Project rules for BlueCarbon-AI

## No paid cloud services (owner's rule, Oct 9 2026)

The Google Cloud project `bluecarbon-ai` may have a billing account attached (for the Earth Engine
noncommercial Contributor tier). Earth Engine noncommercial use is free; every OTHER Google Cloud
service would be billed. Therefore:

- Code may call Earth Engine only (the `earthengine-api` package: computePixels, getInfo, etc.).
- Never use or add: Cloud Storage (`google.cloud.storage`, `gs://`, `Export.*.toCloudStorage`),
  BigQuery, Compute Engine, Cloud Run/Functions, Vertex AI, Maps APIs, or any `google.cloud.*` client.
- Never enable a Google Cloud API on the project.
- Files go to Google Drive (Colab mount) or GitHub, never a Cloud bucket.
- `tests/test_no_paid_services.py` enforces this; do not weaken it.
- If a feature seems to need another service, stop and ask the owner first.

## Product plan (owner approved, Oct 10 2026) - follow it, don't drop steps

Full plan: Claude Doc "BlueCarbon-AI Product Plan (free tier)",
https://claude.ai/code/artifact/28c6da6b-873b-458b-91be-2b4b2e8475ef (read it at the start of planning work).
Order of work:
1. Colab run 642934e (check_surveys.py first, then retrain.py), then the kelp run (results_kelp.zip).
2. Ingest + publish only on a PUBLISH verdict; publish the Kelp page with held-out error.
3. Per-site Olofsson fix (seagrass inflation) + Monte Carlo carbon ranges.
4. Competitor-matching inputs: GEDI L2A/L4A biomass points, Lang 10 m + Meta 1 m canopy height, ALOS PALSAR,
   Sentinel-2 + ICESat-2 bathymetry and water-column correction, seafloor-visible seagrass labels; then next run.
5. Next site batch (WA DNR, NJ/NH eelgrass, Baltic, Posidonia, Wales/Scotland/Ireland marsh, Fundy, China,
   Global Tidal Marsh, more GMW mangrove coasts; more held-out estuaries); then next run.
6. Kelp: Oregon/Washington/Baja, Landsat 1984-2016 history, SST context.
7. Website: model card + accuracy page, version/date on pages, per-number source labels, kelp page.
Known issues to fix: Plum Island false seagrass, Mission Bay salt marsh drop, murky-water seagrass.
Every run: no-paid-services test, compare before publishing, held-out table, 200-point spot-check per test site.
