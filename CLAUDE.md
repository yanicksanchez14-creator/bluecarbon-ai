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
Order of work (status locked in Oct 12 2026; the Doc's "Status" table is the detailed version):
1. DONE: run 642934e published; kelp v1 published; area-correction cap; accuracy page + model card; kelp explorer.
2. RUNNING: main run (95 sites, labels v11, L-band, Stumpf depth, seafloor-visible seagrass, 7 held-out
   estuaries) and the kelp run (11 beds Baja-Washington, giant/bull kelp zones, every bed scored leave-one-out,
   Landsat 1984-2015, sea temperature). On results: install only on PUBLISH, then docs/NEXT_RUN_CHECKLIST.md.
3. NOW (no Colab): source label on every number; Sanderman 2018 mangrove soil carbon as 2nd soil source;
   Global Mangrove Watch change check vs our Landsat history; live-tool sources CSV; page weight + mobile checks.
4. NEXT RUN: years 2019 + 2023 and wet/dry seasons; hard-example mining; U-Net + LightGBM ensemble (keep only
   if held-out improves); tide-relative elevation; Plum Island false seagrass; Mission Bay edges.
   DONE Oct 12: source labels, GMW change check, live-tool sources CSV, page weight/mobile. Then the owner said:
   no new code until this run's results are in.
5. LATER: Sanderman 2018 mangrove soil carbon (not in Earth Engine: the owner downloads the 30 m map from Harvard
   Dataverse, ~1 GB, then wire it in as a 2nd soil source with a disagreement flag); kelp Tasmania/Chile/South Africa; Puget Sound eelgrass (needs a better survey); Netherlands and China
   official marsh maps. Paid upgrades only with the owner's decision.
Known issues to fix: Plum Island false seagrass, Mission Bay salt marsh drop, murky-water seagrass.
Every run: no-paid-services test, compare before publishing, held-out table, 200-point spot-check per test site.
