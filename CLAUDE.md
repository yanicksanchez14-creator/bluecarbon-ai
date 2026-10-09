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
