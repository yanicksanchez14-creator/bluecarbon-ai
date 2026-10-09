"""Guard: the code must only use Earth Engine from Google Cloud (owner's rule; other services would be billed)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    r"google\.cloud", r"from google import cloud", r"gs://", r"toCloudStorage", r"storage\.googleapis\.com",
    r"bigquery", r"compute\.googleapis", r"aiplatform", r"vertexai", r"maps\.googleapis",
    r"googleapiclient\.discovery",
]


def test_no_paid_google_cloud_services():
    hits = []
    for folder in ("src", "scripts", "app", "notebooks"):
        for p in (ROOT / folder).rglob("*"):
            if p.suffix not in {".py", ".ipynb", ".yaml", ".yml", ".toml", ".txt"} or "__pycache__" in p.parts:
                continue
            text = p.read_text(errors="ignore")
            hits += [f"{p.relative_to(ROOT)}: {pat}" for pat in FORBIDDEN if re.search(pat, text, re.I)]
    for f in ("pyproject.toml", "requirements.txt"):
        if (ROOT / f).exists():
            text = (ROOT / f).read_text()
            hits += [f"{f}: {pat}" for pat in FORBIDDEN if re.search(pat, text, re.I)]
    assert not hits, "paid Google Cloud service referenced (see CLAUDE.md):\n" + "\n".join(hits)
