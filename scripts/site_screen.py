"""Build the client-facing site screen PDF(s) for one area.

    python scripts/site_screen.py configs/screens/laguna_terminos.yaml            # English + Spanish
    python scripts/site_screen.py configs/screens/laguna_terminos.yaml --lang es

Writes reports/screens/<site>_<lang>.pdf from the demo page (demo_data/<site>) and the hand-researched
rights and status flags in the YAML file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bluecarbon.screen import build_pdf

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--lang", choices=["en", "es", "both"], default="both")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "screens")
    a = ap.parse_args()
    site = yaml.safe_load(a.spec.read_text())["site"]
    for lang in (["en", "es"] if a.lang == "both" else [a.lang]):
        p = build_pdf(a.spec, a.out / f"{site}_{lang}.pdf", lang, ROOT / "demo_data")
        print("wrote", p)


if __name__ == "__main__":
    main()
