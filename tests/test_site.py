"""The public website: pages reference files that exist, and a site record builds from demo data."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_build():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pages_reference_existing_assets():
    for page in ("index.html", "explore.html", "methodology.html", "analyze.html"):
        html = (ROOT / "site" / page).read_text()
        for ref in re.findall(r'(?:src|href)="((?:css|js)/[^"]+)"', html):
            assert (ROOT / "site" / ref).exists(), f"{page} -> {ref}"
    assert "<!-- METHODOLOGY -->" in (ROOT / "site" / "methodology.html").read_text()
    for page in ("index.html", "explore.html", "methodology.html", "analyze.html"):  # one shared header and footer
        html = (ROOT / "site" / page).read_text()
        assert "<!-- HEADER -->" in html and "<header" not in html
        assert page == "explore.html" or "<!-- FOOTER -->" in html  # the full-screen map has no footer


def test_site_record(tmp_path):
    pytest.importorskip("PIL")
    page = ROOT / "demo_data" / "mission_bay_change"
    if not (page / "meta.json").exists():
        pytest.skip("demo data not present")
    rec, model = _load_build().site_record(page, tmp_path)
    assert rec["kind"] == "change" and rec["change"]["t0"]
    for path in rec["images"].values():
        assert (tmp_path.parent / tmp_path.name / Path(path).relative_to("data")).exists()
    assert "iou" in (model.get("test") or {})
