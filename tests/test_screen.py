"""Site screen: credit arithmetic, avoided-loss lens, verdict and PDF build."""

import json

import pytest

from bluecarbon.screen import CreditAssumptions, credits_estimate, verdict

REP = {"areas_ha": {"mangrove": 1000.0},
       "carbon": {"classes": {"mangrove": {
           "area_ha": 1000.0,
           "stock_tCO2e": {"mean": 1_000_000.0, "p05": 900_000.0, "p95": 1_100_000.0},
           "sequestration_tCO2e_per_yr": {"mean": 6000.0, "p05": 5000.0, "p95": 7000.0}}}}}


def test_credit_formula():
    a = CreditAssumptions(buffer=0.15, leakage=0.10, uncertainty=0.10)
    out = credits_estimate(REP, {"mangrove": 1.0}, a, habitats=("mangrove",))
    assert abs(out["sequestration"]["mid"] - 6000 * 0.85 * 0.9 * 0.9) < 1e-6
    assert out["avoided_loss"] is None
    assert abs(out["credits_30yr"]["mid"] - out["sequestration"]["mid"] * 30) < 1e-6
    assert out["value_usd_per_yr"]["mid"] == pytest.approx(out["sequestration"]["mid"] * 36.5)


def test_avoided_loss_lens():
    change = {"delta_ha": {"mangrove": -60.0}, "years": 6.0}
    out = credits_estimate(REP, {"mangrove": 1.0}, CreditAssumptions(), change, habitats=("mangrove",))
    loss = out["avoided_loss"]
    assert loss["per_habitat"]["mangrove"]["ha_per_yr"] == 10.0
    # 10 ha/yr x 1000 tCO2e/ha x 0.75 assumed avoided x deductions
    assert loss["mid"] == pytest.approx(10 * 1000 * 0.75 * CreditAssumptions().keep)


def test_verdict():
    assert verdict([{"key": "protected_area", "severity": "risk"}], 5000)[0] == "needs_investigation"
    assert verdict([{"key": "x", "severity": "ok"}], 5000)[0] == "promising"
    assert verdict([], 10)[0] == "unlikely"


def test_pdf_builds(tmp_path):
    pytest.importorskip("reportlab")
    from bluecarbon.screen import build_pdf

    page = tmp_path / "demo" / "site"
    page.mkdir(parents=True)
    (page / "meta.json").write_text(json.dumps({"title": "Test site", "region": "Somewhere",
                                                "model": {"arch": "Unet", "encoder": "resnet34",
                                                          "test": {"f1": {"mangrove": 0.9}}},
                                                "report": REP}))
    spec = tmp_path / "s.yaml"
    spec.write_text("site: site\nfocus: mangrove\nflags:\n  - {key: tenure, severity: caution, en: Check, es: Revisar}\n")
    for lang in ("en", "es"):
        out = build_pdf(spec, tmp_path / f"o_{lang}.pdf", lang, tmp_path / "demo")
        assert out.stat().st_size > 1000
