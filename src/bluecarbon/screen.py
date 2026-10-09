"""Site screen: a client-facing PDF for one area (habitat, carbon, credit range, change, rights flags).

    python scripts/site_screen.py configs/screens/laguna_terminos.yaml --lang en

Everything here is a screening estimate. The credit numbers follow the business plan's formula
(credits = area x rate x (1 - deductions)) with illustrative deductions; real numbers come from the
chosen methodology's tools and site measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schema import BLUE_CARBON_KEYS, CLASSES

CLS = {c.key: c for c in CLASSES}


@dataclass
class CreditAssumptions:
    buffer: float = 0.15        # non-permanence risk buffer (registry risk tool; often 10-20%)
    leakage: float = 0.10       # activity displaced elsewhere
    uncertainty: float = 0.10   # deduction for measurement uncertainty
    years: int = 30
    prices_usd: tuple[float, float, float] = (20.0, 36.5, 50.0)
    extra: dict = field(default_factory=dict)

    @property
    def keep(self) -> float:
        return (1 - self.buffer) * (1 - self.leakage) * (1 - self.uncertainty)


def area_range(report: dict, key: str, f1: float | None) -> tuple[float, float, float]:
    """(low, best, high) hectares. Best = accuracy-corrected area; the range is +/- (1 - F1 on unseen
    test areas), a plain-language stand-in for map error (the statistical interval that treats pixels
    as independent is far too narrow)."""
    adj = (report.get("error_adjusted_areas_ha") or {}).get(key)
    best = adj["adjusted_ha"] if adj else report["areas_ha"].get(key, 0.0)
    e = 1 - f1 if f1 is not None else 0.2
    return best * (1 - e), best, best * (1 + e)


def credits_estimate(report: dict, f1: dict, a: CreditAssumptions, change: dict | None = None,
                     habitats: tuple[str, ...] = BLUE_CARBON_KEYS) -> dict:
    """Two lenses, each low / mid / high:

    - sequestration: yearly carbon burial of the habitat that exists (restoration / protection of growth)
    - avoided loss: only if the change study shows a net loss; yearly loss rate x carbon stock per hectare
      (what a conservation project would claim if protection stops the loss)
    """
    cls = report["carbon"]["classes"]
    seq = {"low": 0.0, "mid": 0.0, "high": 0.0}
    for k in habitats:
        if k not in cls:
            continue
        s = cls[k]["sequestration_tCO2e_per_yr"]
        lo, best, hi = area_range(report, k, f1.get(k))
        scale = (lo / best, 1.0, hi / best) if best else (1, 1, 1)
        seq["low"] += s["p05"] * scale[0]
        seq["mid"] += s["mean"]
        seq["high"] += s["p95"] * scale[2]
    out = {"assumptions": a, "keep": a.keep,
           "sequestration": {k: v * a.keep for k, v in seq.items()}, "avoided_loss": None}
    if change:
        years = max(1.0, change.get("years", 1.0))
        loss = {}
        for k in habitats:
            d = change["delta_ha"].get(k, 0.0)
            if d < 0 and k in cls and cls[k]["area_ha"] > 0:
                per_ha = {q: cls[k]["stock_tCO2e"][q] / cls[k]["area_ha"] for q in ("p05", "mean", "p95")}
                loss[k] = {"ha_per_yr": -d / years, "per_ha": per_ha}
        if loss:
            av = {"low": 0.0, "mid": 0.0, "high": 0.0}
            for v in loss.values():
                av["low"] += v["ha_per_yr"] * v["per_ha"]["p05"] * 0.5   # assume protection halves the loss
                av["mid"] += v["ha_per_yr"] * v["per_ha"]["mean"] * 0.75
                av["high"] += v["ha_per_yr"] * v["per_ha"]["p95"]
            out["avoided_loss"] = {"per_habitat": loss, **{k: v * a.keep for k, v in av.items()}}
    lenses = [out["sequestration"]] + ([out["avoided_loss"]] if out["avoided_loss"] else [])
    best = max(lenses, key=lambda d: d["mid"])
    lo, mid, hi = a.prices_usd
    out["value_usd_per_yr"] = {"low": best["low"] * lo, "mid": best["mid"] * mid, "high": best["high"] * hi}
    out["credits_30yr"] = {k: best[k] * a.years for k in ("low", "mid", "high")}
    return out


def verdict(flags: list[dict], area_ha: float, min_area_ha: float = 500) -> tuple[str, list[str]]:
    reasons = []
    risks = [f for f in flags if f.get("severity") == "risk"]
    if area_ha < min_area_ha:
        return "unlikely", [f"less than {min_area_ha:,.0f} ha of the focus habitat"]
    if risks:
        reasons = [f["key"] for f in risks]
        return "needs_investigation", reasons
    return "promising", reasons


# --------------------------------------------------------------------------- text (English / Spanish)
T = {
    "en": dict(
        title="Blue carbon site screen", prepared="Prepared", model="Habitat model",
        verdict_lbl="Screening verdict",
        verdicts={"promising": "Promising", "needs_investigation": "Needs investigation",
                  "unlikely": "Unlikely to qualify"},
        why="Why", maps="Satellite image and habitat map", sat="Satellite image (Sentinel-2)", hab="Habitat map",
        habitats="Habitats found", hab_col="Habitat", area_col="Area (ha)", range_col="Likely range (ha)",
        carbon="Carbon", stock_col="Stored (t CO2e)", seq_col="Absorbed per year (t CO2e)", soil_col="Soil carbon source",
        soil_measured="measured cores ({n} within {r} km, {s} studies)", soil_ipcc="IPCC global default",
        credits="Credit potential (illustrative)", lens_col="Basis", low="Low", mid="Mid", high="High",
        seq_lens="Sequestration of existing habitat (credits per year)",
        loss_lens="Avoided loss, if protection stops the observed loss (credits per year)",
        thirty="Over {y} years (best basis)", value="Value per year at ${lo:g} / ${mid:g} / ${hi:g} per credit",
        deductions="Deductions applied: risk buffer {b:.0%}, leakage {lk:.0%}, uncertainty {u:.0%} (total {t:.0%}).",
        formula="Credits = area x carbon rate x (1 - deductions), as in the plan; real deductions come from the "
                "registry's tools and the methodology.",
        change="Change over time", change_none="Change study not yet available for this site.",
        change_col="Change", per_yr="per year",
        flags="Rights and status screen", flag_col="Check", status_col="Status", note_col="Finding",
        sev={"ok": "OK", "caution": "Caution", "risk": "Risk"},
        sources="Sources", methods="Methods and limits",
        methods_text=[
            "Habitats are mapped from free Sentinel-2 imagery with a deep-learning model (BlueCarbon-AI), "
            "trained on 39 coastal sites and scored on areas it never saw.",
            "Map accuracy on unseen areas for this habitat: F1 {f1}. Areas are corrected for the model's "
            "known errors; the range shown is a plain-language margin, not a statistical interval.",
            "Carbon: soil carbon from measured cores where available (Smithsonian Coastal Carbon Library), "
            "otherwise IPCC 2013 Wetlands Supplement Tier 1 defaults; biomass and burial rates IPCC Tier 1.",
            "Seagrass is mapped conservatively and is a lower bound; murky or deep meadows are often missed.",
        ],
        disclaimer="Screening and estimates only. Not legal, financial or investment advice. Only a registry "
                   "issues credits, after independent validation and verification. Confirm land tenure and "
                   "carbon rights with a local lawyer.",
        ha="ha"),
    "es": dict(
        title="Evaluación preliminar de sitio de carbono azul", prepared="Preparado", model="Modelo de hábitat",
        verdict_lbl="Resultado preliminar",
        verdicts={"promising": "Prometedor", "needs_investigation": "Requiere investigación",
                  "unlikely": "Poco probable que califique"},
        why="Por qué", maps="Imagen satelital y mapa de hábitats", sat="Imagen satelital (Sentinel-2)",
        hab="Mapa de hábitats", habitats="Hábitats encontrados", hab_col="Hábitat", area_col="Área (ha)",
        range_col="Rango probable (ha)", carbon="Carbono", stock_col="Almacenado (t CO2e)",
        seq_col="Absorbido por año (t CO2e)", soil_col="Fuente del carbono en suelo",
        soil_measured="núcleos medidos ({n} a menos de {r} km, {s} estudios)", soil_ipcc="valor global del IPCC",
        credits="Potencial de créditos (ilustrativo)", lens_col="Base", low="Bajo", mid="Medio", high="Alto",
        seq_lens="Secuestro del hábitat existente (créditos por año)",
        loss_lens="Pérdida evitada, si la protección detiene la pérdida observada (créditos por año)",
        thirty="En {y} años (mejor base)", value="Valor por año a ${lo:g} / ${mid:g} / ${hi:g} por crédito",
        deductions="Deducciones aplicadas: reserva de riesgo {b:.0%}, fugas {lk:.0%}, incertidumbre {u:.0%} (total {t:.0%}).",
        formula="Créditos = área x tasa de carbono x (1 - deducciones), como en el plan; las deducciones reales "
                "salen de las herramientas del registro y de la metodología.",
        change="Cambio en el tiempo", change_none="El estudio de cambio aún no está disponible para este sitio.",
        change_col="Cambio", per_yr="por año",
        flags="Revisión de derechos y estatus", flag_col="Revisión", status_col="Estatus", note_col="Hallazgo",
        sev={"ok": "OK", "caution": "Precaución", "risk": "Riesgo"},
        sources="Fuentes", methods="Métodos y límites",
        methods_text=[
            "Los hábitats se mapean con imágenes gratuitas de Sentinel-2 y un modelo de aprendizaje profundo "
            "(BlueCarbon-AI), entrenado en 39 sitios costeros y evaluado en zonas que nunca vio.",
            "Precisión del mapa en zonas no vistas para este hábitat: F1 {f1}. Las áreas se corrigen según los "
            "errores conocidos del modelo; el rango es un margen aproximado, no un intervalo estadístico.",
            "Carbono: carbono en suelo de núcleos medidos cuando existen (Smithsonian Coastal Carbon Library); "
            "si no, valores Nivel 1 del IPCC (Suplemento de Humedales 2013); biomasa y tasas de entierro IPCC Nivel 1.",
            "Los pastos marinos se mapean de forma conservadora y son un mínimo; praderas turbias o profundas "
            "suelen no detectarse.",
        ],
        disclaimer="Solo evaluación preliminar y estimaciones. No es asesoría legal, financiera ni de inversión. "
                   "Solo un registro emite créditos, tras validación y verificación independientes. Confirme la "
                   "tenencia de la tierra y los derechos de carbono con un abogado local.",
        ha="ha"),
}
HAB_ES = {"water": "Agua abierta", "mangrove": "Manglar", "saltmarsh": "Marisma salada", "seagrass": "Pastos marinos",
          "tidal_flat": "Planicie de marea", "other_land": "Otra tierra", "freshwater": "Humedal de agua dulce"}


def hab_name(k: str, lang: str) -> str:
    return HAB_ES[k] if lang == "es" else CLS[k].name


def fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:,.1f}"


def load_change(page_dir: Path | None) -> dict | None:
    if page_dir is None or not (page_dir / "meta.json").exists():
        return None
    m = json.loads((page_dir / "meta.json").read_text())
    ch = m.get("change") or {}
    inner = ch.get("change") if "change" in ch else ch
    if not inner or "delta_ha" not in inner:
        return None
    import re

    labels = m.get("periods") or {}
    yrs = [re.findall(r"(19|20)\d\d", str(labels.get(k, ""))) and int(re.search(r"(?:19|20)\d\d", str(labels[k]))[0])
           for k in ("t0", "t1")]
    years = float(yrs[1] - yrs[0]) if all(yrs) else 1.0
    return {"delta_ha": inner["delta_ha"], "areas_t0": ch.get("areas_t0_ha", {}), "areas_t1": ch.get("areas_t1_ha", {}),
            "labels": labels, "years": years or 1.0, "dir": page_dir}


def build_pdf(spec_path: Path, out_path: Path, lang: str = "en", demo_root: Path = Path("demo_data")) -> Path:
    import yaml
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    t = T[lang]
    spec = yaml.safe_load(Path(spec_path).read_text())
    page = demo_root / spec["site"]
    meta = json.loads((page / "meta.json").read_text())
    rep = meta["report"]
    model = meta.get("model") or {}
    f1 = ((model.get("test") or {}).get("f1")) or {}
    focus = spec.get("focus", "mangrove")
    change = load_change(demo_root / spec["change"]) if spec.get("change") else None
    a = CreditAssumptions(**{k: v for k, v in (spec.get("credits") or {}).items() if k != "extra"})
    cr = credits_estimate(rep, f1, a, change)
    flags = spec.get("flags", [])
    _, focus_area, _ = area_range(rep, focus, f1.get(focus))
    v, why = verdict(flags, focus_area)

    ss = getSampleStyleSheet()
    teal = colors.HexColor("#0f5c63")
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, textColor=teal, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=teal, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#4a5560"))
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)

    def table(rows, widths, header=True):
        rows = [[Paragraph(str(c), cell) if not isinstance(c, Paragraph) else c for c in r] for r in rows]
        tb = Table(rows, colWidths=widths, hAlign="LEFT")
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d2d8")),
              ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f2")))
        tb.setStyle(TableStyle(st))
        return tb

    story = [Paragraph(t["title"], h1),
             Paragraph(f"<b>{meta['title']}</b>, {meta.get('region', '')} &nbsp;·&nbsp; {t['prepared']} "
                       f"{spec.get('prepared', '')} &nbsp;·&nbsp; {t['model']}: {model.get('arch', '')} "
                       f"({model.get('encoder', '')})", body), Spacer(1, 6)]
    sev_color = {"promising": "#2f7d32", "needs_investigation": "#b26a00", "unlikely": "#b3261e"}[v]
    why_txt = "; ".join(next((f[lang] for f in flags if f["key"] == k), k).split(".")[0] for k in why) or "-"
    story.append(table([[Paragraph(f"<b>{t['verdict_lbl']}:</b> <font color='{sev_color}'><b>{t['verdicts'][v]}"
                                   f"</b></font>", body)], [Paragraph(f"<b>{t['why']}:</b> {why_txt}.", body)]],
                       [7.0 * inch], header=False))

    story.append(Paragraph(t["maps"], h2))
    imgs = []
    for f, cap in ((page / "rgb.png", t["sat"]), (page / "classes.png", t["hab"])):
        if f.exists():
            im = Image(str(f))
            r = im.imageHeight / im.imageWidth
            im.drawWidth, im.drawHeight = 3.3 * inch, 3.3 * inch * r
            imgs.append([im, Paragraph(cap, small)])
    if imgs:
        story.append(Table([[i[0] for i in imgs], [i[1] for i in imgs]], hAlign="LEFT"))
    legend = " &nbsp; ".join(f"<font color='{CLS[k].color}'>■</font> {hab_name(k, lang)}" for k in CLS)
    story.append(Paragraph(legend, small))

    story.append(Paragraph(t["habitats"], h2))
    rows = [[t["hab_col"], t["area_col"], t["range_col"]]]
    for k in BLUE_CARBON_KEYS:
        lo, best, hi = area_range(rep, k, f1.get(k))
        if best >= 1:
            rows.append([f"<b>{hab_name(k, lang)}</b>", fmt(best), f"{fmt(lo)} – {fmt(hi)}"])
    story.append(table(rows, [2.6 * inch, 1.6 * inch, 2.8 * inch]))

    story.append(Paragraph(t["carbon"], h2))
    rows = [[t["hab_col"], t["stock_col"], t["seq_col"], t["soil_col"]]]
    for k, c in rep["carbon"]["classes"].items():
        if c["area_ha"] < 1:
            continue
        sl = c.get("soil") or {}
        src = (t["soil_measured"].format(n=sl["n_cores"], r=sl["radius_km"], s=sl["n_studies"])
               if sl.get("source") == "measured" else t["soil_ipcc"])
        rows.append([hab_name(k, lang), f"{fmt(c['stock_tCO2e']['mean'])} ({fmt(c['stock_tCO2e']['p05'])}–"
                     f"{fmt(c['stock_tCO2e']['p95'])})", fmt(c["sequestration_tCO2e_per_yr"]["mean"]), src])
    story.append(table(rows, [1.4 * inch, 2.2 * inch, 1.5 * inch, 1.9 * inch]))

    story.append(Paragraph(t["credits"], h2))
    rows = [[t["lens_col"], t["low"], t["mid"], t["high"]],
            [t["seq_lens"]] + [fmt(cr["sequestration"][q]) for q in ("low", "mid", "high")]]
    if cr["avoided_loss"]:
        rows.append([t["loss_lens"]] + [fmt(cr["avoided_loss"][q]) for q in ("low", "mid", "high")])
    rows.append([t["thirty"].format(y=a.years)] + [fmt(cr["credits_30yr"][q]) for q in ("low", "mid", "high")])
    lo, mid, hi = a.prices_usd
    rows.append([t["value"].format(lo=lo, mid=mid, hi=hi)] + ["$" + fmt(cr["value_usd_per_yr"][q])
                                                             for q in ("low", "mid", "high")])
    story.append(table(rows, [3.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Paragraph(t["deductions"].format(b=a.buffer, lk=a.leakage, u=a.uncertainty, t=1 - a.keep) + " "
                           + t["formula"], small))

    story.append(Paragraph(t["change"], h2))
    if change:
        lab = change["labels"]
        rows = [[t["hab_col"], str(lab.get("t0", "")), str(lab.get("t1", "")), t["change_col"]]]
        for k in BLUE_CARBON_KEYS:
            a0, a1 = change["areas_t0"].get(k, 0.0), change["areas_t1"].get(k, 0.0)
            if max(a0, a1) < 1:
                continue
            d = a1 - a0
            pct = f" ({d / a0:+.0%})" if a0 else ""
            rows.append([hab_name(k, lang), fmt(a0), fmt(a1), f"{d:+,.0f} {t['ha']}{pct}; "
                         f"{d / change['years']:+,.0f} {t['ha']} {t['per_yr']}"])
        story.append(table(rows, [1.8 * inch, 1.4 * inch, 1.4 * inch, 2.4 * inch]))
        cd = change["dir"]
        pics = [f for f in (cd / "rgb_t1.png", cd / "change.png") if f.exists()]
        if pics:
            ims = []
            for f in pics:
                im = Image(str(f))
                r = im.imageHeight / im.imageWidth
                im.drawWidth, im.drawHeight = 3.3 * inch, 3.3 * inch * r
                ims.append(im)
            story.append(Table([ims], hAlign="LEFT"))
    else:
        story.append(Paragraph(t["change_none"], body))

    story.append(Paragraph(t["flags"], h2))
    rows = [[t["flag_col"], t["status_col"], t["note_col"]]]
    sev_c = {"ok": "#2f7d32", "caution": "#b26a00", "risk": "#b3261e"}
    for f in flags:
        note = f[lang] + (f" <font size=6.5 color='#4a5560'>[{f['source']}]</font>" if f.get("source") else "")
        rows.append([f["key"].replace("_", " ").capitalize(),
                     Paragraph(f"<font color='{sev_c[f['severity']]}'><b>{t['sev'][f['severity']]}</b></font>", cell),
                     note])
    story.append(table(rows, [1.3 * inch, 0.9 * inch, 4.8 * inch]))

    story.append(Paragraph(t["methods"], h2))
    for line in t["methods_text"]:
        story.append(Paragraph("• " + line.format(f1=f"{f1.get(focus, 0):.2f}"), small))
    story.append(Spacer(1, 6))
    story.append(table([[Paragraph(t["disclaimer"], small)]], [7.0 * inch], header=False))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=f"{t['title']}: {meta['title']}",
                            author="BlueCarbon-AI")
    doc.build(story)
    return out_path
