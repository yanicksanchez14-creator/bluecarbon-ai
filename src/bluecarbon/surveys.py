"""Regional seagrass survey maps, fetched from public web services and burned into the labels.

Global seagrass labels (Allen Coral Atlas) only cover tropical reefs, so meadows in murky or temperate
water (Florida's Gulf coast, Moreton Bay) were unlabelled. Official survey polygons fix that:

  arcgis  an ArcGIS REST feature layer, queried by bounding box (e.g. FWC "Seagrass Statewide")
  wfs     an OGC WFS layer, downloaded as GeoJSON (e.g. Seamap Australia, Moreton Bay 2015)

Polygons are burned as seagrass only over pixels that are open water or unlabelled water, never over
land, mangrove or salt marsh. A survey that cannot be reached is skipped with a message, so a network
hiccup never stops a training run.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .schema import IGNORE_INDEX, KEY_TO_ID

UA = {"User-Agent": "BlueCarbon-AI (https://github.com/yanicksanchez14-creator/bluecarbon-ai)"}


def _get_json(url: str, params: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def fetch_arcgis(layer_url: str, bbox: list[float], where: str = "1=1", page: int = 1000) -> list[dict]:
    """GeoJSON features (EPSG:4326) of an ArcGIS feature layer that intersect bbox, all pages."""
    feats, offset = [], 0
    while True:
        d = _get_json(layer_url.rstrip("/") + "/query", {
            "where": where, "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326,
            "returnGeometry": "true", "maxAllowableOffset": 0.00003, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": page,
        })
        if "error" in d:
            raise RuntimeError(str(d["error"])[:200])
        batch = d.get("features", [])
        feats += batch
        more = d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit")
        if not batch or (len(batch) < page and not more):
            return feats
        offset += len(batch)


def fetch_wfs(url: str, layer: str, bbox: list[float] | None = None) -> list[dict]:
    """GeoJSON features of a WFS layer (EPSG:4326), limited to bbox when given (WFS 1.0.0: lon/lat order)."""
    q = {"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": layer,
         "outputFormat": "application/json", "srsName": "EPSG:4326"}
    if bbox is not None:
        q["bbox"] = ",".join(map(str, bbox)) + ",EPSG:4326"
    return _get_json(url, q, timeout=300).get("features", [])


def filter_features(feats: list[dict], keep: dict | None = None, drop: dict | None = None) -> list[dict]:
    """keep / drop = {field: [substrings]} (case-insensitive). A feature is kept when every `keep` field
    contains one of its substrings and no `drop` field does."""
    def has(f, field, subs):
        v = str((f.get("properties") or {}).get(field, "")).lower()
        return any(x.lower() in v for x in subs)

    out = feats
    for field, subs in (keep or {}).items():
        out = [f for f in out if has(f, field, subs)]
    for field, subs in (drop or {}).items():
        out = [f for f in out if not has(f, field, subs)]
    return out


def seagrass_features(feats: list[dict]) -> tuple[list[dict], str]:
    """Keep seagrass polygons. If some text field names habitats, keep the rows that say seagrass
    (and not 'no seagrass' / 'absent'); otherwise the whole layer is seagrass."""
    if not feats:
        return [], "no features"
    props = [f.get("properties") or {} for f in feats]
    for key in sorted({k for p in props for k in p}):
        vals = [str(p.get(key, "")).lower() for p in props]
        if any("seagrass" in v for v in vals) and not all("seagrass" in v for v in vals):
            keep = [f for f, v in zip(feats, vals, strict=True)
                    if "seagrass" in v and not any(n in v for n in ("no seagrass", "absent", "non-seagrass"))]
            return keep, f"kept {len(keep)}/{len(feats)} features where '{key}' mentions seagrass"
    return feats, f"all {len(feats)} features treated as seagrass"


def burn_seagrass(label: np.ndarray, image: np.ndarray, transform, crs, feats: list[dict],
                  negatives_px: int = 0) -> tuple[np.ndarray, int]:
    """Set seagrass inside the polygons, over water / unlabelled water pixels only.

    negatives_px > 0: a survey maps every meadow in the area it covers, so unlabelled water within
    this distance of a surveyed meadow but outside every polygon is labelled open water. Without these
    negatives the model only ever sees murky water that IS seagrass, and paints whole bays as seagrass.
    """
    from rasterio import features as rfeatures
    from rasterio.warp import transform_geom

    from .features import compute_features

    geoms = [transform_geom("EPSG:4326", crs, f["geometry"]) for f in feats if f.get("geometry")]
    if not geoms:
        return label, 0
    inside = rfeatures.rasterize([(g, 1) for g in geoms], out_shape=label.shape, transform=transform,
                                 fill=0, dtype="uint8").astype(bool)
    mndwi = compute_features(image)[12]  # MNDWI > 0: the pixel looks like water in the image
    wet = (label == KEY_TO_ID["water"]) | ((label == IGNORE_INDEX) & (mndwi > 0))
    m = inside & wet
    out = label.copy()
    out[m] = KEY_TO_ID["seagrass"]
    if negatives_px > 0 and inside.any():
        from scipy.ndimage import distance_transform_edt

        near = distance_transform_edt(~inside) <= negatives_px
        out[near & ~inside & (label == IGNORE_INDEX) & (mndwi > 0)] = KEY_TO_ID["water"]
    return out, int(m.sum())


def apply_surveys(label_path: str | Path, image_path: str | Path, bbox: list[float], surveys: list[dict],
                  report: dict | None = None, log=print) -> int:
    """Burn every survey that overlaps bbox into label.tif. Returns the number of seagrass pixels added."""
    import rasterio

    todo = [s for s in surveys if _intersects(bbox, s.get("extent", [-180, -90, 180, 90]))]
    if not todo:
        return 0
    with rasterio.open(image_path) as src:
        image = src.read()
    total = 0
    with rasterio.open(label_path, "r+") as ds:
        lab = ds.read(1)
        for s in todo:
            try:
                if s["kind"] == "arcgis":
                    feats = fetch_arcgis(s["url"], bbox, s.get("where", "1=1"))
                else:
                    feats = [f for f in fetch_wfs(s["url"], s["layer"], bbox if s.get("bbox_query", True) else None)
                             if f.get("geometry")]
                if s.get("keep") or s.get("drop"):
                    n0 = len(feats)
                    feats = filter_features(feats, s.get("keep"), s.get("drop"))
                    how = f"kept {len(feats)}/{n0} features matching {s.get('keep') or ''} {s.get('drop') or ''}".strip()
                else:
                    feats, how = seagrass_features(feats)
                neg_px = int(s.get("negatives_within_m", 0) / abs(ds.transform.a))
                lab, n = burn_seagrass(lab, image, ds.transform, ds.crs, feats, neg_px)
                total += n
                log(f"  seagrass survey {s['name']}: {how}; {n * 0.01:,.0f} ha labelled as seagrass")
                if report is not None and n:
                    report.setdefault("label_sources", []).append(s["name"])
            except Exception as e:  # never let an unreachable survey stop the run
                log(f"  seagrass survey {s['name']} skipped ({str(e)[:120]})")
        ds.write(lab, 1)
    return total


def ignore_deep_seagrass(label_path, ancillary_path, max_depth_m: float, log=print) -> int:
    """Seagrass labels in water deeper than max_depth_m (GEBCO) -> unlabelled. A satellite cannot see
    the seafloor there, so those meadows can neither be learned nor fairly scored."""
    import rasterio

    from .features import DEPTH_BAND
    from .schema import IGNORE_INDEX, KEY_TO_ID

    with rasterio.open(ancillary_path) as a:
        if a.count <= DEPTH_BAND:
            return 0
        depth = a.read(DEPTH_BAND + 1)
    with rasterio.open(label_path, "r+") as ds:
        lab = ds.read(1)
        m = (lab == KEY_TO_ID["seagrass"]) & (depth > max_depth_m)
        lab[m] = IGNORE_INDEX
        ds.write(lab, 1)
    if m.any():
        log(f"  {m.sum() * 0.01:,.0f} ha of seagrass deeper than {max_depth_m:g} m left unlabelled (not visible)")
    return int(m.sum())
