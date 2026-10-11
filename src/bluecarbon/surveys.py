"""Regional survey maps (seagrass, salt marsh, mangrove), fetched from public web services and burned
into the labels.

Global seagrass labels (Allen Coral Atlas) only cover tropical reefs, so meadows in murky or temperate
water (Florida's Gulf coast, Moreton Bay) were unlabelled. Official survey polygons fix that:

  arcgis  an ArcGIS REST feature layer, queried by bounding box (e.g. FWC "Seagrass Statewide")
  wfs     an OGC WFS layer, downloaded as GeoJSON (e.g. Seamap Australia, Moreton Bay 2015)

Seagrass polygons are burned only over pixels that are open water or unlabelled water, never over
land, mangrove or salt marsh. Salt marsh and mangrove polygons (e.g. the US National Wetlands
Inventory, NSW estuarine macrophytes) are burned only over pixels that look vegetated in the image and
are not open water. With `authoritative: true`, the survey is taken as the complete map of that class
inside its extent: pixels the global maps call that class but the survey does not are left unlabelled
(the sources disagree, so the pixel is neither trained nor scored on). A survey that cannot be reached is skipped with a message, so a network
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


def _ring_area(r) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0, *_), (x1, y1, *_) in zip(r, r[1:] + r[:1]))


def esri_to_geojson(f: dict) -> dict:
    """An Esri JSON feature -> GeoJSON. Esri polygons list rings; clockwise rings are outer boundaries
    and counter-clockwise rings are holes of the outer ring before them."""
    g = f.get("geometry") or {}
    geom = None
    if "rings" in g:
        polys = []
        for r in g["rings"]:
            r = [list(p[:2]) for p in r]
            if _ring_area(r) <= 0 or not polys:  # clockwise (negative signed area) = outer ring
                polys.append([r])
            else:
                polys[-1].append(r)
        geom = {"type": "MultiPolygon", "coordinates": polys}
    elif "x" in g:
        geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
    return {"type": "Feature", "properties": f.get("attributes") or {}, "geometry": geom}


def fetch_arcgis(layer_url: str, bbox: list[float], where: str = "1=1", page: int = 500) -> list[dict]:
    """GeoJSON features (EPSG:4326) of an ArcGIS feature layer that intersect bbox, all pages.
    Layers that can't answer in GeoJSON (some older MapServers) are read as Esri JSON and converted."""
    feats, offset, fmt = [], 0, "geojson"
    while True:
        q = {"where": where, "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
             "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326,
             "returnGeometry": "true", "maxAllowableOffset": 0.00003, "f": fmt,
             "resultOffset": offset, "resultRecordCount": page}
        try:
            d = _get_json(layer_url.rstrip("/") + "/query", q)
        except (ValueError, OSError):
            if fmt == "json" or offset:
                raise
            fmt = "json"
            continue
        if "error" in d and fmt == "geojson" and not offset:
            fmt = "json"
            continue
        if "error" in d:
            raise RuntimeError(str(d["error"])[:200])
        batch = d.get("features", [])
        if fmt == "json":
            batch = [esri_to_geojson(f) for f in batch]
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


SURVEY_CACHE = Path("runs/default/surveys")


def fetch_zip(url: str, bbox: list[float], cache: Path = SURVEY_CACHE) -> list[dict]:
    """GeoJSON features (EPSG:4326) inside bbox from a zipped shapefile published as a plain download
    (e.g. MassGIS). Downloaded once per run folder, then read from the cache."""
    import geopandas as gpd

    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / url.rstrip("/").rsplit("/", 1)[-1]
    if not dst.exists():
        req = urllib.request.Request(url, headers=UA)
        import os

        part = dst.with_name(f"{dst.name}.{os.getpid()}.part")  # several sites download in parallel
        with urllib.request.urlopen(req, timeout=600) as r:
            part.write_bytes(r.read())
        part.replace(dst)
    gdf = gpd.read_file(f"zip://{dst}")
    gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    w, s, e, n = bbox
    gdf = gdf.cx[w:e, s:n]
    return json.loads(gdf.to_json()).get("features", []) if len(gdf) else []


def fetch_survey(s: dict, bbox: list[float]) -> list[dict]:
    if s["kind"] == "arcgis":
        return fetch_arcgis(s["url"], bbox, s.get("where", "1=1"))
    if s["kind"] == "zip":
        return fetch_zip(s["url"], bbox)
    return [f for f in fetch_wfs(s["url"], s["layer"], bbox if s.get("bbox_query", True) else None) if f.get("geometry")]


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


def burn_wetland(label: np.ndarray, image: np.ndarray, transform, crs, feats: list[dict], cls: str,
                 authoritative: bool = False, min_ndvi: float = 0.2) -> tuple[np.ndarray, int]:
    """Salt marsh or mangrove inside the polygons, over pixels that look vegetated (NDVI >= min_ndvi)
    and are not open water. authoritative: that class outside every polygon (beyond a 2 px margin for
    misregistration) is left unlabelled, because the official map and the global maps disagree."""
    from rasterio import features as rfeatures
    from rasterio.warp import transform_geom
    from scipy.ndimage import binary_dilation

    from .features import compute_features

    geoms = [transform_geom("EPSG:4326", crs, f["geometry"]) for f in feats if f.get("geometry")]
    k = KEY_TO_ID[cls]
    out = label.copy()
    inside = (rfeatures.rasterize([(g, 1) for g in geoms], out_shape=label.shape, transform=transform,
                                  fill=0, dtype="uint8").astype(bool) if geoms else np.zeros(label.shape, bool))
    fx = compute_features(image)
    ndvi, mndwi = fx[10], fx[12]
    other_blue = [KEY_TO_ID[c] for c in ("mangrove", "saltmarsh") if c != cls]
    can = np.isin(label, [IGNORE_INDEX, KEY_TO_ID["other_land"], KEY_TO_ID["freshwater"], KEY_TO_ID["tidal_flat"], k])
    m = inside & can & (ndvi >= min_ndvi) & ~(mndwi > 0.3)
    out[m] = k
    if authoritative and inside.any():  # an empty answer means no coverage here, not 'no marsh'
        near = binary_dilation(inside, iterations=2)
        out[(label == k) & ~near] = IGNORE_INDEX
    # a polygon of the *other* blue carbon class never gets overwritten
    out[np.isin(label, other_blue)] = label[np.isin(label, other_blue)]
    return out, int((m & (label != k)).sum())


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
                feats = fetch_survey(s, bbox)
                if s.get("keep") or s.get("drop"):
                    n0 = len(feats)
                    feats = filter_features(feats, s.get("keep"), s.get("drop"))
                    how = f"kept {len(feats)}/{n0} features matching {s.get('keep') or ''} {s.get('drop') or ''}".strip()
                else:
                    feats, how = seagrass_features(feats)
                cls = s.get("class", "seagrass")
                if cls == "seagrass":
                    neg_px = int(s.get("negatives_within_m", 0) / abs(ds.transform.a))
                    lab, n = burn_seagrass(lab, image, ds.transform, ds.crs, feats, neg_px)
                else:
                    lab, n = burn_wetland(lab, image, ds.transform, ds.crs, feats, cls,
                                          bool(s.get("authoritative")), float(s.get("min_ndvi", 0.2)))
                total += n
                log(f"  {cls} survey {s['name']}: {how}; {n * 0.01:,.0f} ha labelled as {cls}")
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
