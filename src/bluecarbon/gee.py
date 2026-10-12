"""Google Earth Engine: Sentinel-2 composites, fused reference labels, and tiled download.

Downloads use `ee.data.computePixels` in fixed-size tiles written straight into a GeoTIFF
in the local UTM zone, so any area size works (the v1 app used getDownloadURL, which fails
above ~32 MB, and exported in EPSG:4326 which broke area calculations).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import Config
from .features import S2_BANDS
from .schema import IGNORE_INDEX, KEY_TO_ID

try:  # optional dependency
    import ee
except ImportError:  # pragma: no cover
    ee = None


def _require_ee():
    if ee is None:
        raise ImportError("earthengine-api is not installed: pip install 'bluecarbon[gee]'")


REQUEST_TIMEOUT_S = 240  # one Earth Engine request may take at most this long before it is retried


def _transport():
    try:
        import httplib2

        return httplib2.Http(timeout=REQUEST_TIMEOUT_S)
    except Exception:
        return None


def init(project: str, service_account_json: str | None = None) -> None:
    """Initialize Earth Engine with user credentials or a service-account key (JSON text).

    Requests get a socket timeout: without one, a request the server never answers blocks forever
    (no error, no progress) and the whole run silently stalls.
    """
    _require_ee()
    creds = None
    if service_account_json:
        import json

        info = json.loads(service_account_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=service_account_json)
        project = project or info.get("project_id")
    kw = {"project": project}
    http = _transport()
    if http is not None:
        kw["http_transport"] = http
    try:
        ee.Initialize(creds, **kw) if creds else ee.Initialize(**kw)
    except TypeError:  # older earthengine-api without http_transport
        kw.pop("http_transport", None)
        ee.Initialize(creds, **kw) if creds else ee.Initialize(**kw)


class _Watchdog:
    """Raise TimeoutError in the main thread if a block takes longer than `seconds` (Linux / macOS)."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.armed = False

    def __enter__(self):
        import signal
        import threading

        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _fire(*_):
                raise TimeoutError(f"Earth Engine request timed out after {self.seconds}s")

            self._old = signal.signal(signal.SIGALRM, _fire)
            signal.alarm(self.seconds)
            self.armed = True
        return self

    def __exit__(self, *exc):
        if self.armed:
            import signal

            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


# --------------------------------------------------------------------------- geometry
def utm_epsg(lon: float, lat: float) -> int:
    lon = (lon + 180) % 360 - 180  # wrapped map longitudes (e.g. 513) -> -180..180
    zone = min(60, int(math.floor((lon + 180) / 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def utm_grid(bbox: list[float], scale: float) -> tuple[str, rasterio.Affine, int, int]:
    """Snap a lon/lat bbox to a UTM pixel grid. Returns (crs, transform, width, height)."""
    from rasterio.warp import transform_bounds

    lon0, lat0, lon1, lat1 = bbox
    epsg = utm_epsg((lon0 + lon1) / 2, (lat0 + lat1) / 2)
    crs = f"EPSG:{epsg}"
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, lon0, lat0, lon1, lat1)
    x0 = math.floor(x0 / scale) * scale
    y1 = math.ceil(y1 / scale) * scale
    w = int(math.ceil((x1 - x0) / scale))
    h = int(math.ceil((y1 - y0) / scale))
    return crs, from_origin(x0, y1, scale, scale), w, h


# --------------------------------------------------------------------------- imagery
def s2_composite(region, start: str, end: str, cfg: Config):
    """Cloud Score+ masked median composite of S2 L2A bands (uint16 reflectance x1e4)."""
    _require_ee()
    ic = cfg.imagery
    s2 = ee.ImageCollection(ic.collection).filterBounds(region).filterDate(start, end)
    cs = ee.ImageCollection(ic.cloud_score_collection)
    linked = s2.linkCollection(cs, [ic.cloud_score_band])
    masked = linked.map(lambda im: im.updateMask(im.select(ic.cloud_score_band).gte(ic.clear_threshold)))
    return masked.select(S2_BANDS).median().toUint16().set("n_images", s2.size())


def image_count(region, start: str, end: str, cfg: Config) -> int:
    _require_ee()
    return (
        ee.ImageCollection(cfg.imagery.collection).filterBounds(region).filterDate(start, end).size().getInfo()
    )


# --------------------------------------------------------------------------- labels
def _asset_bands(asset_id: str) -> list[str] | None:
    """Band names of an image / first image of a collection, or None if the asset is unavailable."""
    try:
        info = ee.data.getAsset(asset_id)
    except Exception:
        return None
    try:
        img = ee.Image(asset_id) if info.get("type") == "IMAGE" else ee.ImageCollection(asset_id).first()
        return img.bandNames().getInfo()
    except Exception:
        return None


def _tidal_band(cfg: Config) -> str | None:
    lc = cfg.labels
    bands = _asset_bands(lc.tidal_wetland) or []
    if lc.tidal_wetland_band in bands:
        return lc.tidal_wetland_band
    prob = sorted((b for b in bands if "prob" in b.lower()), key=lambda b: ("end" not in b, b))
    return prob[0] if prob else None


CLEAR_FRACTION_PCT = 20  # clear-water image = median of the clearest 20% of observations per pixel


def clear_water_composite(region, start: str, end: str, cfg: Config):
    """Per pixel, the median of the clearest ~20% of cloud-free observations (lowest NIR = least sun
    glint, haze and white water). A single clearest observation (the earlier method) is noisy; the
    median of several clear days is stable while still letting a shallow seafloor show through."""
    _require_ee()
    from .features import CLEAR_BANDS

    ic = cfg.imagery
    s2 = ee.ImageCollection(ic.collection).filterBounds(region).filterDate(start, end)
    cs = ee.ImageCollection(ic.cloud_score_collection)
    linked = s2.linkCollection(cs, [ic.cloud_score_band])
    clear = linked.map(lambda im: im.updateMask(im.select(ic.cloud_score_band).gte(ic.clear_threshold))
                       .select(CLEAR_BANDS))
    nir_cut = clear.select("B8").reduce(ee.Reducer.percentile([CLEAR_FRACTION_PCT])).rename("cut")
    best = clear.map(lambda im: im.updateMask(im.select("B8").lte(nir_cut))).median()
    # pixels with no observation under the cut (very few images): fall back to the plain median
    best = best.unmask(clear.median())
    return best.unmask(0).clamp(0, 10000).rename([f"{b}_clear" for b in CLEAR_BANDS])


def water_depth(cfg: Config):
    """GEBCO water depth in metres (0 on land), or 0 everywhere if the asset is unavailable."""
    try:
        img = ee.ImageCollection(cfg.labels.bathymetry).mosaic().select([0])
        img.bandNames().getInfo()
        return img.multiply(-1).max(0).unmask(0).clamp(0, 6000).rename("depth")
    except Exception as e:
        print(f"[bluecarbon] bathymetry unavailable ({str(e)[:80]}); depth set to 0", flush=True)
        return ee.Image(0).rename("depth")


def ancillary_image(cfg: Config, region=None, start: str | None = None, end: str | None = None):
    """int16 context layers: elevation (m), tidal wetland probability (0-100), |latitude| x 100,
    the clear-water bands for the given period, and water depth (m)."""
    _require_ee()
    from .features import ANCILLARY_BANDS

    lc = cfg.labels
    elev = ee.Image(lc.dem).select("elevation").unmask(0).clamp(-100, 3000)
    band = _tidal_band(cfg)
    tidal = (ee.Image(lc.tidal_wetland).select(band).unmask(0) if band else ee.Image(0))
    abs_lat = ee.Image.pixelLonLat().select("latitude").abs().multiply(100)
    clear = clear_water_composite(region, start, end, cfg)
    return ee.Image.cat([elev, tidal, abs_lat, clear, water_depth(cfg)]).rename(ANCILLARY_BANDS).toInt16()


def download_ancillary(bbox: list[float], start: str, end: str, out_path: str | Path, cfg: Config,
                       progress=None) -> Path:
    from .features import ANCILLARY_BANDS

    img = ancillary_image(cfg, bbox_geometry(bbox), start, end)
    return download(img, bbox, out_path, cfg, "int16", None, ANCILLARY_BANDS, progress=progress)


def ancillary_ok(path: str | Path) -> bool:
    """True if an ancillary.tif exists and has the current set of bands."""
    from .features import ANCILLARY_BANDS

    p = Path(path)
    if not p.exists():
        return False
    with rasterio.open(p) as ds:
        return ds.count == len(ANCILLARY_BANDS)


def tidal_zone(cfg: Config):
    """Where the tide actually reaches. Salt marsh is only labelled inside this zone.

    Primary: Murray et al. (2022) tidal wetland probability (tidal flat + marsh + mangrove, 30 m).
    Fallback: low-lying (<= 3 m) land within 1 km of open water.
    """
    lc = cfg.labels
    bands = _asset_bands(lc.tidal_wetland) or []
    band = lc.tidal_wetland_band if lc.tidal_wetland_band in bands else None
    if band is None:  # tolerate renamed bands: prefer the most recent probability band
        prob = sorted((b for b in bands if "prob" in b.lower()), key=lambda b: ("end" not in b, b))
        band = prob[0] if prob else None
    if band:
        return ee.Image(lc.tidal_wetland).select(band).gte(lc.tidal_wetland_min_prob), f"murray-gic:{band}"
    if bands:
        print(f"[bluecarbon] {lc.tidal_wetland} has no probability band (bands: {bands}); using fallback")
    dem = ee.Image(lc.dem).select("elevation")
    wc = ee.ImageCollection(lc.worldcover).first().select("Map")
    # Computed at 100 m: a 1 km neighbourhood at 10 m is ~30k pixels per output pixel and times out.
    near_water = (wc.eq(80).reduceResolution(ee.Reducer.max(), maxPixels=1024)
                  .reproject(crs="EPSG:4326", scale=100)
                  .focalMax(radius=10, kernelType="circle", units="pixels"))
    return dem.lte(3).And(near_water), "elevation-fallback"


LABEL_VERSION = "labels-v11"  # v11 = v10 + NJ, NH, EMODnet Europe, Torres Strait, Wales, Scotland, Ireland surveys; v10 = v9 with survey fixes; v9: mangrove = WorldCover AND Global Mangrove Watch; NWI/NSW/EA wetland surveys  # bump when the label rules change, so `fetch --labels-only` rebuilds


def wetland_map(cfg: Config):
    """GWL_FCS30 wetland classes for the label year (0 where unmapped), or None if the asset is unavailable."""
    lc = cfg.labels
    try:
        col = ee.ImageCollection(lc.wetland_map)
        yr = col.filter(ee.Filter.calendarRange(lc.wetland_year, lc.wetland_year, "year"))
        if yr.size().getInfo() == 0:
            print(f"[bluecarbon] {lc.wetland_map}: no {lc.wetland_year} image, using the full collection", flush=True)
            yr = col
        img = yr.select([0]).mosaic()
        img.bandNames().getInfo()  # fail here, not mid-download, if the asset is not readable
        return img.unmask(0).rename("gwl")
    except Exception as e:
        print(f"[bluecarbon] wetland map unavailable ({str(e)[:80]}); using the tidal-zone salt marsh rule", flush=True)
        return None


def reference_labels(region, cfg: Config, report: dict | None = None, seagrass_unmapped: bool = False):
    """Fuse published global products into the class schema (uint8, 255 = ignore).

    Priority (later overrides earlier):
      WorldCover base -> Murray tidal flats -> salt marsh -> freshwater marsh -> WorldCover mangrove
      -> Allen Coral Atlas / survey-polygon seagrass.

    Salt marsh = WorldCover herbaceous / grass / shrub that GWL_FCS30 maps as salt marsh. Tidal-zone
    vegetation (Murray tidal wetland zone, below 5 m) that GWL does not call salt marsh is labelled
    only if GWL has no opinion there, and left unlabelled where GWL calls it non-wetland: when the two
    sources disagree, the pixel is not trained or scored on.

    seagrass_unmapped: the site has seagrass that no reference product covers (e.g. Florida Bay,
    Tampa Bay). Its water outside the Allen Coral Atlas footprint is left unlabelled instead of
    being called "water", so the model is not taught that seagrass is open water.
    """
    _require_ee()
    lc = cfg.labels
    wc = ee.ImageCollection(lc.worldcover).first().select("Map")
    ign = IGNORE_INDEX
    k = KEY_TO_ID
    used = [LABEL_VERSION]

    # WorldCover: 10 tree,20 shrub,30 grass,40 crop,50 built,60 bare,70 snow,80 water,90 herb. wetland,
    #             95 mangrove,100 moss/lichen
    base = wc.remap(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        [k["other_land"]] * 6 + [ign, k["water"], ign, k["mangrove"], k["other_land"]],
        ign,
    )
    used.append("worldcover")

    intertidal = ee.ImageCollection(lc.intertidal).sort("system:time_start", False).first().select(0)
    lab = base.where(intertidal.eq(1).And(wc.neq(95)), k["tidal_flat"])
    used.append("murray-tidal-flats")

    tz, tz_src = tidal_zone(cfg)
    used.append(tz_src)
    dem = ee.Image(lc.dem).select("elevation")
    herb = wc.eq(90).Or(wc.eq(30)).Or(wc.eq(20)).And(wc.neq(95))
    tidal_veg = herb.And(tz.unmask(0)).And(dem.lte(lc.marsh_max_elev_m))
    gwl = wetland_map(cfg)
    if gwl is not None:
        used.append("gwl-fcs30")
        gwl_marsh = gwl.eq(186)
        gwl_fresh = gwl.eq(181).Or(gwl.eq(182))
        marsh = herb.And(gwl_marsh).Or(tidal_veg.And(gwl.eq(184)))
        # tidal-zone vegetation GWL calls non-wetland or freshwater: sources disagree -> ignore
        lab = lab.where(tidal_veg.And(marsh.Not()), ign)
        lab = lab.where(marsh, k["saltmarsh"])
        lab = lab.where(wc.eq(90).And(marsh.Not()).And(tidal_veg.Not().Or(gwl_fresh)), k["freshwater"])
    else:
        marsh = tidal_veg
        lab = lab.where(marsh, k["saltmarsh"])
        # herbaceous wetland outside the tidal zone = freshwater marsh (not blue carbon)
        lab = lab.where(wc.eq(90).And(marsh.Not()), k["freshwater"])
    lab = lab.where(wc.eq(95), k["mangrove"])
    if lc.mangrove_map:
        # Second, independent mangrove map (Global Mangrove Watch v3, ALOS radar + Landsat, Bunting et al.
        # 2022). Mangrove is labelled only where both maps agree; where only one says mangrove the pixel
        # is left unlabelled, so the answer key holds only mangrove two published maps confirm.
        try:
            ee.data.getAsset(lc.mangrove_map)  # fail here (not mid-download) if the asset is gone
            gmw = ee.Image(0).paint(ee.FeatureCollection(lc.mangrove_map).filterBounds(region), 1).eq(1)
            lab = lab.where(wc.eq(95).And(gmw.Not()), ign)
            lab = lab.where(gmw.And(wc.neq(95)).And(wc.neq(80)), ign)
            used.append("gmw-v3-2020")
        except Exception as e:
            print(f"[bluecarbon] mangrove map unavailable ({str(e)[:80]})", flush=True)

    aca_footprint = ee.Image(0)
    if _asset_bands(lc.reef_habitat):
        benthic = ee.Image(lc.reef_habitat).select("benthic")
        aca_footprint = benthic.mask().unmask(0).gt(0)
        lab = lab.where(benthic.eq(lc.seagrass_value), k["seagrass"])
        used.append("allen-coral-atlas")
    if seagrass_unmapped:
        lab = lab.where(lab.eq(k["water"]).And(aca_footprint.Not()), ign)
        used.append("unmapped-seagrass-water-ignored")
    for fc_id in lc.seagrass_vectors:
        try:
            fc = ee.FeatureCollection(fc_id).filterBounds(region)
            sg = ee.Image(0).paint(fc, 1).selfMask()
            lab = lab.where(sg.unmask(0).eq(1).And(wc.eq(80).Or(lab.eq(ign))), k["seagrass"])
            used.append(fc_id)
        except Exception:
            pass
    if report is not None:
        report["label_sources"] = used
    return lab.unmask(ign).clip(region).toUint8().rename("label")


# --------------------------------------------------------------------------- download
_RETRYABLE = ("timed out", "timeout", "memory", "too many", "limit", "503", "500", "internal error")


def _compute_tile(image, crs: str, transform: rasterio.Affine, x0: int, y0: int, w: int, h: int,
                  attempt: int = 0) -> np.ndarray:
    """Fetch one tile; on a server timeout, split it into four smaller tiles (recursively) and retry."""
    import time

    try:
        with _Watchdog(REQUEST_TIMEOUT_S + 60):
            return _compute_tile_once(image, crs, transform, x0, y0, w, h)
    except Exception as e:
        msg = str(e).lower()
        if not any(k in msg for k in _RETRYABLE):
            raise
        if min(w, h) >= 128:
            hw, hh = w // 2, h // 2
            print(f"[bluecarbon] tile {w}x{h} at ({x0},{y0}) failed ({str(e)[:60]}); splitting", flush=True)
            top = np.concatenate([_compute_tile(image, crs, transform, x0, y0, hw, hh),
                                  _compute_tile(image, crs, transform, x0 + hw, y0, w - hw, hh)], axis=2)
            bot = np.concatenate([_compute_tile(image, crs, transform, x0, y0 + hh, hw, h - hh),
                                  _compute_tile(image, crs, transform, x0 + hw, y0 + hh, w - hw, h - hh)], axis=2)
            return np.concatenate([top, bot], axis=1)
        if attempt < 4:
            time.sleep(5 * (attempt + 1))
            return _compute_tile(image, crs, transform, x0, y0, w, h, attempt + 1)
        raise


def _compute_tile_once(image, crs: str, transform: rasterio.Affine, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    tx = transform @ rasterio.Affine.translation(x0, y0)
    req = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": tx.a,
                "shearX": tx.b,
                "translateX": tx.c,
                "shearY": tx.d,
                "scaleY": tx.e,
                "translateY": tx.f,
            },
            "crsCode": crs,
        },
    }
    arr = ee.data.computePixels(req)
    if isinstance(arr, bytes | bytearray):  # older clients return raw .npy bytes
        arr = np.load(io.BytesIO(arr))
    return np.stack([arr[n] for n in arr.dtype.names])


def download(image, bbox: list[float], out_path: str | Path, cfg: Config, dtype: str = "uint16",
             nodata: int | None = 0, band_names: list[str] | None = None, progress=None) -> Path:
    """Download an ee.Image over `bbox` to a tiled, compressed GeoTIFF in local UTM."""
    _require_ee()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crs, transform, W, H = utm_grid(bbox, cfg.imagery.scale_m)
    band_names = band_names or image.bandNames().getInfo()
    t = cfg.imagery.tile_px
    tiles = [(x, y) for y in range(0, H, t) for x in range(0, W, t)]
    profile = dict(driver="GTiff", width=W, height=H, count=len(band_names), dtype=dtype, crs=crs,
                   transform=transform, nodata=nodata, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.descriptions = tuple(band_names)
        step = max(1, len(tiles) // 4)
        for i, (x, y) in enumerate(tiles):
            w, h = min(t, W - x), min(t, H - y)
            block = _compute_tile(image, crs, transform, x, y, w, h).astype(dtype)
            dst.write(block, window=rasterio.windows.Window(x, y, w, h))
            if progress:
                progress((i + 1) / len(tiles))
            elif len(tiles) >= 8 and (i + 1) % step == 0:
                print(f"    {out_path.name}: {i + 1}/{len(tiles)} tiles", flush=True)
    return out_path


def bbox_geometry(bbox: list[float]):
    _require_ee()
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


MANGROVE_AGB = "projects/sat-io/open-datasets/global_mangrove_distribution/agb"  # Simard et al. 2019, Mg/ha


def site_biomass_stats(bbox: list[float]) -> dict:
    """Measured mangrove aboveground biomass in a box (NASA map, Simard et al. 2019) and mean annual rainfall
    (WorldClim, for the IPCC climate zone). One small server-side summary, no imagery download."""
    _require_ee()
    region = bbox_geometry(bbox)
    agb = ee.ImageCollection(MANGROVE_AGB).mosaic().select([0], ["agb"])
    red = (ee.Reducer.mean().combine(ee.Reducer.percentile([10, 90]), "", True)
           .combine(ee.Reducer.count(), "", True))
    a = agb.reduceRegion(red, region, 30, maxPixels=int(1e10), bestEffort=True)
    p = ee.Image("WORLDCLIM/V1/BIO").select("bio12").reduceRegion(ee.Reducer.mean(), region, 1000, bestEffort=True)
    with _Watchdog(REQUEST_TIMEOUT_S + 60):
        d = ee.Dictionary({"a": a, "p": p}).getInfo()
    a, p = d.get("a") or {}, d.get("p") or {}
    return {"agb_mean": a.get("agb_mean"), "agb_p10": a.get("agb_p10"), "agb_p90": a.get("agb_p90"),
            "agb_count": a.get("agb_count") or 0, "precip_mm": p.get("bio12")}


PALSAR_YEARLY = "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"   # ALOS-2 PALSAR-2 25 m yearly mosaics (JAXA), 2015-
PALSAR_YEARLY_CLASSIC = "JAXA/ALOS/PALSAR/YEARLY/SAR"  # the original yearly mosaics, 2007-2010 and 2015-2021


def lband_image(region, year: int):
    """ALOS PALSAR-2 L-band HH/HV, gamma0 dB x 100 as int16 (dB = 10 log10(DN^2) - 83, JAXA).

    Per pixel, the valid observation from the mosaic year closest to `year` (both JAXA collections). DN = 0 means
    no data in those mosaics: it is masked, never turned into dB (log of 0 gave -327 dB in the first version).
    No coverage at all reads -3000; values are clamped to -50..10 dB."""
    _require_ee()
    t0 = ee.Date.fromYMD(year, 7, 1)

    def prep(im):
        dn = im.select(["HH", "HV"]).toFloat()
        valid = dn.select("HH").gt(0).And(dn.select("HV").gt(0))
        db = dn.pow(2).log10().multiply(10).subtract(83).updateMask(valid)
        near = (ee.Image.constant(ee.Number(im.date().difference(t0, "day")).abs().multiply(-1)).toFloat()
                .rename("near").updateMask(valid))
        return db.addBands(near)

    col = (ee.ImageCollection(PALSAR_YEARLY).select(["HH", "HV"])
           .merge(ee.ImageCollection(PALSAR_YEARLY_CLASSIC).select(["HH", "HV"]))
           .filterBounds(region).map(prep))
    img = col.qualityMosaic("near").select(["HH", "HV"])
    return img.clamp(-50, 10).multiply(100).unmask(-3000).toInt16().rename(["alos_hh", "alos_hv"])


def lband_ok(path) -> bool:
    """False for a missing L-band file or one from the first (broken) version, full of -327 dB."""
    p = Path(path)
    if not p.exists():
        return False
    with rasterio.open(p) as ds:
        a = ds.read(2)
    return float((a < -5000).mean()) < 0.05


def download_lband(bbox: list[float], year: int, out_path: str | Path, cfg: Config) -> Path:
    return download(lband_image(bbox_geometry(bbox), year), bbox, out_path, cfg, "int16", None, ["alos_hh", "alos_hv"])


GEDI_L4A = "LARSE/GEDI/GEDI04_A_002_MONTHLY"   # GEDI lidar footprint biomass (Dubayah et al.), rasterised 25 m
CANOPY_HEIGHT = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"   # Lang et al. 2023, 10 m


def lidar_biomass_check(bbox: list[float]) -> dict:
    """Independent check of the NASA mangrove biomass map inside a box: GEDI lidar footprint biomass
    (quality-filtered, 2019-2023) and ETH 10 m canopy height, both over the map's mangrove pixels."""
    _require_ee()
    region = bbox_geometry(bbox)
    mang = ee.ImageCollection(MANGROVE_AGB).mosaic().select(0).gt(0)
    gedi = (ee.ImageCollection(GEDI_L4A).filterBounds(region).filterDate("2019-01-01", "2024-01-01")
            .map(lambda im: im.updateMask(im.select("l4_quality_flag").eq(1)).select("agbd")).mosaic().updateMask(mang))
    red = (ee.Reducer.mean().combine(ee.Reducer.percentile([10, 90]), "", True)
           .combine(ee.Reducer.count(), "", True))
    g = gedi.reduceRegion(red, region, 25, maxPixels=int(1e10), bestEffort=True)
    h = ee.Image(CANOPY_HEIGHT).select(0).rename("h").updateMask(mang).reduceRegion(red, region, 10,
                                                                                    maxPixels=int(1e10), bestEffort=True)
    with _Watchdog(REQUEST_TIMEOUT_S + 60):
        d = ee.Dictionary({"g": g, "h": h}).getInfo()
    g, h = d.get("g") or {}, d.get("h") or {}
    out = {"gedi_agbd_mean": g.get("agbd_mean"), "gedi_agbd_p10": g.get("agbd_p10"), "gedi_agbd_p90": g.get("agbd_p90"),
           "gedi_footprints": g.get("agbd_count") or 0, "canopy_h_mean": h.get("h_mean"), "canopy_h_p90": h.get("h_p90")}
    # Meta / WRI 1 m canopy height (Tolan et al. 2024): a second, much finer height source
    try:
        meta = ee.ImageCollection(META_CANOPY_HEIGHT).filterBounds(region).mosaic().select(0).rename("h")
        with _Watchdog(REQUEST_TIMEOUT_S + 60):
            mh = meta.updateMask(mang).reduceRegion(red, region, 10, maxPixels=int(1e10), bestEffort=True).getInfo() or {}
        out.update(meta_h_mean=mh.get("h_mean"), meta_h_p90=mh.get("h_p90"))
    except Exception as e:  # dataset moved or not visible to this project: skip, keep the rest
        print(f"  Meta canopy height skipped ({str(e)[:80]})", flush=True)
    # Local height-to-biomass model fitted on this site's own GEDI footprints (AGB = a * H^b), so biomass can be
    # mapped from 10 m canopy height everywhere, with its error against the lidar reported.
    try:
        pts = (gedi.rename("agbd").addBands(ee.Image(CANOPY_HEIGHT).select(0).rename("h"))
               .sample(region, 25, numPixels=4000, seed=0, geometries=False))
        with _Watchdog(REQUEST_TIMEOUT_S + 60):
            rows = pts.limit(3000).getInfo().get("features", [])
        out.update(fit_height_biomass([(f["properties"].get("h"), f["properties"].get("agbd")) for f in rows]))
        if out.get("hb_a"):
            hb = (ee.Image(CANOPY_HEIGHT).select(0).max(0.5).pow(out["hb_b"]).multiply(out["hb_a"]).rename("agb_h").updateMask(mang)
                  .reduceRegion(ee.Reducer.mean(), region, 10, maxPixels=int(1e10), bestEffort=True))
            with _Watchdog(REQUEST_TIMEOUT_S + 60):
                out["height_agb_mean"] = (hb.getInfo() or {}).get("agb_h")
    except Exception as e:
        print(f"  height-biomass fit skipped ({str(e)[:80]})", flush=True)
    return out


META_CANOPY_HEIGHT = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"


def fit_height_biomass(pairs: list[tuple]) -> dict:
    """log AGB = log a + b log H over footprints with both values. Returns a, b, footprints used, r2 and RMSE
    (Mg/ha, back-transformed, from 5-fold cross-validation) or {} if too few."""
    import numpy as np

    hp = np.array([(h, a) for h, a in pairs if h is not None and a is not None and h > 1 and a > 1], float)
    if len(hp) < 30:
        return {}
    x, y = np.log(hp[:, 0]), np.log(hp[:, 1])
    pred = np.empty_like(y)
    folds = np.arange(len(y)) % 5
    for k in range(5):
        tr = folds != k
        b, la = np.polyfit(x[tr], y[tr], 1)
        pred[~tr] = la + b * x[~tr]
    b, la = np.polyfit(x, y, 1)
    agb, agb_hat = np.exp(y), np.exp(pred)
    r2 = 1 - np.sum((agb - agb_hat) ** 2) / np.sum((agb - agb.mean()) ** 2)
    return {"hb_a": round(float(np.exp(la)), 4), "hb_b": round(float(b), 4), "hb_footprints": int(len(y)),
            "hb_r2_cv": round(float(r2), 3), "hb_rmse_cv": round(float(np.sqrt(np.mean((agb - agb_hat) ** 2))), 1)}


def built_up_image(cfg: Config):
    """1 where ESA WorldCover 2021 says built up (class 50), else 0: towns and roads a wetland can't move into."""
    _require_ee()
    return ee.ImageCollection(cfg.labels.worldcover).first().select("Map").eq(50).unmask(0).rename("built").toUint8()


def download_built(bbox: list[float], out_path: str | Path, cfg: Config) -> Path:
    return download(built_up_image(cfg), bbox, out_path, cfg, "uint8", None, ["built"])


def radar_image(region, start: str, end: str):
    """Sentinel-1 GRD (IW, VV + VH) median backscatter over the period, dB x 100 as int16.
    Pixels with no radar pass read -3000 (-30 dB)."""
    _require_ee()
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return col.select(["VV", "VH"]).median().multiply(100).unmask(-3000).toInt16().rename(["s1_vv", "s1_vh"])


def download_radar(bbox: list[float], start: str, end: str, out_path: str | Path, cfg: Config) -> Path:
    return download(radar_image(bbox_geometry(bbox), start, end), bbox, out_path, cfg, "int16", None, ["s1_vv", "s1_vh"])


# Roy et al. (2016) ETM+ -> OLI surface reflectance harmonisation (OLS), applied to Landsat 5 and 7.
_ROY_SLOPE = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
_ROY_ITCP = [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
_LS = {"LANDSAT/LT05/C02/T1_L2": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
       "LANDSAT/LE07/C02/T1_L2": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
       "LANDSAT/LC08/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
       "LANDSAT/LC09/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]}


def landsat_composite(region, year: int, half_window: int = 1):
    """Three-year (year +/- 1) median of cloud-masked Landsat 5/7/8/9 surface reflectance, harmonised to
    OLI, as uint16 reflectance x 1e4 in bands blue, green, red, nir, swir1, swir2."""
    _require_ee()
    from .history import LANDSAT_BANDS

    start, end = f"{year - half_window}-01-01", f"{year + half_window + 1}-01-01"
    cols = []
    for cid, bands in _LS.items():
        def prep(im, bands=bands, oli=cid.startswith("LANDSAT/LC")):
            qa = im.select("QA_PIXEL")
            clear = qa.bitwiseAnd(0b11010).eq(0)  # dilated cloud, cloud, cloud shadow
            sr = im.select(bands).multiply(0.0000275).add(-0.2)
            if not oli:
                sr = sr.multiply(ee.Image.constant(_ROY_SLOPE)).add(ee.Image.constant(_ROY_ITCP))
            return sr.rename(LANDSAT_BANDS).updateMask(clear).toFloat()  # same band type in every sensor
        cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).map(prep))
    merged = cols[0].merge(cols[1]).merge(cols[2]).merge(cols[3])
    return merged.median().clamp(0, 1).multiply(10000).toUint16()


def download_landsat(bbox: list[float], year: int, out_path: str | Path, cfg: Config) -> Path:
    from .history import LANDSAT_BANDS, SCALE_M

    c30 = cfg.model_copy(update={"imagery": cfg.imagery.model_copy(update={"scale_m": SCALE_M})})
    return download(landsat_composite(bbox_geometry(bbox), year), bbox, out_path, c30, "uint16", 0, LANDSAT_BANDS)


def kelp_quarter_image(region, start: str, end: str, cfg: Config):
    """Kelp canopy inputs for one quarter at 20 m: 90th-percentile Floating Algae Index over the clear
    Sentinel-2 scenes (x 1e4, int16; -32768 = no clear scene), number of clear scenes, and land (WorldCover
    not water). FAI = B8A - (B4 + (B11 - B4) * (865 - 665) / (1610 - 665)), Hu (2009)."""
    _require_ee()
    ic = cfg.imagery
    s2 = ee.ImageCollection(ic.collection).filterBounds(region).filterDate(start, end)
    cs = ee.ImageCollection(ic.cloud_score_collection)
    linked = s2.linkCollection(cs, [ic.cloud_score_band])

    def fai(im):
        r = im.select(["B4", "B8A", "B11"]).divide(10000)
        f = r.select("B8A").subtract(r.select("B4").add(r.select("B11").subtract(r.select("B4"))
                                                        .multiply((865 - 665) / (1610 - 665))))
        return f.rename("fai").updateMask(im.select(ic.cloud_score_band).gte(ic.clear_threshold))

    col = linked.map(fai)
    p90 = col.reduce(ee.Reducer.percentile([90])).rename("fai").multiply(10000).round().unmask(-32768).toInt16()
    n = col.count().rename("n").unmask(0).toInt16()
    wc = ee.ImageCollection(cfg.labels.worldcover).first().select("Map")
    land = wc.neq(80).And(wc.neq(0)).unmask(0).rename("land").toInt16()
    return ee.Image.cat([p90, n, land])


def download_kelp_quarter(bbox: list[float], year: int, q: int, out_path: str | Path, cfg: Config) -> Path:
    from .kelp import SCALE_M, quarter_dates

    a, b = quarter_dates(year, q)
    c20 = cfg.model_copy(update={"imagery": cfg.imagery.model_copy(update={"scale_m": SCALE_M})})
    return download(kelp_quarter_image(bbox_geometry(bbox), a, b, cfg), bbox, out_path, c20, "int16", None,
                    ["fai", "n", "land"])


def kelp_rgb_image(region, year: int, cfg: Config):
    """Summer (June to September) true-colour picture of a kelp coast for the website's year slider, as 8-bit
    RGB. 2016 on: Sentinel-2 (cloud-masked median, 10 m bands). Before: Landsat 5/7/8 surface reflectance
    (median of that summer's clear scenes; 2012 is Landsat 7 only and shows its scan-line gaps)."""
    _require_ee()
    start, end = f"{year}-06-01", f"{year}-10-01"
    if year >= 2016:
        ic = cfg.imagery
        s2 = ee.ImageCollection(ic.collection).filterBounds(region).filterDate(start, end)
        linked = s2.linkCollection(ee.ImageCollection(ic.cloud_score_collection), [ic.cloud_score_band])
        rgb = (linked.map(lambda im: im.updateMask(im.select(ic.cloud_score_band).gte(ic.clear_threshold))
                          .select(["B4", "B3", "B2"]).divide(10000).toFloat())
               .median().rename(["r", "g", "b"]))
    else:
        cols = []
        for cid, bands in _LS.items():
            rgb_b = [bands[2], bands[1], bands[0]]   # red, green, blue in that sensor's band names

            def prep(im, rgb_b=rgb_b):
                clear = im.select("QA_PIXEL").bitwiseAnd(0b11010).eq(0)
                return im.select(rgb_b).multiply(0.0000275).add(-0.2).rename(["r", "g", "b"]).updateMask(clear).toFloat()
            cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).map(prep))
        rgb = cols[0].merge(cols[1]).merge(cols[2]).merge(cols[3]).median()
    return rgb.visualize(bands=["r", "g", "b"], min=0, max=0.16, gamma=1.25).rename(["r", "g", "b"]).unmask(0).toUint8()


def download_kelp_rgb(bbox: list[float], year: int, out_path: str | Path, cfg: Config) -> Path:
    scale = 20 if year >= 2016 else 30
    c = cfg.model_copy(update={"imagery": cfg.imagery.model_copy(update={"scale_m": scale})})
    return download(kelp_rgb_image(bbox_geometry(bbox), year, cfg), bbox, out_path, c, "uint8", 0, ["r", "g", "b"])


# Landsat Floating Algae Index, same method as Sentinel-2 but with each sensor's red / NIR / SWIR1 bands and centre
# wavelengths (nm): TM and ETM+ (Landsat 5, 7) 660 / 835 / 1650, OLI (Landsat 8, 9) 655 / 865 / 1610.
_LS_FAI = {"LANDSAT/LT05/C02/T1_L2": (("SR_B3", "SR_B4", "SR_B5"), (660, 835, 1650)),
           "LANDSAT/LE07/C02/T1_L2": (("SR_B3", "SR_B4", "SR_B5"), (660, 835, 1650)),
           "LANDSAT/LC08/C02/T1_L2": (("SR_B4", "SR_B5", "SR_B6"), (655, 865, 1610)),
           "LANDSAT/LC09/C02/T1_L2": (("SR_B4", "SR_B5", "SR_B6"), (655, 865, 1610))}


def kelp_quarter_image_landsat(region, start: str, end: str, cfg: Config):
    """Kelp canopy inputs for one quarter from Landsat 5/7/8/9 at 30 m (1984 on), in the same three bands as the
    Sentinel-2 version: 90th-percentile FAI x 1e4 (-32768 = no clear scene), clear-scene count, land."""
    _require_ee()
    cols = []
    for cid, (bands, (lr, ln, ls)) in _LS_FAI.items():
        def fai(im, bands=bands, lr=lr, ln=ln, ls=ls):
            clear = im.select("QA_PIXEL").bitwiseAnd(0b11010).eq(0)
            r = im.select(list(bands)).multiply(0.0000275).add(-0.2)
            red, nir, swir = r.select(0), r.select(1), r.select(2)
            f = nir.subtract(red.add(swir.subtract(red).multiply((ln - lr) / (ls - lr))))
            return f.rename("fai").updateMask(clear).toFloat()
        cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).map(fai))
    col = cols[0].merge(cols[1]).merge(cols[2]).merge(cols[3])
    p90 = col.reduce(ee.Reducer.percentile([90])).rename("fai").multiply(10000).round().unmask(-32768).toInt16()
    n = col.count().rename("n").unmask(0).toInt16()
    wc = ee.ImageCollection(cfg.labels.worldcover).first().select("Map")
    land = wc.neq(80).And(wc.neq(0)).unmask(0).rename("land").toInt16()
    return ee.Image.cat([p90, n, land])


def download_kelp_quarter_landsat(bbox: list[float], year: int, q: int, out_path: str | Path, cfg: Config) -> Path:
    from .kelp import quarter_dates

    a, b = quarter_dates(year, q)
    c30 = cfg.model_copy(update={"imagery": cfg.imagery.model_copy(update={"scale_m": 30})})
    return download(kelp_quarter_image_landsat(bbox_geometry(bbox), a, b, cfg), bbox, out_path, c30, "int16", None,
                    ["fai", "n", "land"])


OISST = "NOAA/CDR/OISST/V2_1"   # NOAA daily optimum-interpolation sea surface temperature, 0.25 deg, 1981 on


def sea_surface_temperature(bbox: list[float], first_year: int, last_year: int) -> list[dict]:
    """Quarterly mean sea surface temperature (deg C) over a kelp bed's box, widened by 15 km so it reaches the
    0.25 deg ocean grid, from NOAA OISST v2.1. One request for all quarters."""
    _require_ee()
    region = bbox_geometry(bbox).buffer(15000)
    qs = ee.List([[y, q] for y in range(first_year, last_year + 1) for q in (1, 2, 3, 4)])

    def one(yq):
        yq = ee.List(yq)
        y, q = ee.Number(yq.get(0)), ee.Number(yq.get(1))
        start = ee.Date.fromYMD(y, q.subtract(1).multiply(3).add(1), 1)
        img = ee.ImageCollection(OISST).filterDate(start, start.advance(3, "month")).select("sst").mean()
        v = img.reduceRegion(ee.Reducer.mean(), region, 27830, maxPixels=int(1e8)).get("sst")
        return ee.Dictionary({"year": y, "quarter": q, "sst": v})

    with _Watchdog(REQUEST_TIMEOUT_S + 120):
        rows = qs.map(one).getInfo()
    return [{"year": int(r["year"]), "quarter": int(r["quarter"]), "c": round(r["sst"] * 0.01, 2)}
            for r in rows if r.get("sst") is not None]


GMW_YEARS = (1996, 2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020)
GMW_VEC = "projects/sat-io/open-datasets/GMW/extent/gmw_v3_{year}_vec"   # Global Mangrove Watch v3 (Bunting 2022)


def gmw_areas(bbox: list[float], years=GMW_YEARS) -> dict[int, float]:
    """Mangrove area (ha) inside a box in each Global Mangrove Watch v3 year, 1996-2020: an independent record of
    mangrove loss and gain to check our Landsat history against. Years whose layer can't be read are skipped."""
    _require_ee()
    region = bbox_geometry(bbox)
    out = {}
    for y in years:
        try:
            fc = ee.FeatureCollection(GMW_VEC.format(year=y)).filterBounds(region)
            img = ee.Image(0).paint(fc, 1).selfMask().multiply(ee.Image.pixelArea()).divide(1e4)
            with _Watchdog(REQUEST_TIMEOUT_S + 60):
                v = img.reduceRegion(ee.Reducer.sum(), region, 25, maxPixels=int(1e10), bestEffort=True).getInfo()
            out[y] = round(float(next(iter(v.values())) or 0), 1)
        except Exception as e:
            print(f"  GMW {y} skipped ({str(e)[:80]})", flush=True)
    return out
