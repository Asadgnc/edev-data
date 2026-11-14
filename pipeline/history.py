"""
EDev-Vision | Historical context layers around the NISAR series.

Language: Python, same reason as the rest of the pipeline (rasterio reads
cloud GeoTIFFs by HTTP range request, so only the bytes over our AOI travel).

NISAR has existed since 2025, so it cannot say whether 2026 is a big or a small
flood. These three open, no-account sources can. They are CONTEXT, never merged
into the NISAR curve: different sensor, different physics, different label.

  wofs     Digital Earth Africa "Water Observations from Space", annual
           summary, Landsat optical, 30 m, 1984-2025. One product and one
           method for the whole period -> history.json + history/<year>.png
  jrc      JRC Global Surface Water v1.5 "occurrence" (1984-2024): how often
           each pixel was water -> jrc/occurrence.png
  drivers  WHY the water changes: modelled river discharge near Mohembo
           (GloFAS via Open-Meteo) and rainfall over the Angolan headwaters
           (NASA POWER) -> drivers.json

Optical sensors see OPEN water. They largely miss water standing under
vegetation, which is exactly what NISAR's L-band adds. Say so in the app.

Usage:  python pipeline/history.py all|wofs|jrc|drivers
"""
from __future__ import annotations

import collections
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.warp import transform as warp_xy
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

UA = {"User-Agent": "python-urllib"}
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def get_json(url: str, timeout: int = 120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=timeout) as r:
        return json.load(r)


def to_web(src, src_tf, src_crs, nodata):
    """Reproject to Web Mercator and return (array, Leaflet/MapLibre bounds)."""
    h, w = src.shape
    xmin, ymin, xmax, ymax = array_bounds(h, w, src_tf)
    dst_tf, dw, dh = calculate_default_transform(
        src_crs, "EPSG:3857", w, h, xmin, ymin, xmax, ymax)
    dst = np.full((dh, dw), nodata, src.dtype)
    reproject(src, dst, src_transform=src_tf, src_crs=src_crs,
              src_nodata=nodata, dst_transform=dst_tf, dst_crs="EPSG:3857",
              dst_nodata=nodata, resampling=Resampling.nearest)
    x0, y0, x1, y1 = array_bounds(dh, dw, dst_tf)
    (lw, le), (ls, ln) = warp_xy("EPSG:3857", "EPSG:4326", [x0, x1], [y0, y1])
    b = {"bounds": [[ls, lw], [ln, le]],
         "coordinates": [[lw, ln], [le, ln], [le, ls], [lw, ls]],
         "crs": "EPSG:3857", "width": int(dw), "height": int(dh)}
    return dst, b


# --- WOfS annual, 1984-2025 --------------------------------------------------
STAC = ("https://explorer.digitalearth.africa/stac/search?"
        "collections=wofs_ls_summary_annual&limit=500&bbox={},{},{},{}")
S3 = "https://deafrica-services.s3.af-south-1.amazonaws.com/"
DEC = 4            # read the 120 m overview of the 30 m product
PIX = 30.0 * DEC   # EPSG:6933 is equal-area, so 120 m pixels are 0.0144 km2
KM2 = (PIX / 1000) ** 2
# A pixel counts as water in a year if it was wet in at least 10% of its
# cloud-free looks AND at least twice (one wet look is usually cloud shadow).
# "Permanent" = wet in at least 80% of looks.
F_ANY, F_PERM, MIN_WET = 0.10, 0.80, 2
SEASONAL, PERMANENT, ND = 1, 2, 255
HCOL = {SEASONAL: (107, 174, 214, 200), PERMANENT: (8, 81, 156, 220)}


def wofs_grid():
    (xa, xb), (ya, yb) = warp_xy("EPSG:4326", "EPSG:6933",
                                 [C.AOI[0], C.AOI[2]], [C.AOI[1], C.AOI[3]])
    x0, y1 = np.floor(xa / PIX) * PIX, np.ceil(yb / PIX) * PIX
    return x0, y1, int(np.ceil((xb - x0) / PIX)), int(np.ceil((y1 - ya) / PIX))


def wofs_items():
    url, out = STAC.format(*C.AOI), []
    while url:
        j = get_json(url)
        out += j["features"]
        url = next((l["href"] for l in j.get("links", []) if l["rel"] == "next"), None)
    by = collections.defaultdict(list)
    for f in out:
        by[int(f["properties"]["datetime"][:4])].append(f)
    return by


def read_tile(href: str):
    url = href.replace("s3://deafrica-services/", S3)
    with rasterio.Env(**GDAL), rasterio.open(url) as ds:
        a = ds.read(1, out_shape=(ds.height // DEC, ds.width // DEC),
                    resampling=Resampling.nearest)
    return a.astype(np.int16)


def wofs_year(year: int, feats, grid, pool):
    cache = C.CACHE / "history" / f"{year}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["wet"], z["clear"]
    x0, y1, W, H = grid
    wet = np.full((H, W), -1, np.int16)
    clear = np.full((H, W), -1, np.int16)
    jobs = [(f, k) for f in feats for k in ("count_wet", "count_clear")]
    for (f, k), a in zip(jobs, pool.map(lambda j: read_tile(j[0]["assets"][j[1]]["href"]), jobs)):
        t = f["properties"]["proj:transform"]
        c0, r0 = round((t[2] - x0) / PIX), round((y1 - t[5]) / PIX)
        dst = wet if k == "count_wet" else clear
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + a.shape[0], H), min(c0 + a.shape[1], W)
        if re > rs and ce > cs:
            dst[rs:re, cs:ce] = a[rs - r0:re - r0, cs - c0:ce - c0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, wet=wet, clear=clear)
    return wet, clear


def run_wofs():
    grid = wofs_grid()
    x0, y1, W, H = grid
    tf = Affine(PIX, 0, x0, 0, -PIX, y1)
    out = C.OUT / "history"
    out.mkdir(parents=True, exist_ok=True)
    items = wofs_items()
    recs, bounds = [], None
    with ThreadPoolExecutor(8) as pool:
        for year in sorted(items):
            wet, clear = wofs_year(year, items[year], grid, pool)
            valid = clear > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                freq = np.where(valid, wet / np.maximum(clear, 1), 0)
            anyw = valid & (freq >= F_ANY) & (wet >= MIN_WET)
            perm = anyw & (freq >= F_PERM)
            cls = np.full((H, W), ND, np.uint8)
            cls[valid] = 0
            cls[anyw] = SEASONAL
            cls[perm] = PERMANENT
            med = float(np.median(clear[valid])) if valid.any() else 0.0
            vpct = 100 * valid.mean()
            recs.append({
                "year": year,
                "water_km2": round(float(anyw.sum()) * KM2, 1),
                "permanent_km2": round(float(perm.sum()) * KM2, 1),
                "seasonal_km2": round(float((anyw & ~perm).sum()) * KM2, 1),
                "valid_pct": round(vpct, 1),
                "median_clear_obs": med,
                # Few Landsat passes were downlinked over southern Africa
                # before ~2000. With few looks a seasonal flood is easily
                # missed, so those years are a LOWER bound, not a small flood.
                # 8 looks a year is still sparse: 2003 (7 looks, Landsat 7
                # scan-line failure) shows 184 km2 between two ~1400-2400 km2
                # years, which is a sampling artefact, not a drought.
                "low_coverage": bool(med < 8 or vpct < 95),
            })
            web, bounds = to_web(cls, tf, "EPSG:6933", ND)
            rgba = np.zeros((*web.shape, 4), np.uint8)
            for k, col in HCOL.items():
                rgba[web == k] = col
            Image.fromarray(rgba, "RGBA").save(out / f"{year}.png", optimize=True)
            Image.fromarray(rgba, "RGBA").save(out / f"{year}.webp", lossless=True, method=4)
            r = recs[-1]
            print(f"{year}  water {r['water_km2']:>7}  perm {r['permanent_km2']:>6}  "
                  f"valid {r['valid_pct']:>5}%  looks {med:>4.0f}"
                  f"{'  LOW COVERAGE' if r['low_coverage'] else ''}")
    (out / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    doc = {
        "source": "Digital Earth Africa, Water Observations from Space (WOfS) "
                  "annual summary, derived from USGS/NASA Landsat, 30 m",
        "sensor": "optical (Landsat 5/7/8/9)",
        "pixel_m": PIX,
        "method": f"Per calendar year. Water = wet in >= {int(F_ANY * 100)}% of "
                  f"cloud-free looks and at least {MIN_WET} times. Permanent = "
                  f"wet in >= {int(F_PERM * 100)}% of looks. Seasonal = water "
                  "minus permanent. Areas counted in equal-area EPSG:6933.",
        "caveat": "Optical sensors map OPEN water. Water under reeds, papyrus "
                  "and trees is largely missed, so these areas are not "
                  "comparable in level with the NISAR flooded-vegetation "
                  "class and must not be plotted as one series with NISAR.",
        "low_coverage_note": "Years flagged low_coverage have fewer than 8 "
                             "cloud-free Landsat looks; their area is a lower "
                             "bound. The flood lasts only a few months, so a "
                             "year with few looks can miss it entirely.",
        "reliable_since": 2013,
        "reliable_note": "From 2013 (Landsat 8) every year has 23+ looks. "
                         "Compare years confidently from 2013 on; treat "
                         "earlier year-to-year jumps with caution.",
        "legend": [
            {"class": "seasonal_water", "color": list(HCOL[SEASONAL]),
             "label": "Seasonal open water"},
            {"class": "permanent_water", "color": list(HCOL[PERMANENT]),
             "label": "Permanent open water"}],
        "overlay": "history/<year>.png with history/bounds.json",
        "years": recs,
    }
    (C.OUT / "history.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {C.OUT / 'history.json'} ({len(recs)} years)")


# --- JRC Global Surface Water v1.5 occurrence --------------------------------
JRC_URL = ("https://s3.waw4-1.cloudferro.com/swift/v1/global-surface-water/"
           "download2024/Aggregated/VER1-5/occurrence/"
           "occurrence_20E_10S_v1_5_2024.tif")
JRC_BIN = 3  # 0.00025 deg (~27 m) -> ~80 m, same as the NISAR overlays


def run_jrc():
    out = C.OUT / "jrc"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(**GDAL), rasterio.open(JRC_URL) as ds:
        win = from_bounds(*C.AOI, ds.transform).round_offsets().round_lengths()
        occ = ds.read(1, window=win)
        tf = ds.window_transform(win)
        print("JRC window", occ.shape, "nodata", ds.nodata, "dtype", occ.dtype)
    # occurrence: 0 = never water, 1..100 = % of valid looks, 255 = no data
    valid = occ <= 100
    lat = tf.f + tf.e * (np.arange(occ.shape[0]) + 0.5)
    row_km2 = (abs(tf.e) * 111.32) * (tf.a * 111.32 * np.cos(np.radians(lat)))
    area = lambda m: float((m.sum(axis=1) * row_km2).sum())
    stats = {
        "ever_water_km2": round(area(valid & (occ >= 1)), 1),
        "water_25pct_km2": round(area(valid & (occ >= 25)), 1),
        "water_75pct_km2": round(area(valid & (occ >= 75)), 1),
        "valid_km2": round(area(valid), 1),
    }
    h, w = (occ.shape[0] // JRC_BIN) * JRC_BIN, (occ.shape[1] // JRC_BIN) * JRC_BIN
    blk = lambda a: a[:h, :w].reshape(h // JRC_BIN, JRC_BIN, w // JRC_BIN, JRC_BIN)
    n = blk(valid).sum(axis=(1, 3))
    s = blk(np.where(valid, occ, 0).astype(np.float32)).sum(axis=(1, 3))
    small = np.where(n > 0, np.round(s / np.maximum(n, 1)), 255).astype(np.uint8)
    tf3 = tf * Affine.scale(JRC_BIN)
    web, bounds = to_web(small, tf3, "EPSG:4326", 255)
    # light -> dark blue with occurrence; never-water stays transparent
    t = np.clip(web.astype(np.float32) / 100.0, 0, 1)[..., None]
    lo, hi = np.array([158, 202, 225.0]), np.array([8, 48, 107.0])
    wet = (web >= 1) & (web <= 100)
    rgba = np.zeros((*web.shape, 4), np.uint8)
    rgba[..., :3] = np.where(wet[..., None], lo + (hi - lo) * t, 0).astype(np.uint8)
    rgba[..., 3] = np.where(wet, 220, 0)
    Image.fromarray(rgba, "RGBA").save(out / "occurrence.png", optimize=True)
    Image.fromarray(rgba, "RGBA").save(out / "occurrence.webp", lossless=True, method=4)
    bounds.update({
        "source": "JRC Global Surface Water v1.5 (European Commission Joint "
                  "Research Centre), occurrence 1984-2024, from Landsat",
        "meaning": "Share of cloud-free observations 1984-2024 in which the "
                   "pixel was open water. Pale = rarely, dark = almost always.",
        "color_ramp": {"1": [158, 202, 225], "100": [8, 48, 107]},
        "stats": stats,
        "caveat": "Optical, open water only. Not a NISAR product.",
    })
    (out / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print("JRC", stats)


# --- drivers: why the water changes ------------------------------------------
MOHEMBO = (-18.275, 21.787)  # where the Okavango River enters the Delta
# Headwaters of the Cubango and Cuito rivers in the Angolan highlands.
RAIN_PTS = {"Huambo (Cubango source)": (-12.77, 15.74),
            "Menongue": (-14.66, 17.69),
            "Cuito Cuanavale": (-15.16, 19.17)}


def era5_monthly_rain(lat: float, lon: float) -> dict[str, float]:
    """Monthly mean rainfall in mm/day at one point, 1983-11 onwards, from the
    ERA5 reanalysis through the Open-Meteo archive API (no account needed).
    Only complete months are returned."""
    import time
    import urllib.error
    # 43 years of daily values is a heavy request for a free API, which answers
    # 429 after a few of them. Keep each point on disk and wait out the limit.
    cache = C.CACHE / f"era5_rain_{lat}_{lon}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
    else:
        url = ("https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}&start_date=1983-11-01"
               "&end_date=2026-10-31&daily=precipitation_sum&timezone=GMT")
        for attempt in range(6):
            try:
                d = get_json(url)["daily"]
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                print("  ERA5 rate limit, waiting 65 s ...", flush=True)
                time.sleep(65)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(d))
    tot, n = collections.defaultdict(float), collections.Counter()
    for t, v in zip(d["time"], d["precipitation_sum"]):
        if v is not None:
            tot[t[:7]] += v
            n[t[:7]] += 1
    return {m: tot[m] / n[m] for m in sorted(tot) if n[m] >= 27}


def run_drivers():
    d = get_json("https://flood-api.open-meteo.com/v1/flood?latitude={}&longitude={}"
                 "&daily=river_discharge&start_date=1984-01-01&end_date=2026-11-13"
                 .format(*MOHEMBO))["daily"]
    m = collections.defaultdict(list)
    for t, q in zip(d["time"], d["river_discharge"]):
        if q is not None:
            m[t[:7]].append(q)
    discharge = [{"month": k, "m3s": round(sum(v) / len(v), 1)} for k, v in sorted(m.items())]

    # Rainfall from ERA5, not NASA POWER. MEASURED: POWER's corrected MERRA-2
    # precipitation has spikes over Angola (March 2020 reads 82 mm/day at
    # Huambo, i.e. 2500 mm in one month; ERA5's wettest month in 43 years is
    # 383 mm) and its 2026 season contradicts the river model. ERA5 is steady.
    rain = collections.defaultdict(list)
    for name, (la, lo) in RAIN_PTS.items():
        for month, mm_day in era5_monthly_rain(la, lo).items():
            rain[month].append(mm_day)
        print("ERA5", name, "ok")
    rainfall = [{"month": k, "mm_day": round(sum(v) / len(v), 2)}
                for k, v in sorted(rain.items()) if len(v) == len(RAIN_PTS)]

    doc = {
        "discharge": {
            "source": "GloFAS river discharge reanalysis via Open-Meteo Flood API",
            "location": {"name": "Okavango River near Mohembo (Delta inflow)",
                         "lat": MOHEMBO[0], "lon": MOHEMBO[1]},
            "caveat": "Hydrological MODEL output, not a gauge measurement. "
                      "Trust timing and year-to-year ranking more than exact m3/s.",
            "monthly": discharge},
        "rainfall": {
            "source": "ERA5 reanalysis (Copernicus Climate Change Service / ECMWF) "
                      "via the Open-Meteo archive API, monthly mean",
            "points": {k: {"lat": v[0], "lon": v[1]} for k, v in RAIN_PTS.items()},
            "caveat": "Mean of three points in the Angolan headwaters, a proxy "
                      "for catchment rainfall, not a basin average.",
            "monthly": rainfall},
    }
    (C.OUT / "drivers.json").write_text(json.dumps(doc), encoding="utf-8")
    print(f"drivers: {len(discharge)} discharge months "
          f"({discharge[0]['month']}..{discharge[-1]['month']}), "
          f"{len(rainfall)} rainfall months "
          f"({rainfall[0]['month']}..{rainfall[-1]['month']})")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("drivers", "all"):
        run_drivers()
    if what in ("jrc", "all"):
        run_jrc()
    if what in ("wofs", "all"):
        run_wofs()
