"""
EDev-Vision | Stage 2 of 2: cached arrays -> PNG overlays + JSON for the app.

Runs offline from cache/*.npz, so changing a threshold costs seconds, not a
re-download. Outputs exactly the agreed backend -> frontend contract:

    data/<YYYY-MM-DD>/class.png    RGBA classification overlay (EPSG:3857)
    data/<YYYY-MM-DD>/hh.png       grayscale HH dB, the raw radar look
    data/<YYYY-MM-DD>/bounds.json  geographic bounds of those PNGs
    data/timeseries.json           one record per date, areas in km2
    data/meta.json                 legend, thresholds, data gaps, caveats

Why PNG + bounds instead of GeoTIFF or tiles: a browser can draw a PNG as a
map overlay with no server and no GDAL. Tiles would be the right answer for a
production app with many zoom levels; for one AOI and 10 dates they are weeks
of work we do not have.

Usage:
    python pipeline/render.py                 # every cached date
    python pipeline/render.py --date 2026-07-20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

NONE, WATER, FLOODED_VEG = 0, 1, 2
NODATA = 255


def to_db(lin: np.ndarray) -> np.ndarray:
    """Linear gamma-naught power -> dB. HHHH/HVHV are diagonal covariance
    terms, i.e. power, so the conversion is 10*log10 (not 20*log10, which is
    for amplitude).

    Everything below the noise floor becomes NaN, i.e. invalid. See
    config.NOISE_FLOOR_DB: these pixels are fill, and left alone they would be
    classified as open water because they are dark.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10.0 * np.log10(np.where(lin > 0, lin, np.nan))
    return np.where(db < C.NOISE_FLOOR_DB, np.nan, db).astype(np.float32)


def classify(hh_db: np.ndarray, hv_db: np.ndarray | None) -> np.ndarray:
    """Three-class flood map. UNCERTAIN - see config.THRESH.

    Physics behind the rules (01_proje_baglami.md):
      open water         calm water reflects the pulse away from the radar like
                         a mirror, so almost nothing returns -> very low HH.
      flooded vegetation the pulse bounces off the water surface and then off
                         trunks/stems (double bounce) and comes straight back
                         -> very high HH. Double bounce keeps the polarization,
                         so HH rises much more than HV; the HH-HV gap is what
                         separates flooded forest from merely rough dry ground.
    """
    t = C.THRESH
    out = np.full(hh_db.shape, NODATA, np.uint8)
    valid = np.isfinite(hh_db)
    if hv_db is not None:
        valid &= np.isfinite(hv_db)
    out[valid] = NONE

    water = valid & (hh_db < t["water_hh_max"])
    if hv_db is not None:
        water &= hv_db < t["water_hv_max"]

    fv = valid & (hh_db > t["fv_hh_min"])
    if hv_db is not None:
        fv &= (hh_db - hv_db) > t["fv_hhhv_min"]

    out[fv] = FLOODED_VEG
    out[water] = WATER  # water wins: a dark pixel cannot be double-bounce
    return out


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """k x k moving average that ignores NaN. Summed-area table, numpy only."""
    r = k // 2
    v = np.isfinite(a)

    def sat(m):
        m = np.pad(m, ((r + 1, r), (r + 1, r)))
        s = m.cumsum(0).cumsum(1)
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

    n = sat(v.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, sat(np.where(v, a, 0).astype(np.float64)) / n,
                        np.nan).astype(np.float32)


DESPECKLE_WIN = 5


def despeckle(files) -> dict | None:
    """Multitemporal speckle filter (Quegan & Yu 2001) over the whole series.

    Speckle is the grainy noise every radar image has: a pixel's brightness is
    a random draw around the true value. One date alone can only be cleaned by
    blurring it, and blurring erases the Delta's thin channels (a plain 5x5
    mean removed 70% of the mapped open water in our test).

    This filter uses the other dates instead. For each date k:
        J_k = <I_k> * mean over dates i of ( I_i / <I_i> )
    <.> is a local 5x5 mean. I_i / <I_i> is "pure speckle texture" with the
    seasonal brightness divided out, so averaging it over 10 dates cancels the
    noise while every date keeps its own radiometry <I_k>. Sharp, stable edges
    (channel banks) survive because they are present in every date's ratio.

    It needs every date on the same grid, which the single track/frame series
    guarantees. Returns {date: (hh_lin, hv_lin)} or None if it cannot run.
    """
    hh, hv, key, grid = [], [], [], set()
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            if "hv_lin" not in z:
                return None
            hh.append(z["hh_lin"].astype(np.float32))
            hv.append(z["hv_lin"].astype(np.float32))
            key.append(str(z["date"]))
            grid.add((z["hh_lin"].shape, tuple(np.round(z["bbox_utm"], 1))))
    if len(files) < 3 or len(grid) != 1:
        return None
    hh, hv = np.stack(hh), np.stack(hv)
    floor = 10 ** (C.NOISE_FLOOR_DB / 10)
    bad = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > floor) & (hv > floor))
    out = []
    for s in (hh, hv):
        s[bad] = np.nan
        loc = np.stack([_box(x, DESPECKLE_WIN) for x in s])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = s / loc
        n = np.isfinite(ratio).sum(axis=0)
        ratio = np.where(n > 0, np.nansum(ratio, axis=0) / np.maximum(n, 1), np.nan)
        f = (loc * ratio).astype(np.float32)
        f[bad] = np.nan
        out.append(f)
    return {d: (out[0][i], out[1][i]) for i, d in enumerate(key)}


def sieve(cls: np.ndarray) -> np.ndarray:
    """Drop classified pixels that touch no other pixel of their own class
    (8 neighbours). A lone 80 m pixel of "water" in dry savanna is far more
    likely to be leftover speckle than a 0.6 ha pond; real channels are
    connected and are kept."""
    out = cls.copy()
    h, w = cls.shape
    for k in (WATER, FLOODED_VEG):
        m = cls == k
        p = np.pad(m, 1)
        n = np.zeros((h, w), np.uint8)
        for dy in range(3):
            for dx in range(3):
                if (dy, dx) != (1, 1):
                    n += p[dy:dy + h, dx:dx + w]
        out[m & (n == 0)] = NONE
    return out


def warp(src: np.ndarray, bbox, epsg: int, nodata, resampling):
    """Reproject an array from the product CRS to Web Mercator.

    Why EPSG:3857: Leaflet and MapLibre stretch an image overlay linearly in
    Web Mercator. Handing them a UTM image with lat/lon bounds would shear it
    by hundreds of metres at the edges.
    """
    h, w = src.shape
    xmin, ymin, xmax, ymax = bbox
    src_tf = Affine(C.OUT_PIXEL_M, 0, xmin, 0, -C.OUT_PIXEL_M, ymax)
    dst_tf, dw, dh = calculate_default_transform(
        f"EPSG:{epsg}", f"EPSG:{C.WEB_EPSG}", w, h, xmin, ymin, xmax, ymax
    )
    dst = np.full((dh, dw), nodata, src.dtype)
    reproject(
        src, dst,
        src_transform=src_tf, src_crs=f"EPSG:{epsg}", src_nodata=nodata,
        dst_transform=dst_tf, dst_crs=f"EPSG:{C.WEB_EPSG}", dst_nodata=nodata,
        resampling=resampling,
    )
    return dst, dst_tf, (dw, dh)


def bounds_latlon(dst_tf, size) -> dict:
    w, h = size
    xmin, ymin, xmax, ymax = array_bounds(h, w, dst_tf)
    tf = Transformer.from_crs(C.WEB_EPSG, 4326, always_xy=True)
    (lon_w, lon_e), (lat_s, lat_n) = tf.transform([xmin, xmax], [ymin, ymax])
    return {
        # Leaflet: L.imageOverlay(url, bounds) wants [[south, west], [north, east]]
        "bounds": [[lat_s, lon_w], [lat_n, lon_e]],
        # MapLibre: image source "coordinates", clockwise from top-left
        "coordinates": [
            [lon_w, lat_n], [lon_e, lat_n], [lon_e, lat_s], [lon_w, lat_s]
        ],
        "crs": f"EPSG:{C.WEB_EPSG}",
        "width": int(w),
        "height": int(h),
    }


def render_one(path: Path, clean: dict | None = None) -> dict:
    with np.load(path, allow_pickle=False) as z:
        hh_db = to_db(z["hh_lin"])
        hv_db = to_db(z["hv_lin"]) if "hv_lin" in z else None
        bbox = tuple(z["bbox_utm"])
        epsg = int(z["epsg"])
        date = str(z["date"])
        granule = str(z["granule"])
        quality = str(z["quality"])
        partial = bool(z["partial"])

    # Classify the despeckled backscatter, then drop lone pixels. hh.png below
    # still shows the UNFILTERED image: it is "what the radar saw".
    if clean and date in clean:
        cls = sieve(classify(to_db(clean[date][0]), to_db(clean[date][1])))
    else:
        cls = classify(hh_db, hv_db)
    # Frame-edge fade-out is dark but is not water: see config.EDGE_BUFFER_PX.
    if C.EDGE_BUFFER_PX:
        k = 2 * C.EDGE_BUFFER_PX + 1
        cls[_box((cls == NODATA).astype(np.float32), k) > 0] = NODATA

    # --- areas: counted in UTM, BEFORE reprojection -------------------------
    # Web Mercator inflates area by 1/cos(lat)^2, which at 19 S would be a ~12%
    # error. UTM 34S is an equal-ish-area local projection, so the count of
    # 80 m pixels times 0.0064 km2 is the honest number.
    n_water = int((cls == WATER).sum())
    n_fv = int((cls == FLOODED_VEG).sum())
    n_valid = int((cls != NODATA).sum())
    rec = {
        "date": date,
        "flooded_km2": round((n_water + n_fv) * C.PIXEL_KM2, 1),
        "open_water_km2": round(n_water * C.PIXEL_KM2, 1),
        "flooded_veg_km2": round(n_fv * C.PIXEL_KM2, 1),
        "valid_km2": round(n_valid * C.PIXEL_KM2, 1),
        "quality": quality,
        "granule": granule,
    }

    out_dir = C.OUT / date
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- class overlay: nearest neighbour, then colorize --------------------
    # Reprojecting class indices with anything but nearest would invent classes
    # that do not exist (e.g. 0.5 between none and water).
    cls_w, dst_tf, size = warp(cls, bbox, epsg, NODATA, Resampling.nearest)
    rgba = np.zeros((*cls_w.shape, 4), np.uint8)
    rgba[cls_w == WATER] = C.COLORS["water"]
    rgba[cls_w == FLOODED_VEG] = C.COLORS["flooded_veg"]
    Image.fromarray(rgba, "RGBA").save(out_dir / "class.png", optimize=True)

    # --- HH dB grayscale: the un-interpreted radar image --------------------
    # Fixed stretch across all dates, otherwise a brighter date would look the
    # same as a darker one and the animation would lie.
    hh_w, _, _ = warp(hh_db, bbox, epsg, np.nan, Resampling.bilinear)
    lo, hi = C.DB_STRETCH
    g = np.clip((hh_w - lo) / (hi - lo), 0, 1)
    # Mode "LA" = 8-bit gray + alpha. Storing a gray image as RGBA would
    # triple the luminance channel for nothing; at ~5 MB per date that is the
    # difference between a date slider that feels instant and one that stalls.
    gray = np.zeros((*hh_w.shape, 2), np.uint8)
    gray[..., 0] = (np.nan_to_num(g) * 255).astype(np.uint8)
    gray[..., 1] = np.where(np.isfinite(hh_w), 255, 0)
    Image.fromarray(gray, "LA").save(out_dir / "hh.png", optimize=True)

    b = bounds_latlon(dst_tf, size)
    b.update(date=date, granule=granule)
    (out_dir / "bounds.json").write_text(json.dumps(b, indent=2), encoding="utf-8")

    flag = "  [PARTIAL CACHE]" if partial else ""
    print(f"{date}  {size[0]}x{size[1]}px  water {rec['open_water_km2']:>7} "
          f"fv {rec['flooded_veg_km2']:>7} total {rec['flooded_km2']:>7} km2  "
          f"valid {100 * n_valid / cls.size:.0f}%  {quality}{flag}")
    rec["partial"] = partial
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--freq", choices=["A", "B"], default="A")
    ap.add_argument("--raw", action="store_true",
                    help="skip the speckle filter and the lone-pixel sieve")
    a = ap.parse_args()

    files = sorted(C.cache_dir(a.freq).glob("*.npz"))
    if not files:
        raise SystemExit(f"no cache in {C.cache_dir(a.freq)} - run fetch.py first")
    # The filter needs the WHOLE series even when one date is rendered.
    clean = None if a.raw else despeckle(files)
    if not a.raw and clean is None:
        print("speckle filter skipped: dates are not on one common grid")
    if a.date:
        files = [p for p in files if p.stem == a.date]
        if not files:
            raise SystemExit(f"no cached date {a.date}")

    C.OUT.mkdir(parents=True, exist_ok=True)
    recs = [render_one(p, clean) for p in files]

    # timeseries.json keeps the agreed shape; "partial" is dropped so the
    # published file only ever carries finished dates.
    series = [{k: v for k, v in r.items() if k != "partial"}
              for r in recs if not r["partial"]]
    (C.OUT / "timeseries.json").write_text(
        json.dumps(series, indent=2), encoding="utf-8")

    meta = {
        "aoi": {"lon_min": C.AOI[0], "lat_min": C.AOI[1],
                "lon_max": C.AOI[2], "lat_max": C.AOI[3]},
        "mission": "NISAR (NASA-ISRO SAR)",
        "product": "L2 GCOV, L-band",
        "band_note": (
            "Frequency B (5 MHz, natively 80 m) was used. It is the same "
            "acquisition and geometry as the 20 MHz frequency A band at 1/16 "
            "the data volume; measured dB percentiles agree with frequency A "
            "to within a few tenths of a dB. Costs: more speckle, and NISAR's "
            "known-issues page reports larger systematic radiometric "
            "calibration residuals for frequency B than for frequency A."
            if a.freq == "B" else
            "Frequency A (20 MHz, natively 20 m) was used, averaged 4x4 in "
            "linear power to 80 m. This is NISAR's primary science band."),
        "despeckle": (
            {"applied": True,
             "method": "Multitemporal speckle filter (Quegan & Yu 2001), "
                       f"{DESPECKLE_WIN}x{DESPECKLE_WIN} local window, over "
                       f"all {len(clean)} dates; then classified pixels with "
                       "no same-class neighbour are removed.",
             "why": "Radar images are grainy (speckle). Unfiltered, about one "
                    "in six mapped pixels stood alone, which is noise, not "
                    "ponds. The filter uses the other dates to clean each "
                    "date without blurring thin channels.",
             "effect": "Mapped areas are smaller than unfiltered ones. The "
                       "dates of the two seasonal peaks do not change.",
             "note": "class.png and timeseries.json are filtered; hh.png is "
                     "the unfiltered radar image."}
            if clean else {"applied": False}),
        "polarizations": ["HH", "HV"],
        "geometry": {"track": C.TRACK, "frame": C.FRAME,
                     "pass": "Ascending", "note":
                     "All dates share one track/frame/pass, so differences "
                     "between dates are real surface change and not a change "
                     "in viewing geometry."},
        "pixel_m": C.OUT_PIXEL_M,
        # Must describe the band that was actually rendered, not frequency A.
        "frequency": a.freq,
        "native_pixel_m": C.NATIVE_M[a.freq],
        "looks": C.LOOKS_BY_FREQ[a.freq],
        "dates": [r["date"] for r in series],
        "gaps": [list(g) for g in C.GAPS],
        "gap_note": "NISAR has a permanent data gap in this window. Show it "
                    "as 'no data', not as 'no flood'.",
        "thresholds_db": C.THRESH,
        "noise_floor_db": C.NOISE_FLOOR_DB,
        "edge_buffer": {
            "pixels": C.EDGE_BUFFER_PX,
            "why": "The radar image fades to dark at the edge of the frame. "
                   "Dark is how open water looks, so the edge was being "
                   "mapped as a straight line of water. Pixels within "
                   f"{C.EDGE_BUFFER_PX} pixels "
                   f"({C.EDGE_BUFFER_PX * C.OUT_PIXEL_M / 1000:.2f} km) of the "
                   "frame edge are now treated as no data."},
        "uncertain": True,
        "uncertainty_note": "Backscatter thresholds are fixed, physically "
                            "motivated values, identical on every date. They "
                            "are not calibrated against ground measurements, "
                            "so treat the areas as a consistent relative "
                            "index of flooding rather than an absolute "
                            "hydrological measurement.",
        "conservative_note": "These thresholds are deliberately strict, so "
                             "the mapped area is a lower bound. Published "
                             "estimates of the Okavango's total seasonal "
                             "flood extent are several times larger; part of "
                             "that difference is real (the area of interest "
                             "and one radar frame do not cover the whole "
                             "delta) and part is the strict thresholds. The "
                             "shape of the seasonal curve is more reliable "
                             "than its absolute level.",
        "quality_note": "Dates up to 2026-10-12 are NISAR PROVISIONAL "
                        "products; 2026-10-24 is VALIDATED. A small "
                        "calibration offset between the two is possible.",
        "legend": [
            {"class": "open_water", "color": list(C.COLORS["water"]),
             "label": "Open water",
             "why": "Calm water mirrors the radar pulse away, so very little "
                    "energy returns: HH is very low."},
            {"class": "flooded_vegetation",
             "color": list(C.COLORS["flooded_veg"]),
             "label": "Flooded vegetation",
             "why": "The pulse bounces off the water and then off trunks and "
                    "stems and returns to the radar (double bounce): HH is "
                    "very high and much higher than HV. L-band sees this "
                    "under the canopy; C-band largely cannot."},
        ],
        "db_stretch": list(C.DB_STRETCH),
    }
    (C.OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{len(series)} complete date(s) -> {C.OUT}")
    if any(r["partial"] for r in recs):
        print("partial dates were rendered for inspection but kept OUT of "
              "timeseries.json")


if __name__ == "__main__":
    main()
