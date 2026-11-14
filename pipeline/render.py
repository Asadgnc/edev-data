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


def render_one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        hh_db = to_db(z["hh_lin"])
        hv_db = to_db(z["hv_lin"]) if "hv_lin" in z else None
        bbox = tuple(z["bbox_utm"])
        epsg = int(z["epsg"])
        date = str(z["date"])
        granule = str(z["granule"])
        quality = str(z["quality"])
        partial = bool(z["partial"])

    cls = classify(hh_db, hv_db)

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
    a = ap.parse_args()

    files = sorted(C.cache_dir(a.freq).glob("*.npz"))
    if a.date:
        files = [p for p in files if p.stem == a.date]
    if not files:
        raise SystemExit(f"no cache in {C.cache_dir(a.freq)} - run fetch.py first")

    C.OUT.mkdir(parents=True, exist_ok=True)
    recs = [render_one(p) for p in files]

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
        "band_note": "Frequency B (5 MHz, natively 80 m) was used. It is the "
                     "same acquisition, geometry and calibration as the "
                     "20 MHz frequency A band, at 1/16 the data volume; "
                     "measured dB percentiles agree with frequency A to "
                     "within a few tenths of a dB. The cost is a coarser "
                     "native range resolution, so slightly more speckle.",
        "polarizations": ["HH", "HV"],
        "geometry": {"track": C.TRACK, "frame": C.FRAME,
                     "pass": "Ascending", "note":
                     "All dates share one track/frame/pass, so differences "
                     "between dates are real surface change and not a change "
                     "in viewing geometry."},
        "pixel_m": C.OUT_PIXEL_M,
        "native_pixel_m": C.PIXEL_M,
        "looks": C.LOOKS,
        "dates": [r["date"] for r in series],
        "gaps": [list(g) for g in C.GAPS],
        "gap_note": "NISAR has a permanent data gap in this window. Show it "
                    "as 'no data', not as 'no flood'.",
        "thresholds_db": C.THRESH,
        "noise_floor_db": C.NOISE_FLOOR_DB,
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
