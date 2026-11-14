"""
EDev-Vision | NASA Space Apps 2026 | Dancing with the SARs
Shared configuration for the Okavango flood-pulse pipeline.

Language: Python. Chosen because the only mature readers for NISAR HDF5
(h5py), NASA streaming auth (earthaccess) and geospatial reprojection
(rasterio/pyproj) are Python. No other language has this stack.

Everything that the fetch and render stages must agree on lives here, so a
threshold or a date list is changed in exactly one place.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
QA = ROOT / "qa"
OUT = ROOT / "edev-data" / "data"
GRANULE_CSV = ROOT / "granules_list.csv"


def cache_dir(freq: str = "A"):
    """One cache folder per frequency band, so a fast frequency-B run and a
    high-quality frequency-A run can coexist and be compared."""
    return CACHE / freq

# --- area of interest -------------------------------------------------------
# lon_min, lat_min, lon_max, lat_max. Panhandle (north) to Maun (south-east).
AOI = (22.0, -20.0, 23.8, -18.3)

# --- the series -------------------------------------------------------------
# Track 172, ascending, frame 168, DHDH (HH + HV).
# Same track + frame + flight direction is MANDATORY: otherwise a difference
# in incidence angle shows up as fake "change". See 01_proje_baglami.md.
TRACK, FRAME, DIRECTION, POL = "172", "168", "A", "DHDH"

# BETA data (Nov 2025 - Jan 2026) is pre-calibration and is excluded from the
# main series. The remaining collections are usable.
QUALITY = {
    "NISAR_L2_GCOV_PROVISIONAL_V1": "PROVISIONAL",
    "NISAR_L2_GCOV_V1": "VALIDATED",
}

# 2026-09-06 exists twice (two processing runs, ..._001 and ..._002).
# Prefer the later run.
PREFER_LATEST_VERSION = True

# Permanent NISAR data gap. The frontend must show this as "no data", not as
# "no flood".
GAPS = [("2026-07-27", "2026-08-10")]

# --- HDF5 layout (verified against gcov_structure.txt) ----------------------
IDENT = "science/LSAR/identification"


def paths(freq: str = "A") -> dict:
    """HDF5 paths for one frequency band.

    frequency A is the 20 MHz band at 20 m - the science band.
    frequency B is the 5 MHz band, already gridded at 80 m. It is the same
    acquisition and the same geometry, just a narrower bandwidth, so it is a
    legitimate fallback when bandwidth runs out: 16x less data for the same
    80 m output. The cost is a coarser native range resolution, i.e. more
    speckle and a slightly higher noise floor.
    """
    g = f"science/LSAR/GCOV/grids/frequency{freq}"
    return {
        "grid": g,
        # HHHH / HVHV are diagonal covariance terms = linear gamma-naught power
        "bands": {"hh": f"{g}/HHHH", "hv": f"{g}/HVHV"},
        "mask": f"{g}/mask",
        "projection": f"{g}/projection",
        "pols": f"{g}/listOfPolarizations",
        "x": f"{g}/xCoordinates",
        "y": f"{g}/yCoordinates",
    }
# Fallback only. The real EPSG is read from the product: frame 168 is UTM 34S
# (32734) because the Delta sits at 22-24 E, west of the 24 E zone boundary.
# Never hardcode this - a neighbouring frame can be in another zone.
SRC_EPSG = 32734
WEB_EPSG = 3857  # Web Mercator, what Leaflet/MapLibre overlays need

# --- processing -------------------------------------------------------------
# Every output is 80 m, whichever band it came from, so the two are
# interchangeable in the published product.
OUT_PIXEL_M = 80.0
# Native pixel size and the block-average factor that takes it to 80 m.
# Averaging in LINEAR power (not in dB) is the physically correct multilook: it
# reduces speckle without biasing the mean, which averaging dB would do.
NATIVE_M = {"A": 20.0, "B": 80.0}
LOOKS_BY_FREQ = {"A": 4, "B": 1}
PIXEL_M = NATIVE_M["A"]
LOOKS = LOOKS_BY_FREQ["A"]
# Area of one output pixel in km^2, computed in UTM where area is true.
PIXEL_KM2 = (OUT_PIXEL_M / 1000.0) ** 2

# Read the HDF5 window in horizontal stripes so peak memory stays small and
# streaming stays sequential. The HDF5 chunks are 512 rows tall, so a large
# stripe wastes less: a 1024-row read still has to pull 3 chunk rows, while a
# 2048-row read pulls 5 for twice the data.
STRIPE_ROWS = 2048

# MEASURED, not assumed. One thread sustained ~1.45 MB/s; three threads
# sustained ~0.83 MB/s IN TOTAL - parallel range requests make the DAAC slower,
# not faster. So we stream one date at a time.
WORKERS = 1

# --- classification thresholds (dB) ----------------------------------------
# UNCERTAIN. These are starting values from radar physics, to be tuned on the
# 2026-07-20 histogram and checked against Sentinel-2. They are published in
# meta.json with an "uncertain" flag so the app can say so out loud.
#
#   open water        : calm water is a mirror -> very little energy comes back
#   flooded vegetation: HH double-bounce off trunks/stems in standing water ->
#                       bright in HH, and much brighter in HH than in HV
#                       because double-bounce preserves polarization
# Noise floor. MEASURED, not guessed: ~2.46% of pixels sit below -40 dB, the
# figure barely moves between June and October, and 98.4% of them are the SAME
# pixels on every date, packed into one vertical strip at the frame's western
# edge. Real water moves with the season; this does not. They are fill, not
# backscatter, and at -18 dB they would have been counted as ~1500 km2 of
# phantom open water. Anything below this is marked invalid, never water.
# NISAR's noise-equivalent backscatter is around -25 dB, so -30 dB is safely
# below any real surface return while still clearing the fill.
NOISE_FLOOR_DB = -30.0

THRESH = {
    "water_hh_max": -18.0,
    "water_hv_max": -22.0,
    "fv_hh_min": -5.0,
    "fv_hhhv_min": 7.0,
}

# Fixed dB stretch for the grayscale HH image, identical on every date so the
# animation shows real brightness change and not a per-date rescaling.
DB_STRETCH = (-25.0, 0.0)

# RGBA colors for the class PNG. Non-flooded stays fully transparent so the
# basemap shows through.
COLORS = {
    "none": (0, 0, 0, 0),
    "water": (33, 113, 181, 210),  # blue
    "flooded_veg": (49, 163, 84, 200),  # green
}

# --- auth -------------------------------------------------------------------


def login():
    """Earthdata Login. Credentials are read from the environment only; they
    are never written to a file in this repo."""
    import earthaccess

    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return earthaccess.login(strategy="environment")
    return earthaccess.login(strategy="interactive")


# --- the date -> granule table ---------------------------------------------


def series() -> list[dict]:
    """Read granules_list.csv and return the frame-168 time series, sorted by
    date, one granule per date."""
    by_date: dict[str, dict] = {}
    with open(GRANULE_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["track"], r["frame"], r["direction"], r["pol"]) != (
                TRACK,
                FRAME,
                DIRECTION,
                POL,
            ):
                continue
            if r["collection"] not in QUALITY:
                continue  # drops BETA
            s = r["start"]
            date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
            row = {
                "date": date,
                "granule": r["granule"],
                "collection": r["collection"],
                "quality": QUALITY[r["collection"]],
                "version": r["granule"].rsplit("_", 1)[-1],
                "gb": float(r["gb"]) if r["gb"] else None,
            }
            prev = by_date.get(date)
            if prev is None or (
                PREFER_LATEST_VERSION and row["version"] > prev["version"]
            ):
                by_date[date] = row
    return [by_date[d] for d in sorted(by_date)]


DATES = [r["date"] for r in series()]

if __name__ == "__main__":
    for r in series():
        print(f"{r['date']}  {r['quality']:<12} v{r['version']}  {r['gb']} GB")
    print(f"\n{len(DATES)} dates, output pixel {OUT_PIXEL_M:.0f} m "
          f"= {PIXEL_KM2:.4f} km2")
