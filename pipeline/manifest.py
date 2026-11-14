"""
EDev-Vision | One manifest for the frontend, one credits file for the rules.

  layers.json    every map layer and chart the app can show. The frontend loops
                 over it; adding a layer here needs no frontend change.
  sources.json   every dataset and tool we used, with provider, licence, link
                 and which submission-form field it belongs in.
  SOURCES.md     the same credits as text, ready to paste into the Space Apps
                 project page (NASA Data / Space Agency Partner Data &
                 References / Use of AI).

Some layers are NOT ours: they are map tiles served live by NASA GIBS, the
service behind NASA Worldview. Nothing is downloaded or processed for them; the
browser asks NASA for the picture. This script only checks, for one of them,
which days actually have data over the Delta, so the app does not offer empty
days.

Usage:  python pipeline/manifest.py
"""
from __future__ import annotations

import io
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

DATA_ROOT = "https://asadgnc.github.io/edev-data/data"
ASSISTANT = "https://edev-vision-assistant.vercel.app/api/chat"
GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
load = lambda name: json.loads((C.OUT / name).read_text(encoding="utf-8"))


def gibs(layer: str, matrix: str, ext: str) -> str:
    # {z}/{y}/{x}: GIBS puts the row before the column, unlike most tile servers.
    return f"{GIBS}/{layer}/default/{{time}}/{matrix}/{{z}}/{{y}}/{{x}}.{ext}"


# --- which days does NASA's water product cover the Delta? ------------------
def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def dswx_coverage(days: list[str], z: int = 8) -> dict[str, float]:
    """Share of the AOI with a clear observation in OPERA DSWx-HLS, per day.
    The product exists only where Landsat/Sentinel-2 passed that day, and grey
    pixels are cloud. Read from 4 small preview tiles per day."""
    cache = C.CACHE / "gibs_dswx_coverage.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    x0, y0 = _tile_xy(C.AOI[0], C.AOI[3], z)  # north-west corner
    x1, y1 = _tile_xy(C.AOI[2], C.AOI[1], z)  # south-east corner
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    px = (slice(round((y0 - ys[0]) * 256), round((y1 - ys[0]) * 256)),
          slice(round((x0 - xs[0]) * 256), round((x1 - xs[0]) * 256)))
    url = gibs("OPERA_L3_Dynamic_Surface_Water_Extent-HLS",
               "GoogleMapsCompatible_Level12", "png")

    def one(day):
        mos = np.zeros((len(ys) * 256, len(xs) * 256, 4), np.uint8)
        for j, ty in enumerate(ys):
            for i, tx in enumerate(xs):
                u = url.format(time=day, z=z, y=ty, x=tx)
                try:
                    with urllib.request.urlopen(u, timeout=40) as r:
                        t = np.array(Image.open(io.BytesIO(r.read())).convert("RGBA"))
                    mos[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = t
                except Exception:
                    pass  # a missing tile is "no data"
        a = mos[px]
        grey = (a[..., 0] == a[..., 1]) & (a[..., 1] == a[..., 2]) & (a[..., 0] < 250)
        return day, round(float(((a[..., 3] > 0) & ~grey).mean()), 3)

    todo = [d for d in days if d not in done]
    with ThreadPoolExecutor(12) as pool:
        for day, cov in pool.map(one, todo):
            done[day] = cov
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(done))
    return {d: done[d] for d in days}


# --- credits -----------------------------------------------------------------
# field = where it goes on the Space Apps project page.
SOURCES = [
    {"id": "nisar", "field": "nasa_data",
     "name": "NISAR L2 Geocoded Polarimetric Covariance (GCOV), L-band",
     "provider": "NASA / ISRO, distributed by the Alaska Satellite Facility DAAC",
     "url": "https://search.asf.alaska.edu/",
     "docs": "https://nisar-docs.asf.alaska.edu/gcov/",
     "license": "Open data (NASA Earth Science Data policy)",
     "used_for": "The core of the project: HH and HV backscatter for 10 dates "
                 "(track 172, frame 168, ascending), classified into open "
                 "water and flooded vegetation.",
     "processing": "Streamed over the AOI only, converted to dB, despeckled "
                   "(Quegan & Yu 2001), thresholded. See meta.json."},
    {"id": "gibs", "field": "nasa_data",
     "name": "NASA Global Imagery Browse Services (GIBS): NISAR GCOV browse "
             "imagery, OPERA Dynamic Surface Water Extent (DSWx-HLS), MODIS "
             "Aqua corrected reflectance",
     "provider": "NASA EOSDIS (OPERA products by NASA JPL)",
     "url": "https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api",
     "license": "Open data; NASA asks for acknowledgement of GIBS",
     "used_for": "Live reference map layers, shown unmodified next to our own "
                 "classification.",
     "processing": "None. Tiles are requested from NASA by the browser.",
     "acknowledgement": "We acknowledge the use of imagery provided by "
                        "services from NASA's Global Imagery Browse Services "
                        "(GIBS), part of NASA's Earth Science Data and "
                        "Information System (ESDIS)."},
    {"id": "power", "field": "nasa_data",
     "name": "NASA POWER (MERRA-2) monthly precipitation, PRECTOTCORR",
     "provider": "NASA Langley Research Center",
     "url": "https://power.larc.nasa.gov/",
     "license": "Open data",
     "used_for": "Rainfall in the Angolan headwaters that feeds the flood.",
     "processing": "Mean of three points; Nov-Apr totals per year."},
    {"id": "wofs", "field": "references",
     "name": "Water Observations from Space (WOfS) annual summary, from "
             "USGS/NASA Landsat",
     "provider": "Digital Earth Africa",
     "url": "https://docs.digitalearthafrica.org/en/latest/data_specs/Landsat_WOfS_specs.html",
     "license": "CC BY 4.0",
     "used_for": "Open-water extent of every year 1984-2025, the long-term "
                 "context NISAR cannot give yet.",
     "processing": "Cropped to the AOI at 120 m; yearly areas. See history.json."},
    {"id": "jrc", "field": "references",
     "name": "JRC Global Surface Water v1.5, occurrence 1984-2024",
     "provider": "European Commission Joint Research Centre",
     "url": "https://global-surface-water.appspot.com/",
     "license": "Copernicus open licence; cite Pekel et al. 2016, Nature 540",
     "citation": "Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S. "
                 "(2016). High-resolution mapping of global surface water and "
                 "its long-term changes. Nature 540, 418-422.",
     "used_for": "How often each pixel was open water over 40 years.",
     "processing": "Cropped to the AOI, averaged to 80 m."},
    {"id": "glofas", "field": "references",
     "name": "GloFAS river discharge reanalysis",
     "provider": "Copernicus Emergency Management Service / ECMWF, accessed "
                 "through the Open-Meteo Flood API",
     "url": "https://open-meteo.com/en/docs/flood-api",
     "license": "CC BY 4.0",
     "used_for": "Modelled inflow of the Okavango River near Mohembo.",
     "processing": "Daily values averaged to months."},
    {"id": "basemap", "field": "references",
     "name": "Base map tiles and data",
     "provider": "OpenStreetMap contributors",
     "url": "https://www.openstreetmap.org/copyright",
     "license": "ODbL",
     "used_for": "Background map.", "processing": "None."},
    {"id": "papers", "field": "references",
     "name": "Scientific references",
     "items": [
         "Quegan, S., Yu, J. J. (2001). Filtering of multichannel SAR images. "
         "IEEE Trans. Geosci. Remote Sens. 39(11), 2373-2379.",
         "Inman, V. L., Lyons, M. B. (2020). Automated inundation mapping over "
         "large areas using Landsat data and Google Earth Engine. Remote "
         "Sensing 12(8), 1348.",
         "Wolski, P., Murray-Hudson, M., Thito, K., Cassidy, L. (2017). "
         "Keeping it simple: monitoring flood extent in large data-poor "
         "wetlands using MODIS SWIR data. Int. J. Appl. Earth Obs. Geoinf. 57, "
         "224-234.",
         "NISAR Provisional Product Known Issues, "
         "https://nisar-docs.asf.alaska.edu/provisional-known-issues/"]},
]

AI_USE = {
    "field": "use_of_ai",
    "tools": [
        {"tool": "Claude (Anthropic), via Claude Code",
         "used_for": "Writing and reviewing the Python data pipeline and the "
                     "serverless assistant function, checking data sources, "
                     "drafting documentation.",
         "human_check": "Every number in the app is computed by the published "
                        "pipeline from the cited data, not written by an AI."},
        {"tool": "OpenAI GPT-5.4 mini via API; open-weight gpt-oss models on "
                 "Groq as backup",
         "used_for": "The in-app 'Ask the Delta' assistant.",
         "human_check": "It may only answer from assistant_context.json, a "
                        "file generated from our data. It is not trained or "
                        "fine-tuned by us."}],
    "not_ai": "No AI-generated images, video or audio are used. All maps are "
              "rendered from satellite data. No NASA logo or insignia is used.",
    "in_app_label": "AI-generated answer based on this project's data files. "
                    "It can be wrong: check the numbers against the charts.",
}


def main() -> None:
    meta, hist = load("meta.json"), load("history.json")
    dates = meta["dates"]
    span = [date.fromisoformat(dates[0]) + timedelta(days=i)
            for i in range((date.fromisoformat(dates[-1])
                            - date.fromisoformat(dates[0])).days + 1)]
    cov = dswx_coverage([d.isoformat() for d in span])
    dswx_days = [d for d, c in cov.items() if c >= 0.6]
    print(f"DSWx-HLS: {len(dswx_days)} of {len(cov)} days cover >=60% of the AOI")
    near = {d: min(dswx_days, key=lambda x: abs((date.fromisoformat(x)
                                                 - date.fromisoformat(d)).days))
            for d in dates} if dswx_days else {}
    for d, n in near.items():
        print(f"  NISAR {d} -> nearest DSWx day {n} (coverage {cov[n]:.0%})")

    years = [str(r["year"]) for r in hist["years"]]
    gibs_nisar = [d for d in dates if d >= "2026-08-10"]  # GIBS archive start
    layers = [
        {"id": "nisar_class", "group": "NISAR radar (2026)", "default": True,
         "title": "Flood classes from NISAR",
         "kind": "image", "times": dates,
         "url": "{time}/class.webp", "bounds": "{time}/bounds.json",
         "opacity": 0.85, "legend": meta["legend"], "source": "nisar",
         "chart": "nisar",
         "note": "Our classification. Blue = open water, green = water under "
                 "vegetation, which only L-band radar sees."},
        {"id": "nisar_rgb", "group": "NISAR radar (2026)",
         "title": "Radar colour image (NISAR)",
         "kind": "image", "times": dates,
         "url": "{time}/rgb.webp", "bounds": "{time}/bounds.json",
         "opacity": 1.0, "source": "nisar",
         "legend": [
             {"label": "Blue: dark, smooth surface (open water, wet flats)",
              "color": [40, 40, 230, 255]},
             {"label": "Red / orange: water standing under vegetation "
                       "(double bounce)", "color": [220, 90, 30, 255]},
             {"label": "Green: vegetation on dry ground",
              "color": [40, 190, 40, 255]}],
         "note": "Our colour picture of the NISAR measurement, cleaned of "
                 "speckle, for all 10 dates. It is the evidence behind the "
                 "flood classes, not a classification: blue also covers "
                 "smooth flats, so it shows more area than the strict "
                 "open-water class. Slide through the dates: blue shrinks "
                 "and red grows as the flood moves into the vegetation."},
        {"id": "nisar_hh", "group": "NISAR radar (2026)",
         "title": "Raw radar image (HH)",
         "kind": "image", "times": dates,
         "url": "{time}/hh.webp", "bounds": "{time}/bounds.json",
         "opacity": 1.0, "source": "nisar",
         "legend": [{"label": "Dark = smooth open water", "color": [20, 20, 20, 255]},
                    {"label": "Bright = flooded vegetation, trees", "color": [235, 235, 235, 255]}],
         "note": "Unfiltered HH backscatter, fixed -25..0 dB stretch."},
        {"id": "gibs_nisar", "group": "NASA reference layers (live)",
         "title": "NISAR as published by NASA",
         "kind": "tiles", "times": gibs_nisar, "tile_order": "zyx",
         "url": gibs("NISAR_L2_Geocoded_Polarimetric_Covariance",
                     "GoogleMapsCompatible_Level13", "png"),
         "max_native_zoom": 13, "opacity": 1.0, "source": "gibs",
         "note": "NASA's own colour rendering of the same NISAR product, "
                 "served live by NASA. Available from 2026-08-10, so not for "
                 "our first three dates, and NASA's tile set has occasional "
                 "empty tiles. Use 'Radar colour image' for the full series."},
        {"id": "gibs_dswx", "group": "NASA reference layers (live)",
         "title": "NASA surface water map (OPERA DSWx-HLS, optical)",
         "kind": "tiles", "times": dswx_days, "tile_order": "zyx",
         "nearest_to_nisar": near,
         "url": gibs("OPERA_L3_Dynamic_Surface_Water_Extent-HLS",
                     "GoogleMapsCompatible_Level12", "png"),
         "max_native_zoom": 12, "opacity": 1.0, "blend": "multiply",
         "source": "gibs",
         "legend": [{"label": "Open water", "color": [0, 0, 255, 255]},
                    {"label": "Partial surface water", "color": [180, 213, 244, 255]},
                    {"label": "Cloud / no clear view", "color": [175, 175, 175, 255]}],
         "note": "Independent NASA product from Landsat and Sentinel-2. It "
                 "sees open water only. 'Not water' is drawn WHITE, so show it "
                 "with CSS mix-blend-mode: multiply. Only days with a clear "
                 "view of at least 60% of the area are listed."},
        {"id": "gibs_truecolor", "group": "NASA reference layers (live)",
         "title": "Satellite photo of the day (MODIS Aqua)",
         "kind": "tiles", "time_range": [dates[0], dates[-1]], "tile_order": "zyx",
         "url": gibs("MODIS_Aqua_CorrectedReflectance_TrueColor",
                     "GoogleMapsCompatible_Level9", "jpg"),
         "max_native_zoom": 9, "opacity": 1.0, "source": "gibs",
         "note": "What a camera sees that day, 250 m. Any day in the range "
                 "works; clouds hide the ground, radar does not care."},
        {"id": "landsat_history", "group": "History 1984-2025 (optical)",
         "title": "Open water per year (Landsat)",
         "kind": "image", "times": years,
         "url": "history/{time}.webp", "bounds": "history/bounds.json",
         "opacity": 0.9, "legend": hist["legend"], "source": "wofs",
         "chart": "history",
         "flag_times": [str(r["year"]) for r in hist["years"] if r["low_coverage"]],
         "flag_label": "Few satellite looks this year: a lower bound.",
         "note": hist["caveat"]},
        {"id": "jrc_occurrence", "group": "History 1984-2025 (optical)",
         "title": "How often was it water? (JRC, 1984-2024)",
         "kind": "image", "url": "jrc/occurrence.webp", "bounds": "jrc/bounds.json",
         "opacity": 0.9, "source": "jrc",
         "legend": [{"label": "Rarely water", "color": [158, 202, 225, 220]},
                    {"label": "Almost always water", "color": [8, 48, 107, 220]}],
         "note": "Optical, open water only. Most of the Delta is vegetated, "
                 "so this map shows far less water than the radar does."},
    ]
    charts = [
        {"id": "nisar", "title": "Flooded area in 2026 (NISAR)", "url": "timeseries.json",
         "x": "date", "series": [
             {"key": "open_water_km2", "label": "Open water", "color": [33, 113, 181]},
             {"key": "flooded_veg_km2", "label": "Flooded vegetation", "color": [49, 163, 84]}],
         "stacked": True, "unit": "km2", "gaps": meta["gaps"], "source": "nisar",
         "note": "Draw the gap as 'no data'. Mark the VALIDATED point differently."},
        {"id": "history", "title": "Open water per year, 1984-2025 (Landsat)",
         "url": "history.json", "path": "years", "x": "year", "series": [
             {"key": "water_km2", "label": "Open water", "color": [107, 174, 214]}],
         "unit": "km2", "flag_key": "low_coverage", "source": "wofs",
         "note": "Do NOT draw on the same axis as the NISAR chart."},
        {"id": "discharge", "title": "River inflow near Mohembo (model)",
         "url": "drivers.json", "path": "discharge.monthly", "x": "month", "series": [
             {"key": "m3s", "label": "Discharge", "color": [80, 80, 80]}],
         "unit": "m3/s", "source": "glofas",
         "note": "Hydrological model, not a gauge."},
        {"id": "rainfall", "title": "Rain in the Angolan headwaters",
         "url": "drivers.json", "path": "rainfall.monthly", "x": "month", "series": [
             {"key": "mm_day", "label": "Rainfall", "color": [66, 146, 198]}],
         "unit": "mm/day", "source": "power"},
    ]
    manifest = {
        "version": 1,
        "data_root": DATA_ROOT,
        "how_to_use": "Relative urls are under data_root. Replace {time} with "
                      "an entry of 'times' (or any YYYY-MM-DD inside "
                      "'time_range'). kind=image -> image overlay placed with "
                      "the 'bounds' json (its 'bounds' field for Leaflet, "
                      "'coordinates' for MapLibre). kind=tiles -> tile layer; "
                      "the url already has {z}/{y}/{x} in the right order.",
        "aoi": meta["aoi"],
        "layers": layers,
        "charts": charts,
        "must_show": [
            "Thresholds are uncertain: this is a visualization of radar "
            "backscatter classes, not a calibrated hydrological product.",
            "NISAR acquired no data between 2026-07-27 and 2026-08-10.",
            "Radar (NISAR) and optical (Landsat, JRC, DSWx) layers measure "
            "different things and are never added together."],
        "assistant": {"url": ASSISTANT, "label": AI_USE["in_app_label"]},
        "credits": "sources.json",
        # Optional "How clean is the data?" panel: summary.plain is 4 sentences.
        "quality": "quality.json",
    }
    (C.OUT / "layers.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (C.OUT / "sources.json").write_text(
        json.dumps({"sources": SOURCES, "ai": AI_USE,
                    "fields": {"nasa_data": "NASA Data",
                               "references": "Space Agency Partner Data & References",
                               "use_of_ai": "Use of AI"}}, indent=1), encoding="utf-8")

    md = ["# Data, references and AI use", "",
          "Generated by `pipeline/manifest.py`. Paste each section into the "
          "matching field of the Space Apps project page.", ""]
    for field, title in (("nasa_data", "NASA Data"),
                         ("references", "Space Agency Partner Data & References")):
        md += [f"## {title}", ""]
        for s in (s for s in SOURCES if s["field"] == field):
            if "items" in s:
                md += [f"- {i}" for i in s["items"]]
                continue
            md.append(f"- **{s['name']}** — {s['provider']}. {s['used_for']} "
                      f"{s['url']} ({s['license']})")
            if "citation" in s:
                md.append(f"  - {s['citation']}")
            if "acknowledgement" in s:
                md.append(f"  - {s['acknowledgement']}")
        md.append("")
    md += ["## Use of AI", ""]
    md += [f"- **{t['tool']}**: {t['used_for']} {t['human_check']}" for t in AI_USE["tools"]]
    md += [f"- {AI_USE['not_ai']}", ""]
    (C.OUT.parent / "SOURCES.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote layers.json ({len(layers)} layers, {len(charts)} charts), "
          f"sources.json, SOURCES.md")


if __name__ == "__main__":
    main()
