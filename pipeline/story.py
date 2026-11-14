"""
EDev-Vision | Why was the water like this that year, and who lived next to it?

Builds data/story.json: one record per year, 1984-2026, joining

  CAUSES   rain in the Angolan headwaters (NASA POWER), modelled river inflow
           near Mohembo (GloFAS), rain over the Delta itself (NASA POWER),
           flood-season heat (NASA POWER), and El Nino / La Nina (NOAA ONI).
           Each cause becomes a level from -2 (far below normal) to +2, a map
           icon with a position, and one plain sentence.
  OUTCOME  open-water area of that year from history.json (Landsat).
  PEOPLE   how many residents live within 5 km of that year's open water
           (WorldPop 2020 on our yearly water maps).
  EVENTS   a short list of documented impacts, each with its source.

Every sentence is assembled by the rules in this file from the numbers, not
written by an AI, so the same data always gives the same text.

Usage:  python pipeline/story.py
"""
from __future__ import annotations

import calendar
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from history import F_ANY, MIN_WET, PIX, era5_monthly_rain, get_json, wofs_grid

load = lambda name: json.loads((C.OUT / name).read_text(encoding="utf-8"))
BASE = (1991, 2020)                 # climate "normal" period
DELTA = (-19.3, 22.9)               # centre of the Delta
NEAR_KM = 5.0
POP_URL = ("https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
           "2020/BWA/bwa_ppp_2020_1km_Aggregated_UNadj.tif")
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Where each cause is drawn. "pos" is inside our map area so the icon is
# always visible; "true_pos" is where the thing really happens.
CAUSES = {
    "headwater_rain": {
        "icon": "\U0001F327", "label": "Rain in the Angolan highlands",
        "pos": [-18.40, 22.12], "true_pos": [-14.2, 17.5],
        "hint": "About 600 km north-west of the map; arrives months later."},
    "inflow": {
        "icon": "\U0001F30A", "label": "River inflow into the Delta",
        "pos": [-18.72, 22.22], "true_pos": [-18.275, 21.787],
        "hint": "Okavango River near Mohembo. Hydrological model, not a gauge."},
    "local_rain": {
        "icon": "☔", "label": "Rain over the Delta",
        "pos": [-18.55, 23.25], "true_pos": [DELTA[0], DELTA[1]],
        "hint": "November-April rain falling on the Delta itself."},
    "heat": {
        "icon": "\U0001F321", "label": "Flood-season heat",
        "pos": [-19.25, 23.55], "true_pos": [DELTA[0], DELTA[1]],
        "hint": "Hotter days evaporate the flood faster."},
    "enso": {
        "icon": "\U0001F30D", "label": "El Nino / La Nina",
        "pos": [-18.40, 23.62], "true_pos": [0.0, -145.0],
        "hint": "Pacific Ocean pattern; El Nino usually means drier southern Africa."},
}
LEVEL_WORD = {-2: "far below normal", -1: "below normal", 0: "near normal",
              1: "above normal", 2: "far above normal"}
LEVEL_COLOR = {-2: [178, 24, 43], -1: [239, 138, 98], 0: [150, 150, 150],
               1: [103, 169, 207], 2: [33, 102, 172]}

# Documented impacts. Only statements we read in the cited source.
EVENTS = [
    {"year": 2009, "kind": "people", "title": "Floods displace families",
     "text": "Floods on the Okavango, Zambezi and Chobe displaced 430 people "
             "and submerged eight villages. In Ngamiland 61 hectares of "
             "ploughed fields went under water and lodges at Sepopa were "
             "flooded.",
     "source": "Mmegi, 30 March 2009",
     "url": "https://www.mmegi.bw/ampArticle/17479"},
    {"year": 2010, "kind": "nature", "title": "Largest floods in recent memory",
     "text": "Between 2010 and 2012 the Delta saw some of its largest floods "
             "in recent memory; channels and lagoons dry for decades refilled "
             "and water reached Lake Ngami.",
     "source": "okavango.com (travel guide, not a scientific source)",
     "url": "https://www.okavango.com/okavango-delta-flood-cycle.php"},
    {"year": 2019, "kind": "people", "title": "The flood did not arrive",
     "text": "Water sources across Ngamiland dried up, leaving communities, "
             "livestock and wildlife in a dire situation. Fishermen and "
             "mokoro polers lost their income and farmers lost cattle.",
     "source": "Botswana Daily News (BOPA), 26 April 2020",
     "url": "https://dailynews.gov.bw/news-detail/55851"},
    {"year": 2020, "kind": "people", "title": "The Delta springs back to life",
     "text": "The flood returned and was expected to reach the Thamalakane "
             "River at Maun in mid-May, restoring water for farming, fishing "
             "and mokoro tourism.",
     "source": "Botswana Daily News (BOPA), 26 April 2020",
     "url": "https://dailynews.gov.bw/news-detail/55851"},
    {"year": 2024, "kind": "wildlife", "title": "Hippos stranded near Maun",
     "text": "During an El Nino drought the Thamalakane River dried up and "
             "herds of hippos were stuck in drying pools near Maun; wildlife "
             "officials warned they were in danger of dying.",
     "source": "AFP, 26 April 2024",
     "url": "https://www.gulf-times.com/article/681371/international/"
            "herds-of-hippos-trapped-in-mud-in-drought-hit-botswana"},
]


def water_year_sum(monthly: dict, y: int):
    """Nov (y-1) .. Apr (y) total in mm from mm/day monthly means."""
    ws = [(y - 1, 11), (y - 1, 12), (y, 1), (y, 2), (y, 3), (y, 4)]
    v = [monthly.get(f"{a}{b:02d}") for a, b in ws]
    if any(x is None or x < -900 for x in v):
        return None
    return round(sum(x * calendar.monthrange(a, b)[1] for x, (a, b) in zip(v, ws)))


def level(v, mean, std):
    if v is None:
        return None
    z = (v - mean) / std
    return int(-2 if z < -1.5 else -1 if z < -0.5 else 2 if z > 1.5 else 1 if z > 0.5 else 0)


def people_near_water() -> tuple[dict, float]:
    """Residents within NEAR_KM of each year's open water. Population is held
    at 2020 on purpose: the year-to-year change then comes from the water
    alone, not from population growth."""
    x0, y1, W, H = wofs_grid()
    k = 8                                    # 8 x 120 m = 960 m cells
    cell_m = PIX * k
    r = int(round(NEAR_KM * 1000 / cell_m))
    h, w = H // k, W // k
    # The population grid reaches NEAR_KM beyond the map on every side: the
    # southern map edge cuts through Maun, and people living just outside it
    # are still within reach of water mapped inside.
    hp, wp = h + 2 * r, w + 2 * r
    tf = Affine(cell_m, 0, x0 - r * cell_m, 0, -cell_m, y1 + r * cell_m)
    pop = np.zeros((hp, wp), np.float32)
    # 3 MB, whole country; this server does not allow partial reads.
    local = C.CACHE / "worldpop_bwa_2020_1km.tif"
    if not local.exists():
        local.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(POP_URL, local)
    with rasterio.open(local) as ds:
        src = ds.read(1).astype(np.float32)
        src[(src < 0) | ~np.isfinite(src)] = 0
        # Resampling.sum keeps the head count when the grid changes.
        reproject(src, pop, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=tf, dst_crs="EPSG:6933", resampling=Resampling.sum)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r ** 2
    out = {}
    for p in sorted((C.CACHE / "history").glob("*.npz")):
        with np.load(p) as z:
            wet, clear = z["wet"], z["clear"]
        with np.errstate(invalid="ignore", divide="ignore"):
            water = (clear > 0) & (wet >= MIN_WET) & (wet / np.maximum(clear, 1) >= F_ANY)
        cell = np.pad(water[:h * k, :w * k].reshape(h, k, w, k).any(axis=(1, 3)), r)
        near = np.zeros_like(cell)
        pad = np.pad(cell, r)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                if disk[dy, dx]:
                    near |= pad[dy:dy + hp, dx:dx + wp]
        out[int(p.stem)] = int(round(float(pop[near].sum()), -2))
    return out, float(pop.sum())


def main() -> None:
    hist, drv = load("history.json"), load("drivers.json")
    years = range(1984, 2027)

    # --- raw yearly numbers --------------------------------------------------
    head = {}
    hm = {r["month"].replace("-", ""): r["mm_day"] for r in drv["rainfall"]["monthly"]}
    q = {r["month"]: r["m3s"] for r in drv["discharge"]["monthly"]}
    # Temperature from NASA POWER; rain over the Delta from ERA5 (see the note
    # on POWER's precipitation spikes in history.run_drivers).
    p = get_json("https://power.larc.nasa.gov/api/temporal/monthly/point?"
                 "parameters=T2M_MAX&community=AG"
                 f"&longitude={DELTA[1]}&latitude={DELTA[0]}&start=1983&end=2026"
                 "&format=JSON")["properties"]["parameter"]
    lm = {k.replace("-", ""): v for k, v in era5_monthly_rain(*DELTA).items()}
    oni = {}
    with urllib.request.urlopen(ONI_URL, timeout=60) as r:
        for line in r.read().decode().splitlines()[1:]:
            s, y, _, a = line.split()
            if s == "DJF":
                oni[int(y)] = float(a)      # DJF of year y = the rainy season feeding y

    raw = {}
    for y in years:
        first = [q.get(f"{y}-{m:02d}") for m in range(1, 7)]
        tmax = [p["T2M_MAX"].get(f"{y}{m:02d}") for m in range(5, 11)]
        tmax = [t for t in tmax if t is not None and t > -900]
        raw[y] = {
            "headwater_rain": water_year_sum(hm, y),
            "inflow": max(first) if all(v is not None for v in first) else None,
            "local_rain": water_year_sum(lm, y),
            "heat": round(float(np.mean(tmax)), 2) if len(tmax) >= 4 else None,
            "enso": oni.get(y),
        }
    stats = {}
    for k in ("headwater_rain", "inflow", "local_rain", "heat"):
        b = [raw[y][k] for y in range(BASE[0], BASE[1] + 1) if raw[y][k] is not None]
        stats[k] = (float(np.mean(b)), float(np.std(b)))
    inflow_rank = sorted((y for y in years if raw[y]["inflow"] is not None),
                         key=lambda y: -raw[y]["inflow"])

    hy = {r["year"]: r for r in hist["years"]}
    good = [r["water_km2"] for r in hist["years"] if r["year"] >= hist["reliable_since"]]
    w_med = float(np.median(good))
    people, pop_total = people_near_water()
    reliable = [people[y] for y in people if y >= hist["reliable_since"]]
    p_med = float(np.median(reliable))
    events = {}
    for e in EVENTS:
        events.setdefault(e["year"], []).append(e)

    out_years = {}
    for y in years:
        r, causes, parts = raw[y], [], []

        def add(key, lvl, value, sentence, state=None):
            c = CAUSES[key]
            causes.append({"id": key, "icon": c["icon"], "label": c["label"],
                           "level": lvl, "color": LEVEL_COLOR[lvl], "value": value,
                           # what it was like, in the cause's own words; the
                           # level/colour say which way it pushed the flood
                           "state": state or LEVEL_WORD[lvl],
                           "text": sentence, "pos": c["pos"], "true_pos": c["true_pos"],
                           "hint": c["hint"]})
            parts.append(sentence)

        pct = lambda v, k: round(100 * (v / stats[k][0] - 1))
        sign = lambda n: ("close to normal" if abs(n) < 3 else
                          f"{abs(n)}% {'above' if n >= 0 else 'below'} normal")
        if r["enso"] is not None:
            e = r["enso"]
            lvl = -2 if e >= 1.5 else -1 if e >= 0.5 else 2 if e <= -1.5 else 1 if e <= -0.5 else 0
            name = ("a strong El Nino" if e >= 1.5 else "an El Nino" if e >= 0.5 else
                    "a strong La Nina" if e <= -1.5 else "a La Nina" if e <= -0.5 else
                    "a neutral Pacific")
            add("enso", lvl, f"ONI {e:+.1f}",
                f"The rainy season came with {name} (index {e:+.1f})"
                + ("; El Nino usually dries southern Africa." if e >= 0.5 else
                   "; La Nina usually brings more rain here." if e <= -0.5 else "."),
                state=name.split(" ", 1)[1] if name[0] == "a" else name)
        if r["headwater_rain"] is not None:
            n = pct(r["headwater_rain"], "headwater_rain")
            add("headwater_rain", level(r["headwater_rain"], *stats["headwater_rain"]),
                f"{r['headwater_rain']} mm",
                f"Rain in the Angolan highlands was {sign(n)} ({r['headwater_rain']} mm).")
        if r["inflow"] is not None:
            add("inflow", level(r["inflow"], *stats["inflow"]),
                f"{round(r['inflow'])} m3/s",
                f"Modelled river inflow peaked at {round(r['inflow'])} m3/s, rank "
                f"{inflow_rank.index(y) + 1} of {len(inflow_rank)} years since 1997.")
        if r["local_rain"] is not None:
            n = pct(r["local_rain"], "local_rain")
            add("local_rain", level(r["local_rain"], *stats["local_rain"]),
                f"{r['local_rain']} mm",
                f"Rain over the Delta itself was {sign(n)} ({r['local_rain']} mm).")
        if r["heat"] is not None:
            d = r["heat"] - stats["heat"][0]
            # hot = more evaporation = less water, so the level is inverted
            lvl = -level(r["heat"], *stats["heat"])
            add("heat", lvl, f"{d:+.1f} C",
                f"Flood-season days were {abs(d):.1f} C "
                f"{'hotter' if d >= 0 else 'cooler'} than normal"
                + (", so more water evaporated." if d >= 0.5 else "."),
                state=("much hotter" if d >= 1.5 else "hotter" if d >= 0.5 else
                       "much cooler" if d <= -1.5 else "cooler" if d <= -0.5 else
                       "normal temperature"))

        rec = {"causes": causes}
        h = hy.get(y)
        if h:
            n = round(100 * (h["water_km2"] / w_med - 1))
            rec["water_km2"] = h["water_km2"]
            rec["low_coverage"] = h["low_coverage"]
            rec["outcome"] = (
                f"Landsat saw too few clear days in {y} to trust the area "
                f"({h['water_km2']:.0f} km2 is a lower bound)." if h["low_coverage"] else
                f"Result: {h['water_km2']:.0f} km2 of open water, "
                f"{abs(n)}% {'above' if n >= 0 else 'below'} a typical recent year.")
        else:
            rec["water_km2"] = None
            rec["outcome"] = ("Result: see the NISAR radar layers; the Landsat "
                              "yearly map for this year is not complete yet."
                              if y == 2026 else "No Landsat water map for this year.")
        # Do the causes agree with what was mapped? Say so when they do not.
        wet_votes = sum(c["level"] for c in causes if c["id"] != "heat")
        if h and not h["low_coverage"]:
            n = h["water_km2"] / w_med - 1
            if (n > 0.25 and wet_votes < -1) or (n < -0.25 and wet_votes > 1):
                rec["mismatch"] = ("The causes and the mapped water disagree "
                                   "this year. The inflow figure is a model and "
                                   "the flood also carries over water from the "
                                   "year before, so treat the explanation with "
                                   "caution.")
        lv = {c["id"]: c["level"] for c in causes}
        if abs(lv.get("headwater_rain", 0) - lv.get("inflow", lv.get("headwater_rain", 0))) >= 2:
            rec["mismatch"] = (rec.get("mismatch", "") + " The rainfall estimate and "
                               "the river model disagree with each other this year; "
                               "both are models, not measurements.").strip()
        if y in people:
            pp = people[y]
            rec["people_near_water"] = pp
            rec["people_text"] = (
                f"About {pp:,} of the {round(pop_total, -3):,.0f} people in and "
                f"around the map area lived within {NEAR_KM:.0f} km of open water "
                "this year"
                + ("." if not h or h["low_coverage"] else
                   ", about the same as in a typical recent year."
                   if abs(pp / p_med - 1) < 0.03 else
                   f", {abs(round(100 * (pp / p_med - 1)))}% "
                   f"{'more' if pp >= p_med else 'fewer'} than in a typical recent year."))
            rec["people_pos"] = [-19.90, 23.42]   # near Maun, the main town
        if y in events:
            rec["events"] = events[y]
        rec["why"] = " ".join(parts)
        out_years[y] = rec

    doc = {
        "how_to_use": "Look up years[<year>]. Draw each entry of 'causes' as a "
                      "map marker at 'pos' (inside the map) using 'icon' and "
                      "'color'; show 'label', 'value' and 'text' on hover. Show "
                      "'why', 'outcome', 'people_text' and 'events' as text. "
                      "Levels run from -2 (pushes towards a small flood) to +2 "
                      "(pushes towards a large flood).",
        "normal_period": list(BASE),
        "levels": {str(k): {"word": v, "color": LEVEL_COLOR[k]} for k, v in LEVEL_WORD.items()},
        "color_meaning": "Red = pushed towards a smaller flood, blue = towards "
                         "a larger flood, grey = no push. Use each cause's "
                         "'state' as its short label.",
        "sources": {
            "headwater_rain": "ERA5 reanalysis via Open-Meteo, mean of three "
                              "points in the Angolan highlands, November-April total",
            "inflow": "GloFAS river discharge reanalysis via Open-Meteo, peak "
                      "monthly mean January-June. A MODEL, not a gauge.",
            "local_rain": f"ERA5 reanalysis via Open-Meteo at {DELTA[0]}, "
                          f"{DELTA[1]}, November-April total",
            "heat": "NASA POWER (MERRA-2) daily maximum temperature, May-October mean",
            "enso": "NOAA Climate Prediction Center, Oceanic Nino Index, "
                    "December-February",
            "people": "WorldPop 2020, 1 km, UN-adjusted (CC BY 4.0), counted "
                      f"within {NEAR_KM:.0f} km of that year's Landsat open water",
        },
        "caveats": [
            "Causes are associations, not proof: a year can be wet for reasons "
            "not listed here, and last year's water carries over.",
            "River inflow is a hydrological model. It disagrees with the "
            "mapped water in some years (for example 2025).",
            "Population is held at its 2020 distribution in every year, so the "
            "change between years comes from the water, not from growth. "
            "Living near water is not the same as being flooded or helped.",
            "Events are from news reports and a travel guide; they are "
            "illustrations, not a complete record."],
        "population_in_map_area_2020": round(pop_total, -3),
        "typical_recent_water_km2": w_med,
        "years": out_years,
        "events": EVENTS,
    }
    (C.OUT / "story.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"population in map area (2020): {pop_total:,.0f}")
    for y in (2009, 2011, 2019, 2020, 2024, 2025, 2026):
        r = out_years[y]
        print(f"\n{y}: " + " ".join(f"{c['icon']}{c['level']:+d}" for c in r["causes"]))
        print("  ", r["why"])
        print("  ", r["outcome"], r.get("mismatch", ""))
        print("  ", r.get("people_text", "(no people figure)"))
    print(f"\nwrote {C.OUT / 'story.json'}")


if __name__ == "__main__":
    main()
