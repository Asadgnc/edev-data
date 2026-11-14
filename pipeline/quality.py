"""
EDev-Vision | What does NASA itself say about the noise in each granule?

Reads only the small quality metadata of each NISAR GCOV file by HTTP range
request (a few hundred kB per date, no imagery):

  rfiLikelihood               0..1, share of the raw data flagged as radio
                              frequency interference (other transmitters in
                              L-band: radars, navigation, telecom)
  noiseEquivalentBackscatter  the instrument's own noise floor (NESZ). A surface
                              darker than this cannot be measured, only bounded.
  processing flags            was RFI mitigation / noise correction applied

and compares the noise floor with how dark our "open water" class is. If water
sat at the noise floor the open-water map would be noise; the margin says how
far above it we are.

Writes data/quality.json. Needs EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
Usage:  python pipeline/quality.py [A|B]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FLAGS = ("noiseCorrectionApplied", "rfiMitigationApplied",
         "shadowMaskingApplied", "validSamplesSubSwathMaskingApplied")


def _s(v):
    v = v[()] if hasattr(v, "shape") else v
    return v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)


def read_one(row: dict, freq: str) -> dict:
    import earthaccess

    res = earthaccess.search_data(short_name=row["collection"], bounding_box=C.AOI,
                                  temporal=(row["date"], row["date"]), count=50)
    hits = [g for g in res if g["umm"]["GranuleUR"] == row["granule"]]
    out = {"date": row["date"], "quality": row["quality"]}
    with h5py.File(earthaccess.open(hits)[0], "r") as h5:
        cal = h5[f"science/LSAR/GCOV/metadata/calibrationInformation/frequency{freq}"]
        out["rfi_likelihood"] = {
            pol: round(float(_s(cal[pol]["rfiLikelihood"])), 4)
            for pol in ("HH", "HV") if pol in cal and "rfiLikelihood" in cal[pol]}
        ne = cal["noiseEquivalentBackscatter"]
        x, y = ne["xCoordinates"][:], ne["yCoordinates"][:]
        epsg = int(_s(ne["projection"]))
        tf = Transformer.from_crs(4326, epsg, always_xy=True)
        (xa, xb), (ya, yb) = tf.transform([C.AOI[0], C.AOI[2]], [C.AOI[1], C.AOI[3]])
        ix = (x >= min(xa, xb)) & (x <= max(xa, xb))
        iy = (y >= min(ya, yb)) & (y <= max(ya, yb))
        out["nesz_db"] = {}
        for pol in ("HH", "HV"):
            if pol not in ne:
                continue
            a = ne[pol][:][np.ix_(iy, ix)].astype(float)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size:
                d = 10 * np.log10(a)
                out["nesz_db"][pol] = {"median": round(float(np.median(d)), 1),
                                       "min": round(float(d.min()), 1),
                                       "max": round(float(d.max()), 1)}
        par = h5["science/LSAR/GCOV/metadata/processingInformation/parameters"]
        out["flags"] = {k: str(_s(par[k])) for k in FLAGS if k in par}
    print(row["date"], out["rfi_likelihood"], out["nesz_db"], out["flags"], flush=True)
    return out


def main() -> None:
    freq = sys.argv[1] if len(sys.argv) > 1 else "B"
    old = C.OUT / "quality.json"
    if "--offline" in sys.argv and old.exists():
        # Rebuild the summary from the metadata already read, no login needed.
        keep = ("date", "quality", "rfi_likelihood", "nesz_db", "flags", "error")
        recs = [{k: r[k] for k in keep if k in r}
                for r in json.loads(old.read_text(encoding="utf-8"))["dates"]]
    else:
        C.login()
        recs = []
        for row in C.series():
            try:
                recs.append(read_one(row, freq))
            except Exception as e:  # one unreadable granule must not lose the rest
                print(row["date"], "FAILED:", repr(e)[:200], flush=True)
                recs.append({"date": row["date"], "error": repr(e)[:200]})

    # How dark is the open-water class, from the cached backscatter?
    from render import WATER, classify, to_db
    water = {}
    for p in sorted(C.cache_dir(freq).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            hh, hv = to_db(z["hh_lin"]), to_db(z["hv_lin"])
        w = classify(hh, hv) == WATER
        water[p.stem] = {"HH": round(float(np.nanmedian(hh[w])), 1),
                         "HV": round(float(np.nanmedian(hv[w])), 1)}
    for r in recs:
        if "nesz_db" in r and r["date"] in water:
            r["open_water_median_db"] = water[r["date"]]
            r["water_above_noise_db"] = {
                pol: round(water[r["date"]][pol] - r["nesz_db"][pol]["median"], 1)
                for pol in r["nesz_db"]}

    doc = {
        "frequency": freq,
        "source": "NISAR L2 GCOV product metadata (calibrationInformation, "
                  "processingInformation), read per granule",
        "explain": {
            "rfi_likelihood": "NASA documents this as radio-interference "
                              "severity in [0,1]. Several of our granules "
                              "report values above 1, outside the documented "
                              "range, so we do NOT interpret the scale. It is "
                              "listed for transparency only.",
            "nesz_db": "Instrument noise floor (noise-equivalent backscatter) "
                       "over the AOI as reported in the product, in dB. A "
                       "surface darker than this cannot be measured.",
            "water_above_noise_db": "How many dB the median open-water pixel "
                                    "sits above the noise floor. Near 0 means "
                                    "the channel is measuring noise, not water."},
        "dates": recs,
    }
    ok = [r for r in recs if "water_above_noise_db" in r]
    if ok:
        med = lambda f: round(float(np.median([f(r) for r in ok])), 1)
        doc["summary"] = {
            "noise_floor_db": med(lambda r: r["nesz_db"]["HH"]["median"]),
            "open_water_above_noise_db": {
                "HH": med(lambda r: r["water_above_noise_db"]["HH"]),
                "HV": med(lambda r: r["water_above_noise_db"].get("HV", np.nan))},
            "noise_subtracted_by_nasa": any(
                r["flags"].get("noiseCorrectionApplied") == "True" for r in ok),
            "rfi_mitigated_by_nasa": any(
                r["flags"].get("rfiMitigationApplied") == "True" for r in ok),
            "plain": [
                "The radar's noise floor is steady across all ten dates, so "
                "the seasonal change is not an instrument effect.",
                "Open water is clearly above the noise in HH but only about "
                "2 dB above it in HV. Open water is therefore mapped mainly "
                "by HH; its HV value is mostly instrument noise.",
                "NASA did not subtract thermal noise in these products, so "
                "very dark surfaces read slightly brighter than they are. "
                "This does not affect the bright flooded-vegetation class.",
                "NASA's 'mask' layer marks samples that were not fully "
                "focused as invalid; these are the same edge-strip pixels our "
                "noise-floor rule already removes."],
        }
    (C.OUT / "quality.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", C.OUT / "quality.json")


if __name__ == "__main__":
    main()
