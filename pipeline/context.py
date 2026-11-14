"""
EDev-Vision | Join the layers into (1) a forecast feasibility test and (2) the
single small file the in-app assistant reads.

Runs offline from the JSON that render.py and history.py already wrote.

  forecast.json            can upstream river flow predict the year's open-water
                           extent? Tested by leaving each year out in turn.
                           "show" is false unless the model beats climatology.
  assistant_context.json   everything the assistant is allowed to know. It is
                           small enough to be sent whole as the system prompt,
                           so there is no model training, no vector database and
                           no retrieval step that could fetch the wrong thing.

Usage:  python pipeline/context.py
"""
from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

load = lambda name: json.loads((C.OUT / name).read_text(encoding="utf-8"))


def annual_drivers(drv: dict) -> dict:
    q = {r["month"]: r["m3s"] for r in drv["discharge"]["monthly"]}
    p = {r["month"]: r["mm_day"] for r in drv["rainfall"]["monthly"]}
    out = {}
    for y in range(1985, 2027):
        rec = {}
        first = [q.get(f"{y}-{m:02d}") for m in range(1, 7)]
        if all(v is not None for v in first):
            rec["peak_m3s"] = max(first)
            rec["peak_month"] = f"{y}-{first.index(max(first)) + 1:02d}"
            rec["feb_apr_m3s"] = round(float(np.mean(first[1:4])), 1)
        # wet season that feeds year y: November (y-1) .. April (y)
        ws = [(y - 1, 11), (y - 1, 12), (y, 1), (y, 2), (y, 3), (y, 4)]
        mm = [p.get(f"{a}-{b:02d}") for a, b in ws]
        if all(v is not None for v in mm):
            rec["headwater_rain_mm"] = round(sum(
                v * calendar.monthrange(a, b)[1] for v, (a, b) in zip(mm, ws)))
        if rec:
            out[y] = rec
    return out


def story_table():
    """Compact per-year causes, people and events from story.json."""
    f = C.OUT / "story.json"
    if not f.exists():
        return None
    st = json.loads(f.read_text(encoding="utf-8"))
    lv = lambda r, k: next((c["level"] for c in r["causes"] if c["id"] == k), None)
    return {
        "what": "Level of each cause per year, -2 (pushes towards a small "
                "flood) to +2 (towards a large flood), against the 1991-2020 "
                "normal. people = residents within 5 km of that year's Landsat "
                "open water (2020 population held fixed). Causes are "
                "associations, not proof.",
        "columns": ["year", "el_nino_la_nina", "headwater_rain", "river_inflow_model",
                    "rain_over_delta", "heat", "people_near_water"],
        "rows": [[int(y), lv(r, "enso"), lv(r, "headwater_rain"), lv(r, "inflow"),
                  lv(r, "local_rain"), lv(r, "heat"), r.get("people_near_water")]
                 for y, r in st["years"].items() if int(y) >= 1997],
        "sentences_2026": st["years"]["2026"]["why"],
        "documented_events": [{k: e[k] for k in ("year", "title", "text", "source")}
                              for e in st["events"]],
        "caveats": st["caveats"],
    }


def loyo(x: np.ndarray, y: np.ndarray):
    """Leave-one-year-out linear regression. Returns predictions and skill
    against the leave-one-out mean (climatology). skill <= 0 means useless."""
    n = len(y)
    pred, clim = np.empty(n), np.empty(n)
    X = np.column_stack([np.ones(n), x])
    for i in range(n):
        k = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(X[k], y[k], rcond=None)
        pred[i], clim[i] = X[i] @ beta, y[k].mean()
    sse, sse0 = ((pred - y) ** 2).sum(), ((clim - y) ** 2).sum()
    return pred, float(1 - sse / sse0), float(np.sqrt(sse / n))


def main() -> None:
    meta, ts = load("meta.json"), load("timeseries.json")
    hist, drv = load("history.json"), load("drivers.json")
    ann = annual_drivers(drv)
    years = {r["year"]: r for r in hist["years"]}

    # --- forecast feasibility -----------------------------------------------
    tests = []
    for name, key in (("Feb-Apr river discharge near Mohembo", "feb_apr_m3s"),
                      ("Nov-Apr rainfall in the Angolan headwaters", "headwater_rain_mm")):
        ys = [y for y in sorted(years)
              if not years[y]["low_coverage"] and key in ann.get(y, {})]
        x = np.array([ann[y][key] for y in ys], float)
        w = np.array([years[y]["water_km2"] for y in ys], float)
        if len(ys) < 8:
            continue
        pred, skill, rmse = loyo(x, w)
        beta = np.polyfit(x, w, 1)
        t = {"predictor": name, "key": key, "n_years": len(ys),
             "years": [ys[0], ys[-1]],
             "correlation_r": round(float(np.corrcoef(x, w)[0, 1]), 2),
             "skill_vs_climatology": round(skill, 2),
             "rmse_km2": round(rmse, 1),
             "hindcast": [{"year": y, "observed_km2": float(o), "predicted_km2": round(float(q), 1)}
                          for y, o, q in zip(ys, w, pred)]}
        if key in ann.get(2026, {}):
            v = float(np.polyval(beta, ann[2026][key]))
            t["prediction_2026"] = {"km2": round(v, 1),
                                    "low_km2": round(v - 1.96 * rmse, 1),
                                    "high_km2": round(v + 1.96 * rmse, 1),
                                    "input": ann[2026][key]}
        tests.append(t)
        print(f"{name}: n={len(ys)} r={t['correlation_r']} "
              f"skill={t['skill_vs_climatology']} rmse={t['rmse_km2']} km2 "
              f"2026={t.get('prediction_2026')}")
    best = max(tests, key=lambda t: t["skill_vs_climatology"], default=None)
    show = bool(best and best["skill_vs_climatology"] >= 0.3)
    forecast = {
        "target": "annual open-water extent in the AOI (optical, history.json)",
        "method": "One-variable linear regression, validated by leaving each "
                  "year out in turn. Skill = 1 - error/error of always "
                  "predicting the long-term mean; 0 or less means no skill.",
        "show": show,
        "verdict": ("The best predictor beats climatology; a forecast with its "
                    "interval may be shown." if show else
                    "No predictor beats climatology clearly enough. Do not show "
                    "a forecast; show the historical ranking instead."),
        "best": best and best["predictor"],
        "tests": tests,
        "not_forecast": "NISAR classes are not forecast: ten dates in one "
                        "season cannot train or validate a model.",
    }
    (C.OUT / "forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf-8")

    # --- where does 2026 sit? -----------------------------------------------
    peaks = {y: a["peak_m3s"] for y, a in ann.items() if "peak_m3s" in a}
    rank = sorted(peaks, key=peaks.get, reverse=True)
    good = [r for r in hist["years"] if not r["low_coverage"]]
    findings = {
        "inflow_2026": {
            "peak_m3s": peaks.get(2026), "peak_month": ann.get(2026, {}).get("peak_month"),
            "rank": rank.index(2026) + 1 if 2026 in rank else None,
            "of_years": len(rank), "period": [min(peaks), max(peaks)],
            "previous_three_peaks_m3s": {y: peaks.get(y) for y in (2023, 2024, 2025)}},
        "optical_history": {
            "largest": [(r := max(good, key=lambda r: r["water_km2"]))["year"], r["water_km2"]],
            "smallest": [(r := min(good, key=lambda r: r["water_km2"]))["year"], r["water_km2"]],
            "median_km2": float(np.median([r["water_km2"] for r in good])),
            "years_used": len(good)},
        "nisar_2026": {
            "open_water_peak": [(r := max(ts, key=lambda r: r["open_water_km2"]))["date"],
                                r["open_water_km2"]],
            "flooded_vegetation_peak": [(r := max(ts, key=lambda r: r["flooded_veg_km2"]))["date"],
                                        r["flooded_veg_km2"]],
            "lag_days": None},
    }
    from datetime import date
    n26 = findings["nisar_2026"]
    n26["lag_days"] = (date.fromisoformat(n26["flooded_vegetation_peak"][0])
                       - date.fromisoformat(n26["open_water_peak"][0])).days
    n26["open_water_first_to_last"] = [
        [ts[0]["date"], ts[0]["open_water_km2"]], [ts[-1]["date"], ts[-1]["open_water_km2"]]]
    i26 = findings["inflow_2026"]
    findings["how_2026_compares"] = (
        f"By modelled peak monthly inflow, 2026 ({i26['peak_m3s']} m3/s) ranks "
        f"{i26['rank']} of {i26['of_years']} years ({i26['period'][0]}-"
        f"{i26['period'][1]}): an average year. Caution: this river model "
        "is unreliable in some years; it shows 2025 as weak although Landsat "
        "mapped a normal-sized flood that year.")

    ctx = {
        "instructions": (
            "You are the assistant of the EDev-Vision Okavango Delta flood app. "
            "Answer ONLY from the data and facts in this document. If the answer "
            "is not here, say you do not have that data. Quote numbers with "
            "their unit, date and source layer. Keep three sources apart and "
            "never add or average them: (1) NISAR L-band radar, 2026, open "
            "water AND flooded vegetation; (2) Landsat optical history, "
            "1984-2025, open water only; (3) modelled river discharge and "
            "headwater rainfall, the drivers. Satellites measure the AREA of "
            "water, not its depth, volume or quality: never state how much "
            "water is drinkable. Mention uncertainty when it applies "
            "(uncalibrated thresholds, low-coverage years, model not gauge). "
            "NISAR km2 and Landsat km2 measure different things: never put "
            "them side by side or say one is larger. To judge whether 2026 "
            "was a big year use ONLY the river-discharge rank; Landsat has no "
            "2026 value. Read each number from its own column; do not mix "
            "open water with flooded vegetation. If asked about the future, "
            "explain the forecast test result instead of predicting. "
            "Answer in the user's language, in plain words for non-experts, "
            "in at most 120 words. Plain sentences only: no tables, no "
            "headings, no citation marks or brackets."),
        "area": {"name": "Okavango Delta, Botswana", "aoi_lonlat": meta["aoi"]},
        "background_facts": [
            {"fact": "The flood is driven by rain that falls months earlier in "
                     "the Angolan highlands and travels down the Okavango "
                     "River; it reaches the Delta in the local dry season.",
             "source": "Inman & Lyons 2020, Remote Sensing 12(8):1348"},
            {"fact": "Mean annual input (river plus local rain) is about 16 "
                     "km3; roughly 96-98% returns to the atmosphere by "
                     "evapotranspiration and only about 2% leaves as surface "
                     "outflow towards Maun.",
             "source": "Okavango water-balance literature (EGU2011-13435; "
                       "Jao/Boro outflow modelling study)"},
            {"fact": "Open water is dark in radar (mirror reflection). Flooded "
                     "vegetation is bright in HH (double bounce between water "
                     "and stems). L-band penetrates the canopy; optical sensors "
                     "and shorter radar wavelengths mostly do not.",
             "source": "SAR physics; NISAR mission documentation"},
            {"fact": "Why the water shrinks within a year: river inflow peaks "
                     "around February-April, the flood wave takes months to "
                     "cross the flat Delta, and after it passes the inflow "
                     "keeps falling while evapotranspiration and seepage into "
                     "the sand remove the water. It is a normal seasonal "
                     "retreat, not a sign of a drying Delta.",
             "source": "discharge_2026_monthly_m3s in this document; "
                       "water-balance literature"},
            {"fact": "HYPOTHESIS (ours, not proven): flooded vegetation peaks "
                     "weeks after open water because the flood first fills "
                     "the open channels and lagoons and only later spreads "
                     "sideways into the vegetated floodplains.",
             "source": "EDev-Vision interpretation of the NISAR series"},
        ],
        # Everything below is written as compact tables, not one object per
        # row: the free tier of the hosted model allows only a few thousand
        # tokens per minute, so every repeated key name costs a question.
        "nisar_2026": {
            "what": "NISAR L-band radar, L2 GCOV, 80 m, one track/frame/pass, "
                    "so differences between dates are real surface change.",
            "columns": ["date", "open_water_km2", "flooded_vegetation_km2", "quality"],
            "rows": [[r["date"], r["open_water_km2"], r["flooded_veg_km2"],
                      r["quality"]] for r in ts],
            "no_data": meta["gaps"],
            "noise_filter": meta.get("despeckle", {}).get("method"),
            "caveats": [
                "Thresholds are physical rules, not calibrated on the ground: "
                "a consistent relative index and a LOWER bound, not an "
                "absolute area. The seasonal shape is more reliable than the level.",
                "The AOI and one radar frame do not cover the whole Delta.",
                "The last date is a VALIDATED product, the rest PROVISIONAL; "
                "its small rise in open water may be a calibration step."]},
        "optical_history": {
            "what": "Landsat optical (Digital Earth Africa WOfS annual), OPEN "
                    "water only, km2 of the AOI wet in >=10% of cloud-free "
                    "looks in that calendar year.",
            "water_km2_by_year": {r["year"]: r["water_km2"] for r in hist["years"]},
            "low_coverage_years": [r["year"] for r in hist["years"] if r["low_coverage"]],
            "low_coverage_meaning": "fewer than 8 looks: a lower bound, the "
                                    "flood may simply have been missed",
            "reliable_since": hist["reliable_since"],
            "caveat": "Misses water under vegetation; never compare its level "
                      "with NISAR's."},
        "jrc_occurrence_1984_2024": load("jrc/bounds.json").get("stats"),
        "drivers": {
            "what": "Peak monthly river discharge near Mohembo (GloFAS MODEL, "
                    "not a gauge) and Nov-Apr rainfall in the Angolan "
                    "headwaters (NASA POWER, three points).",
            "columns": ["year", "peak_discharge_m3s", "peak_month", "headwater_rain_mm"],
            "rows": [[y, a.get("peak_m3s"), a.get("peak_month", "")[5:] or None,
                      a.get("headwater_rain_mm")] for y, a in sorted(ann.items())],
            "discharge_2026_monthly_m3s": {
                r["month"][5:]: round(r["m3s"]) for r in drv["discharge"]["monthly"]
                if r["month"].startswith("2026")}},
        "findings": findings,
        "causes_and_people_by_year": story_table(),
        "data_quality": (load("quality.json").get("summary")
                         if (C.OUT / "quality.json").exists() else None),
        "forecast": {
            "shown_in_app": show, "verdict": forecast["verdict"],
            "tests": [{k: t[k] for k in ("predictor", "n_years", "correlation_r",
                                         "skill_vs_climatology")} for t in tests],
            "nisar": forecast["not_forecast"]},
    }
    out = C.OUT / "assistant_context.json"
    out.write_text(json.dumps(ctx, separators=(",", ":")), encoding="utf-8")
    print(f"\nforecast.show = {show}")
    print(f"assistant_context.json: {out.stat().st_size / 1024:.0f} KB")
    print(json.dumps(findings, indent=1)[:1500])


if __name__ == "__main__":
    main()
