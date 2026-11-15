"""
EDev-Vision | RSynthesis estimation engine: where open water is likely to be
in 2026-2030, as a probability (0 to 1) for every 120 m pixel.

Language: Python + NumPy, like the rest of the pipeline. Runs offline from
files that history.py, story.py and the inflow/rain fetchers already produced:
  cache/history/<year>.npz   Landsat/WOfS wet and clear-look counts per pixel
  data/history.json          yearly open-water area and coverage flags
  data/drivers.json, data/story.json, cache/era5_rain_*   the causes

Stage 1 - HOW MUCH water a year holds (one number per year)
  RSynthesis v2.1, pipeline/rsynthesis.py (full description there; writes
  rsynthesis.json). Causes reduced by exhaustive Bayesian subset search; an
  equal mixture of robust, censored Bayesian regressions on a Box-Cox scale.
  Validation repeats cause screening inside each held-out fold and uses
  rolling-origin forecasts.
  For 2027-2030 the causes are drawn from past years, so the band widens.

Stage 2 - WHERE that water goes (one probability per pixel)
  P(pixel i is water in year y) = sigmoid(a_i + b_i * z_y),
  z_y = the year's open-water area, standardised.
  A Bayesian logistic regression for every pixel that held water in at least
  one well-observed year, with Gaussian priors a ~ N(0, 5^2), b ~ N(0, 2.5^2).
  All pixels are fitted at once by Newton's method (a 2x2 linear system per
  pixel); the Laplace approximation gives each pixel's parameter uncertainty.
  Tested by leaving each year out: is our map better than the plain
  "how often was this pixel wet" map (Brier skill score)?

Stage 3 - The future map
  P_i(y) = E[ sigmoid(a_i + b_i * z) ], averaged over the stage-1 forecast of
  z and over the pixel's parameter uncertainty, using the probit
  approximation (MacKay 1992). Dry and wet scenarios use the 10th and 90th
  percentile of that year's forecast area.

Outputs (C.OUT):  future.json, future/<year>_<scenario>.webp, future/bounds.json

What this is NOT: a hydrological model, a forecast of NISAR flooded
vegetation, or a statement about water depth. Landsat sees open water only.

Usage:  python pipeline/future.py              (fit, validate, draw; ~17 min)
        python pipeline/future.py --maps-only  (reuse the saved map validation; ~6 min)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import history as H
import rsynthesis as RS

load = lambda name: json.loads((C.OUT / name).read_text(encoding="utf-8"))

YEARS_AHEAD = [2026, 2027, 2028, 2029, 2030]
SCENARIOS = ("expected", "dry", "wet")
TAU_A, TAU_B = 5.0, 2.5          # prior standard deviations (intercept, slope)
NODATA = 255
# Probability -> colour on the dark basemap. Below 5 % stays transparent.
RAMP = [(0.05, (34, 86, 120, 90)), (0.25, (46, 128, 182, 150)),
        (0.50, (78, 176, 226, 205)), (0.75, (150, 218, 246, 235)),
        (0.95, (232, 249, 255, 255))]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------- stage 1
def stage1():
    """Stage 1 = RSynthesis v2 (pipeline/rsynthesis.py). Also writes rsynthesis.json.
    Returns the summary for future.json (v1 field names kept so the site reads it),
    the leave-one-out hindcast, the 2026-2030 rows, and the forecast nodes:
    40 equal-probability yearly areas per year (projection and leave-one-out)."""
    doc, ens, loo, nodes = RS.engine()
    (C.OUT / "rsynthesis.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    val, eff = doc["step4_validation"], doc["step2_engine"]["effects"]
    lo, ro, sp = val["leave_one_out"], val["rolling_origin"], val["single_split"]
    k = eff["km2_per_unit_at_normal_year"]
    reliable = doc["step2_engine"]["years_fitted"]["exact"]
    screen = doc["step1_cause_reduction"]
    cause_names = [RS.CANDIDATES[c] for c in doc["summary"]["causes_used"]]
    core_names = [RS.CANDIDATES[c] for c in screen["core"]]
    contested_names = [RS.CANDIDATES[c] for c in screen["contested"]]
    formula = (f"Equal mixture of {doc['summary']['models']} robust Bayesian models on a Box-Cox scale; "
               f"core causes: {', '.join(core_names) or 'none'}; "
               f"optional causes: {', '.join(contested_names) or 'none'}. "
               "Student-t errors, low-coverage years as lower bounds, Gibbs sampling. "
               "Cause screening is repeated inside each validation fold.")
    info = {
        "engine": f"RSynthesis v{doc['version']} (pipeline/rsynthesis.py, details in rsynthesis.json)",
        "models": doc["summary"]["models"],
        "formula": formula,
        "causes_used": doc["summary"]["causes_used"],
        "causes_dropped": doc["summary"]["causes_dropped"],
        # v1 field names, filled with the v2 equivalents at a normal year:
        "coef": {"c1_km2_per_mm": k.get("headwater_rain"), "c2_km2_per_year": round(k["trend"], 1),
                 "local_rain_km2_per_mm": k.get("local_rain"), "enso_km2_per_point": k.get("enso")},
        "t_values": {"year": eff["trend_posterior_z"]},
        "n_years": len(reliable), "years": [reliable[0], reliable[-1]],
        "loo_skill": lo["v2"]["skill_vs_climatology"], "loo_rmse_km2": lo["v2"]["rmse_km2"],
        "loo_mae_km2": lo["v2"]["mae_km2"], "loo_crps_km2": lo["v2"]["crps_km2"],
        "chrono_skill": sp["v2"]["skill"], "chrono_rmse_km2": sp["v2"]["rmse_km2"],
        "chrono_split": sp["split"],
        "rolling_skill": ro["v2"]["skill_vs_climatology"], "rolling_rmse_km2": ro["v2"]["rmse_km2"],
        "rolling_from": RS.ROLL_FROM,
        "v1": {"loo_skill": lo["v1"]["skill_vs_climatology"], "loo_rmse_km2": lo["v1"]["rmse_km2"],
               "chrono_skill": sp["v1"]["skill"], "rolling_rmse_km2": ro["v1"]["rmse_km2"]},
        "rain_trend_mm_per_year": eff["rain_trend_mm_per_year"], "rain_trend_t": eff["rain_trend_t"],
        "outlier_years": doc["step3_outlier_logic"]["outlier_years"],
        "explained_outlier_years": doc["step3_outlier_logic"]["explained_years"],
    }
    hindcast = [{"year": r["year"], "observed_km2": round(float(r["observed_km2"]), 1),
                 "predicted_km2": r["loo_km2"], "low_km2": r["loo_p10_km2"], "high_km2": r["loo_p90_km2"]}
                for r in doc["years"] if r["year"] in loo]
    loo_nodes = {y: loo[y]["nodes"] for y in loo}
    return info, hindcast, doc["step5_projection"]["years"], nodes, loo_nodes


# --------------------------------------------------------------- stage 2
def year_masks(years):
    """Water and valid masks per well-observed year, same rule as history.py."""
    W, V = [], []
    for y in years:
        with np.load(C.CACHE / "history" / f"{y}.npz") as z:
            wet, clear = z["wet"].astype(np.int32), z["clear"].astype(np.int32)
        valid = clear > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(valid, wet / np.maximum(clear, 1), 0)
        W.append(valid & (freq >= H.F_ANY) & (wet >= H.MIN_WET))
        V.append(valid)
    return np.stack(W), np.stack(V)


def fit_pixels(Wf, Vf, z, iters=30):
    """MAP Bayesian logistic regression for every pixel at once.
    Wf, Vf: (years, pixels) water / valid; z: (years,) standardised area.
    Returns a, b and the Laplace covariance (saa, sab, sbb)."""
    m = Vf.astype(np.float64)
    w_obs = Wf.astype(np.float64) * m
    n_valid = m.sum(0)
    freq = (w_obs.sum(0) + 0.5) / (n_valid + 1.0)
    a = np.log(freq / (1 - freq))
    b = np.zeros_like(a)
    zc = z[:, None]
    for _ in range(iters):
        p = sigmoid(a[None, :] + b[None, :] * zc)
        r = w_obs - p * m
        ga = r.sum(0) - a / TAU_A ** 2
        gb = (r * zc).sum(0) - b / TAU_B ** 2
        w = p * (1 - p) * m
        haa = w.sum(0) + 1 / TAU_A ** 2
        hab = (w * zc).sum(0)
        hbb = (w * zc ** 2).sum(0) + 1 / TAU_B ** 2
        det = haa * hbb - hab ** 2
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a += da
        b += db
        if max(np.abs(da).max(), np.abs(db).max()) < 1e-7:
            break
    p = sigmoid(a[None, :] + b[None, :] * zc)
    w = p * (1 - p) * m
    haa = w.sum(0) + 1 / TAU_A ** 2
    hab = (w * zc).sum(0)
    hbb = (w * zc ** 2).sum(0) + 1 / TAU_B ** 2
    det = haa * hbb - hab ** 2
    return a, b, hbb / det, -hab / det, haa / det


def logit_at(a, b, saa, sab, sbb, z):
    """Log-odds of water at standardised area z, shrunk by each pixel's own
    parameter uncertainty (probit approximation, MacKay 1992)."""
    var = saa + 2 * z * sab + z ** 2 * sbb
    return (a + b * z) / np.sqrt(1 + math.pi * var / 8)


def calibrate(eta, target_km2):
    """One global shift d in log-odds so that sum(sigmoid(eta + d)) * pixel area
    equals the target area (bisection). Keeps the map and the yearly number
    in agreement. eta: (..., pixels); target_km2: scalar or (...,)."""
    target = np.asarray(target_km2, float)[..., None] / H.KM2
    lo = np.full(target.shape, -30.0)
    hi = np.full(target.shape, 30.0)
    for _ in range(45):
        mid = (lo + hi) / 2
        big = sigmoid(eta + mid).sum(-1, keepdims=True) > target
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    p = sigmoid(eta + (lo + hi) / 2)
    return np.where(target > 0, p, 0.0)


def expected_map(params, nodes_km2, mu_a, sd_a):
    """Average of the calibrated maps over the equal-probability nodes of the
    stage-1 forecast distribution of the yearly area."""
    a, b, saa, sab, sbb = params
    out = np.zeros_like(a)
    for target in nodes_km2:
        out += calibrate(logit_at(a, b, saa, sab, sbb, (target - mu_a) / sd_a), target)
    return out / len(nodes_km2)


def brier(p, y):
    return float(((p - y) ** 2).mean())


def iou(p, truth):
    pred, t = p >= 0.5, truth > 0
    return float((pred & t).sum() / max((pred | t).sum(), 1))


def validate(Wf, Vf, area, years, loo_nodes):
    """Leave each year out and redraw it from the other 26.
    'where': both maps are scaled to the year's true area, so only the
             placement of water is tested (ours vs. the 'how often was this
             pixel wet' map).
    'full':  ours gets only the stage-1 forecast distribution of the area
             (RSynthesis v2, refitted without the year);
             the reference is the plain frequency map (an 'average year')."""
    n = len(years)
    rows, sums = [], {"m": 0.0, "c": 0.0, "f": 0.0, "fc": 0.0}
    ious_m, ious_c = [], []
    for k in range(n):
        keep = np.arange(n) != k
        mu, sd = area[keep].mean(), area[keep].std(ddof=1)
        zk = (area - mu) / sd
        params = fit_pixels(Wf[keep], Vf[keep], zk[keep], iters=20)
        ok = Vf[k]
        truth = Wf[k][ok].astype(float)
        true_km2 = truth.sum() * H.KM2
        a, b, saa, sab, sbb = (v[ok] for v in params)
        cnt = (Wf[keep] & Vf[keep]).sum(0)[ok]
        freq = (cnt + 0.5) / (Vf[keep].sum(0)[ok] + 1.0)
        p_m = calibrate(logit_at(a, b, saa, sab, sbb, zk[k]), true_km2)
        p_c = calibrate(np.log(freq / (1 - freq)), true_km2)
        m, c = brier(p_m, truth), brier(p_c, truth)
        sums["m"] += m
        sums["c"] += c
        ious_m.append(iou(p_m, truth))
        ious_c.append(iou(p_c, truth))
        row = {"year": int(years[k]), "true_km2": round(true_km2, 1),
               "brier_skill_where": round(1 - m / c, 3),
               "iou": round(ious_m[-1], 3), "iou_frequency_map": round(ious_c[-1], 3)}
        if int(years[k]) in loo_nodes:
            nodes_k = loo_nodes[int(years[k])]
            pf = expected_map((a, b, saa, sab, sbb), nodes_k, mu, sd)
            f, fc = brier(pf, truth), brier(freq, truth)
            sums["f"] += f
            sums["fc"] += fc
            row["predicted_km2"] = round(float(np.mean(nodes_k)), 1)
            row["brier_skill_full"] = round(1 - f / fc, 3)
        rows.append(row)
    summary = {
        "brier_skill_where": round(1 - sums["m"] / sums["c"], 3),
        "brier_skill_full": round(1 - sums["f"] / sums["fc"], 3) if sums["fc"] else None,
        "mean_iou": round(float(np.mean(ious_m)), 3),
        "mean_iou_frequency_map": round(float(np.mean(ious_c)), 3),
    }
    return summary, rows


# --------------------------------------------------------------- rendering
def colourise(p8):
    """uint8 probability (0..250, 255 = no data) -> RGBA."""
    p = p8.astype(np.float32) / 250.0
    rgba = np.zeros((*p8.shape, 4), np.uint8)
    xs = [s for s, _ in RAMP]
    for ch in range(4):
        ys = [c[ch] for _, c in RAMP]
        rgba[..., ch] = np.interp(p, xs, ys).astype(np.uint8)
    rgba[(p < RAMP[0][0]) | (p8 == NODATA)] = 0
    return rgba


def main() -> None:
    t0 = time.time()
    hist = load("history.json")

    s1, hindcast, proj1, proj_nodes, loo_nodes = stage1()
    print(f"stage 1  LOO skill {s1['loo_skill']}  chrono {s1['chrono_skill']}  rolling RMSE "
          f"{s1['rolling_rmse_km2']}  trend {s1['coef']['c2_km2_per_year']} km2/yr (z {s1['t_values']['year']})")

    # Stage 2 uses every well-observed year (the area of each year is the target's scale).
    rows = {r["year"]: r for r in hist["years"]}
    years = np.array([y for y in sorted(rows) if not rows[y]["low_coverage"]])
    area = np.array([rows[y]["water_km2"] for y in years], float)
    Wst, Vst = year_masks(years)
    Hh, Ww = Wst.shape[1:]
    domain = (Wst & Vst).any(0)
    idx = np.flatnonzero(domain.ravel())
    Wf = Wst.reshape(len(years), -1)[:, idx]
    Vf = Vst.reshape(len(years), -1)[:, idx]
    print(f"stage 2  {len(years)} years, {idx.size:,} candidate pixels "
          f"({idx.size * H.KM2:,.0f} km2 ever wet)")

    if "--maps-only" in sys.argv and (C.OUT / "future.json").exists():
        saved = load("future.json")["stage2"]["validation"]
        per_year = saved.pop("per_year")
        saved.pop("meaning", None)
        summary = saved
    else:
        summary, per_year = validate(Wf, Vf, area, years, loo_nodes)
    print(f"         LOO Brier skill (where) {summary['brier_skill_where']}  "
          f"(full) {summary['brier_skill_full']}  IoU {summary['mean_iou']} "
          f"vs frequency map {summary['mean_iou_frequency_map']}")

    mu_a, sd_a = area.mean(), area.std(ddof=1)
    z = (area - mu_a) / sd_a
    params = fit_pixels(Wf, Vf, z)

    # Draw only where Landsat saw the ground in each of the last five years.
    seen = Vst[-5:].all(0)
    on = seen.ravel()[idx]
    params = tuple(v[on] for v in params)
    idx = idx[on]
    a, b, saa, sab, sbb = params
    x0, y1, gw, gh = H.wofs_grid()
    tf = Affine(H.PIX, 0, x0, 0, -H.PIX, y1)
    out = C.OUT / "future"
    out.mkdir(parents=True, exist_ok=True)

    projection, bounds = [], None
    for year in YEARS_AHEAD:
        f = next(r for r in proj1 if r["year"] == year)
        q = {k: f[f"{k}_km2"] for k in ("p025", "p10", "p50", "p90", "p975")}
        rec = {"year": year, "rain_mm": round(f["headwater_rain"]) if f["causes_known"] else None,
               "rain_known": f["causes_known"],
               "delta_rain_mm": round(f["local_rain"]) if f["causes_known"] else None,
               "enso": f.get("enso"),
               "mean_km2": f["mean_km2"], "sd_km2": f["sd_km2"],
               **{f"{k}_km2": v for k, v in q.items()},
               "nodes_km2": [round(float(v), 1) for v in proj_nodes[year]], "map_km2": {}}
        for scen in SCENARIOS:
            if scen == "expected":
                p = expected_map(params, proj_nodes[year], mu_a, sd_a)
            else:
                target = q["p10"] if scen == "dry" else q["p90"]
                p = calibrate(logit_at(a, b, saa, sab, sbb, (target - mu_a) / sd_a), target)
            grid = np.zeros(Hh * Ww, np.float64)
            grid[idx] = p
            grid = grid.reshape(Hh, Ww)
            rec["map_km2"][scen] = round(float(grid.sum() * H.KM2), 1)
            if scen == "expected":
                rec["likely_km2"] = round(float((grid >= 0.5).sum() * H.KM2), 1)
            p8 = np.where(seen, np.round(grid * 250), NODATA).astype(np.uint8)
            web, bounds = H.to_web(p8, tf, "EPSG:6933", NODATA)
            # Shown at about 180 m: two thirds of the grid is already finer than the screen.
            # Resample the probability (no-data as 0) before colouring, so edges get no dark halo.
            prob = Image.fromarray(np.where(web == NODATA, 0, web).astype(np.uint8), "L")
            small = np.asarray(prob.resize((web.shape[1] * 2 // 3, web.shape[0] * 2 // 3), Image.BOX))
            Image.fromarray(colourise(small), "RGBA").save(out / f"{year}_{scen}.webp", quality=70, method=6)
        projection.append(rec)
        print(f"{year}  causes {'known' if f['causes_known'] else 'unknown'}  "
              f"median {rec['p50_km2']:>7}  80% {rec['p10_km2']}-{rec['p90_km2']}  "
              f"map {rec['map_km2']}")
    (out / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")

    doc = {
        "name": "RSynthesis estimation engine",
        "version": RS.VERSION,
        "generated": time.strftime("%Y-%m-%d"),
        "what": "Probability (0 to 1) that each 120 m pixel of the study area holds open "
                "water in a given year, 2026-2030, with the yearly open-water area it implies.",
        "stage1": {
            "question": "How much open water will the year hold?",
            "method": (f"RSynthesis v{RS.VERSION}: causes reduced by exhaustive Bayesian subset "
                       f"search ({len(RS.CANDIDATES)} candidates); an equal mixture of {s1['models']} "
                       "robust Bayesian regressions on a Box-Cox scale, with Student-t errors and "
                       "low-coverage years as lower bounds. Cause screening is repeated inside each "
                       "validation fold. coef and t_values give the model equivalents at a normal year "
                       "in v1's units."),
            **s1,
        },
        "stage2": {
            "question": "Where will that water be?",
            "method": "Bayesian logistic regression for every pixel that held open water in at least "
                      "one well-observed year: P(water) = sigmoid(a + b * standardised yearly area), "
                      "priors a ~ N(0, 5^2), b ~ N(0, 2.5^2), fitted for all pixels at once by "
                      "Newton's method; Laplace approximation for parameter uncertainty.",
            "pixels": int(idx.size), "pixel_m": H.PIX, "years_used": [int(y) for y in years],
            "validation": {**summary, "per_year": per_year,
                           "meaning": "Brier skill > 0 means the map beats the plain 'how often was "
                                      "this pixel wet' map. 'where' gives the model the true yearly "
                                      "area; 'full' gives it only the stage-1 forecast, refitted without the year."},
        },
        "stage3": {
            "method": "For a given yearly area the pixel log-odds a + b z (shrunk by each pixel's "
                      "parameter uncertainty, probit approximation, MacKay 1992) get one global "
                      "shift, found by bisection, so the map holds exactly that area. The expected "
                      "map averages these maps over 40 equal-probability nodes of the stage-1 "
                      "forecast; dry and wet use its 10th and 90th percentiles.",
            "scenarios": {"expected": "averaged over the whole forecast range",
                          "dry": "10th percentile of that year's forecast area",
                          "wet": "90th percentile of that year's forecast area"},
        },
        "projection": projection,
        "hindcast": hindcast,
        "climatology": {"mean_km2": round(float(mu_a), 1), "sd_km2": round(float(sd_a), 1),
                        "years": [int(years[0]), int(years[-1])], "n_years": int(len(years))},
        "images": {"pattern": "future/<year>_<scenario>.webp", "bounds": "future/bounds.json",
                   "years": YEARS_AHEAD, "scenarios": list(SCENARIOS)},
        "legend": [{"p": p, "color": list(c)} for p, c in RAMP],
        "caveats": [
            "A statistical projection, not a certainty: rain and El Nino after April 2026 are unknown, "
            "so 2027-2030 use the causes of past years and the band is wide.",
            "Landsat sees OPEN water only. Water under reeds and papyrus (the NISAR flooded-"
            "vegetation class) is not part of this projection.",
            "The downward trend is measured in this study area only. Part of it may be water "
            "moving between channels of the Delta, or a long dry phase, rather than a permanent loss.",
            "Pixels never wet in the 27 well-observed years stay near zero by design.",
        ],
    }
    (C.OUT / "future.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"future.json written in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
