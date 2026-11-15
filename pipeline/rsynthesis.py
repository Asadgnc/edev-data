"""
EDev-Vision | RSynthesis v2.1: how much open water a year will hold, with as
few causes as the data allow and a logic for the outlier years.

Replaces stage 1 of future.py (v1 was: area = c0 + c1 * Angolan rain + c2 * year,
least squares). Runs offline from files the pipeline already wrote:
  data/history.json      yearly Landsat open-water area + low_coverage flag
  data/drivers.json      ERA5 rain in the Angolan headwaters, GloFAS discharge
  data/story.json        heat and El Nino values
  data/inflow_daily.csv  measured Mohembo discharge (only tested as a cause)
  cache/era5_rain_*      ERA5 rain over the Delta and 17 basin points

1  REDUCE THE CAUSES (discrete search + Bayesian evidence)
   All 2^6 = 64 subsets of the candidate causes are scored by their exact
   marginal likelihood under Zellner's g-prior (g = n) with the Scott-Berger
   beta-binomial prior on model size. Each cause gets a posterior inclusion
   probability (PIP), i.e. the odds that the data want it:
     PIP >= 2/3   (odds >= 2:1 for)      -> core, in every model
     PIP <= 1/3   (odds >= 2:1 against)  -> dropped
     otherwise                           -> contested
   Causes that exist for only part of the record (GloFAS, the gauge) and the
   ideas for the unexplained years get an exact leave-one-out added-value
   test (hat matrix, e_i / (1 - h_ii)). Partial correlations (inverse
   covariance) show which causes act directly.

2  THE ENGINE (probability + linear algebra)
   For each model: z = BoxCox(area, lambda), lambda by profile likelihood
   (the flood answers rain multiplicatively, so a straight line in km2 is the
   wrong shape at the extremes); z ~ Student-t(x b, s2, nu) written as a scale
   mixture of normals, so every year has a latent weight w (w << 1: the
   causes do not explain that year). Low-coverage years are lower bounds and
   enter as right-censored observations. Gibbs sampling with data augmentation:
     censored z | ...  truncated normal (inverse CDF)
     nu | ...          discrete grid, Student-t likelihood with w integrated out
     w | nu, ...       Gamma((nu+1)/2, (nu + r^2/s2)/2)
     b | ...           Gaussian, Cholesky factor of the weighted normal equations
     s2 | ...          inverse Gamma
   THE ENSEMBLE: core causes x every on/off pattern of the contested causes
   (a 2^k factorial of models), combined with EQUAL weights. Weights learnt
   from past forecast scores were tested and lost out of time (too few years
   to learn them: the "forecast combination puzzle").

3  OUTLIER LOGIC
   Outliers are the years v1 missed by more than its own RMSE. We test what
   they share (correlation of the v1 miss with each cause and with the
   curvature term) and whether v2 removes it; then we test six ideas for the
   years v2 still cannot explain. Each year gets exact Shapley contributions
   (all orderings of the causes) that split its departure from a normal year
   into km2 per cause.

4  HONEST ERROR
   (a) Nested leave-one-year-out: cause screening and the model are refitted
       without the held-out year. (b) Rolling origin: every year from 2006
       is forecast from the years before it only. v1, a pre-specified
       baseline (v1's causes on the new engine) and v2 are scored on RMSE,
       MAE, skill against climatology, CRPS (proper score for the whole
       forecast distribution) and coverage of the 80 % interval.

5  PROJECTION 2026-2030
   2026: its rain and its El Nino state are known, so it is a real
   prediction. 2027-2030: each draw takes the causes of one random past year
   together (keeps their correlations); the memory check uses consecutive
   years only. The trend is extrapolated with its posterior uncertainty.

Output: data/rsynthesis.json. future.py imports engine() for stage 1.
Usage:  python pipeline/rsynthesis.py          (~3 min)
"""
from __future__ import annotations

import calendar
import csv
import itertools
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from context import annual_drivers

ND = NormalDist()
load = lambda name: json.loads((C.OUT / name).read_text(encoding="utf-8"))

FIRST, LAST = 1984, 2025            # years with a Landsat area
AHEAD = [2026, 2027, 2028, 2029, 2030]
ROLL_FROM = 2006                    # rolling-origin forecasts start here
DELTA = (-19.3, 22.9)               # same point as story.py
NU_GRID = np.array([4, 5, 6, 8, 12, 20, 40, 1000.0])
ITERS, BURN = 6000, 1500            # final fit, per model
ITERS_CV, BURN_CV = 2000, 500       # each validation fold, per model
NODES = 40                          # equal-probability nodes handed to future.py
SEED = 2026
VERSION = "2.1"
CORE, DROP = 2 / 3, 1 / 3           # PIP thresholds (2:1 odds)
RIVER_SIGNAL = 0.10                 # river must differ >= 10 % from its rain to count
# ERA5 points across the Cubango (west) and Cuito (east) catchments above Mohembo,
# used only to test whether WHERE the rain fell explains the outlier years.
BASIN_POINTS = [(-12.77, 15.74), (-13.5, 16.2), (-14.0, 16.8), (-14.66, 17.69), (-15.5, 17.6),
                (-16.4, 18.3), (-17.4, 19.0), (-12.8, 18.6), (-13.6, 18.9), (-14.4, 19.1),
                (-15.16, 19.17), (-16.0, 19.8), (-16.9, 20.3), (-17.6, 20.8), (-13.2, 17.3),
                (-14.3, 18.0), (-15.6, 18.6)]
CANDIDATES = {                      # id -> label (all known for every year)
    "headwater_rain": "Rain in the Angolan highlands (Nov-Apr)",
    "local_rain": "Rain over the Delta (Nov-Apr)",
    "heat": "Flood-season heat (May-Oct)",
    "enso": "El Nino / La Nina (Dec-Feb ocean index)",
    "prev_headwater_rain": "Last year's rain in the Angolan highlands",
    "trend": "Calendar year (slow change)",
}
BASELINE = ["headwater_rain", "trend"]   # v1's causes, pre-specified


# ------------------------------------------------------------------ data
def season_total(monthly_mm_day: dict, y: int):
    """Nov (y-1) .. Apr (y) rain in mm; None if a month is missing."""
    s = 0.0
    for a, b in [(y - 1, 11), (y - 1, 12), (y, 1), (y, 2), (y, 3), (y, 4)]:
        v = monthly_mm_day.get(f"{a}-{b:02d}")
        if v is None:
            return None
        s += v * calendar.monthrange(a, b)[1]
    return s


def gauge_feb_may():
    """Measured Mohembo discharge, Feb-May mean of monthly means (m3/s)."""
    f = C.OUT / "inflow_daily.csv"
    if not f.exists():
        return {}
    by = defaultdict(list)
    with f.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["discharge_m3_s"] and r["quality_flag"] == "measured":
                by[r["date"][:7]].append(float(r["discharge_m3_s"]))
    out = {}
    for y in range(1975, 2026):
        ms = [by.get(f"{y}-{m:02d}") for m in (2, 3, 4, 5)]
        if all(ms):
            out[y] = float(np.mean([np.mean(v) for v in ms]))
    return out


def load_table():
    from history import era5_monthly_rain
    hist, drv, story = load("history.json"), load("drivers.json"), load("story.json")
    ann = annual_drivers(drv)
    local = era5_monthly_rain(*DELTA)
    num = lambda s: float(re.findall(r"[-+]?\d+\.?\d*", s.replace(",", ""))[0])
    sv = {int(y): {c["id"]: num(c["value"]) for c in r["causes"]} for y, r in story["years"].items()}
    rows = {r["year"]: r for r in hist["years"]}
    gauge = gauge_feb_may()
    T = {}
    for y in range(FIRST - 1, AHEAD[-1] + 1):
        rec = {"year": y}
        if y in rows:
            rec["area"] = float(rows[y]["water_km2"])
            rec["low_coverage"] = bool(rows[y]["low_coverage"])
        a = ann.get(y, {})
        rec["headwater_rain"] = a.get("headwater_rain_mm")
        rec["glofas_feb_apr"] = a.get("feb_apr_m3s")
        lr = season_total(local, y)
        rec["local_rain"] = round(lr, 1) if lr is not None else None
        rec["heat"] = sv.get(y, {}).get("heat")
        rec["enso"] = sv.get(y, {}).get("enso")
        rec["gauge_feb_may"] = gauge.get(y)
        rec["trend"] = float(y - 2000)
        T[y] = rec
    for y in T:
        T[y]["prev_headwater_rain"] = T.get(y - 1, {}).get("headwater_rain")
    return T


def has(T, y, causes):
    return all(T.get(y, {}).get(c) is not None for c in causes)


# ------------------------------------------------------------- transforms
def boxcox(x, lam):
    x = np.maximum(np.asarray(x, float), 1.0)
    return np.log(x) if abs(lam) < 1e-9 else (x ** lam - 1) / lam


def inv_boxcox(z, lam):
    z = np.asarray(z, float)
    return np.exp(z) if abs(lam) < 1e-9 else np.maximum(lam * z + 1, 0) ** (1 / lam)


def profile_lambda(area, X, grid=np.linspace(-0.3, 1.2, 61)):
    """Box-Cox lambda maximising the profile likelihood (with the Jacobian)."""
    Xd = np.column_stack([np.ones(len(area)), X])
    best = (-np.inf, 1.0)
    for lam in grid:
        z = boxcox(area, lam)
        b = np.linalg.lstsq(Xd, z, rcond=None)[0]
        s2 = max(float(((z - Xd @ b) ** 2).mean()), 1e-12)
        ll = -len(area) / 2 * math.log(s2) + (lam - 1) * float(np.log(np.maximum(area, 1)).sum())
        if ll > best[0]:
            best = (ll, float(lam))
    return best[1]


# ---------------------------------------------------------- 1 cause search
def log_evidence(X, z, g):
    """log marginal likelihood of y = a + X b under Zellner's g-prior (constants dropped)."""
    n = len(z)
    zc = z - z.mean()
    tss = float(zc @ zc)
    if X.shape[1] == 0:
        return -(n - 1) / 2 * math.log(tss)
    Xc = X - X.mean(0)
    b = np.linalg.lstsq(Xc, zc, rcond=None)[0]
    r2 = 1 - float(((zc - Xc @ b) ** 2).sum()) / tss
    p = X.shape[1]
    # The closed-form g-prior evidence assumes a proper residual subspace.
    # A saturated model can drive R² to one and make comparison meaningless,
    # especially in early rolling-origin folds.
    if p >= n - 1:
        return -np.inf
    return ((n - 1 - p) / 2 * math.log(1 + g) - (n - 1) / 2 * math.log(1 + g * (1 - r2))
            - (n - 1) / 2 * math.log(tss))


def press_skill(X, z):
    """Exact leave-one-out skill of least squares via the hat matrix: e_i / (1 - h_ii)."""
    n = len(z)
    Xd = np.column_stack([np.ones(n), X])
    h = np.einsum("ij,ji->i", Xd, np.linalg.pinv(Xd))
    e = (z - Xd @ np.linalg.lstsq(Xd, z, rcond=None)[0]) / (1 - h)
    clim = (z - z.mean()) * n / (n - 1)
    return 1 - float(e @ e) / float(clim @ clim)


def screen_causes(T, years):
    names = list(CANDIDATES)
    area = np.array([T[y]["area"] for y in years])
    X = np.array([[T[y][k] for k in names] for y in years], float)
    Xs = (X - X.mean(0)) / X.std(0)
    lam = profile_lambda(area, Xs[:, [names.index("headwater_rain")]])
    z = boxcox(area, lam)
    n, P = len(z), len(names)
    models = []
    for k in range(P + 1):
        for s in itertools.combinations(range(P), k):
            prior = -math.log(P + 1) - math.log(math.comb(P, k))   # Scott-Berger
            models.append((s, log_evidence(Xs[:, list(s)], z, g=n) + prior))
    lml = np.array([m[1] for m in models])
    w = np.exp(lml - lml.max())
    w /= w.sum()
    pip = {names[j]: float(sum(wi for (s, _), wi in zip(models, w) if j in s)) for j in range(P)}
    core = [k for k in names if pip[k] >= CORE]
    contested = [k for k in names if DROP < pip[k] < CORE]
    dropped = [k for k in names if pip[k] <= DROP]
    sets = [core + [c for c in contested if c in on]
            for r in range(len(contested) + 1) for on in itertools.combinations(contested, r)]
    top = sorted(zip(models, w), key=lambda t: -t[1])[:6]

    # Partial correlations from the inverse covariance (Gaussian graphical model).
    M = np.column_stack([z, Xs])
    # Keep the descriptive summary finite when predictors are collinear in a
    # small training fold.
    prec = np.linalg.pinv(np.cov(M, rowvar=False))
    d = np.sqrt(np.diag(prec))
    pcor = -prec / np.outer(d, d)
    partial = {names[j]: round(float(pcor[0, j + 1]), 2) for j in range(P)}
    marginal = {names[j]: round(float(np.corrcoef(z, Xs[:, j])[0, 1]), 2) for j in range(P)}
    ce = lambda k: round(float(np.corrcoef(Xs[:, names.index("enso")], Xs[:, names.index(k)])[0, 1]), 2)
    enso_with = {"headwater_rain": ce("headwater_rain"), "local_rain": ce("local_rain"), "heat": ce("heat")}

    # Partial-record causes: do they add to the full v2 cause set on their own years?
    full = core + contested
    added = {}
    for cid, label in (("glofas_feb_apr", "GloFAS modelled inflow, Feb-Apr (from 1997)"),
                       ("gauge_feb_may", "Measured Mohembo inflow, Feb-May (gauge)")):
        ys = [y for y in years if T[y].get(cid)]
        if len(ys) < 12:
            continue
        a = boxcox(np.array([T[y]["area"] for y in ys]), lam)
        base = np.array([[T[y][k] for k in full] for y in ys], float)
        q = np.log([T[y][cid] for y in ys])
        added[cid] = {"label": label, "n_years": len(ys), "years": [ys[0], ys[-1]],
                      "loo_skill_v2_causes": round(press_skill(base, a), 3),
                      "loo_skill_v2_causes_plus_this": round(press_skill(np.column_stack([base, q]), a), 3),
                      "loo_skill_this_alone": round(press_skill(q[:, None], a), 3)}
    return {
        "candidates": CANDIDATES,
        "n_years": n, "lambda_used": round(lam, 3),
        "models_scored": len(models),
        "inclusion_probability": {k: round(v, 3) for k, v in pip.items()},
        "core": core, "contested": contested, "dropped": dropped,
        "ensemble_cause_sets": sets,
        "rule": "PIP >= 2/3 (odds 2:1 for): core, in every model. PIP <= 1/3 (odds 2:1 against): "
                "dropped. In between: contested; the ensemble holds every on/off pattern of the "
                "contested causes with equal weight.",
        "top_models": [{"causes": [names[j] for j in s], "posterior": round(float(wi), 3)}
                       for (s, _), wi in top],
        "partial_correlation_with_area": partial,
        "plain_correlation_with_area": marginal,
        "enso_correlation_with": enso_with,
        "partial_record_causes": added,
    }


# --------------------------------------------------------------- 2 engine
def gibbs(X, z, cens, iters, burn, seed, robust=True):
    """Bayesian Box-Cox-scale regression with Student-t errors and right-censored
    observations (z_i is only a lower bound where cens_i). Returns posterior draws."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    zl = z.copy()
    w = np.ones(n)
    b = np.linalg.lstsq(X, z, rcond=None)[0]
    s2 = float(np.var(z - X @ b))
    nu = 8.0
    P0 = np.eye(p) / (10 * np.std(z)) ** 2       # weak N(0, (10 sd(z))^2) on scaled slopes
    P0[0, 0] = 1e-10                              # flat on the intercept
    ci = np.flatnonzero(cens)
    keep_b, keep_s2, keep_nu, keep_w = [], [], [], []
    for it in range(iters):
        mu = X @ b
        for i in ci:                              # truncated normal above the lower bound
            sd = math.sqrt(s2 / w[i])
            lo = ND.cdf((z[i] - mu[i]) / sd)
            u = min(max(lo + (1 - lo) * rng.random(), 1e-12), 1 - 1e-12)
            zl[i] = mu[i] + sd * ND.inv_cdf(u)
        r = zl - mu
        if robust:
            # nu from the Student-t likelihood of the residuals with w integrated
            # out (collapsed step: mixes far better than nu | w), then w | nu.
            u = r ** 2 / s2
            lp = np.array([n * (math.lgamma((v + 1) / 2) - math.lgamma(v / 2) - 0.5 * math.log(v))
                           - (v + 1) / 2 * float(np.log1p(u / v).sum()) for v in NU_GRID])
            pr = np.exp(lp - lp.max())
            nu = float(rng.choice(NU_GRID, p=pr / pr.sum()))
            w = rng.gamma((nu + 1) / 2, 2 / (nu + u))
        Q = X.T @ (X * w[:, None]) / s2 + P0
        L = np.linalg.cholesky(Q)
        m = np.linalg.solve(Q, X.T @ (w * zl) / s2)
        b = m + np.linalg.solve(L.T, rng.standard_normal(p))
        r = zl - X @ b
        s2 = 1 / rng.gamma(n / 2, 2 / float((w * r ** 2).sum()))
        if it >= burn:
            keep_b.append(b)
            keep_s2.append(s2)
            keep_nu.append(nu)
            keep_w.append(w.copy())
    return {"b": np.array(keep_b), "s2": np.array(keep_s2), "nu": np.array(keep_nu),
            "w": np.array(keep_w)}


class Engine:
    """One fitted model: its causes -> posterior predictive of the area (km2)."""

    def __init__(self, T, years, causes, iters=ITERS, burn=BURN, seed=SEED):
        self.causes = list(causes)
        F = np.array([[T[y][k] for k in causes] for y in years], float)
        self.mu, self.sd = F.mean(0), F.std(0)
        area = np.array([T[y]["area"] for y in years])
        cens = np.array([T[y]["low_coverage"] for y in years])
        self.lam = profile_lambda(area[~cens], self.scale(F[~cens]))
        self.post = gibbs(self.design(F), boxcox(area, self.lam), cens, iters, burn, seed)
        self.years = list(years)

    def scale(self, F):
        return (np.asarray(F, float) - self.mu) / self.sd

    def design(self, F):
        F = np.atleast_2d(F)
        return np.column_stack([np.ones(len(F)), self.scale(F)])

    def rows(self, x):
        """x: dict cause -> value (one setting) or cause -> array (one per draw)."""
        return np.column_stack([np.broadcast_to(np.asarray(x[c], float), (len(self.post["b"]),))
                                for c in self.causes])

    def draws(self, x, seed=0):
        rng = np.random.default_rng(seed)
        b, s2, nu = self.post["b"], self.post["s2"], self.post["nu"]
        X = self.design(self.rows(x))
        w = rng.gamma(nu / 2, 2 / nu)
        z = (b * X).sum(1) + np.sqrt(s2 / w) * rng.standard_normal(len(b))
        return inv_boxcox(z, self.lam)

    def point(self, x):
        """Back-transformed linear predictor at the posterior median (no noise)."""
        b = np.median(self.post["b"], 0)
        return float(inv_boxcox(self.design([x[c] for c in self.causes])[0] @ b, self.lam))


class Ensemble:
    """Equal-weight mixture of Engines (one per cause set)."""

    def __init__(self, T, years, sets, iters=ITERS, burn=BURN, seed=SEED):
        self.members = [Engine(T, years, s, iters, burn, seed + 7 * k) for k, s in enumerate(sets)]
        self.causes = sorted({c for s in sets for c in s}, key=list(CANDIDATES).index)
        self.years = list(years)

    def draws(self, x, seed=0):
        return np.concatenate([m.draws(x, seed + k) for k, m in enumerate(self.members)])

    def point(self, x):
        return float(np.mean([m.point(x) for m in self.members]))

    def weight(self, y):
        """Mean latent weight of a fitted year over the members (< 1: an outlier)."""
        vals = [float(m.post["w"][:, m.years.index(y)].mean()) for m in self.members if y in m.years]
        return float(np.mean(vals)) if vals else None


def summarise(d):
    q = np.quantile(d, [0.025, 0.10, 0.50, 0.90, 0.975])
    nodes = np.quantile(d, (np.arange(NODES) + 0.5) / NODES)
    return {"mean": float(d.mean()), "sd": float(d.std()), "p025": float(q[0]), "p10": float(q[1]), "p50": float(q[2]),
            "p90": float(q[3]), "p975": float(q[4]), "nodes": nodes}


def crps_samples(d, y):
    s = np.sort(d)
    m = len(s)
    i = np.arange(1, m + 1)
    return float(np.abs(s - y).mean() - (2 * i - m - 1) @ s / m ** 2)


def crps_normal(mu, sd, y):
    zz = (y - mu) / sd
    return float(sd * (zz * (2 * ND.cdf(zz) - 1) + 2 * math.exp(-zz * zz / 2) / math.sqrt(2 * math.pi)
                       - 1 / math.sqrt(math.pi)))


# ------------------------------------------------------------- v1 for comparison
def v1_predict(T, train, year):
    """The v1 stage 1: least squares on headwater rain and year, normal predictive."""
    X = np.array([[1, T[y]["headwater_rain"], T[y]["trend"]] for y in train], float)
    a = np.array([T[y]["area"] for y in train])
    b, *_ = np.linalg.lstsq(X, a, rcond=None)
    s2 = float(((a - X @ b) ** 2).sum() / (len(a) - 3))
    x0 = np.array([1, T[year]["headwater_rain"], T[year]["trend"]], float)
    var = s2 * (1 + x0 @ np.linalg.inv(X.T @ X) @ x0)
    return float(x0 @ b), math.sqrt(var)


# ------------------------------------------------------------- 4 validation
def scores(rec):
    obs, mean, med, lo, hi, clim = (np.asarray(rec[k]) for k in ("obs", "mean", "med", "lo", "hi", "clim"))
    return {"rmse_km2": round(float(np.sqrt(((mean - obs) ** 2).mean())), 1),
            "mae_km2": round(float(np.abs(med - obs).mean()), 1),
            "skill_vs_climatology": round(float(1 - ((mean - obs) ** 2).sum() / ((clim - obs) ** 2).sum()), 3),
            "crps_km2": round(float(np.mean(rec["crps"])), 1),
            "coverage_80pct": round(float(((obs >= lo) & (obs <= hi)).mean()), 2),
            "mean_width_80pct_km2": round(float((hi - lo).mean()), 0),
            "n_years": int(len(obs))}


def score_round(T, reliable, targets, train_rule, label):
    """Forecast each target year using training-only cause selection, v2, baseline and v1."""
    recs = {k: defaultdict(list) for k in ("v2", "baseline", "v1")}
    per_year = {}
    for k, y in enumerate(targets):
        tr_rel = [t for t in reliable if train_rule(t, y)]
        obs = T[y]["area"]
        clim = float(np.mean([T[t]["area"] for t in tr_rel]))
        x = {c: T[y][c] for c in CANDIDATES}
        # Re-select causes from this fold's training years. Selecting once on
        # the full record leaks each held-out target into the feature set and
        # makes the reported validation score optimistic.
        if len(tr_rel) >= 4:
            fold_screen = screen_causes(T, tr_rel)
            v2_sets = fold_screen["ensemble_cause_sets"]
        else:
            fold_screen = None
            # Too little data for screening: use the pre-specified baseline,
            # never a model selected with information from the held-out year.
            v2_sets = [BASELINE]
        for name, model_sets in (("v2", v2_sets), ("baseline", [BASELINE])):
            used = sorted({c for cause_set in model_sets for c in cause_set},
                          key=list(CANDIDATES).index)
            tr_cens = [t for t in range(FIRST, LAST + 1)
                       if t in T and T[t].get("low_coverage") and has(T, t, used)
                       and train_rule(t, y)]
            tr = tr_rel + tr_cens
            ens = Ensemble(T, tr, model_sets, ITERS_CV, BURN_CV, seed=SEED + 31 * k)
            d = ens.draws(x, seed=k)
            s = summarise(d)
            for key, val in (("obs", obs), ("mean", s["mean"]), ("med", s["p50"]), ("lo", s["p10"]),
                             ("hi", s["p90"]), ("crps", crps_samples(d, obs)), ("clim", clim)):
                recs[name][key].append(val)
            if name == "v2":
                per_year[y] = s
                per_year[y]["selected_causes"] = used
                if fold_screen is not None:
                    per_year[y]["cause_inclusion_probability"] = fold_screen["inclusion_probability"]
        m1, sd1 = v1_predict(T, tr_rel, y)
        for key, val in (("obs", obs), ("mean", m1), ("med", m1), ("lo", m1 - 1.2816 * sd1),
                         ("hi", m1 + 1.2816 * sd1), ("crps", crps_normal(m1, sd1, obs)), ("clim", clim)):
            recs["v1"][key].append(val)
        per_year[y]["v1_mean"] = m1
    out = {k: scores(v) for k, v in recs.items()}
    out["what"] = label
    return out, per_year


def split_test(T, reliable, split=2012):
    """The v1 chronological test: fit up to 2012, forecast 2013-2025 (kept so the
    site's 'trained on 1989-2012, tested on 2013-2025' stays true)."""
    tr_rel = [y for y in reliable if y <= split]
    test = [y for y in reliable if y > split]
    selected = (screen_causes(T, tr_rel)["ensemble_cause_sets"]
                if len(tr_rel) >= 4 else [BASELINE])
    used = sorted({c for cause_set in selected for c in cause_set}, key=list(CANDIDATES).index)
    tr_cens = [y for y in range(FIRST, split + 1)
               if y in T and T[y].get("low_coverage") and has(T, y, used)]
    ens = Ensemble(T, tr_rel + tr_cens, selected, ITERS_CV, BURN_CV, seed=SEED)
    clim = float(np.mean([T[y]["area"] for y in tr_rel]))
    o = np.array([T[y]["area"] for y in test])
    m2 = np.array([ens.draws({c: T[y][c] for c in CANDIDATES}, seed=y).mean() for y in test])
    m1 = np.array([v1_predict(T, tr_rel, y)[0] for y in test])
    sk = lambda m: round(float(1 - ((m - o) ** 2).sum() / ((clim - o) ** 2).sum()), 3)
    rm = lambda m: round(float(np.sqrt(((m - o) ** 2).mean())), 1)
    return {"split": f"fit to {split}, forecast {split + 1}-{LAST}",
            "v2": {"skill": sk(m2), "rmse_km2": rm(m2)}, "v1": {"skill": sk(m1), "rmse_km2": rm(m1)},
            "per_year": [{"year": y, "observed_km2": round(float(a), 1), "v2_km2": round(float(b), 1),
                          "v1_km2": round(float(c), 1)} for y, a, b, c in zip(test, o, m2, m1)]}


def marginal_effects(ens, T, reliable):
    """km2 change per unit of each cause at a normal year (central difference of the
    ensemble's point prediction), and the trend's posterior z-value."""
    base = {c: float(np.mean([T[y][c] for y in reliable])) for c in ens.causes}
    step = {"headwater_rain": 10.0, "local_rain": 10.0, "enso": 0.1, "trend": 1.0}
    eff = {}
    for c in ens.causes:
        hi, lo = dict(base), dict(base)
        hi[c] += step[c] / 2
        lo[c] -= step[c] / 2
        eff[c] = (ens.point(hi) - ens.point(lo)) / step[c]
    # Trend per posterior draw in km2/year at a normal year: the members use
    # different Box-Cox scales, so their raw coefficients cannot be pooled.
    hi, lo = dict(base), dict(base)
    hi["trend"] += 0.5
    lo["trend"] -= 0.5
    bt = []
    for m in ens.members:
        xh = m.design([hi[c] for c in m.causes])[0]
        xl = m.design([lo[c] for c in m.causes])[0]
        bt.append(inv_boxcox(m.post["b"] @ xh, m.lam) - inv_boxcox(m.post["b"] @ xl, m.lam))
    bt = np.concatenate(bt)
    rr = [(y, T[y]["headwater_rain"]) for y in range(FIRST, LAST + 1) if T[y].get("headwater_rain")]
    X = np.column_stack([np.ones(len(rr)), [y - 2000 for y, _ in rr]])
    v = np.array([r for _, r in rr], float)
    b = np.linalg.lstsq(X, v, rcond=None)[0]
    s2 = float(((v - X @ b) ** 2).sum() / (len(v) - 2))
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return {"km2_per_unit_at_normal_year": {c: round(float(e), 3) for c, e in eff.items()},
            "units": {"headwater_rain": "mm", "local_rain": "mm", "enso": "index point", "trend": "year"},
            "trend_posterior_z": round(float(np.mean(bt) / np.std(bt)), 2),
            "trend_prob_negative": round(float((bt < 0).mean()), 3),
            "rain_trend_mm_per_year": round(float(b[1]), 2), "rain_trend_t": round(float(b[1] / se), 2),
            "normal_year": {c: round(v, 1) for c, v in base.items()}}


# ------------------------------------------------------------- 3 outlier logic
def shapley(ens, x, base):
    """Exact Shapley split of point(x) - point(base) over the ensemble's causes."""
    names = ens.causes
    k = len(names)
    val = {}
    for r in range(k + 1):
        for S in itertools.combinations(range(k), r):
            v = dict(base)
            for j in S:
                v[names[j]] = x[names[j]]
            val[S] = ens.point(v)
    phi = np.zeros(k)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        for r in range(k):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                phi[j] += wgt * (val[tuple(sorted(S + (j,)))] - val[S])
    return {names[j]: round(float(phi[j]), 1) for j in range(k)}, val[()], val[tuple(range(k))]


def outlier_logic(T, reliable, loo, ens):
    obs = np.array([T[y]["area"] for y in reliable])
    e1 = np.array([loo[y]["v1_mean"] for y in reliable]) - obs
    e2 = np.array([loo[y]["mean"] for y in reliable]) - obs
    rmse1 = float(np.sqrt((e1 ** 2).mean()))
    out = np.abs(e1) > rmse1
    z = lambda v: (v - v.mean()) / v.std()
    links = {}
    for cid in ("headwater_rain", "local_rain", "enso", "heat", "prev_headwater_rain"):
        v = z(np.array([T[y][cid] for y in reliable], float))
        links[cid] = {"r_with_v1_miss": round(float(np.corrcoef(e1, v)[0, 1]), 2),
                      "r_with_v2_miss": round(float(np.corrcoef(e2, v)[0, 1]), 2)}
    vr = z(np.array([T[y]["headwater_rain"] for y in reliable], float))
    curvature = {"r_with_v1_miss": round(float(np.corrcoef(e1, vr ** 2)[0, 1]), 2),
                 "r_with_v2_miss": round(float(np.corrcoef(e2, vr ** 2)[0, 1]), 2)}
    shared = max(links, key=lambda c: abs(links[c]["r_with_v1_miss"]))

    # Did the river carry more or less than the year's headwater rain implies?
    gy = [y for y in range(FIRST, LAST + 1) if T[y].get("gauge_feb_may") and T[y].get("headwater_rain")]
    G = np.array([[1, T[y]["headwater_rain"]] for y in gy], float)
    lq = np.log([T[y]["gauge_feb_may"] for y in gy])
    gb = np.linalg.lstsq(G, lq, rcond=None)[0]
    river = {y: float(np.exp(lq[i] - G[i] @ gb) - 1) for i, y in enumerate(gy)}

    rows, agree, with_gauge = [], 0, 0
    for i, y in enumerate(reliable):
        if not out[i]:
            continue
        rec = {"year": y, "observed_km2": round(float(obs[i]), 1),
               "v1_miss_km2": round(float(e1[i]), 1), "v2_miss_km2": round(float(e2[i]), 1),
               "explained_by_v2": bool(abs(e2[i]) < 0.6 * abs(e1[i])),
               "weight": round(ens.weight(y), 2),
               "headwater_rain_mm": round(T[y]["headwater_rain"]),
               "local_rain_mm": round(T[y]["local_rain"]), "enso": T[y]["enso"]}
        if y in river:
            rec["river_vs_rain_pct"] = round(100 * river[y])
            if not rec["explained_by_v2"]:
                with_gauge += 1
                rec["river_agrees"] = bool(abs(river[y]) >= RIVER_SIGNAL and (e2[i] < 0) == (river[y] > 0))
                agree += int(rec["river_agrees"])
        rec["text"] = outlier_sentence(rec)
        rows.append(rec)
    m_out = lambda e: round(float(np.abs(e[out]).mean()), 1)
    m_in = lambda e: round(float(np.abs(e[~out]).mean()), 1)
    return {
        "definition": f"An outlier year is one that v1 (rain + trend, least squares) missed by more than "
                      f"its own leave-one-out RMSE ({rmse1:.0f} km2).",
        "outlier_years": [r["year"] for r in rows],
        "explained_years": [r["year"] for r in rows if r["explained_by_v2"]],
        "unexplained_years": [r["year"] for r in rows if not r["explained_by_v2"]],
        "shared_cause": shared,
        "links": links,
        "curvature": curvature,
        "mean_abs_miss_outlier_years": {"v1": m_out(e1), "v2": m_out(e2)},
        "mean_abs_miss_other_years": {"v1": m_in(e1), "v2": m_in(e2)},
        "unexplained_with_gauge": {"years": with_gauge, "river_agrees": agree},
        "river_rule": "river_vs_rain_pct: measured Feb-May inflow at Mohembo against what that year's "
                      "headwater rain implies (log-linear fit over gauge years). It counts only beyond "
                      f"+/-{100 * RIVER_SIGNAL:.0f}%.",
        "years": rows,
    }


def outlier_sentence(r):
    y = r["year"]
    if r["explained_by_v2"]:
        why = []
        if r["local_rain_mm"] >= 700:
            why.append(f"a very wet Delta ({r['local_rain_mm']} mm of rain on the Delta itself)")
        elif r["local_rain_mm"] <= 350:
            why.append(f"a very dry Delta ({r['local_rain_mm']} mm on the Delta itself)")
        if r["enso"] is not None and abs(r["enso"]) >= 1.0:
            why.append(f"{'La Nina' if r['enso'] < 0 else 'El Nino'} (index {r['enso']:+.1f})")
        if r["headwater_rain_mm"] <= 700 or r["headwater_rain_mm"] >= 1040:
            why.append(f"extreme headwater rain ({r['headwater_rain_mm']} mm) on a multiplicative scale")
        reason = ", ".join(why) if why else "the causes together on a multiplicative scale"
        return (f"{y}: v1 missed by {abs(r['v1_miss_km2']):.0f} km2, v2 by "
                f"{abs(r['v2_miss_km2']):.0f} km2. Explained by {reason}.")
    s = (f"{y}: still not explained (v1 missed by {abs(r['v1_miss_km2']):.0f} km2, v2 by "
         f"{abs(r['v2_miss_km2']):.0f} km2).")
    if "river_vs_rain_pct" in r:
        d = r["river_vs_rain_pct"]
        if abs(d) < 100 * RIVER_SIGNAL:
            s += f" The measured river inflow was close to what the rain implies ({d:+d}%), so the river does not explain it either."
        else:
            s += (f" The measured river inflow was {abs(d)}% {'above' if d > 0 else 'below'} what the rain "
                  f"implies{', which matches the miss' if r.get('river_agrees') else ', which does not match the miss'}.")
    else:
        s += " No measured river inflow exists for this year."
    return s


def basin_eofs(years):
    """Nov-Apr rain at the 17 basin points -> SVD of the standardised year x point
    matrix (empirical orthogonal functions). Returns {year: (mean, pc1, pc2, pc3)}
    and the share of variance of each mode, or (None, None) if rain is missing."""
    from history import era5_monthly_rain
    try:
        M = [era5_monthly_rain(*p) for p in BASIN_POINTS]
    except Exception as e:                       # no network and no cache
        print(f"  basin rain unavailable ({e}); pattern test skipped")
        return None, None
    ys = [y for y in years if all(season_total(m, y) is not None for m in M)]
    X = np.array([[season_total(m, y) for m in M] for y in ys])
    Z = (X - X.mean(0)) / X.std(0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U * s
    share = (s ** 2 / (s ** 2).sum())[:3]
    return {y: (float(X[i].mean()), *map(float, pcs[i, :3])) for i, y in enumerate(ys)}, \
        [round(float(v), 3) for v in share]


def hypotheses(T, reliable, causes, lam):
    """What else could link the unexplained years? Each idea is added to the
    v2 causes and scored by exact leave-one-out skill on the same years."""
    eof, share = basin_eofs(list(range(FIRST, AHEAD[0] + 1)))
    tests = []

    def run(name, label, extra, replace=None):
        ys = [y for y in reliable if extra(y) is not None]
        z = boxcox(np.array([T[y]["area"] for y in ys]), lam)
        base = np.array([[T[y][c] for c in causes] for y in ys], float)
        alt = np.column_stack([np.array([[T[y][c] for c in causes if c != replace] for y in ys], float),
                               np.array([extra(y) for y in ys], float)])
        s0, s1 = press_skill(base, z), press_skill(alt, z)
        tests.append({"id": name, "idea": label, "n_years": len(ys),
                      "loo_skill_without": round(s0, 3), "loo_skill_with": round(s1, 3),
                      "verdict": "helps" if s1 > s0 + 0.01 else "does not help"})

    run("memory_last_year_water", "Memory: last year's open water carries over",
        lambda y: T.get(y - 1, {}).get("area") or None)
    run("memory_last_year_rain", "Memory: last year's headwater rain (slow groundwater of the Cuito)",
        lambda y: T[y].get("prev_headwater_rain"))
    run("heat", "Heat: hot flood seasons evaporate more", lambda y: T[y].get("heat"))
    run("river_gauge", "River: measured Mohembo inflow, Feb-May (log)",
        lambda y: math.log(T[y]["gauge_feb_may"]) if T[y].get("gauge_feb_may") else None)
    run("river_glofas", "River: GloFAS modelled inflow, Feb-Apr (log)",
        lambda y: math.log(T[y]["glofas_feb_apr"]) if T[y].get("glofas_feb_apr") else None)
    if eof:
        run("where_rain_fell", "Where in the basin the rain fell (EOF mode 2 of 17 ERA5 points: "
            "north-west highlands against the south-east)", lambda y: eof[y][2] if y in eof else None)
        run("basin_total", "Whole-basin rain (17 points) instead of 3 points",
            lambda y: eof[y][0] if y in eof else None, replace="headwater_rain")
    return {"tests": tests, "basin_eof_variance_share": share,
            "basin_points": [list(p) for p in BASIN_POINTS] if eof else None}


# ------------------------------------------------------------- 5 projection
def markov_test(T, causes):
    """Permutation test of tercile transitions across consecutive observed years.

    A permutation reference avoids relying on a sparse-table chi-square
    approximation for this short annual record.
    """
    out = {}
    for cause_no, cid in enumerate(causes):
        if cid == "trend":
            continue
        ys = [y for y in range(FIRST, AHEAD[0] + 1) if T[y].get(cid) is not None]
        v = np.array([T[y][cid] for y in ys])
        s = np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
        pairs = [(i, i + 1) for i in range(len(ys) - 1) if ys[i + 1] == ys[i] + 1]
        pi = np.array([i for i, _ in pairs], dtype=int)
        pj = np.array([j for _, j in pairs], dtype=int)

        def transition_table(state):
            return np.bincount(state[pi] * 3 + state[pj], minlength=9).reshape(3, 3)

        def transition_stat(state):
            table = transition_table(state)
            row, col = table.sum(1), table.sum(0)
            active = (row[:, None] > 0) & (col[None, :] > 0)
            if not active.any():
                return 0.0
            expected = row[:, None] * col[None, :] / max(float(table.sum()), 1.0)
            return float((((table - expected) ** 2) / np.where(active, expected, 1))[active].sum())

        N = transition_table(s)
        row, col = N.sum(1), N.sum(0)
        active_rows, active_cols = int((row > 0).sum()), int((col > 0).sum())
        dof = (active_rows - 1) * (active_cols - 1)
        chi2, p = transition_stat(s), None
        if dof > 0 and pairs:
            rng = np.random.default_rng(SEED + cause_no)
            n_perm = 9999
            exceed = sum(transition_stat(rng.permutation(s)) >= chi2
                         for _ in range(n_perm))
            p = (exceed + 1) / (n_perm + 1)
        lag_pairs = [(v[i], v[j]) for i, j in pairs]
        lag1 = (float(np.corrcoef(np.asarray(lag_pairs).T)[0, 1])
                if len(lag_pairs) > 1 and np.std(np.asarray(lag_pairs), axis=0).min() > 0 else None)
        out[cid] = {"transitions_low_mid_high": N.astype(int).tolist(),
                    "n_consecutive_transitions": len(pairs),
                    "chi2": round(chi2, 2),
                    "dof": dof, "p_value": round(p, 3) if p is not None else None,
                    "p_value_method": "9999 random permutations of annual terciles",
                    "lag1_r": round(lag1, 2) if lag1 is not None else None}
    return out


def project(ens, T):
    causes = [c for c in ens.causes if c != "trend"]
    past = [y for y in range(FIRST, AHEAD[0] + 1) if has(T, y, causes)]
    rng = np.random.default_rng(SEED)
    S = len(ens.members[0].post["b"])
    rows, nodes = [], {}
    for year in AHEAD:
        known = has(T, year, causes)
        if known:
            x = {c: T[year][c] for c in ens.causes}
        else:
            pick = rng.choice(past, size=S)
            x = {c: np.array([T[p][c] for p in pick], float) for c in causes}
            x["trend"] = float(year - 2000)
        s = summarise(ens.draws(x, seed=year))
        nodes[year] = s["nodes"]
        rows.append({"year": year, "causes_known": known,
                     **({c: T[year][c] for c in causes} if known else {}),
                     "mean_km2": round(s["mean"], 1), "sd_km2": round(s["sd"], 1),
                     "p025_km2": round(s["p025"], 1),
                     "p10_km2": round(s["p10"], 1), "p50_km2": round(s["p50"], 1),
                     "p90_km2": round(s["p90"], 1), "p975_km2": round(s["p975"], 1)})
    return rows, nodes, past


# ------------------------------------------------------------------ main
def engine(verbose=True):
    """Run everything. Returns (doc, Ensemble, per-year leave-one-out summaries,
    projection nodes {year: 40 equal-probability areas})."""
    t0 = time.time()
    T = load_table()
    reliable = [y for y in range(FIRST, LAST + 1) if "area" in T[y] and not T[y]["low_coverage"]
                and has(T, y, CANDIDATES)]
    screen = screen_causes(T, reliable)
    sets = screen["ensemble_cause_sets"]
    used = sorted({c for s in sets for c in s}, key=list(CANDIDATES).index)
    censored = [y for y in range(FIRST, LAST + 1) if "area" in T[y] and T[y]["low_coverage"]
                and has(T, y, used)]
    if verbose:
        print(f"PIP {screen['inclusion_probability']}\ncore {screen['core']}  contested "
              f"{screen['contested']}  dropped {screen['dropped']}  -> {len(sets)} models")
    ens = Ensemble(T, reliable + censored, sets)

    loo_scores, loo = score_round(
        T, reliable, reliable, lambda t, y: t != y,
        "Nested leave-one-year-out: cause screening, lambda, coefficients, weights and nu refitted "
        "without the held-out year.")
    if verbose:
        print("LOO     ", {k: v["rmse_km2"] for k, v in loo_scores.items() if k != "what"}, f"{time.time() - t0:.0f} s")
    roll_scores, _ = score_round(
        T, reliable, [y for y in reliable if y >= ROLL_FROM], lambda t, y: t < y,
        f"Rolling origin: each year from {ROLL_FROM} forecast only from the years before it.")
    if verbose:
        print("rolling ", {k: v["rmse_km2"] for k, v in roll_scores.items() if k != "what"}, f"{time.time() - t0:.0f} s")

    split = split_test(T, reliable)
    effects = marginal_effects(ens, T, reliable)
    logic = outlier_logic(T, reliable, loo, ens)
    logic["hypotheses_for_unexplained_years"] = hypotheses(T, reliable, used, screen["lambda_used"])
    markov = markov_test(T, used)
    proj, nodes, past = project(ens, T)

    base = {c: float(np.mean([T[y][c] for y in reliable])) for c in used}
    years_out = []
    for y in range(FIRST, AHEAD[0] + 1):
        if not has(T, y, used):
            continue
        phi, at_base, at_x = shapley(ens, {c: T[y][c] for c in used}, base)
        rec = {"year": y, "observed_km2": T[y].get("area"), "low_coverage": T[y].get("low_coverage"),
               "fitted_km2": round(at_x, 1), "normal_year_km2": round(at_base, 1),
               "contribution_km2": phi, **{c: T[y][c] for c in used if c != "trend"}}
        if y in loo:
            rec.update({"loo_km2": round(loo[y]["mean"], 1), "loo_p10_km2": round(loo[y]["p10"], 1),
                        "loo_p90_km2": round(loo[y]["p90"], 1), "v1_loo_km2": round(loo[y]["v1_mean"], 1)})
            rec["loo_selected_causes"] = loo[y]["selected_causes"]
            rec["loo_cause_inclusion_probability"] = loo[y].get("cause_inclusion_probability")
        if y in ens.years:
            rec["weight"] = round(ens.weight(y), 2)
        years_out.append(rec)

    members = []
    for m in ens.members:
        b = m.post["b"]
        members.append({
            "causes": m.causes, "lambda": round(m.lam, 3),
            "coefficients_scaled": {("intercept" if j == 0 else m.causes[j - 1]): {
                "median": round(float(np.median(b[:, j])), 4),
                "p05": round(float(np.quantile(b[:, j], 0.05)), 4),
                "p95": round(float(np.quantile(b[:, j], 0.95)), 4),
                "prob_positive": round(float((b[:, j] > 0).mean()), 3)} for j in range(b.shape[1])},
            "nu_posterior": {str(int(v)): round(float((m.post["nu"] == v).mean()), 3) for v in NU_GRID},
            "standardisation": {c: [round(float(mu), 2), round(float(sd), 2)]
                                for c, mu, sd in zip(m.causes, m.mu, m.sd)}})
    v1s, v2s = loo_scores["v1"], loo_scores["v2"]
    r1, r2 = roll_scores["v1"], roll_scores["v2"]
    pct = lambda a, b: round(100 * (b / a - 1), 1)
    doc = {
        "name": "RSynthesis estimation engine",
        "version": VERSION,
        "generated": time.strftime("%Y-%m-%d"),
        "target": "Yearly open-water area of the study area from Landsat (history.json), km2",
        "summary": {
            "causes_used": used, "core": screen["core"], "contested": screen["contested"],
            "causes_dropped": screen["dropped"] + ["river inflow (gauge and GloFAS)", "where the rain fell"],
            "models": len(sets),
            "leave_one_out": {"v1_rmse": v1s["rmse_km2"], "v2_rmse": v2s["rmse_km2"],
                              "rmse_change_pct": pct(v1s["rmse_km2"], v2s["rmse_km2"]),
                              "mae_change_pct": pct(v1s["mae_km2"], v2s["mae_km2"]),
                              "crps_change_pct": pct(v1s["crps_km2"], v2s["crps_km2"])},
            "rolling_origin": {"v1_rmse": r1["rmse_km2"], "v2_rmse": r2["rmse_km2"],
                               "rmse_change_pct": pct(r1["rmse_km2"], r2["rmse_km2"]),
                               "mae_change_pct": pct(r1["mae_km2"], r2["mae_km2"]),
                               "crps_change_pct": pct(r1["crps_km2"], r2["crps_km2"])},
        },
        "step1_cause_reduction": screen,
        "step2_engine": {
            "formula": "For each model m: BoxCox(area, lambda_m) = b0 + sum_k b_k (cause_k - mean_k) / sd_k + e, "
                       "e ~ Student-t(nu, 0, s2). Forecast = equal mixture of the models' predictive draws.",
            "members": members,
            "effects": effects,
            "years_fitted": {"exact": reliable, "lower_bound_only": censored},
            "sampler": {"method": "Gibbs sampling with data augmentation", "iterations": ITERS,
                        "burn_in": BURN, "seed": SEED},
            "why_equal_weights": "Weights learnt from earlier forecast scores (stacking) lost to equal "
                                 "weights out of time: 20 years are too few to learn them.",
        },
        "step3_outlier_logic": logic,
        "step4_validation": {"leave_one_out": loo_scores, "rolling_origin": roll_scores,
                             "single_split": split,
                             "baseline": f"v1's causes {BASELINE} on the v2 engine (no cause screening "
                                         "involved): the part of the gain that comes from the engine alone."},
        "step5_projection": {
            "cause_memory_test": markov,
            "future_causes": f"2027-2030: each posterior draw takes the causes of one random year of "
                             f"{past[0]}-{past[-1]} together, keeping their correlations.",
            "years": proj,
        },
        "years": years_out,
        "caveats": [
            "Landsat sees open water only; water under reeds and papyrus is not in these numbers.",
            "Cause screening is repeated inside each validation fold; the full-record cause set is "
            "used only for the final fit and projection.",
            "Rain over the Delta sits at the border (inclusion probability close to 2/3); "
            "the ensemble keeps it in half of its models for that reason.",
            "The trend term is extrapolated to 2030 with its uncertainty; it describes this study area only.",
            "Causes are associations in 27 well-observed years, not proof.",
            "2027-2030 rain and El Nino are unknown; their bands use past years.",
        ],
    }
    if verbose:
        print(f"done in {time.time() - t0:.0f} s")
    return doc, ens, loo, nodes


def main() -> None:
    doc, ens, loo, nodes = engine()
    (C.OUT / "rsynthesis.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    s = doc["summary"]
    print(f"causes used {s['causes_used']}  dropped {s['causes_dropped']}")
    print(f"leave-one-out {s['leave_one_out']}\nrolling origin {s['rolling_origin']}")
    lg = doc["step3_outlier_logic"]
    print(f"outliers {lg['outlier_years']}  shared cause {lg['shared_cause']}  "
          f"|miss| on them {lg['mean_abs_miss_outlier_years']}")
    for r in lg["years"]:
        print("  ", r["text"])
    for p in doc["step5_projection"]["years"]:
        print(f"  {p['year']}  median {p['p50_km2']:.0f}  80% {p['p10_km2']:.0f}-{p['p90_km2']:.0f}")


if __name__ == "__main__":
    main()
