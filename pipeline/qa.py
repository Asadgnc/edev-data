"""
EDev-Vision | Quality assurance: pick thresholds from the data, not from a guess.

Produces, for each cached date:
  qa/hist_<date>.png   HH and HV dB histograms with the current thresholds drawn
  qa/scatter.png       HH vs HH-HV density, where the three classes should
                       separate into visible lobes
and prints:
  - percentiles per date, so a calibration offset between PROVISIONAL and
    VALIDATED products shows up as a shift in the dry-land median
  - the share of pixels each class would claim at the current thresholds

Read the histogram before trusting any number in timeseries.json. If the
distribution is not clearly multi-modal, the classification is weak and the app
must say so.

Usage:  python pipeline/qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI on this machine; write files only
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from render import FLOODED_VEG, NODATA, WATER, classify, to_db


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "date": str(z["date"]),
            "quality": str(z["quality"]),
            "partial": bool(z["partial"]),
            "hh": to_db(z["hh_lin"]),
            "hv": to_db(z["hv_lin"]) if "hv_lin" in z else None,
            "mask_hist": z["mask_hist"],
        }


def main() -> None:
    freq = sys.argv[1] if len(sys.argv) > 1 else "A"
    files = sorted(C.cache_dir(freq).glob("*.npz"))
    if not files:
        raise SystemExit("no cache - run fetch.py first")
    C.QA.mkdir(exist_ok=True)
    t = C.THRESH
    rows, pts = [], []

    for p in files:
        d = load(p)
        hh, hv = d["hh"], d["hv"]
        v = np.isfinite(hh)
        if hv is not None:
            v &= np.isfinite(hv)
        hhv, hvv = hh[v], (hv[v] if hv is not None else None)

        cls = classify(hh, hv)
        n = max(int((cls != NODATA).sum()), 1)
        rows.append({
            "date": d["date"], "quality": d["quality"], "partial": d["partial"],
            "valid_pct": 100 * v.mean(),
            "hh_p": np.percentile(hhv, [1, 5, 25, 50, 75, 95, 99]).round(1),
            "hv_p": (np.percentile(hvv, [1, 5, 25, 50, 75, 95, 99]).round(1)
                     if hvv is not None else None),
            # Dry-land reference: the 60-80th percentile band is dominated by
            # unflooded savanna, which should not change much between dates.
            # A jump here between PROVISIONAL and VALIDATED means calibration,
            # not flooding.
            "dry_ref": float(np.percentile(hhv, 70).round(2)),
            "water_pct": 100 * int((cls == WATER).sum()) / n,
            "fv_pct": 100 * int((cls == FLOODED_VEG).sum()) / n,
        })

        fig, ax = plt.subplots(1, 2 if hvv is not None else 1,
                               figsize=(11, 4), squeeze=False)
        ax[0][0].hist(hhv, bins=250, range=(-35, 10), color="#1f78b4")
        ax[0][0].axvline(t["water_hh_max"], color="red", ls="--",
                         label=f"water < {t['water_hh_max']}")
        ax[0][0].axvline(t["fv_hh_min"], color="green", ls="--",
                         label=f"flooded veg > {t['fv_hh_min']}")
        ax[0][0].set(title=f"HH dB  {d['date']}", xlabel="dB", ylabel="pixels")
        ax[0][0].legend(fontsize=8)
        if hvv is not None:
            ax[0][1].hist(hvv, bins=250, range=(-40, 5), color="#6a51a3")
            ax[0][1].axvline(t["water_hv_max"], color="red", ls="--",
                             label=f"water < {t['water_hv_max']}")
            ax[0][1].set(title=f"HV dB  {d['date']}", xlabel="dB")
            ax[0][1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(C.QA / f"hist_{d['date']}.png", dpi=110)
        plt.close(fig)

        if hvv is not None:
            k = np.random.default_rng(0).choice(
                hhv.size, size=min(60000, hhv.size), replace=False)
            pts.append((d["date"], hhv[k], hhv[k] - hvv[k]))

    print(f"{'date':<12}{'qual':<12}{'valid%':>7}{'dry70':>8}"
          f"{'water%':>8}{'fveg%':>7}   HH dB p1/5/25/50/75/95/99")
    for r in rows:
        flag = " PARTIAL" if r["partial"] else ""
        print(f"{r['date']:<12}{r['quality']:<12}{r['valid_pct']:>7.1f}"
              f"{r['dry_ref']:>8.2f}{r['water_pct']:>8.2f}{r['fv_pct']:>7.2f}"
              f"   {r['hh_p']}{flag}")
    if rows:
        dry = np.array([r["dry_ref"] for r in rows])
        print(f"\ndry-land (HH p70) spread across dates: "
              f"{dry.min():.2f} .. {dry.max():.2f} dB  "
              f"(range {np.ptp(dry):.2f})")
        print("A range above ~1 dB means part of the 'change' is calibration, "
              "not water. Note it in the app.")

    if pts:
        cols = min(len(pts), 5)
        rws = (len(pts) + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(3.1 * cols, 3 * rws),
                                 squeeze=False)
        for axx, (date, x, y) in zip(axes.ravel(), pts):
            axx.hexbin(x, y, gridsize=60, bins="log", cmap="magma",
                       extent=(-30, 5, -5, 20))
            axx.axvline(t["water_hh_max"], color="cyan", ls="--", lw=0.8)
            axx.axvline(t["fv_hh_min"], color="lime", ls="--", lw=0.8)
            axx.axhline(t["fv_hhhv_min"], color="lime", ls="--", lw=0.8)
            axx.set(title=date, xlabel="HH dB", ylabel="HH-HV dB")
        for axx in axes.ravel()[len(pts):]:
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(C.QA / "scatter.png", dpi=110)
        plt.close(fig)

    print(f"\nwrote {C.QA}")
    mh = rows and load(files[0])["mask_hist"]
    if mh is not None and len(mh):
        vals = {int(i): int(mh[i]) for i in np.nonzero(mh)[0]}
        print(f"mask layer values seen (first date): {vals}")
        print("Validity in this pipeline comes from HH/HV being finite and "
              "positive, not from assuming what a mask value means.")


if __name__ == "__main__":
    main()
