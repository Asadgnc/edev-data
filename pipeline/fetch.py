"""
EDev-Vision | Stage 1 of 2: stream one NISAR GCOV granule and cache the AOI.

Why two stages? Downloading is the slow part (minutes per date) and threshold
tuning is the part we will redo many times. Separating them means we pay the
network cost once and can re-render in seconds.

What this does, per date:
  1. find the granule by name (track 172 / frame 168 / ascending / DHDH)
  2. open it with earthaccess + h5py WITHOUT downloading the 1.9 GB file
     (fsspec issues HTTP range requests; HDF5 is chunked 512x512 so only the
     chunks covering our AOI travel over the wire)
  3. read HHHH, HVHV and mask inside the AOI window, in horizontal stripes
  4. multilook 4x4 in LINEAR power -> 80 m, then cache as .npz

Usage:
    python pipeline/fetch.py --date 2026-07-20
    python pipeline/fetch.py --all
    python pipeline/fetch.py --date 2026-07-20 --stripes 2   # speed probe
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def aoi_utm_bbox(epsg: int) -> tuple[float, float, float, float]:
    """AOI lon/lat box -> projected bounding box in the product's own CRS.

    The lon/lat box is not a rectangle in UTM, so we densify its edges and take
    the extremes. Using only the 4 corners would clip the bulging edges.
    """
    lon0, lat0, lon1, lat1 = C.AOI
    n = 100
    lons = np.linspace(lon0, lon1, n)
    lats = np.linspace(lat0, lat1, n)
    edge_lon = np.concatenate([lons, lons, np.full(n, lon0), np.full(n, lon1)])
    edge_lat = np.concatenate([np.full(n, lat0), np.full(n, lat1), lats, lats])
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    ex, ey = tf.transform(edge_lon, edge_lat)
    return float(np.min(ex)), float(np.min(ey)), float(np.max(ex)), float(np.max(ey))


def window(h5: h5py.File, p: dict, looks: int) -> dict:
    """Map the AOI onto array indices, aligned to the multilook factor."""
    epsg = int(h5[p["projection"]][()])
    x = h5[p["x"]][:]
    y = h5[p["y"]][:]  # descending (yCoordinateSpacing is negative)
    xmin, ymin, xmax, ymax = aoi_utm_bbox(epsg)

    i0 = int(np.clip(np.searchsorted(x, xmin) - 1, 0, x.size - 1))
    i1 = int(np.clip(np.searchsorted(x, xmax) + 1, 1, x.size))
    # y descends, so search in the negated array to keep it ascending
    j0 = int(np.clip(np.searchsorted(-y, -ymax) - 1, 0, y.size - 1))
    j1 = int(np.clip(np.searchsorted(-y, -ymin) + 1, 1, y.size))

    # snap to whole multilook blocks
    i0 -= i0 % looks
    j0 -= j0 % looks
    cols = ((i1 - i0) // looks) * looks
    rows = ((j1 - j0) // looks) * looks
    if rows < looks or cols < looks:
        raise SystemExit("AOI does not overlap this granule's grid.")

    xs = x[i0 : i0 + cols].reshape(-1, looks).mean(axis=1)
    ys = y[j0 : j0 + rows].reshape(-1, looks).mean(axis=1)
    half = C.OUT_PIXEL_M / 2.0
    return {
        "i0": i0, "j0": j0, "rows": rows, "cols": cols,
        "x_out": xs, "y_out": ys, "epsg": epsg,
        # outer edges of the output grid, in UTM 35S
        "bbox_utm": (
            float(xs[0] - half), float(ys[-1] - half),
            float(xs[-1] + half), float(ys[0] + half),
        ),
    }


def block_reduce(a: np.ndarray, valid: np.ndarray, looks: int):
    """Sum of valid linear-power values per looks x looks block, plus count."""
    r, c = a.shape
    shp = (r // looks, looks, c // looks, looks)
    s = np.where(valid, a, np.float32(0.0)).reshape(shp).sum(axis=(1, 3))
    n = valid.reshape(shp).sum(axis=(1, 3), dtype=np.int32)
    return s.astype(np.float64), n


def fetch_one(row: dict, max_stripes: int | None = None, freq: str = "A") -> Path:
    import earthaccess

    date, name = row["date"], row["granule"]
    p = C.paths(freq)
    looks = C.LOOKS_BY_FREQ[freq]
    print(f"\n=== {date}  {row['quality']}  freq{freq}  {name[:55]}...")
    t0 = time.time()

    res = earthaccess.search_data(
        short_name=row["collection"],
        bounding_box=C.AOI,
        temporal=(date, date),
        count=50,
    )
    hits = [g for g in res if g["umm"]["GranuleUR"] == name]
    if not hits:
        raise SystemExit(f"granule not found in CMR: {name}")
    fobj = earthaccess.open(hits)[0]
    print(f"  opened stream in {time.time() - t0:.1f}s")

    with h5py.File(fobj, "r") as h5:
        w = window(h5, p, looks)
        print(f"  EPSG:{w['epsg']}  window rows={w['rows']} cols={w['cols']} "
              f"({w['rows'] * w['cols'] / 1e6:.1f} Mpx per band) "
              f"-> out {w['rows'] // looks}x{w['cols'] // looks}")

        pols = [q.decode() if isinstance(q, bytes) else str(q)
                for q in h5[p["pols"]][:]]
        # Trust the file, not the granule name: only take bands that exist.
        bands = {k: v for k, v in p["bands"].items() if v in h5}
        print(f"  polarizations {pols}  bands used {list(bands)}")

        ro, co = w["rows"] // looks, w["cols"] // looks
        acc = {k: (np.zeros((ro, co)), np.zeros((ro, co), np.int32)) for k in bands}
        mask_bad = np.zeros((ro, co), np.int32)
        mask_hist = np.zeros(256, np.int64)

        stripe = max(C.STRIPE_ROWS, looks)
        stripe -= stripe % looks
        starts = list(range(w["j0"], w["j0"] + w["rows"], stripe))
        if max_stripes:
            starts = starts[:max_stripes]
        ds_mask = h5[p["mask"]]
        for k, js in enumerate(starts):
            je = min(js + stripe, w["j0"] + w["rows"])
            oj = (js - w["j0"]) // looks
            ts = time.time()
            m = ds_mask[js:je, w["i0"] : w["i0"] + w["cols"]]
            mask_hist += np.bincount(m.ravel(), minlength=256)
            _, bn = block_reduce(m.astype(np.float32), m != 0, looks)
            mask_bad[oj : oj + bn.shape[0]] = bn
            for bk, path in bands.items():
                a = h5[path][js:je, w["i0"] : w["i0"] + w["cols"]]
                v = np.isfinite(a) & (a > 0)
                s, n = block_reduce(a, v, looks)
                acc[bk][0][oj : oj + s.shape[0]] += s
                acc[bk][1][oj : oj + n.shape[0]] += n
            el = time.time() - ts
            mb = (je - js) * w["cols"] * 4 * (len(bands) + 1) / 1e6
            print(f"    stripe {k + 1}/{len(starts)} rows {js}-{je} "
                  f"{el:.1f}s  {mb / max(el, 0.1):.2f} MB/s", flush=True)

    out = {}
    for bk, (s, n) in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            lin = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[f"{bk}_lin"] = lin.astype(np.float32)
        out[f"{bk}_n"] = n
    nz = int(mask_hist[1:].sum())
    print(f"  mask: {int(mask_hist[0])} zero, {nz} non-zero "
          f"({100 * nz / max(mask_hist.sum(), 1):.2f}%), "
          f"values present: {np.nonzero(mask_hist)[0].tolist()}")

    cdir = C.cache_dir(freq)
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"{date}.npz"
    np.savez_compressed(
        cpath,
        x_out=w["x_out"], y_out=w["y_out"], epsg=w["epsg"],
        bbox_utm=np.array(w["bbox_utm"]),
        mask_bad=mask_bad, mask_hist=mask_hist,
        looks=looks, freq=freq, partial=bool(max_stripes),
        granule=name, date=date, quality=row["quality"],
        **out,
    )
    hh = out["hh_lin"]
    good = np.isfinite(hh)
    print(f"  saved {freq}/{cpath.name}  {cpath.stat().st_size / 1e6:.1f} MB  "
          f"valid {100 * good.mean():.1f}%  "
          f"HH dB p5/p50/p95 = "
          f"{np.nanpercentile(10 * np.log10(hh[good]), [5, 50, 95]).round(1)}")
    print(f"  TOTAL {(time.time() - t0) / 60:.1f} min")
    return cpath


def is_complete(date: str, freq: str = "A") -> bool:
    """True only for a cached file that was written by a full (non-partial)
    read. A speed-probe file must not be mistaken for a finished one."""
    p = C.cache_dir(freq) / f"{date}.npz"
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return not bool(z["partial"])
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stripes", type=int, help="read only the first N stripes")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--freq", choices=["A", "B"], default="A",
                    help="A = 20 MHz / 20 m (science). B = 5 MHz, already at "
                         "80 m, 16x less data - the bandwidth fallback.")
    a = ap.parse_args()

    rows = C.series()
    if a.date:
        rows = [r for r in rows if r["date"] == a.date]
        if not rows:
            raise SystemExit(f"{a.date} is not in the series: {C.DATES}")
    elif not a.all:
        raise SystemExit("pass --date YYYY-MM-DD or --all")

    if a.skip_existing:
        keep = [r for r in rows if not is_complete(r["date"], a.freq)]
        for r in rows:
            if r not in keep:
                print(f"skip {r['date']} (already complete)")
        rows = keep
    if not rows:
        print("nothing to do")
        return

    C.login()
    t0 = time.time()
    done, failed = [], []

    def run(r):
        try:
            return fetch_one(r, a.stripes, a.freq)
        except Exception as e:  # one bad date must not kill the whole run
            print(f"  !! FAILED {r['date']}: {type(e).__name__}: {e}", flush=True)
            return None

    workers = max(1, min(a.workers, len(rows)))
    if workers == 1:
        results = [run(r) for r in rows]
    else:
        from concurrent.futures import ThreadPoolExecutor

        print(f"streaming {len(rows)} dates on {workers} threads")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run, rows))
    for r, p in zip(rows, results):
        (done if p else failed).append(r["date"])

    print(f"\n{len(done)} cached in {(time.time() - t0) / 60:.1f} min: {done}")
    if failed:
        print(f"FAILED: {failed}  -> re-run with --all --skip-existing")


if __name__ == "__main__":
    main()
