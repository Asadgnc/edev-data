"""
EDev-Vision | Step 4: discover the real structure of one NISAR GCOV granule.

Goal: print actual HDF5 group/dataset names, shapes, dtypes and key attributes
(polarizations, projection, coordinates) BEFORE writing the processing code.
Streams via earthaccess/fsspec (reads only the HDF5 metadata blocks, not the
whole multi-GB file).

Run:  python discover_gcov.py
Needs an Earthdata Login (first run prompts for username/password; use
`earthaccess.login(strategy="interactive", persist=True)` once to save it).
Output: prints to console AND writes gcov_structure.txt -> paste that file back.
"""

import sys

import earthaccess
import h5py

AOI_BBOX = (22.0, -20.0, 23.8, -18.3)  # lon_min, lat_min, lon_max, lat_max
OUT = "gcov_structure.txt"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def main():
    # Uses EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars if set,
    # otherwise falls back to ~/.netrc or an interactive prompt.
    import os

    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="interactive")  # prompts for user/password

    # Short names are not guaranteed, so list candidates first.
    log("--- Collections matching 'NISAR GCOV' ---")
    colls = earthaccess.search_datasets(keyword="NISAR GCOV")
    cands = []
    for c in colls:
        sn = c.summary().get("short-name")
        log(f"  {sn}  | {c.summary().get('version')} | {c.get_umm('EntryTitle')}")
        cands.append(sn)
    if not cands:
        log("No collection found. Check Earthdata login / NISAR access.")
        sys.exit(1)

    granules = []
    for sn in cands:
        if "GCOV" not in str(sn).upper():
            continue
        g = earthaccess.search_data(
            short_name=sn, bounding_box=AOI_BBOX,
            temporal=("2026-06-17", "2026-11-14"), count=5,
        )
        log(f"\n{sn}: {len(g)} granule(s) found in AOI")
        if g:
            granules = g
            break
    if not granules:
        log("No GCOV granule in AOI.")
        sys.exit(1)

    g = granules[0]
    log(f"\nUsing granule: {g['umm']['GranuleUR']}")
    f = earthaccess.open([g])[0]  # fsspec file object, streamed lazily

    with h5py.File(f, "r") as h:
        log("\n--- Top-level groups ---")
        log(list(h.keys()))

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                log(f"[D] {name}  shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks}")

        log("\n--- All datasets (name, shape, dtype, chunks) ---")
        h.visititems(visit)

        # Print attributes of interesting datasets (projection, spacing, bands)
        log("\n--- Key attributes ---")
        for name in h:
            pass
        for path in [
            "science/LSAR/GCOV/grids/frequencyA",
            "science/LSAR/identification",
        ]:
            if path in h:
                log(f"\n[{path}] attrs:")
                for k, v in h[path].attrs.items():
                    log(f"  {k}: {v}")
                if path.endswith("frequencyA"):
                    for k in h[path].keys():
                        d = h[path][k]
                        if isinstance(d, h5py.Dataset) and d.size < 50:
                            log(f"  {k} = {d[()]}")
                        elif isinstance(d, h5py.Dataset):
                            log(f"  {k}: shape={d.shape} first={d[0]} last={d[-1]}"
                                if d.ndim == 1 else f"  {k}: shape={d.shape}")
        ident = h.get("science/LSAR/identification")
        if ident is not None:
            log("\n--- identification (track/frame/direction/orbit) ---")
            for k in ident.keys():
                d = ident[k]
                if isinstance(d, h5py.Dataset) and d.size < 20:
                    log(f"  {k} = {d[()]}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"\nSaved: {OUT}  <- paste this file's content back")


if __name__ == "__main__":
    main()
