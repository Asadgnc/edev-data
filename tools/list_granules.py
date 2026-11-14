"""
EDev-Vision | List all NISAR GCOV granules over the Okavango AOI.
Parses track/frame/direction/polarization mode from the granule name:
NISAR_L2_PR_GCOV_<cycle>_<track>_<A|D>_<frame>_<mode>_<pol>_..._<start>_<end>_...
Writes granules_list.csv and prints grouped summary.
Auth: EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars, else interactive prompt.
"""
import os
from collections import defaultdict

import earthaccess
import pandas as pd

AOI_BBOX = (22.0, -20.0, 23.8, -18.3)
COLLECTIONS = ["NISAR_L2_GCOV_BETA_V1", "NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_V1"]

if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
    earthaccess.login(strategy="environment")
else:
    earthaccess.login(strategy="interactive")

rows = []
for sn in COLLECTIONS:
    res = earthaccess.search_data(
        short_name=sn, bounding_box=AOI_BBOX,
        temporal=("2025-10-01", "2026-11-14"), count=500,
    )
    print(f"{sn}: {len(res)}")
    for g in res:
        ur = g["umm"]["GranuleUR"]
        p = ur.split("_")
        size = g["umm"].get("DataGranule", {}).get("ArchiveAndDistributionInformation", [{}])[0].get("SizeInBytes")
        rows.append({
            "collection": sn, "granule": ur, "cycle": p[4], "track": p[5],
            "direction": p[6], "frame": p[7], "pol": p[9], "start": p[11],
            "gb": round(size / 1e9, 2) if size else None,
        })

df = pd.DataFrame(rows).sort_values("start")
df.to_csv("granules_list.csv", index=False)
print("\n--- by track/frame/direction/pol ---")
for k, d in df.groupby(["track", "frame", "direction", "pol"]):
    print(k, len(d), "granules")
    for _, r in d.iterrows():
        print(f"   {r['start'][:8]}  {r['collection']:<30} {r['gb']} GB")
