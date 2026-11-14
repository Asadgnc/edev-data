"""Print footprint bounds of track 172 frames over the Okavango AOI (metadata only)."""
import os

import earthaccess

AOI = (22.0, -20.0, 23.8, -18.3)
if os.environ.get("EARTHDATA_USERNAME"):
    earthaccess.login(strategy="environment")
else:
    earthaccess.login(strategy="interactive")

res = earthaccess.search_data(
    short_name="NISAR_L2_GCOV_PROVISIONAL_V1", bounding_box=AOI,
    temporal=("2026-07-20", "2026-07-21"), count=50,
)
for g in res:
    ur = g["umm"]["GranuleUR"]
    p = ur.split("_")
    if p[5] != "172":
        continue
    poly = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    lons = [q["Longitude"] for q in poly]
    lats = [q["Latitude"] for q in poly]
    print(f"frame {p[7]}: lon {min(lons):.2f}..{max(lons):.2f}  lat {min(lats):.2f}..{max(lats):.2f}")
    for q in poly:
        print(f"    ({q['Longitude']:.2f}, {q['Latitude']:.2f})")
