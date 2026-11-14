# Okavango Flood Pulse — NISAR data layers

Open data layers produced by **team EDev-Vision** for the NASA Space Apps
Challenge 2026 challenge *Dancing with the SARs*. They show the seasonal flood
pulse of the Okavango Delta, Botswana, as seen by **NISAR L-band radar**.

This repository holds **only the data**, served as static files over GitHub
Pages so the web app needs no backend and stays free and always online.

## Why NISAR, and why L-band

The Okavango flood arrives from the Angolan highlands in the middle of
Botswana's dry season and spreads through papyrus reed beds and riparian
woodland. Most of that water is **under vegetation**, where an optical
satellite sees only leaves.

L-band radar (~24 cm wavelength) penetrates the canopy and reaches the stems
and trunks. Two signatures matter:

| Surface | What the radar sees | Appearance |
|---|---|---|
| Calm open water | The pulse reflects away from the radar like a mirror; almost nothing returns | **Dark** in HH |
| Flooded vegetation | The pulse bounces off the water, then off trunks/stems, and returns to the radar (**double bounce**) | **Very bright** in HH, and much brighter in HH than in HV |
| Dry land | Diffuse surface and volume scattering | Intermediate |

Double bounce also preserves polarization, so the **HH − HV gap** is what
separates flooded woodland from merely rough dry ground. Sentinel-1 C-band
(~5.6 cm) largely stops at the canopy top and cannot do this reliably — that is
the specific reason this project needs NISAR.

## Source data

- **Mission:** NISAR (NASA–ISRO Synthetic Aperture Radar)
- **Product:** L2 GCOV — geocoded polarimetric covariance, frequency A (L-band)
- **Polarizations:** HH and HV (`HHHH`, `HVHV` diagonal covariance terms,
  linear γ⁰ power)
- **Geometry:** track 172, frame 168, **ascending** — identical for every date
- **Native grid:** 20 m, UTM zone 34S (EPSG:32734)
- **Access:** NASA Earthdata / ASF DAAC via `earthaccess`
- **Processing level flags:** dates up to 2026-10-12 are NISAR **PROVISIONAL**
  products; 2026-10-24 is **VALIDATED**

Using one track, one frame and one flight direction for the whole series is not
a convenience — it is a correctness requirement. Comparing dates from different
viewing geometries makes a difference in incidence angle look like surface
change.

## Method

1. **Stream, don't download.** Each granule is ~1.9 GB. The HDF5 arrays are
   chunked 512×512, so only the chunks covering the area of interest are
   transferred over HTTP range requests.
2. **Multilook.** 4×4 blocks are averaged **in linear power** (not in dB, which
   would bias the mean low), giving 80 m pixels with roughly 16× less speckle.
3. **Convert to dB.** `dB = 10·log10(γ⁰)`, the power convention.
4. **Classify** into open water / flooded vegetation / other using fixed
   backscatter thresholds, identical on every date.
5. **Measure area in UTM**, before reprojection. Web Mercator inflates area by
   ~12% at this latitude, so areas are counted as 80 m pixels in UTM 34S.
6. **Reproject to EPSG:3857** and write PNG overlays, which a browser map can
   draw directly.

Validity is checked against Sentinel-2 optical imagery on matching dates, and
against a dry-land reference backscatter that should not change between dates.

## Known limitations

- **The thresholds are uncertain.** They are tuned on one date's histogram and
  sanity-checked optically. This is a visualization of radar backscatter
  classes, **not** a calibrated hydrological product. `meta.json` carries
  `"uncertain": true` and the threshold values used.
- **Permanent data gap: 2026-07-27 to 2026-08-10.** NISAR acquired nothing in
  this window. The app shows it as *no data*, never as *no flood*.
- **Mixed product maturity.** One VALIDATED date among PROVISIONAL ones may
  carry a small calibration offset.
- **Frame edges.** A single frame does not cover the far west of the area of
  interest; edge pixels are marked invalid rather than guessed.
- Very smooth dry surfaces (salt pans, bare clay) can be as dark as water, and
  dry rough terrain can be bright. Both are known L-band confusions.

## File layout

```
data/
  meta.json                  legend, thresholds, gaps, caveats, date list
  timeseries.json            one record per date, areas in km2
  <YYYY-MM-DD>/
    class.png                RGBA classification overlay, EPSG:3857
    hh.png                   grayscale HH dB (fixed -25..0 dB stretch)
    bounds.json              geographic bounds for both PNGs
```

`timeseries.json` record:

```json
{
  "date": "2026-07-20",
  "flooded_km2": 0.0,
  "open_water_km2": 0.0,
  "flooded_veg_km2": 0.0,
  "valid_km2": 0.0,
  "quality": "PROVISIONAL",
  "granule": "NISAR_L2_PR_GCOV_..."
}
```

`bounds.json` gives both map-library conventions: `bounds` as
`[[south, west], [north, east]]` for Leaflet, and `coordinates` clockwise from
top-left for MapLibre.

## Reuse

Everything here is produced by the open pipeline in the project repository
(`pipeline/config.py`, `fetch.py`, `render.py`, `qa.py`). Pointing `AOI` and the
track/frame selection at another wetland — the Pantanal, the Sudd, the Mekong
delta — reproduces the same layers for that site without changing the code.

## Credits

NISAR data courtesy of NASA/JPL-Caltech and ISRO, distributed by the Alaska
Satellite Facility DAAC. Basemap © OpenStreetMap contributors.
