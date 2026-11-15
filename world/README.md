# World data: country water, causes and the RSynthesis world engine

Since 2026-11-15 the EDev-Vision website covers the whole world rather than
the Okavango Delta. This folder holds the data and code behind that version.
The website's copies live in `observatory/public/data/` of
[yamenkabalan/EDev-Vision](https://github.com/yamenkabalan/EDev-Vision)
(branch `global-water-explorer`).

Over GitHub Pages:
`https://Asadgnc.github.io/edev-data/world/<file>`

## Files

| File | What it holds |
|---|---|
| `global-water.json` | Inland surface water for 266 countries and territories, 1984–2018, in km². Per year: `[permanent, seasonal, projectedPermanent, projectedSeasonal]`. The projected fields estimate water in areas Landsat did not see. Outlines come from Natural Earth 50 m. |
| `world.json` | The causes for each country: ERA5 annual precipitation (`pr`, 1984–2025) and temperature anomaly (`ta`), UN SDG 6.6.1 permanent and seasonal water (`sdg`, 2000–2022), ISO3 code, continent and subregion. It also holds sums for the world and 6 continents, the NOAA ONI (El Niño) index and the source links. |
| `engine-summary.json` | Test results for the engine under its default options. |
| `world-context.json` | Everything the AI assistant is allowed to say about each place (256 places): measured water, causes, fit, 2030 range and tests. |
| `engine/world.ts` | The RSynthesis world engine (TypeScript, no dependencies). |
| `engine/world.test.mjs` | Tests for the engine on these files. |
| `scripts/build-world-data.py` | Builds `world.json` from the public sources below. |
| `scripts/engine-summary.mts`, `scripts/assistant-context.mts` | Build `engine-summary.json` and `world-context.json`. |

The scripts and tests use paths inside the website repository
(`public/data/`, `lib/world.ts`). Run them from there:

```
python scripts/build-world-data.py
node --experimental-strip-types scripts/engine-summary.mts
node --experimental-strip-types scripts/assistant-context.mts
npm test
```

## Sources

| Data | Source |
|---|---|
| Country water 1984–2018 | EC JRC Global Surface Water, GAUL-0 statistics: <https://storage.googleapis.com/global-surface-water-stats/gaul0-all-2018.csv>. Method: Pekel et al. 2016, *Nature* 540, 418–422. |
| Water 2000–2022 (check) | UN SDG indicator 6.6.1, series EN_LKRV_PWAN / EN_LKRV_SWAN: <https://unstats.un.org/sdgs/dataportal/database> |
| Precipitation | ERA5 via Our World in Data: <https://ourworldindata.org/grapher/average-precipitation-per-year> |
| Temperature anomaly | ERA5 via Our World in Data: <https://ourworldindata.org/grapher/annual-temperature-anomalies> |
| El Niño (ONI) | NOAA CPC: <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt> |
| Boundaries, continents | Natural Earth 50 m: <https://github.com/nvkelso/natural-earth-vector> |

## How the engine works

1. **Response:** the logarithm of a country's open water each year.
2. **Low-coverage years are left out.** When the area Landsat could not see is
   larger than 15% of the measured water, that year is only a lower bound, so
   it is not used for fitting. This affects most years before 2000. The option
   `mode: 'gapfilled'` uses the JRC estimate for the unseen area instead.
3. **Causes:** rain this year, rain last year, heat, El Niño (ONI) and a time
   trend. The user can choose any subset of them.
4. **Every combination of causes is a model.** Each model is weighed by its
   closed-form evidence under a Zellner g-prior with g = max(n, K²) and a
   Scott–Berger prior over model size. A cause's share of that weight is its
   *inclusion probability*.
5. **Projection:** Student-t predictive draws (1,200 draws, fixed seed).
   Future rain, heat and ONI are drawn from the country's own past years. The
   output gives the 2.5, 10, 50, 90 and 97.5% points for each year up to 2030.

## How good it is (default options)

| Test | Result |
|---|---|
| Countries with enough clear years to fit | 208 |
| Leave-one-year-out skill, median country | 0.26 (positive in 69%) |
| Leave-one-year-out skill, whole world | 0.64 |
| Trained only up to 2008, scored on 2009–2018, median country | 0.05 |
| UN SDG 2018 → 2022 change inside our 80% range | 51% of 184 countries |

Skill is 1 − (error of the engine ÷ error of always guessing the country's
mean). The engine explains the world total well. For a single country it is
only a little better than the mean. These numbers are shown on the site as
they are.

## Licences

JRC Global Surface Water: free, with attribution to EC JRC / Google. UN SDG
data: UN terms of use. ERA5: Copernicus licence (via OWID, CC BY). NOAA ONI:
public domain. Natural Earth: public domain.
