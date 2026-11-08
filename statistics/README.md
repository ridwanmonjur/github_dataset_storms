# Statistics

Two analysis scripts, run from the project root with `.venv\Scripts\python.exe statistics/<script>.py`, on the cleaned and linked tables (EM-DAT's 11 storm subtypes since 1990: 2,752 storms, 3,870 event × country rows; 2026-10-28 build). Outputs in `csv/`.

The papers' figures come from `figure1_storms.py` (storms per subtype and decade, and the map), `figure_workflow.py` and `figure_kg.py`.

| Script | Question | Outputs |
|---|---|---|
| `subtype_relations.py` | Within each subtype, is each impact related to intensity, exposure and development? Which impact is most strongly connected? | `subtype_correlations.csv`, `subtype_regressions.csv`, `subtype_outcome_connection.csv` |
| `technical_validation.py` | Are the database's values and links right, against independent records and known answers? | `validation_summary.csv`, `validation_track_names.csv`, `validation_glide_dates.csv` |

**Impacts:** deaths, injured, people affected, homeless and damage (EM-DAT; damage in nominal US$, not inflation-adjusted), and displacement (IDMC, 2008 on: the IDMC rows matched to the storm in each country). An unreported value adds nothing to a sum and is never 0; the share of events reporting each value is given beside it.

---

## 1. Subtype relations (`subtype_relations.py`)

**How a relation is shown:**
1. **Spearman rank correlation** (ρ, from −1 to 1) with a 95% bootstrap interval: the relation is real when the interval excludes 0.
2. **Log-log regression** per subtype and impact, with all the variables at once: ln impact = a + b₁ ln intensity + b₂ ln area + b₃ ln population density + b₄ ln GDP per capita + b₅ year, with robust (HC3) errors, the form of Bakkensen & Mendelsohn (2016) and Kahn (2005). Each b is an elasticity: the % change of the impact for a 1% change of the variable, the others held fixed.
3. **Holm correction** across all tests: only p-values under 0.05 after correction count.

**Variables:**
- **Intensity:** for tropical cyclones, the modelled wind of the matched IBTrACS storm over the affected units (`track_wind_kt`, Eberenz et al. 2021; 760 matched rows); for the other subtypes, EM-DAT's own magnitude (km/h).
- **Exposure:** the event's Geo-Disasters units (geodesic area); the country's population and density (WDI).
- **Country:** HDI, GDP per capita, government effectiveness.
- **Not yet:** the Earth Engine measures (being extracted).

**86 of 267 correlations and 27 of 153 regression coefficients are significant after Holm.** Significant correlations by subtype: tropical cyclone 34, storm (general) 14, lightning 10, tornado 10, severe weather 9, hail 6, blizzard 3; none for extra-tropical, dust, surge or derecho (too few rows).

**Tropical cyclones (elasticity, 95% interval, all significant after Holm):**

| Impact | n | Adjusted R² | Track wind | Area | Population density | GDP per capita |
|---|---|---|---|---|---|---|
| deaths | 555 | 0.30 | **+1.11** (0.70–1.52) | +0.27 | +0.41 | **−0.37** (−0.45 to −0.29) |
| people affected | 599 | 0.32 | +2.84 (2.24–3.43) | +0.44 | +0.66 | −0.74 |
| displaced (per storm and country) | 247 | 0.31 | +2.86 (2.05–3.66) | +0.48 | +0.93 | −0.43 |

- **Every cyclone deaths relation holds, in the expected direction:** stronger wind, more exposure and lower income go with more deaths. Bakkensen & Mendelsohn's fatality regressions give +0.38 to +0.65 for ln wind, −0.13 to −0.65 for ln income and +0.11 to +0.23 for ln density; our wind elasticity is higher, as our wind is the modelled wind over the affected units rather than the best-track wind at landfall.
- **Development goes with fewer deaths for thunderstorms too:** lightning deaths against HDI ρ = −0.58 and GDP per capita −0.57; hail affected against GDP per capita −0.57.
- **Damage goes the other way:** richer countries record more nominal damage (lightning ρ = +0.71 with GDP per capita, severe weather +0.67, tornado +0.67, hail +0.69 with HDI), as there is more to destroy.

**Which impact is most strongly connected (adjusted R², 95% interval):**
- **Tropical cyclone:** affected 0.32 (0.28–0.39), displaced 0.31 (0.24–0.43), deaths 0.30 (0.26–0.38); overlapping intervals, so a tie.
- **Thunderstorms:** damage first (tornado 0.53, lightning 0.47, hail 0.38).
- **Storm (general):** deaths 0.32.
- **Blizzard:** injured 0.52.

---

## 2. Technical validation (`technical_validation.py`)

Not statistics of the storms but checks of the database (checks 6a, 6d, 6e, 6f); each runs on what is built so far and names what it skipped.

| Check | Result (2026-10-28) |
|---|---|
| 6e Track matches against EM-DAT's own storm names | **95.9%** of the 728 linked rows where both name the storm agree (level 1: 98.3%, 2: 87.5%, **3: 42.9%**, 4: 96.5%, 5: 100%); the 30 disagreements are in `validation_track_names.csv` |
| 6f IDMC date matches against GLIDE matches | of 359 IDMC rows matched by GLIDE, dates alone find the same storm for **70.2%**, another for 12.5%, none for 17.3% |
| 6a ERA5 wind against best-track wind | waits for the Earth Engine extraction |
| 6d WorldPop against GPWv4 | dropped: WorldPop was removed (commit 74cad8d), GPWv4 is the only population |

---

**Caveats:**
- R² of 0.1–0.6: the variables explain part of the variation, not most of it.
- These are associations, not causes.
- Only reported impacts are tested.
- The Earth Engine measures are not in yet; the tables update when they are.
