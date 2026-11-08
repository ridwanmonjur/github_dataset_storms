# Pipeline

Run in this order, one script per step of the paper's workflow figure; each step writes CSVs, and only step 5 builds the database and the knowledge graph.

| Step | Script | Output |
|------|--------|--------|
| 1 Download | `../dataset_download/run1_download_datasets.py` (raw files only) | `datasets/raw/<source>/` |
| 2 Clean | `run2_clean_datasets.py` | `datasets/clean/<source>.csv`; `datasets/clean/geodisasters_areas.gpkg`; `datasets/linked/event_points.csv` (each storm × country's coordinate) |
| 3 Link | `run3_link_records.py` (needs run2's full tables) | each record's event area (`area_source`, `area_quality` in `event_points.csv`); IDMC / IFRC rows matched to their EM-DAT storms (in `clean/idmc.csv`, `clean/ifrc.csv`); `datasets/linked/cascades.csv`; every table cut to the storms' rows; best tracks, `track_features.csv` and `track_points.csv` |
| 4 Extract | `run4_extract_measures.py` (needs `secrets/gee_key.json` and `secrets/cds_key.json`; resumable) | `datasets/linked/gtsm_features.csv` (storm surge), then `gee_features.csv` (Earth Engine, deadliest first, 4 rows at a time), split into the five feature tables and the area's columns of `event_points.csv` |
| 5 Build | `run5_build_outputs.py` | `datasets/schema.yaml`; keys checked; `datasets/outputs/storms.sqlite`; `datasets/outputs/storms.ttl.gz` (RDF) with its vocabulary, SHACL shapes and VoID; the release without EM-DAT (`datasets/outputs/release/`) |

`lib_cleans.py` holds only what several scripts share (the keys `SCHEMA`, the status rule, the storm rows, the step runner).

```bash
python ../dataset_download/run1_download_datasets.py   # every raw file; cached ones are reused, so a rerun is quick
python run2_clean_datasets.py                          # the cleans
python run3_link_records.py                            # IDMC / IFRC matches, cascades, best tracks
python run4_extract_measures.py                        # GTSM, Earth Engine (about a day)
python run5_build_outputs.py
```

`run1`, `run2` and `run3` write `logs/run<n>_<timestamp>.log` and end with a table of each step's status; failed downloads are listed with a pointer to this README. `run2` cleans EM-DAT before Geo-Disasters, whose units are checked against their event's EM-DAT countries; `run3` needs `run2`'s full tables, because it matches IDMC / IFRC rows and finds the cascades before cutting every table to the storms' rows.

## Rebuild

Get the EM-DAT export from the author (its licence forbids publishing it) as `datasets/emdat_raw/emdat_raw.xlsx`, then run `run1`–`run5`. Every source is saved raw in `datasets/raw/<source>/` and reused, so a rerun reads the same files. Most sources serve only their current edition (World Bank, IDMC, IFRC), so deleting a raw file and downloading it again can change the data.

## What the Clean Does

Each dataset has one entry in `run2_clean_datasets.SPECS`; `standardize()` applies it and `save()` writes it.

- **Columns:** only the columns the database uses are kept, renamed (`DisNo.` → `disasterno`, `ISO` → `iso3`, `Start Day` → `start_day`, World Bank codes → indicator names). Missing raw columns fail with an assertion.
- **Types:** numbers coerced (a blank stays null, never 0), dates parsed with the timezone dropped, `year` taken from any form (`2013`, `YR2013`, a date).
- **Signs and units:** counts and amounts are made positive (|x|; one IFRC `amount_funded` was negative); EM-DAT damage is converted from '000 US$ to US$.
- **Ids:** EM-DAT ids lose the country suffix; GLIDE codes are pulled out of EM-DAT `External IDs` and IDMC `event_codes`.
- **Grids:** HDI, WDI and WGI are expanded to every country-year (1990–now).
- **Storms:** EM-DAT's 11 storm subtypes get `hazard = storm` (`lib_cleans.HAZARDS`; coastal floods, EM-DAT type Flood, do not); every other disaster has none. IDMC and IFRC rows are storm rows by their own hazard types (an IFRC appeal with a storm word in its name is a storm appeal).
- **Matches (`run3`):** IDMC and IFRC rows get the EM-DAT storm they match (`disasterno`, `match_method`, `idmc_confidence` / `ifrc_confidence`): GLIDE, else the same country with start dates at most 30 days apart (FLODIS); only to a storm of the row's own country.
- **Source-specific:** HDI wide → long; IDMC / IFRC `granularity` (event row vs yearly aggregate); Geo-Disasters exact duplicates dropped, country codes mapped to EM-DAT's (SRB / MNE → SCG, SSD → SDN, HKG → CHN, ESP → SPI, PRT → AZO, IND → PAK for Kashmir) and units outside their event's countries dropped.

## Datasets

| Name | Source | How |
|------|--------|-----|
| EM-DAT | CRED | **Manual** |
| Geo-Disasters | Zenodo record 15487667 | Scripted (Zenodo API) |
| IBTrACS v04r01 (CSV and NetCDF) | NOAA NCEI | Scripted (download) |
| IDMC | HDX | Scripted (HDX API) |
| IFRC | IFRC GO API | Scripted (API) |
| HDI | UNDP | Scripted (CSV) |
| WDI | World Bank | Scripted (`wbgapi`; the codes come from `SPECS`) |
| WGI | World Bank | Scripted (`wbgapi`, source 3, `GOV_WGI_*` codes from `SPECS`) |
| LECZ v3 | NASA Earthdata (CIESIN) | Scripted (`earthaccess`, token in `secrets/earthdata_key.json`) |
| Wikidata countries | Wikidata SPARQL (ISO 3166 alpha-3, P298) | Scripted (one query); the knowledge graph's links out |
| ERA5, ERA5-Land, GPWv4, GHSL, SRTM, ESA WorldCover | Google Earth Engine | read in `run4`, not downloaded: only the numbers per storm × country come back |
| GTSM v3 reanalysis | Copernicus CDS | Scripted in `run4` (one request per variable and year, cached as month files in `datasets/gtsm_raw/`) |

## Manual Downloads

Put each file in the folder shown and keep the stated format.

| Dataset | Website | Link to click | File | Save to |
|---------|---------|---------------|------|---------|
| EM-DAT | https://public.emdat.be (free registration) | Access Data → Natural disasters, all countries, 1990 to now → Download | `public_emdat_<date>.xlsx` | `datasets/emdat_raw/emdat_raw.xlsx` (rename to this) |

## If a Scripted Download Fails

Each scripted dataset reuses its raw file if one is present, so you can download it by hand and rerun. Delete the raw file to force a fresh download.

| Dataset | Website | Link to click | File | Save to |
|---------|---------|---------------|------|---------|
| HDI | https://hdr.undp.org/data-center/documentation-and-downloads | All composite indices and components time series, CSV | `HDR25_Composite_indices_complete_time_series.csv` | `datasets/hdi_raw/composite_hdi.csv` (rename to this) |
| IDMC | https://data.humdata.org/dataset/idmc-internal-displacements-new-displacements-associated-with-disasters | Internal displacements associated with disasters → Download | `internal-displacements-new-displacements-associated-with-disasters.csv` | `datasets/idmc_raw/idmc_all.csv` (rename to this) |
| Geo-Disasters | https://zenodo.org/records/15487667 | Files → `disaster_subnational_90_23.gpkg` → Download | `disaster_subnational_90_23.gpkg` | `datasets/geodisasters_raw/` |
| WDI, WGI | https://data.worldbank.org | API only (`wbgapi`) | — | Check the internet connection or World Bank API status |
| IFRC | https://go.ifrc.org | API only (GO API v2) | — | Check the internet connection or GO API status |
