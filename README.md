# Storm Disaster Database (dataset paper)

How much impact does a storm have in a place, and why? This project links EM-DAT's storms (its 11 storm subtypes, 1990 to now, worldwide) to their hazard, exposure, vulnerability and outcomes. Every value says why it is present or absent, every inferred link has a method and a confidence, raw values come before derived ones, and every method follows a published precedent. Self-contained: nothing is shared with `../github_chatbot_cyclones/`.

## Outputs

| Form | Where | What |
|---|---|---|
| CSV tables | `datasets/<source>/`, `datasets/linked/`, keys in `datasets/schema.yaml` | the complete tables, each kept as built, linked only by their keys (`disasterno` and `iso3` for a storm in a country, `iso3` and `year` for a country's context) |
| Storm context | `datasets/storm_context/` | the rows connected to an EM-DAT storm (1.4 of 6.6 million values) |
| Database | `datasets/storms.sqlite` | the storm context, every key enforced |
| Knowledge graph | `datasets/storms.ttl.gz` (RDF, Turtle) with its vocabulary, SHACL shapes and VoID description | the storm context as things (events, countries linked to Wikidata, tracks, units) and the rows about them |
| Release | `datasets/release/` | every table but EM-DAT's and the public graph, keys only: EM-DAT's licence forbids redistributing its data |

## Running it

`uv sync` in this folder, then run each script with `..\.venv\Scripts\python.exe <script>` from its folder. The **Get first** column lists what a step cannot fetch by itself.

| Step | Folder → script | Get first (by hand) | Gives |
|---|---|---|---|
| 1 | `dataset_download/` → `run1_download_datasets.py` | **EM-DAT**, the only manual download. Reviewers: the author's export, sent privately, saved as `datasets/emdat_raw/emdat_raw.xlsx`. A new export: [public.emdat.be](https://public.emdat.be) (free registration) → Access Data → Natural disasters, all countries, 1990 to now → Download → same path. **NASA Earthdata** token for LECZ v3 in `secrets/earthdata_key.json` | every raw file in `datasets/raw/<source>/` |
| 2–3 | `dataset_exp/` → `run2_clean_datasets.py`, `run3_link_records.py` | — | the cleaned sources in `datasets/clean/`, IDMC / IFRC rows matched to their storms, `event_points` (every storm × country and its coordinate), cascades, best tracks |
| 4 | `dataset_exp/` → `run4_extract_measures.py` (resumable, about a day) | **Google Earth Engine key**: ask the author for `gee_key.json`, or [register a Cloud project](https://code.earthengine.google.com/register) and [create a service account key](https://developers.google.com/earth-engine/guides/service_account); save as `secrets/gee_key.json` (git-ignored). **Copernicus key**: a free ECMWF account's personal access token in `secrets/cds_key.json` as `{"key": "<token>"}`, with the `sis-water-level-change-timeseries-cmip6` licence accepted at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) | `datasets/linked/track_features.csv` (best tracks), `gee_features.csv` (Earth Engine), `gtsm_features.csv` (storm surge) |
| 5 | `dataset_exp/` → `run5_build_outputs.py` | — | `schema.yaml`; keys checked; `storms.sqlite`, `storms.ttl.gz` (with SHACL and competency-question checks), `release/` |


**Viewing:** `storms.sqlite` in any SQLite viewer (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/dl/)); `storms.ttl.gz` in any RDF store (Oxigraph, GraphDB Free, Apache Jena) with SPARQL; the CSVs in a spreadsheet or `pandas`.

## Documents

| File | What |
|---|---|
| `docs/data_dictionary.csv` | Every column, each `_gap` and `_status` column included: table, kind (raw, derived, key, link, meta, status), unit, meaning; then each status and gap value (table `*`, kind `status value` / `gap value`); checked by `run5` |
| `datasets/outputs/release/supplementary_information.pdf` | The supplementary information (S1–S5): how each column is read, the values left to users with their formulas, the methods, validation, the data dictionary |
| `dataset_exp/README.md` | Sources, data versions, manual downloads, what to do when a download fails |
| `statistics/README.md` | The statistics and the technical validation |
| `CLAUDE.md` | Data rules and code style |
