# Secrets

Put keys here; git ignores every file in this folder except this README and `.gitignore`.

| File | Used by | Get it |
|---|---|---|
| `gee_key.json` | `dataset_exp/run2a_download_gee.py` (Google Earth Engine service account; its `client_email` and `project_id` are the account and project used) | Ask the author for theirs (sent privately), or make your own (see the main README, step 2) |
| `cds_key.json` | `dataset_exp/run2b_download_copernicus.py` (GTSM storm surge on the Copernicus CDS) | Free ECMWF account: copy the personal access token from your profile into this file as `{"key": "<token>"}`, and accept the `sis-water-level-change-timeseries-cmip6` licence (https://cds.climate.copernicus.eu) on its Download tab |

| `earthdata_key.json` | the LECZ v3 download (SEDAC, low elevation coastal zone) | Free NASA Earthdata account (https://urs.earthdata.nasa.gov/users/new): Profile → Generate Token, saved here as `{"key": "<token>"}`; tokens expire after about 60 days |

API keys for the models (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`) are environment variables or Kaggle secrets, not files here.
