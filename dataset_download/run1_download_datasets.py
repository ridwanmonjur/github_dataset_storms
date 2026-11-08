"""Download every source's raw file into datasets/raw/<source>/ (EM-DAT is a manual export, checked to be in place); cached files are
not downloaded again. Run first: python run1_download_datasets.py"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import wbgapi as wb

BASE_DIR = Path(__file__).parent.parent
EMDAT_RAW_FILE = BASE_DIR / "datasets" / "raw" / "emdat" / "emdat_raw.xlsx"
# Each source's raw file (Geo-Disasters: its folder), which lib_cleans reads
RAW = {
    "HDI": BASE_DIR / "datasets" / "raw" / "hdi" / "composite_hdi.csv",
    "WDI": BASE_DIR / "datasets" / "raw" / "wdi" / "wdi_raw.csv",
    "WGI": BASE_DIR / "datasets" / "raw" / "wgi" / "wgi_raw.csv",
    "IDMC": BASE_DIR / "datasets" / "raw" / "idmc" / "idmc_all.csv",
    "IFRC": BASE_DIR / "datasets" / "raw" / "ifrc" / "ifrc_appeals_raw.csv",
    "IBTrACS": BASE_DIR / "datasets" / "raw" / "ibtracs" / "ibtracs.since1980.csv",
    "Geo-Disasters": BASE_DIR / "datasets" / "raw" / "geodisasters",
    "Wikidata": BASE_DIR / "datasets" / "raw" / "wikidata" / "countries.csv",
}
# World Bank indicator codes per dataset; run2_clean_datasets.SPECS renames them
WORLD_BANK_CODES = {"WDI": ["NY.GDP.PCAP.CD", "EN.POP.DNST", "SP.POP.TOTL"], "WGI": ["GOV_WGI_PV.EST", "GOV_WGI_GE.EST"]}


def check_emdat():
    """The manual EM-DAT export is in place (it cannot be downloaded by script)."""
    assert EMDAT_RAW_FILE.exists(), f"EM-DAT is a manual download: save it as {EMDAT_RAW_FILE} (dataset_exp/README.md)"


# === HDI ===
HDI_URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"

def download_hdi() -> Path:
    raw_path = RAW["HDI"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print("[HDI] Downloading...")
        r = requests.get(HDI_URL, timeout=120)
        r.raise_for_status()
        raw_path.write_bytes(r.content)
    return raw_path


# === WDI / WGI === (WORLD_BANK_CODES)
def download_world_bank(dataset_name: str, years: range, db: int) -> Path:
    """World Bank API: the dataset's indicator codes (WORLD_BANK_CODES), as economy x time x code."""
    codes, raw_path = WORLD_BANK_CODES[dataset_name], RAW[dataset_name]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print(f"[{dataset_name}] Fetching {codes} from World Bank API (source {db})...")
        wb.data.DataFrame(codes, time=years, db=db, skipBlanks=True, columns="series").reset_index().to_csv(raw_path, index=False)
    return raw_path


def download_wdi() -> Path:
    return download_world_bank("WDI", range(1990, 2025), db=2)


# World Bank source 3 (Worldwide Governance Indicators); the old GE.EST-style codes were archived
def download_wgi() -> Path:
    return download_world_bank("WGI", range(1996, 2026), db=3)


# === IDMC ===
IDMC_HDX_API = "https://data.humdata.org/api/3/action"
IDMC_PACKAGE = "idmc-internal-displacements-new-displacements-associated-with-disasters"

def download_idmc() -> Path:
    """Download the global IDMC disaster displacement file (all countries, event rows)."""
    raw_path = RAW["IDMC"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"[IDMC] Using existing: {raw_path} (delete it to re-download)")
        return raw_path
    print("[IDMC] Downloading global disaster displacement from HDX...")
    r = requests.get(f"{IDMC_HDX_API}/package_show?id={IDMC_PACKAGE}", timeout=30)
    r.raise_for_status()
    csv_res = [res for res in r.json()["result"]["resources"] if res.get("format", "").upper() == "CSV"]
    assert len(csv_res) == 1, f"Expected 1 CSV resource, got {len(csv_res)}"
    df = pd.read_csv(StringIO(requests.get(csv_res[0]["url"], timeout=300).text))
    assert "event_name" in df.columns and "hazard_type_name" in df.columns, "Not the disaster events file"
    df.to_csv(raw_path, index=False)
    return raw_path


# === IFRC ===
IFRC_BASE_URL = "https://goadmin.ifrc.org/api/v2"

def download_ifrc() -> Path:
    raw_path = RAW["IFRC"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"[IFRC] Using existing: {raw_path} (delete it to re-download)")
        return raw_path
    print("[IFRC] Fetching appeals...")

    # Fetch disaster types
    dtypes = {}
    params = {'limit': 1000, 'offset': 0}
    while True:
        r = requests.get(f"{IFRC_BASE_URL}/disaster_type/", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        for d in data.get('results', []):
            dtypes[d['id']] = d['name']
        if not data.get('next'):
            break
        params['offset'] += 1000
        time.sleep(0.3)

    # Fetch countries
    countries = {}
    params = {'limit': 1000, 'offset': 0}
    while True:
        r = requests.get(f"{IFRC_BASE_URL}/country/", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        for c in data.get('results', []):
            countries[c['id']] = {'name': c.get('name', ''), 'iso3': c.get('iso3', '')}
        if not data.get('next'):
            break
        params['offset'] += 1000
        time.sleep(0.3)

    # Fetch appeals
    appeals = []
    params = {'limit': 1000, 'offset': 0, 'start_date__gte': '1990-01-01'}
    while True:
        r = requests.get(f"{IFRC_BASE_URL}/appeal/", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        appeals.extend(data.get('results', []))
        print(f"\r  {len(appeals):,} appeals...", end="", flush=True)
        if not data.get('next'):
            break
        params['offset'] += 1000
        time.sleep(0.3)
    print()

    records = []
    for a in appeals:
        country = a.get('country', {})
        cid = country.get('id') if isinstance(country, dict) else country
        cinfo = countries.get(cid, {})
        dtype = a.get('dtype', {})
        dtype_id = dtype.get('id') if isinstance(dtype, dict) else dtype
        records.append({
            'appeal_id': a.get('id'), 'appeal_name': a.get('name'),
            'disaster_type': dtypes.get(dtype_id, 'Unknown'),
            'start_date': a.get('start_date'), 'end_date': a.get('end_date'),
            'num_beneficiaries': a.get('num_beneficiaries'),
            'amount_requested': a.get('amount_requested'),
            'amount_funded': a.get('amount_funded'),
            'iso3': cinfo.get('iso3', ''),
        })
    pd.DataFrame(records).to_csv(raw_path, index=False)
    return raw_path


# === IBTrACS ===
IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"


def download_ibtracs() -> Path:
    raw_csv = RAW["IBTrACS"]
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if not raw_csv.exists():
        print("[IBTrACS] Downloading...")
        with requests.get(IBTRACS_URL, stream=True, timeout=600) as r, open(raw_csv, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return raw_csv


IBTRACS_NC_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"


def download_ibtracs_netcdf() -> list[Path]:
    """IBTrACS since 1980 as NetCDF, the file CLIMADA's TCTracks.from_ibtracs_netcdf reads for the wind fields of run3_link_records.match_tracks()."""
    raw_nc = BASE_DIR / "datasets" / "raw" / "ibtracs" / "IBTrACS.since1980.v04r01.nc"
    raw_nc.parent.mkdir(parents=True, exist_ok=True)
    if not raw_nc.exists():
        with requests.get(IBTRACS_NC_URL, stream=True, timeout=600) as r, open(raw_nc, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return [raw_nc]


# === LECZ ===
# NASA CMR collection of LECZ Urban-Rural Population and Land Area Estimates v3 (SEDAC)
LECZ_COLLECTION = "C3540930157-ESDIS"


def download_lecz() -> list[Path]:
    """LECZ v3 low elevation coastal zones on MERIT-DEM (MacManus et al. 2021) through NASA Earthdata (earthaccess, the token in
    secrets/earthdata_key.json); its global grid (9 arc-seconds; 5 = contiguous 0-5 m, 10 = 5-10 m, 31 = other land, empty = sea) and world
    file extracted to datasets/raw/lecz/ as published, for an Earth Engine asset."""
    import earthaccess
    import zipfile
    raw_dir, folder = BASE_DIR / "datasets" / "raw" / "lecz", BASE_DIR / "datasets" / "lecz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    granules = [g for g in earthaccess.search_data(concept_id=LECZ_COLLECTION) if "geotiff" in g["umm"]["GranuleUR"]]
    assert len(granules) == 1, [g["umm"]["GranuleUR"] for g in granules]
    raw_zip = raw_dir / granules[0]["umm"]["GranuleUR"]
    if not raw_zip.exists():
        os.environ["EARTHDATA_TOKEN"] = json.loads((BASE_DIR / "secrets" / "earthdata_key.json").read_text())["key"]
        earthaccess.login(strategy="environment")
        earthaccess.download(granules, local_path=str(raw_dir))
    with zipfile.ZipFile(raw_zip) as z:
        for n in [n for n in z.namelist() if n.startswith("lecz_v3_spatial_data/data/merit_leczs.t")]:
            (folder / Path(n).name).write_bytes(z.read(n))
    assert (folder / "merit_leczs.tif").stat().st_size > 100e6
    return sorted(folder.iterdir())


# === Wikidata === (the knowledge graph's links out: each ISO 3166 alpha-3 code to its Wikidata item)
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_QUERY = "SELECT ?item ?iso3 WHERE { ?item wdt:P298 ?iso3 }"


def download_wikidata_countries() -> Path:
    """Wikidata's items with an ISO 3166-1 alpha-3 code (property P298), as iso3 -> item."""
    raw_path = RAW["Wikidata"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print("[Wikidata] Downloading...")
        r = requests.get(WIKIDATA_SPARQL, params={"query": WIKIDATA_QUERY, "format": "json"}, timeout=120,
                         headers={"User-Agent": "storm-disaster-database/1.0 (research; github_dataset_storms)"})
        r.raise_for_status()
        rows = [(b["iso3"]["value"], b["item"]["value"]) for b in r.json()["results"]["bindings"]]
        pd.DataFrame(rows, columns=["iso3", "wikidata"]).drop_duplicates("iso3").sort_values("iso3").to_csv(raw_path, index=False)
    return raw_path


# === Geo-Disasters ===


GEODISASTERS_ZENODO_ID = "15487667"

def download_geodisasters() -> Path:
    raw_dir = RAW["Geo-Disasters"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("[Geo-Disasters] Downloading from Zenodo...")

    existing = list(raw_dir.glob("*.gpkg"))
    if existing:
        gpkg_path = existing[0]
        print(f"  Using existing: {gpkg_path}")
    else:
        r = requests.get(f"https://zenodo.org/api/records/{GEODISASTERS_ZENODO_ID}", timeout=30)
        r.raise_for_status()
        files = r.json().get("files", [])
        gpkg_files = [f for f in files if f["key"].endswith(".gpkg")]
        if not gpkg_files:
            raise RuntimeError("No GeoPackage found on Zenodo")
        best = next((f for f in gpkg_files if "subnational" in f["key"].lower()), gpkg_files[0])
        gpkg_path = raw_dir / best["key"]
        print(f"  Downloading {best['key']} ({best.get('size', 0)/1e6:.1f} MB)...")
        r = requests.get(best["links"]["self"], stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(gpkg_path, "wb") as f:
            done = 0
            for chunk in r.iter_content(65536):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/total*100:.1f}%", end="", flush=True)
        print()

    return gpkg_path


README = Path(__file__).parent.parent / "dataset_exp" / "README.md"

# (name, download): each returns its raw file; cached files are not downloaded again
STEPS = [
    ("HDI", download_hdi),
    ("WDI", download_wdi),
    ("WGI", download_wgi),
    ("IDMC", download_idmc),
    ("IFRC", download_ifrc),
    ("EM-DAT", check_emdat),
    ("Geo-Disasters", download_geodisasters),
    ("IBTrACS", download_ibtracs),
    ("IBTrACS NetCDF", download_ibtracs_netcdf),
    ("LECZ", download_lecz),
    ("Wikidata", download_wikidata_countries),
]


LOG_DIR = Path(__file__).parent.parent / "logs"


class Tee:
    """Write to the console and the log file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def notify(msg: str):
    """Print notification with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{ts}] {msg}")
    print('='*60)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    print(f"[LOG] Writing to {log_path}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n Steps: {[n for n, _ in STEPS]}")

    results = []
    for name, step in STEPS:
        notify(f"DOWNLOADING: {name}")
        try:
            out = step()
            results.append({"name": name, "out": f"{len(out):,} rows" if hasattr(out, "columns") else "done", "error": None})
        except KeyboardInterrupt:
            print("\n\n[ABORT] Interrupted by user")
            break
        except Exception as e:
            results.append({"name": name, "out": None, "error": f"{type(e).__name__}: {e}"})
            print(f"\n[ERROR] {name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    ok = [r["name"] for r in results if not r["error"]]
    fail = [r["name"] for r in results if r["error"]]
    print(f"\n[DONE] {len(ok)}/{len(results)} OK" + (f" | FAILED: {', '.join(fail)}" if fail else ""))
    for r in results:
        print(f"  {r['name']:14s} {r['out'] or '-'}  {r['error'] or ''}")
    if fail:
        print(f"\nGet these by hand (website, link, file name, where to save): {README} -> Manual Downloads; then rerun run1_download_datasets.py")
    print(f"\n[LOG] Full output and tracebacks: {log_path}")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
