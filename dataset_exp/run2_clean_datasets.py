"""Clean every raw file in datasets/raw/ into datasets/clean/<source>.csv, with the event areas and points (datasets/linked/event_points.csv).
Run after run1: python run2_clean_datasets.py"""

from functools import partial

import numpy as np
import pandas as pd
import pycountry

from lib_cleans import AREAS, BASE_DIR, EVENT_POINTS, HAZARDS, cascade_candidates, run_steps, status
from run1_download_datasets import EMDAT_RAW_FILE, RAW, WORLD_BANK_CODES  # on the path through lib_cleans

# World Bank aggregates (regions, income groups) and small territories
EXCLUDE_CODES = ['AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS',
    'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC',
    'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS',
    'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN',
    'TSA', 'TSS', 'UMC', 'WLD', 'ABW', 'AND', 'ASM', 'BMU', 'CHI', 'CUW', 'CYM', 'FRO',
    'GIB', 'GRL', 'GUM', 'IMN', 'LIE', 'MAC', 'MAF', 'MCO', 'MNP', 'NCL', 'PYF', 'SMR',
    'SXM', 'TCA', 'VGB', 'VIR']
EMDAT_OUTCOMES = ["deaths", "injured", "affected", "homeless", "total_affected", "total_damage_usd"]
GLIDE_PATTERN = r"([A-Z]{2}-\d{4}-\d{6})"
# The database covers 1990-2023: the HDI starts in 1990; later years lack the event-area extracts and HDI, and EM-DAT still revises recent
# impacts
FIRST_YEAR, LAST_YEAR = 1990, 2023
GRID_YEARS = range(FIRST_YEAR, LAST_YEAR + 1)
IDMC_HAZARDS = {"Storm": "storm"}
IFRC_HAZARDS = {"Cyclone": "storm", "Storm Surge": "storm"}
# IBTrACS columns kept as reported, beside the 1-minute wind and pressure: distances in km, radii in nautical miles, speed in knots
IBTRACS_RAW = {"DIST2LAND": "dist2land_km", "LANDFALL": "landfall_km", "USA_SSHS": "usa_sshs", "USA_RMW": "usa_rmw_nmi", "USA_ROCI": "usa_roci_nmi",
               "USA_POCI": "usa_poci_mb", "USA_R34_NE": "usa_r34_ne_nmi", "USA_R34_SE": "usa_r34_se_nmi", "USA_R34_SW": "usa_r34_sw_nmi",
               "USA_R34_NW": "usa_r34_nw_nmi", "USA_GUST": "usa_gust_kt", "STORM_SPEED": "storm_speed_kt", "STORM_DIR": "storm_dir_deg"}
# One spec per dataset. columns: raw name -> database name, the only columns kept; name: the cleaned file, datasets/clean/<name>.csv;
# numeric: numbers (blank stays null, never 0);
# nonnegative: counts and amounts, |x| for negative entries; scale: unit factors; dates: parsed, timezone dropped; year_from: date
# column giving the year; dated_by: the column that dates a row, cut to FIRST_YEAR-LAST_YEAR (a year, or a date's year); indicator: country-year
# table, expanded to every country-year; min_rows: row-count assertion
SPECS = {
    # The HDI holds income and life expectancy; area population replaces national density; government effectiveness is the WGI measure
    # tied to disaster outcomes
    "HDI": dict(name="hdi", indicator=True, dated_by="year", numeric=["hdi"], columns={"iso3": "iso3", "year": "year", "hdi": "hdi"}),
    "WDI": dict(name="wdi", indicator=True, dated_by="year", numeric=["gdp_per_capita", "population"],
                columns={"economy": "iso3", "time": "year", WORLD_BANK_CODES["WDI"][0]: "gdp_per_capita", WORLD_BANK_CODES["WDI"][2]: "population"}),
    "WGI": dict(name="wgi", indicator=True, dated_by="year", numeric=["wgi_gov_effectiveness"],
                columns={"economy": "iso3", "time": "year", WORLD_BANK_CODES["WGI"][1]: "wgi_gov_effectiveness"}),
    "IBTrACS": dict(name="ibtracs", numeric=["lat", "lon", "wmo_wind_kt", "usa_wind_kt", "wmo_pres_mb", "usa_pres_mb", *IBTRACS_RAW.values()],
                    nonnegative=["wmo_wind_kt", "usa_wind_kt", "wmo_pres_mb", "usa_pres_mb"], dates=["time"], dated_by="time", min_rows=100000,
                    columns={"SID": "sid", "NAME": "name", "SEASON": "year", "ISO_TIME": "time", "BASIN": "basin", "NATURE": "nature", "LAT": "lat",
                             "LON": "lon", "WMO_WIND": "wmo_wind_kt", "WMO_AGENCY": "wmo_agency", "USA_WIND": "usa_wind_kt", "WMO_PRES": "wmo_pres_mb",
                             "USA_PRES": "usa_pres_mb", **IBTRACS_RAW}),
    "IDMC": dict(name="idmc", numeric=["displacement"], nonnegative=["displacement"], dates=["start_date"], dated_by="year",
                 columns={"iso3": "iso3", "year": "year", "start_date": "start_date", "start_date_accuracy": "start_date_accuracy",
                          "event_name": "event_name", "hazard_type_name": "hazard_type", "new_displacement": "displacement", "event_codes": "glide"}),
    "IFRC": dict(name="ifrc", numeric=["amount_requested", "amount_funded"], nonnegative=["amount_requested", "amount_funded"],
                 dates=["start_date"], year_from="start_date", dated_by="start_date",
                 columns={"appeal_id": "appeal_id", "iso3": "iso3", "start_date": "start_date", "appeal_name": "appeal_name",
                          "disaster_type": "disaster_type", "amount_requested": "amount_requested", "amount_funded": "amount_funded"}),
    # geocoding_q: the unit's quality flag (Teber et al. 2025): 1 EM-DAT's GAUL id, 2 a GeoNames name matching a GAUL unit, 3 matched only
# through its province's name, 4 GeoNames only
    "Geo-Disasters": dict(name="geodisasters", numeric=["lat", "lon", "area_km2", "adm1_code", "adm2_code", "geocoding_q"], min_rows=10000,
                          columns={"DisNo.": "disasterno", "ISO": "iso3", "admin_level": "admin_level", "ADM1_CODE": "adm1_code",
                                   "ADM1_NAME": "adm1_name", "ADM2_CODE": "adm2_code", "ADM2_NAME": "adm2_name",
                                   "latitude": "lat", "longitude": "lon", "area_km2": "area_km2", "geocoding_q": "geocoding_q"}),
    "EM-DAT": dict(name="emdat", numeric=[*EMDAT_OUTCOMES, "lat", "lon", "start_month", "start_day", "end_year", "end_month", "end_day"],
                   nonnegative=EMDAT_OUTCOMES, scale={"total_damage_usd": 1000}, dated_by="year", min_rows=10000,
                   columns={"DisNo.": "disasterno", "ISO": "iso3", "Country": "country_name", "Event Name": "event_name",
                            "Disaster Type": "disaster_type", "Disaster Subtype": "disaster_subtype",
                            "Associated Types": "associated_types", "External IDs": "glide",
                            "Location": "location",
                            "Latitude": "lat", "Longitude": "lon", "Start Year": "year", "Start Month": "start_month", "Start Day": "start_day",
                            "End Year": "end_year", "End Month": "end_month", "End Day": "end_day",
                            "Total Deaths": "deaths", "No. Injured": "injured", "No. Affected": "affected", "No. Homeless": "homeless",
                            "Total Affected": "total_affected", "Total Damage ('000 US$)": "total_damage_usd"}),
}


def standardize(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """The one place for columns and types: keep and rename the spec's columns (all others dropped), then dates, iso3 / year,
    numbers, |x| for counts and amounts, unit scales, EM-DAT ids without the country suffix, GLIDE codes; rows dated outside
    FIRST_YEAR-LAST_YEAR by the spec's dated_by column dropped."""
    spec = SPECS[dataset_name]
    print(f"[{dataset_name}] Raw rows: {len(df):,}")
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in spec["columns"] if c not in df.columns]
    assert not missing, f"[{dataset_name}] Columns missing from the raw data: {missing}"
    df = df[list(spec["columns"])].rename(columns=spec["columns"])
    for col in spec.get("dates", []):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    if "year_from" in spec:
        df["year"] = df[spec["year_from"]].dt.year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype("string").str.strip().str.upper().replace("", pd.NA)
        df = df.dropna(subset=[c for c in ("iso3", "year") if c in df.columns])
        if "year" in df.columns:
            df["year"] = df["year"].astype(int)
    for col in spec.get("numeric", []):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in spec.get("nonnegative", []):
        if (df[col] < 0).any():
            print(f"[{dataset_name}] {col}: {(df[col] < 0).sum()} negative values -> absolute")
        df[col] = df[col].abs()
    for col, factor in spec.get("scale", {}).items():
        df[col] = df[col] * factor
    if "disasterno" in df.columns:
        df["disasterno"] = df["disasterno"].astype(str).str.strip().str.replace(r"-[A-Z]{3}$", "", regex=True)
    if "glide" in df.columns:
        glide = df["glide"].str.findall(GLIDE_PATTERN).str.join("|")
        df["glide"] = glide.where(glide != "")
    if "dated_by" not in spec:
        return df
    date = df[spec["dated_by"]]
    year = date.dt.year if pd.api.types.is_datetime64_any_dtype(date) else date
    inside = year.between(FIRST_YEAR, LAST_YEAR)
    print(f"[{dataset_name}] {(~inside).sum():,} rows dated outside {FIRST_YEAR}-{LAST_YEAR} by {spec['dated_by']} dropped")
    return df[inside]


def country_year_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Every country (ISO 3166 plus any code in the data, aggregates removed) x every year of GRID_YEARS."""
    iso3s = sorted(({c.alpha_3 for c in pycountry.countries} | set(df["iso3"])) - set(EXCLUDE_CODES))
    return pd.MultiIndex.from_product([iso3s, GRID_YEARS], names=["iso3", "year"]).to_frame(index=False)


def save(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Indicator grid, row count asserted, CSV written."""
    spec = SPECS[dataset_name]
    folder = BASE_DIR / "datasets" / "clean"
    folder.mkdir(parents=True, exist_ok=True)
    if spec.get("indicator"):
        df = country_year_grid(df).merge(df.drop_duplicates(["iso3", "year"], keep="last"), on=["iso3", "year"], how="left")
    assert len(df) >= spec.get("min_rows", 1), f"[{dataset_name}] Too few rows: {len(df)}"
    df.to_csv(folder / f"{spec['name']}.csv", index=False)
    print(f"[{dataset_name}] Clean: {len(df):,} rows, columns {list(df.columns)} -> {folder / spec['name']}.csv")
    return df


def clean_indicator(dataset_name: str) -> pd.DataFrame:
    """Country-year indicator table (WDI, WGI): nothing beyond its spec."""
    return save(standardize(pd.read_csv(RAW[dataset_name]), dataset_name), dataset_name)


def point_and_area(geom) -> pd.DataFrame:
    """Point inside the largest part of each polygon + geodesic area: centroids of island groups fall at sea, date-line units wrap the globe."""
    import numpy as np
    from pyproj import Geod
    from shapely.geometry.polygon import orient
    from shapely.ops import transform
    parts = geom.to_crs(epsg=4326).explode(index_parts=True)
    wraps = (parts.bounds["maxx"] - parts.bounds["minx"]) > 180
    parts[wraps] = parts[wraps].apply(lambda g: transform(lambda x, y: (np.where(x < 0, x + 360, x), y), g))
    geod = Geod(ellps="WGS84")
    area = parts.apply(lambda g: geod.geometry_area_perimeter(orient(g))[0] / 1e6)
    point = parts.loc[area.groupby(level=0).idxmax()].representative_point().droplevel(1)
    return pd.DataFrame({"lat": point.y, "lon": ((point.x + 180) % 360) - 180,
                         "area_km2": area.groupby(level=0).sum().round(1)}).reindex(geom.index)


# === HDI === (wide -> long reshape first)
def clean_hdi() -> pd.DataFrame:
    df = pd.read_csv(RAW["HDI"], encoding="latin-1", low_memory=False)
    id_cols = ['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2023']
    metrics = {col.rsplit('_', 1)[0] for col in df.columns if col not in id_cols and col.rsplit('_', 1)[-1].isdigit()}
    df = pd.wide_to_long(df, stubnames=list(metrics), i=[c for c in id_cols if c in df.columns],
                         j='year', sep='_', suffix=r'\d+').reset_index()
    return save(standardize(df, "HDI"), "HDI")
# === IDMC === (matched to storms in run3)
def clean_idmc() -> pd.DataFrame:
    df = pd.read_csv(RAW["IDMC"], low_memory=False)
    assert "event_name" in df.columns and "hazard_type_name" in df.columns, f"{RAW['IDMC']} is not the disaster events file"
    df = standardize(df, "IDMC")
    df["hazard"] = df["hazard_type"].map(IDMC_HAZARDS)
    # Event rows (dated to the day or week) can be matched to EM-DAT events; the rest are yearly / country aggregates
    df["granularity"] = np.where(df["start_date_accuracy"].isin(["Day", "Week"]), "event", "aggregate")
    return save(df, "IDMC")
# === IFRC ===
IFRC_STORM_NAME = r"cyclone|hurricane|typhoon|storm|tornado|\bTS\b|\bTC\b"


def clean_ifrc() -> pd.DataFrame:
    df = pd.read_csv(RAW["IFRC"])
    df = standardize(df, "IFRC")
    # Hazard from the disaster type; a storm word in the appeal name makes it a storm appeal (cyclone floods are appealed as floods)
    storm_name = df["appeal_name"].str.contains(IFRC_STORM_NAME, case=False, na=False)
    df["hazard"] = df["disaster_type"].map(IFRC_HAZARDS).where(~storm_name, "storm")
    # Aggregate appeals: type Other with a bare country name ("India") or a minor-emergency fund; the rest are event appeals
    bare_country = ~df["appeal_name"].str.contains(r"[-:\d]", na=False) & (df["appeal_name"].str.split().str.len() <= 3)
    aggregate = df["disaster_type"].eq("Other") & (bare_country | df["appeal_name"].str.contains("minor em", case=False, na=False))
    df["granularity"] = np.where(aggregate, "aggregate", "event")
    return save(df, "IFRC")


# === Geo-Disasters === (after EM-DAT: units are checked against their event's EM-DAT countries)
# Geo-Disasters uses current codes where EM-DAT keeps historical / territory ones (Serbia and Montenegro, pre-2011 Sudan, Hong Kong,
# Canary Islands, Azores, Pakistan-administered Kashmir coded IND in GAUL)
ISO_TO_EMDAT = {"SRB": "SCG", "MNE": "SCG", "SSD": "SDN", "HKG": "CHN", "ESP": "SPI", "PRT": "AZO", "IND": "PAK"}
AREA_TOLERANCE_DEG = 0.01


def clean_geodisasters() -> pd.DataFrame:
    """Exact duplicates dropped; a unit's country mapped to EM-DAT's code for its event when EM-DAT lists only the alias; units in a
    country that is not one of their event's countries dropped (geocoding errors); units dated outside FIRST_YEAR-LAST_YEAR dropped (Geo-Disasters has no
    date: a unit takes its event's EM-DAT start year, else the year in its EM-DAT id). Units of events missing from EM-DAT are kept. The kept
    units are also dissolved into one area per event x country (simplified to AREA_TOLERANCE_DEG), the area run3, run4 and the cascades read, with its
    worst unit's quality flag (AREAS); and every storm x country gets a row with one coordinate, a point inside its largest affected unit,
    else EM-DAT's own point, else none (coord_status not_recorded) (EVENT_POINTS)."""
    import geopandas as gpd
    df = gpd.read_file(next(RAW["Geo-Disasters"].glob("*.gpkg")))
    df = df.set_crs(epsg=4326) if df.crs is None else df.to_crs(epsg=4326)
    pa = point_and_area(df.geometry)
    df["longitude"], df["latitude"], df["area_km2"] = pa["lon"], pa["lat"], pa["area_km2"]
    geometry = df.geometry
    df = standardize(df, "Geo-Disasters").drop_duplicates()
    emdat = pd.read_csv(BASE_DIR / "datasets" / "clean" / "emdat.csv", usecols=["disasterno", "iso3", "year"])
    year = df["disasterno"].map(emdat.drop_duplicates("disasterno").set_index("disasterno")["year"]).fillna(df["disasterno"].str[:4].astype(int))
    inside = year.between(FIRST_YEAR, LAST_YEAR)
    print(f"[Geo-Disasters] {(~inside).sum():,} units dated outside {FIRST_YEAR}-{LAST_YEAR} dropped")
    df = df[inside]
    pairs = set(zip(emdat["disasterno"], emdat["iso3"]))
    in_emdat = lambda iso3s: pd.Series([(d, i) in pairs for d, i in zip(df["disasterno"], iso3s)], index=df.index)
    alias = df["iso3"].map(ISO_TO_EMDAT)
    remap = ~in_emdat(df["iso3"]) & in_emdat(alias)
    df.loc[remap, "iso3"] = alias[remap]
    wrong = df["disasterno"].isin(emdat["disasterno"]) & ~in_emdat(df["iso3"])
    print(f"[Geo-Disasters] {remap.sum()} units recoded to EM-DAT's country code, {wrong.sum()} units outside their event's countries dropped: "
          f"{df.loc[wrong, ['disasterno', 'iso3', 'adm1_name']].to_dict('records')}")
    kept = df[~wrong].drop_duplicates()
    # Only the units the database uses: storms' and those of the records that may cascade with one (cascade_candidates())
    used = pd.MultiIndex.from_frame(kept[["disasterno", "iso3"]]).isin(cascade_candidates(pd.read_csv(BASE_DIR / "datasets" / "clean" / "emdat.csv", low_memory=False)))
    print(f"[Geo-Disasters] {used.sum():,} of {len(kept):,} units belong to a storm or a possible cascade partner")
    kept = kept[used]
    areas = gpd.GeoDataFrame(kept[["disasterno", "iso3", "geocoding_q"]], geometry=geometry.loc[kept.index].simplify(AREA_TOLERANCE_DEG).make_valid(), crs=4326)
    # An area's flag is its worst unit's (Teber et al. 2025)
    areas = areas.dissolve(["disasterno", "iso3"], aggfunc="max").reset_index()
    areas.to_file(AREAS, driver="GPKG")
    print(f"[Geo-Disasters] {len(areas):,} event x country areas -> {AREAS.name}")
    kept = save(kept, "Geo-Disasters")
    storms = pd.read_csv(BASE_DIR / "datasets" / "clean" / "emdat.csv", usecols=["disasterno", "iso3", "hazard", "lat", "lon"])
    storms = storms[storms["hazard"].notna()].drop(columns="hazard")
    largest = kept.dropna(subset=["lat", "lon"]).sort_values("area_km2", ascending=False).drop_duplicates(["disasterno", "iso3"])
    points = storms.merge(largest[["disasterno", "iso3", "lat", "lon", "area_km2"]], on=["disasterno", "iso3"], how="left", suffixes=("_emdat", ""))
    unit = points["lat"].notna()
    points["lat"], points["lon"] = points["lat"].where(unit, points["lat_emdat"]), points["lon"].where(unit, points["lon_emdat"])
    point = points["lat"].notna()
    points["coord_source"] = np.select([unit, point], ["geodisasters", "emdat"], None)
    points["coord_anchor"] = np.select([unit, point], ["largest_unit", "emdat_point"], None)
    points["coord_status"] = status(points["lat"])
    points = points.rename(columns={"area_km2": "unit_area_km2"})
    EVENT_POINTS.parent.mkdir(parents=True, exist_ok=True)
    points[["disasterno", "iso3", "lat", "lon", "coord_source", "coord_anchor", "coord_status", "unit_area_km2"]].to_csv(EVENT_POINTS, index=False)
    print(f"[Geo-Disasters] {len(points):,} storm x country rows, coordinates {points['coord_anchor'].value_counts(dropna=False).to_dict()} "
          f"-> {EVENT_POINTS.name}")
    return kept


# === IBTrACS ===
def clean_ibtracs() -> pd.DataFrame:
    """Track points with a position and time (dated by the point's time); the reported winds and pressures (WMO, US agencies) and
    IBTRACS_RAW as IBTrACS gives them."""
    df = pd.read_csv(RAW["IBTrACS"], skiprows=[1], low_memory=False, keep_default_na=False, na_values=["", " "], dtype=str)
    df = standardize(df, "IBTrACS").dropna(subset=["sid", "lat", "lon", "time"])
    df = df.copy()
    df["name"] = df["name"].str.strip().str.upper()
    print(f"[IBTrACS] {df['sid'].nunique():,} storms, {len(df):,} track points")
    return save(df[["sid", "name", "year", "time", "basin", "nature", "lat", "lon", "wmo_agency", "wmo_wind_kt", "wmo_pres_mb",
                    "usa_wind_kt", "usa_pres_mb", *IBTRACS_RAW.values()]], "IBTrACS")


# === EM-DAT ===
def clean_emdat() -> pd.DataFrame:
    """Read the manually downloaded EM-DAT export (see README) and clean it; storms get their hazard (HAZARDS), the rest none."""
    if not EMDAT_RAW_FILE.exists():
        raise FileNotFoundError(f"[EM-DAT] {EMDAT_RAW_FILE} not found: manual download, see dataset_exp/README.md")
    df = standardize(pd.read_excel(EMDAT_RAW_FILE, engine="openpyxl"), "EM-DAT")
    df["hazard"] = df["disaster_subtype"].map(HAZARDS)
    return save(df, "EM-DAT")


# (name, clean) in run order: EM-DAT before Geo-Disasters, whose units are checked against it
STEPS = [
    ("HDI", clean_hdi),
    ("WDI", partial(clean_indicator, dataset_name="WDI")),
    ("WGI", partial(clean_indicator, dataset_name="WGI")),
    ("EM-DAT", clean_emdat),
    ("Geo-Disasters", clean_geodisasters),
    ("IDMC", clean_idmc),
    ("IFRC", clean_ifrc),
    ("IBTrACS", clean_ibtracs),
]


if __name__ == "__main__":
    run_steps("run2", STEPS)
