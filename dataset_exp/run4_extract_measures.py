"""Hazard and exposure per storm x country from Google Earth Engine and GTSM storm surge -> datasets/linked/gee_features.csv,
gtsm_features.csv, then split into one table per entity. Resumable, about a day. Run after run3: python run4_extract_measures.py"""

import json
import logging
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
import rasterio.mask
from datetime import datetime
from pathlib import Path

import cdsapi
import ee
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import MultiPolygon, mapping, shape
from tqdm import tqdm

from lib_cleans import AREAS, EVENT_POINTS, HAZARDS, extract_table, save_extract, storm_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
OUTPUT = BASE_DIR / "datasets" / "linked" / "gee_features.csv"
KEY_FILE = BASE_DIR / "secrets" / "gee_key.json"
CDS_KEY_FILE = BASE_DIR / "secrets" / "cds_key.json"
LECZ = BASE_DIR / "datasets" / "raw" / "lecz" / "merit_leczs.tif"

GEE_ASSETS = ["ESA/WorldCover/v200", "USGS/SRTMGL1_003", "CIESIN/GPWv411/GPW_Population_Count", "JRC/GHSL/P2023A/GHS_BUILT_S",
              "ECMWF/ERA5/HOURLY", "ECMWF/ERA5_LAND/DAILY_AGGR", "NASA/GSFC/MERRA/aer/2", "UCSB-CHG/CHIRPS/DAILY",
              "projects/sat-io/open-datasets/GMW/extent"]

# Subtypes read by the Copernicus storm-surge extract
GTSM_SUBTYPES = ["Storm surge", "Tropical cyclone", "Extra-tropical storm", "Storm (General)"]
STORM = ("srtm", "ghsl", "era5", "windloss")
LAYERS = {"Tropical cyclone": STORM + ("waves", "lowland", "gust", "chirps", "mangrove"),
          "Extra-tropical storm": STORM + ("waves", "lowland", "gust"),
          "Storm (General)": STORM + ("rainrate", "convective", "waves", "lowland", "gust", "chirps"),
          **dict.fromkeys(["Tornado", "Hail", "Lightning/Thunderstorms"], STORM + ("rainrate", "convective")),
          **dict.fromkeys(["Severe weather", "Derecho"], STORM + ("rainrate", "convective", "gust")),
          "Sand/Dust storm": STORM + ("gust", "dust"),
          "Blizzard/Winter storm": STORM + ("snow", "gust"),
          "Storm surge": STORM + ("waves", "lowland", "gust", "mangrove")}
assert set(LAYERS) == set(HAZARDS), set(HAZARDS) ^ set(LAYERS)
ALL_LAYERS = sorted({g for layers in LAYERS.values() for g in layers})
LAND_LAYERS = {"srtm", "lowland", "ghsl", "chirps"}
DAILY_LAYERS = {"era5", "windloss", "convective", "waves", "rainrate", "snow", "gust", "dust", "chirps"}

DEFAULT_BUFFER_M = 50_000
MIN_BUFFER_M, MAX_BUFFER_M, SHRINK_STEP_M = 5_000, 300_000, 10_000
MIN_LAND_FRAC = 0.5
MAX_WINDOW_DAYS, MAX_MONTH_WINDOW_DAYS = 14, 31
GHSL_CELL_M2 = 100 * 100
# Loss index threshold: each ERA5 cell's 98th percentile of daily maximum 10 m wind, 1991-2020, an asset built once (too large per event)
WIND_P98_ASSET = "assets/hdi/era5_wind10m_dailymax_p98_1991_2020"
# Each gridded product is read on its own grid
SCALE_M = {"era5": 27_830, "era5land": 11_132, "era5wave": 55_660, "merra2": 55_660, "chirps": 5_566, "gmw": 100}
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
GMW_EPOCHS, GMW = [1996, 2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020], "projects/sat-io/open-datasets/GMW/extent/gmw_v3_{}_vec"
# Fields of each layer; the first empty means the layer returned nothing
FIELDS = {"srtm": ["srtm_elev_mean"],
          "lowland": ["lecz_below_10m_share"], "ghsl": ["ghsl_built_km2"],
          "era5": ["era5_mslp_min_hpa", "era5_precip_mm", "era5_precip_max_mm", "era5_wind_max_ms", "era5_t2m_min_c", "era5_n_hours"],
          "windloss": ["era5_wind_excess_max", "era5_storm_loss_index", "era5_wind_p98_mean_ms", "gpw_count"],
          "convective": ["era5_cape_max_jkg", "era5_shear500_max_ms", "era5_shear850_max_ms", "era5_totals_index_max_k", "era5_conv_precip_max_mm",
                         "era5_wmaxshear_max"],
          "waves": ["era5_wave_height_max_m"],
          "rainrate": ["era5_rain_rate_max_mmh"], "snow": ["era5_snowfall_mm", "era5land_snow_depth_max_m"],
          "gust": ["era5_gust_max_ms"], "dust": ["merra2_dust_smass_max_ugm3", "merra2_dust_aod_max"],
          "chirps": ["chirps_precip_mm", "chirps_precip_max_mm"], "mangrove": ["gmw_mangrove_km2"]}
assert set(FIELDS) == set(ALL_LAYERS)
# The read split into one table per entity, each keyed by disasterno and iso3 with its fields' _gap and _status: the event area (into
# event_points.csv, beside the storm's coordinate), exposure, the weather over the window (with the wind loss index), the convective
# environment, the sea state, snow and dust
GEE_TABLES = {"event_points": ["area_km2", "buffer_km", "area_land_frac", "gee_date"],
              "exposure_features": ["gpw_count", "ghsl_built_km2", "ghsl_epoch", "ghsl_extrapolated", *FIELDS["srtm"], *FIELDS["lowland"],
                                    *FIELDS["mangrove"], "gmw_epoch"],
              "weather_features": [*FIELDS["era5"], "era5_wind_p98_mean_ms", "era5_wind_excess_max", "era5_storm_loss_index", *FIELDS["gust"],
                                   *FIELDS["chirps"]],
              "convection_features": [*FIELDS["convective"], *FIELDS["rainrate"]],
              "wave_features": FIELDS["waves"],
              "snow_features": FIELDS["snow"],
              "dust_features": FIELDS["dust"]}
COLUMNS = ["disasterno", "iso3", "area_km2", "buffer_km", "area_land_frac", "ghsl_epoch", "ghsl_extrapolated", "gmw_epoch",
           *[f for fs in FIELDS.values() for f in fs], *[f"{f}_gap" for fs in FIELDS.values() for f in fs], "gee_date"]
# GTSM at the CDS: requests in the dataset's queue at once (its limit is about 3)
STORE, GTSM, GTSM_WORKERS = "https://cds.climate.copernicus.eu/api", "sis-water-level-change-timeseries-cmip6", 3
# Earth Engine: rows sent at once (each is one computation on Google's servers; the quota allows parallel requests)
GEE_WORKERS = 4
BOX_MARGIN_KM, YEARS = 50, range(1950, 2025)
GTSM_RAW = BASE_DIR / "datasets" / "raw" / "gtsm"
GTSM_OUTPUT = BASE_DIR / "datasets" / "linked" / "gtsm_features.csv"
GTSM_FIELDS = ["gtsm_surge_max_m", "gtsm_surge_max_date", "gtsm_twl_max_m", "gtsm_n_stations"]
GTSM_COLUMNS = ["disasterno", "iso3", "box_km", "gtsm_lat", "gtsm_lon", *GTSM_FIELDS, *[f"{f}_gap" for f in GTSM_FIELDS], "gtsm_date"]
GTSM_VARIABLES = ["storm_surge_residual", "total_water_level"]


def pending(table: pd.DataFrame) -> pd.Index:
    """The rows of an extract still to read."""
    return table.index[table[[c for c in table.columns if c.endswith("_gap")]].eq("not_extracted").any(axis=1)]


def rnd(v, nd=1, scale=1.0, offset=0.0):
    """Round a GEE value; None stays None so missing is never written as 0."""
    return None if v is None else round(v * scale + offset, nd)


def event_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Event window: EM-DAT dates padded 1 day each side, at most 14 days, else the whole start month."""
    start = pd.to_datetime(dict(year=df["year"], month=df["start_month"], day=df["start_day"].fillna(1)), errors="coerce")
    end = pd.to_datetime(dict(year=df["end_year"], month=df["end_month"], day=df["end_day"]), errors="coerce") + pd.Timedelta(days=2)
    known_days = df["start_day"].notna() & end.notna()
    month_start = start.dt.to_period("M").dt.to_timestamp()
    df["window_start"] = start.where(~known_days, start - pd.Timedelta(days=1))
    df["window_end"] = end.where(known_days, month_start + pd.offsets.MonthBegin(1)).clip(upper=df["window_start"] + pd.to_timedelta(
        known_days.map({True: MAX_WINDOW_DAYS, False: MAX_MONTH_WINDOW_DAYS}), unit="D"))
    log.info(f"  Event windows: {known_days.sum():,} exact days, {(~known_days & start.notna()).sum():,} month, {start.isna().sum():,} none")
    return df


def buffer_radius_m(area_km2) -> int:
    """Buffer radius of a unit's area (circle of the same area, 5-300 km); 50 km without a unit."""
    return max(MIN_BUFFER_M, min(int(math.sqrt(area_km2 * 1e6 / math.pi)), MAX_BUFFER_M)) if pd.notna(area_km2) and area_km2 > 0 else DEFAULT_BUFFER_M


def init_gee() -> str:
    """Earth Engine with the service account in secrets/gee_key.json; returns its project."""
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(credentials=ee.ServiceAccountCredentials(key["client_email"], str(KEY_FILE)), project=key["project_id"])
    return key["project_id"]


def speed(im: ee.Image, level: str = "10m") -> ee.Image:
    """Wind speed of an ERA5 hourly image at a level (10m, 100m, 850hPa, 500hPa)."""
    return im.select(f"u_component_of_wind_{level}").hypot(im.select(f"v_component_of_wind_{level}")).rename(f"wind{level}")


def collection_stats(ic: ee.ImageCollection, image: ee.Image, reducer: ee.Reducer, geom, scale: int) -> ee.Dictionary:
    """Number of images in a collection and, when there are any, the reduced region of the image built from it."""
    return ee.Dictionary({"n": ic.size(), "s": ee.Algorithms.If(ic.size().gt(0), image.reduceRegion(reducer, geom, scale, bestEffort=True), ee.Dictionary({}))})


def lecz_share(polygon):
    """Share of the area's land in LECZ v3's contiguous 0-10 m zone (grid values 5 and 10; 31 other land), cells weighted by cos latitude;
    None without a land cell."""
    with rasterio.open(LECZ) as src:
        cells, transform = rasterio.mask.mask(src, [mapping(polygon)], crop=True, filled=False)
    lat = transform.f + transform.e * (np.arange(cells.shape[1]) + 0.5)
    weight = np.cos(np.radians(lat))[:, None] * ~np.ma.getmaskarray(cells[0])
    land = (weight * np.isin(cells[0].data, [5, 10, 31])).sum()
    return float((weight * np.isin(cells[0].data, [5, 10])).sum() / land) if land else None


def averaged(density: ee.Image, native: ee.Projection) -> ee.Image:
    """A density averaged from its native cells to a coarser scale (GHSL's pyramid does not average), empty cells as 0."""
    return density.unmask(0).setDefaultProjection(native).reduceResolution(ee.Reducer.mean(), maxPixels=65535)


def wind_p98_image(project: str) -> ee.Image:
    """The wind 98th-percentile asset; where it does not exist yet, its export is started (hours on Google's servers) and the run stops."""
    asset = f"projects/{project}/{WIND_P98_ASSET}"
    if ee.data.getInfo(asset):
        return ee.Image(asset)
    hourly, start = ee.ImageCollection("ECMWF/ERA5/HOURLY"), ee.Date("1991-01-01")
    day = lambda i: hourly.filterDate(start.advance(i, "day"), start.advance(ee.Number(i).add(1), "day")).map(speed).max()
    p98 = ee.ImageCollection(ee.List.sequence(0, 10956).map(day)).reduce(ee.Reducer.percentile([98])).rename("wind_p98").toFloat()
    if not ee.data.getInfo(asset.rsplit("/", 1)[0]):
        ee.data.createAsset({"type": "FOLDER"}, asset.rsplit("/", 1)[0])
    ee.batch.Export.image.toAsset(image=p98, description="era5_wind_p98", assetId=asset, scale=SCALE_M["era5"], crs="EPSG:4326",
                                  region=ee.Geometry.Rectangle([-180, -90, 180, 90], None, False), maxPixels=1e10).start()
    raise SystemExit(f"Wind 98th percentile export started -> {asset}; run again when it has finished (the service account needs roles/earthengine.writer)")


def month_file(variable: str, month: pd.Timestamp) -> Path:
    """The cached GTSM file of a variable and month."""
    return GTSM_RAW / f"{variable}_daily_max_{month.year}_{month.month:02d}.nc"


def gtsm_year(variable: str, year: int):
    """One CDS request for a variable's whole year, split into the cached month files; sent again a minute later while the dataset's queue
    is full."""
    client = cdsapi.Client(url=STORE, key=json.loads(CDS_KEY_FILE.read_text())["key"], quiet=True)
    tmp = GTSM_RAW / f"{variable}_daily_max_{year}.download"
    while not tmp.exists():
        try:
            client.retrieve(GTSM, {"variable": [variable], "experiment": "reanalysis", "temporal_aggregation": "daily_maximum", "year": [str(year)],
                                   "month": [f"{m:02d}" for m in range(1, 13)], "version": "v3"}, str(tmp.with_suffix(".part")))
            tmp.with_suffix(".part").rename(tmp)
        except Exception as e:
            if "queued requests" not in str(e):
                raise
            time.sleep(60)
    parts = [tmp]
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            parts = [tmp.with_suffix(f".{i}.nc") for i, n in enumerate(n for n in z.namelist() if n.endswith(".nc"))]
            for part, n in zip(parts, [n for n in z.namelist() if n.endswith(".nc")]):
                part.write_bytes(z.read(n))
        tmp.unlink()
    # A file holding one month is kept as served; a file holding several is split by month
    for part in parts:
        with xr.open_dataset(part) as ds:
            months = pd.DatetimeIndex(ds["time"].values).to_period("M").unique()
            if len(months) > 1:
                for m in months:
                    if not month_file(variable, m.to_timestamp()).exists():
                        ds.sel(time=str(m)).to_netcdf(month_file(variable, m.to_timestamp()))
        if len(months) == 1 and not month_file(variable, months[0].to_timestamp()).exists():
            part.rename(month_file(variable, months[0].to_timestamp()))
        else:
            part.unlink()


def gtsm_row(row, half_km: float) -> dict:
    """The highest daily surge (and its day) and total water level in the event window among GTSM stations in the box, from the cached month files."""
    out = {}
    for variable, field in zip(GTSM_VARIABLES, ("gtsm_surge_max_m", "gtsm_twl_max_m")):
        months = pd.date_range(row["window_start"].to_period("M").to_timestamp(), row["window_end"] - pd.Timedelta(days=1), freq="MS")
        ds = xr.concat([xr.open_dataset(month_file(variable, m)) for m in months], "time")
        level = ds[next(iter(ds.data_vars))].sel(time=slice(row["window_start"], row["window_end"] - pd.Timedelta(days=1)))
        lat, lon, half = ds["station_y_coordinate"].values, ds["station_x_coordinate"].values, half_km / 111
        near = np.flatnonzero((np.abs(lat - row["coord_lat"]) <= half) &
                              (np.abs((lon - row["coord_lon"] + 180) % 360 - 180) <= half / max(np.cos(np.radians(row["coord_lat"])), 0.1)))
        if len(near) == 0:
            return {"gtsm_n_stations": 0, **{f"{f}_gap": "no_coastal_station" for f in GTSM_FIELDS if f != "gtsm_n_stations"}}
        peak = level.isel(stations=near).max("time").values
        best = near[int(np.nanargmax(peak))]
        out.update({"gtsm_n_stations": len(near), field: round(float(np.nanmax(peak)), 2)})
        if variable == "storm_surge_residual":
            out.update({"gtsm_lat": round(float(lat[best]), 3), "gtsm_lon": round(float(lon[best]), 3),
                        "gtsm_surge_max_date": pd.Timestamp(level.isel(stations=best).idxmax("time").values).strftime("%Y-%m-%d")})
    return out


def extract_gtsm(events: pd.DataFrame):
    """GTSM per storm x country of GTSM_SUBTYPES: the years still missing from the cache fetched first (GTSM_WORKERS requests at once), then
    every row read from the cache -> GTSM_OUTPUT."""
    assert CDS_KEY_FILE.exists(), (f"Copernicus needs a key: copy your ECMWF personal access token (profile page of cds.climate.copernicus.eu) "
                                   f"into {CDS_KEY_FILE} as {{\"key\": \"<token>\"}} and accept the {GTSM} licence on its Download tab")
    events = event_windows(events[events["coord_source"].notna()].copy())
    GTSM_RAW.mkdir(parents=True, exist_ok=True)
    table = extract_table(events, GTSM_COLUMNS, events["disaster_subtype"].isin(GTSM_SUBTYPES).to_numpy(), GTSM_OUTPUT)
    todo = events[pd.MultiIndex.from_frame(events[["disasterno", "iso3"]].astype(str)).isin(pending(table))].sort_values("deaths", ascending=False, na_position="last")
    dated = todo[todo["window_start"].notna() & todo["year"].isin(YEARS)]
    months = {m for s, e in zip(dated["window_start"], dated["window_end"]) for m in pd.date_range(s.to_period("M").to_timestamp(), e - pd.Timedelta(days=1), freq="MS")}
    years = sorted({(v, m.year) for v in GTSM_VARIABLES for m in months if not month_file(v, m).exists()})
    log.info(f"[gtsm] {len(todo):,} of {len(events):,} rows to read; {len(years)} variable-years to request")
    with ThreadPoolExecutor(GTSM_WORKERS) as pool:
        jobs = {pool.submit(gtsm_year, v, y): (v, y) for v, y in years}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="gtsm years"):
            if job.exception():
                log.warning(f"  [gtsm {jobs[job]}] {type(job.exception()).__name__}: {str(job.exception())[:200]}")

    for _, row in todo.iterrows():
        half_km = buffer_radius_m(row["unit_area_km2"]) / 1000 + BOX_MARGIN_KM
        out = {"box_km": round(2 * half_km, 1), "gtsm_date": datetime.now().strftime("%Y-%m-%d"), **{f"{f}_gap": None for f in GTSM_FIELDS}}
        reason = "no_event_date" if pd.isna(row["window_start"]) else "outside_coverage" if row["year"] not in YEARS else None
        if reason is None:
            try:
                out.update(gtsm_row(row, half_km))
            except Exception as e:
                log.warning(f"  [gtsm {row['disasterno']} {row['iso3']}] {type(e).__name__}: {str(e)[:200]}")
                reason = "error"
        if reason:
            out.update({f"{f}_gap": reason for f in GTSM_FIELDS})
        table.loc[(str(row["disasterno"]), str(row["iso3"]))] = [out.get(c) for c in table.columns]
    save_extract(table, GTSM_OUTPUT)
    log.info(f"[gtsm] {len(table):,} rows -> {GTSM_OUTPUT}; {len(pending(table)):,} not yet read")


def extract_gee(events: pd.DataFrame):
    """Earth Engine layers per storm x country with a coordinate (event_points.csv: Geo-Disasters largest unit, else EM-DAT point) -> OUTPUT."""
    df = events[events["coord_source"].notna()].rename(columns={"unit_area_km2": "coord_area_km2"}).reset_index(drop=True)
    log.info(f"  With coords: {len(df):,} {df['coord_anchor'].value_counts().to_dict()}")
    df = event_windows(df)
    # Sliver parts left by the dissolve (< 1e-8 square degrees, at most 0.003% of an area) dropped: Earth Engine reads some as the whole globe
    units = gpd.read_file(AREAS).set_index(["disasterno", "iso3"])
    units["geometry"] = units.geometry.apply(lambda g: MultiPolygon([p for p in g.geoms if p.area >= 1e-8]) if g.geom_type == "MultiPolygon" else g)
    areas = units.geometry
    assert df.loc[df["coord_source"] == "geodisasters", ["disasterno", "iso3"]].apply(tuple, axis=1).isin(areas.index).all()
    wind_p98 = wind_p98_image(init_gee())

    land = ee.ImageCollection("ESA/WorldCover/v200").first().neq(80).unmask(0).rename("land")
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    era5land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    # The day after ERA5's last day: the per-day loss index needs every day of the window
    era5_end = pd.Timestamp(ee.ImageCollection("ECMWF/ERA5/HOURLY").limit(1, "system:time_start", False).first().get("system:time_start").getInfo(),
                            unit="ms").normalize() + pd.Timedelta(days=1)
    min_mean_max = ee.Reducer.min().combine(ee.Reducer.mean(), sharedInputs=True).combine(ee.Reducer.max(), sharedInputs=True)
    # Highest value over every cell the area touches (with the mean, as min_mean_max: a plain max counts only cells whose centre is inside,
    # none for an area smaller than an ERA5 cell); its keys are <band>_max
    area_max = ee.Reducer.max().combine(ee.Reducer.mean(), sharedInputs=True)
    region = lambda image, geom, scale, reducer=ee.Reducer.mean(): image.reduceRegion(reducer, geom, scale, bestEffort=True)

    # The whole table, not_extracted until read; deadliest first
    table = extract_table(df, COLUMNS, [True] * len(df), OUTPUT)
    todo = df[pd.MultiIndex.from_frame(df[["disasterno", "iso3"]].astype(str)).isin(pending(table))].sort_values("deaths", ascending=False, na_position="last")
    log.info(f"  To process: {len(todo):,} {todo['disaster_subtype'].value_counts().to_dict()}")

    def read_row(i: int, row) -> tuple:
        """One row's Earth Engine values and gaps, as (key, fields)."""
        event_id, subtype, layers = row["disasterno"], row["disaster_subtype"], LAYERS[row["disaster_subtype"]]
        lat, lon = float(row["coord_lat"]), ((float(row["coord_lon"]) + 180) % 360) - 180
        year = int(row["year"])
        ghsl_year = min(GHSL_EPOCHS, key=lambda x: abs(x - year))
        gmw_year = min(GMW_EPOCHS, key=lambda x: abs(x - year))
        gaps = {g: "not_applicable" for g in ALL_LAYERS if g not in layers}

        # Area: the event's units in that country, else (no units, or units across the date line) a buffer shrunk until >= 50% land
        unit = areas.get((event_id, row["iso3"]))
        buffer_m = None
        if unit is not None and unit.bounds[2] - unit.bounds[0] <= 180:
            geom = ee.Geometry(mapping(unit), None, False)
            land_frac = region(land, geom, 1000).get("land").getInfo()
        else:
            point = ee.Geometry.Point([lon, lat])
            r0 = buffer_radius_m(row["coord_area_km2"])
            radii = list(range(r0, MIN_BUFFER_M, -SHRINK_STEP_M)) + [MIN_BUFFER_M]
            fc = ee.FeatureCollection([ee.Feature(point.buffer(r), {"r": r}) for r in radii])
            fracs = dict(zip(radii, land.reduceRegions(fc, ee.Reducer.mean(), scale=max(100, r0 // 150)).aggregate_array("mean").getInfo()))
            buffer_m = next((r for r in radii if (fracs.get(r) or 0) >= MIN_LAND_FRAC), MIN_BUFFER_M)
            geom, land_frac = point.buffer(buffer_m), fracs.get(buffer_m)
        area_m2 = geom.area(maxError=1000).getInfo()
        log.info(f"[{i}/{len(todo)}] {event_id} {row['iso3']} {subtype} {'units' if buffer_m is None else 'buffer'} {area_m2 / 1e6:,.0f} km2, land {land_frac}")

        # Coverage and dates decide which layers are requested
        has_window = pd.notna(row["window_start"])
        for g in set(layers) - set(gaps):
            if g in LAND_LAYERS and not land_frac:
                gaps[g] = "coord_off_land"
            elif g in DAILY_LAYERS and not has_window:
                gaps[g] = "no_event_date"
            elif (g == "srtm" and not -56 <= lat <= 60) or (g == "lowland" and not -59.49 <= lat <= 84) or (g == "chirps" and abs(lat) > 50):
                gaps[g] = "outside_coverage"
            elif g == "windloss" and row["window_end"] > era5_end:
                gaps[g] = "not_yet_available"
        want = set(layers) - set(gaps)
        if has_window:
            w0, w1 = row["window_start"].strftime("%Y-%m-%d"), row["window_end"].strftime("%Y-%m-%d")

        # All requested layers in one request
        requests = {}
        if "srtm" in want:
            requests["srtm"] = region(dem, geom, 1000)
        if "ghsl" in want:
            built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{ghsl_year}").select("built_surface")
            requests["ghsl"] = region(averaged(built.divide(GHSL_CELL_M2), built.projection()), geom, 1000)
        if want & DAILY_LAYERS:
            hourly = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(w0, w1)
        if "era5" in want:
            # Per cell over the window: lowest pressure, total rain, strongest hourly 10 m wind, lowest temperature
            era5 = ee.Image.cat(hourly.select("mean_sea_level_pressure").min(), hourly.select("total_precipitation").sum(), hourly.map(speed).max(),
                                hourly.select("minimum_2m_temperature_since_previous_post_processing").min().rename("t2m_min"))
            requests["era5"] = collection_stats(hourly, era5, min_mean_max, geom, SCALE_M["era5"])
        if "windloss" in want:
            # Klawa & Ulbrich (2003) Eq. 2 summed over days: GPWv4 population (as Cusack 2023) x (daily max / v98 - 1)^3 above v98, on GPW's grid
            ratios = ee.ImageCollection([hourly.filterDate(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")).map(speed).max()
                                         .divide(wind_p98).rename("r") for d in pd.date_range(row["window_start"], row["window_end"], freq="D", inclusive="left")])
            gpw = ee.Image(f"CIESIN/GPWv411/GPW_Population_Count/gpw_v4_population_count_rev11_{min(range(2000, 2021, 5), key=lambda y: abs(y - year))}_30_sec")
            requests["windloss"] = ee.Dictionary({"n": hourly.size(), "s": ee.Dictionary({
                "ratio": region(ratios.max(), geom, SCALE_M["era5"], area_max).get("r_max"),
                "p98": region(wind_p98, geom, SCALE_M["era5"]).get("wind_p98"),
                "gpw": gpw.reduceRegion(ee.Reducer.sum(), geom, crs=gpw.projection(), maxPixels=1e11, tileScale=16).get("population_count"),
                "index": ratios.map(lambda r: r.subtract(1).max(0).pow(3)).sum().multiply(gpw.unmask(0)).reduceRegion(
                    ee.Reducer.sum(), geom, crs=gpw.projection(), maxPixels=1e11, tileScale=16).get("r")})})
        if "convective" in want:
            # Per hour: CAPE, wind change from 10 m to 500 hPa (about 5.5 km, for Taszarek et al.'s 0-6 km) and to 850 hPa (about 1.5 km),
            # the total totals index; WMAXSHEAR = sqrt(2 CAPE) x 10 m-500 hPa shear (Taszarek et al. 2020b; 500 hPa as Ben Bouallegue et al. 2024)
            def environment(im):
                cape = im.select("convective_available_potential_energy").rename("cape")
                shear = lambda p: (im.select(f"u_component_of_wind_{p}hPa").subtract(im.select("u_component_of_wind_10m"))
                                   .hypot(im.select(f"v_component_of_wind_{p}hPa").subtract(im.select("v_component_of_wind_10m"))).rename(f"shear{p}"))
                return ee.Image.cat(cape, shear(500), shear(850), im.select("total_totals_index").rename("totals"),
                                    cape.multiply(2).sqrt().multiply(shear(500)).rename("wmaxshear"))
            conv = hourly.map(environment).max().addBands(hourly.select("convective_precipitation").sum().rename("conv_precip"))
            requests["convective"] = collection_stats(hourly, conv, area_max, geom, SCALE_M["era5"])
        if "waves" in want:
            # Waves at sea off the area (ERA5's wave model, open sea only, about 0.5 degrees): the area widened by BOX_MARGIN_KM, as for GTSM
            waves = hourly.select(["significant_height_of_combined_wind_waves_and_swell"]).max()
            requests["waves"] = collection_stats(hourly, waves, area_max, geom.buffer(BOX_MARGIN_KM * 1000, 1000), SCALE_M["era5wave"])
        if "rainrate" in want:
            requests["rainrate"] = collection_stats(hourly, hourly.select("maximum_total_precipitation_rate_since_previous_post_processing").max().rename("rate"),
                                                    area_max, geom, SCALE_M["era5"])
        if "snow" in want:
            land_days = era5land.filterDate(w0, w1)
            snowfall = hourly.select("snowfall").sum()
            snow = snowfall.addBands(land_days.select("snow_depth_max").max().rename("depth"))
            requests["snow"] = collection_stats(hourly, snow, ee.Reducer.mean(), geom, SCALE_M["era5land"])
        if "gust" in want:
            requests["gust"] = collection_stats(hourly, hourly.select("wind_gust_since_previous_post_processing_10m").max().rename("gust"), area_max, geom, SCALE_M["era5"])
        if "dust" in want:
            # MERRA-2: highest dust surface concentration and dust optical depth at 550 nm
            merra = ee.ImageCollection("NASA/GSFC/MERRA/aer/2").filterDate(w0, w1)
            requests["dust"] = collection_stats(merra, merra.select(["DUSMASS", "DUEXTTAU"]).max(), area_max, geom, SCALE_M["merra2"])
        if "chirps" in want:
            # CHIRPS rain over the window per cell: the area's mean and its wettest cell
            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(w0, w1)
            requests["chirps"] = collection_stats(chirps, chirps.select("precipitation").sum(), area_max, geom, SCALE_M["chirps"])
        if "mangrove" in want:
            # Mangrove area within BOX_MARGIN_KM of the area, the epoch nearest the storm's year
            requests["mangrove"] = ee.Image.pixelArea().multiply(ee.Image(0).paint(ee.FeatureCollection(GMW.format(gmw_year)), 1)).reduceRegion(
                ee.Reducer.sum(), geom.buffer(BOX_MARGIN_KM * 1000, 1000), SCALE_M["gmw"], maxPixels=1e11, tileScale=16)

        res = {}
        try:
            res = ee.Dictionary(requests).getInfo() if requests else {}
        except Exception as e:
            log.warning(f"  [{event_id}] GEE request {sorted(requests)}: {e}")
            gaps.update({g: "error" for g in requests})

        values = {
            "srtm": lambda s: {"srtm_elev_mean": rnd(s.get("elevation"))},
            "ghsl": lambda s: {"ghsl_built_km2": rnd(s.get("built_surface"), 3, area_m2 / 1e6)},
            "era5": lambda s: {"era5_mslp_min_hpa": rnd(s.get("mean_sea_level_pressure_min"), 1, 0.01), "era5_precip_mm": rnd(s.get("total_precipitation_mean"), 1, 1000),
                               "era5_precip_max_mm": rnd(s.get("total_precipitation_max"), 1, 1000), "era5_wind_max_ms": rnd(s.get("wind10m_max")),
                               "era5_t2m_min_c": rnd(s.get("t2m_min_min"), 1, 1, -273.15)},
            "windloss": lambda s: {"era5_wind_excess_max": rnd(s.get("ratio"), 2), "era5_storm_loss_index": rnd(s.get("index"), 1),
                                   "era5_wind_p98_mean_ms": rnd(s.get("p98")), "gpw_count": rnd(s.get("gpw"), 0)},
            "convective": lambda s: {"era5_cape_max_jkg": rnd(s.get("cape_max"), 0), "era5_shear500_max_ms": rnd(s.get("shear500_max")),
                                     "era5_shear850_max_ms": rnd(s.get("shear850_max")), "era5_totals_index_max_k": rnd(s.get("totals_max")),
                                     "era5_conv_precip_max_mm": rnd(s.get("conv_precip_max"), 1, 1000),
                                     "era5_wmaxshear_max": rnd(s.get("wmaxshear_max"), 0)},
            "waves": lambda s: {"era5_wave_height_max_m": rnd(s.get("significant_height_of_combined_wind_waves_and_swell_max"), 2)},
            "rainrate": lambda s: {"era5_rain_rate_max_mmh": rnd(s.get("rate_max"), 1, 3600)},
            "snow": lambda s: {"era5_snowfall_mm": rnd(s.get("snowfall"), 1, 1000), "era5land_snow_depth_max_m": rnd(s.get("depth"), 2)},
            "gust": lambda s: {"era5_gust_max_ms": rnd(s.get("gust_max"))},
            "dust": lambda s: {"merra2_dust_smass_max_ugm3": rnd(s.get("DUSMASS_max"), 1, 1e9), "merra2_dust_aod_max": rnd(s.get("DUEXTTAU_max"), 3)},
            "chirps": lambda s: {"chirps_precip_mm": rnd(s.get("precipitation_mean")), "chirps_precip_max_mm": rnd(s.get("precipitation_max"))},
            "mangrove": lambda s: {"gmw_mangrove_km2": rnd(s.get("area"), 2, 1e-6)},
        }
        gee_data = {}
        for g in res:
            gee_data.update(values[g](res[g]["s"] if "s" in res[g] else res[g]))
            if g == "era5":
                gee_data["era5_n_hours"] = res[g].get("n")
            if res[g].get("n") == 0:
                gaps[g] = "not_yet_available"

        if "lowland" in want:
            try:
                gee_data["lecz_below_10m_share"] = rnd(lecz_share(unit if buffer_m is None else shape(geom.getInfo())), 3)
            except Exception as e:
                log.warning(f"  [{event_id}] LECZ: {e}")
                gaps["lowland"] = "error"

        # Reason for every requested layer that came back empty without one
        for g in want:
            if g not in gaps and gee_data.get(FIELDS[g][0]) is None:
                gaps[g] = "no_land_cell" if g in LAND_LAYERS else "no_sea_cell" if g == "waves" else "no_data"

        out = {"area_km2": round(area_m2 / 1e6, 1),
               "buffer_km": None if buffer_m is None else round(buffer_m / 1000, 1), "area_land_frac": rnd(land_frac, 2),
               **gee_data,
               "ghsl_epoch": ghsl_year if "ghsl" in want else None, "ghsl_extrapolated": (1 if year > 2020 else 0) if "ghsl" in want else None,
               "gmw_epoch": gmw_year if "mangrove" in want else None,
               **{f"{f}_gap": gaps.get(g) for g, fs in FIELDS.items() for f in fs}, "gee_date": datetime.now().strftime("%Y-%m-%d")}
        return (str(event_id), str(row["iso3"])), out

    # GEE_WORKERS rows in Earth Engine at once; only this thread writes the table, saved every 50 rows; a row that fails stays not_extracted
    with ThreadPoolExecutor(GEE_WORKERS) as pool:
        jobs = [pool.submit(read_row, i, row) for i, (_, row) in enumerate(todo.iterrows(), 1)]
        for n, job in enumerate(tqdm(as_completed(jobs), total=len(jobs)), 1):
            try:
                key, out = job.result()
            except Exception as e:
                log.warning(f"  a row failed and stays not_extracted: {type(e).__name__}: {str(e)[:200]}")
                continue
            table.loc[key] = [out.get(c) for c in table.columns]
            if n % 50 == 0:
                save_extract(table, OUTPUT)
    save_extract(table, OUTPUT)
    log.info(f"{OUTPUT}: {len(table):,} rows, {len(pending(table)):,} not yet read")


def split_gee():
    """The Earth Engine read (OUTPUT) split into GEE_TABLES, each keyed by disasterno and iso3 with its fields' _gap and _status; the area's
    columns join event_points.csv (EVENT_POINTS, every storm x country; empty where no area was read)."""
    df = pd.read_csv(OUTPUT, dtype={"disasterno": str, "iso3": str}, low_memory=False)
    placed = [c for cols in GEE_TABLES.values() for c in cols]
    assert len(placed) == len(set(placed)) and {f for fs in FIELDS.values() for f in fs} <= set(placed), "a field in no table, or in two"
    for name, cols in GEE_TABLES.items():
        out = df[["disasterno", "iso3", *[c for col in cols for c in (col, f"{col}_gap", f"{col}_status") if c in df.columns]]]
        if name == "event_points":
            points = pd.read_csv(EVENT_POINTS)
            out = points.drop(columns=[c for c in out.columns[2:] if c in points.columns]).merge(out, on=["disasterno", "iso3"], how="left", validate="1:1")
            assert len(out) == len(points)
        out.to_csv(OUTPUT.with_name(f"{name}.csv"), index=False)
        log.info(f"[GEE] {name}: {len(out):,} rows, {out.shape[1] - 2} columns")


def main():
    """GTSM first (its CDS years GTSM_WORKERS at once, then rows from the cache: minutes once cached), then Earth Engine (GEE_WORKERS rows at
    once) and its split."""
    events = storm_events()
    extract_gtsm(events.copy())
    extract_gee(events.copy())
    split_gee()


if __name__ == "__main__":
    main()
