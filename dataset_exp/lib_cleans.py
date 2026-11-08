"""Code shared by run2 to run5 and the statistics: paths, subtypes, every CSV's keys (SCHEMA), the status rule, the storm x country
rows, an extract's table and the step runner with its log. Imported, not run."""

import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
# The raw files' paths live with the downloads (dataset_download/)
sys.path.append(str(BASE_DIR / "dataset_download"))
LOG_DIR = BASE_DIR / "logs"

# Written by clean_geodisasters(), read by run3, run4 and the cascades: one area and one coordinate per event x country
AREAS = BASE_DIR / "datasets" / "clean" / "geodisasters_areas.gpkg"
EVENT_POINTS = BASE_DIR / "datasets" / "linked" / "event_points.csv"
# Storms: EM-DAT's 11 storm subtypes are the events; every other disaster (coastal floods too, EM-DAT type Flood) stays in the table
# without a hazard, as a cascade partner only
HAZARDS = dict.fromkeys(["Tropical cyclone", "Storm (General)", "Tornado", "Severe weather", "Blizzard/Winter storm", "Lightning/Thunderstorms",
                         "Extra-tropical storm", "Hail", "Sand/Dust storm", "Storm surge", "Derecho"], "storm")
# Gap reasons meaning the product does not cover the place or time (status not_covered); the rest are not_recorded
NOT_COVERED = {"outside_coverage", "no_land_cell", "no_sea_cell", "coord_off_land", "not_yet_available", "no_coastal_station", "no_track_match"}
# Gap reasons that are their own status
KEPT_AS_STATUS = ["not_applicable", "not_extracted"]
# Cascades (Jaeger et al. 2025): records of Jaeger's hazard types only (Table 2: no wildfires, epidemics or technological disasters), start
# dates at most LAG_DAYS apart
LAG_DAYS = 91
CASCADE_TYPES = ["Storm", "Flood", "Earthquake", "Drought", "Extreme temperature", "Mass movement (wet)", "Mass movement (dry)", "Volcanic activity"]
# Event-area extracts (run3's best tracks; run4's Earth Engine read split by entity, GTSM), each a complete table of the storm x country rows
# with a coordinate, statuses on every field
EXTRACTS = ["exposure_features", "weather_features", "convection_features", "wave_features", "snow_features", "dust_features", "track_features",
            "gtsm_features"]
# Every CSV: path under datasets/, primary key, foreign keys as (columns, table); a foreign key points at the table's primary key
SCHEMA = {
    "emdat": ("clean/emdat.csv", ["disasterno", "iso3"], []),
    "geodisasters": ("clean/geodisasters.csv", [], [(["disasterno", "iso3"], "emdat")]),
    "track_points": ("linked/track_points.csv", ["sid", "time"], []),
    "idmc": ("clean/idmc.csv", [], [(["disasterno", "iso3"], "emdat")]),
    "ifrc": ("clean/ifrc.csv", ["appeal_id"], [(["disasterno", "iso3"], "emdat")]),
    "hdi": ("clean/hdi.csv", ["iso3", "year"], []),
    "wdi": ("clean/wdi.csv", ["iso3", "year"], []),
    "wgi": ("clean/wgi.csv", ["iso3", "year"], []),
    "event_points": ("linked/event_points.csv", ["disasterno", "iso3"], [(["disasterno", "iso3"], "emdat")]),
    **{name: (f"linked/{name}.csv", ["disasterno", "iso3"], [(["disasterno", "iso3"], "emdat")]) for name in EXTRACTS},
    "cascades": ("linked/cascades.csv", ["from_disasterno", "to_disasterno", "iso3"],
                 [(["from_disasterno", "iso3"], "emdat"), (["to_disasterno", "iso3"], "emdat")]),
}


def status(values: pd.Series, gaps: pd.Series = None) -> np.ndarray:
    """The one status rule for every table, from a value and its gap reason, if any: KEPT_AS_STATUS as given, not_covered (NOT_COVERED),
    not_recorded (any other reason, or a blank), zero, reported; a blank is never read as 0."""
    gaps = pd.Series(None, index=values.index, dtype=object) if gaps is None else pd.Series(gaps, index=values.index, dtype=object)
    return np.select([gaps.isin(KEPT_AS_STATUS), gaps.isin(NOT_COVERED), gaps.notna() | values.isna(), values == 0],
                     [gaps, "not_covered", "not_recorded", "zero"], "reported")


def event_dates(year: pd.Series, month: pd.Series, day: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Earliest and latest possible start date: the day when known (clipped to the month), else the whole month; NaT without a month."""
    month_start = pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")
    month_end = month_start + pd.offsets.MonthEnd(0)
    lo = month_start + pd.to_timedelta(pd.to_numeric(day).fillna(1) - 1, unit="D")
    lo = lo.where(lo <= month_end, month_end)
    return lo, lo.where(day.notna(), month_end)


def storm_events() -> pd.DataFrame:
    """EM-DAT storms (HAZARDS), one row per event x country, with its coordinate from EVENT_POINTS (coord_lat, coord_lon; none without)
    and its possible start dates (date_lo, date_hi)."""
    emdat = pd.read_csv(BASE_DIR / "datasets" / "clean" / "emdat.csv", low_memory=False)
    points = pd.read_csv(EVENT_POINTS, usecols=["disasterno", "iso3", "lat", "lon", "coord_source", "coord_anchor", "unit_area_km2"])
    points = points.rename(columns={"lat": "coord_lat", "lon": "coord_lon"})
    events = emdat[emdat["hazard"].notna()].merge(points, on=["disasterno", "iso3"], how="left").reset_index(drop=True)
    events["date_lo"], events["date_hi"] = event_dates(events["year"], events["start_month"], events["start_day"])
    assert not events.duplicated(["disasterno", "iso3"]).any() and set(events["hazard"]) == {"storm"}
    print(f"[EVENTS] {len(events):,} storm event x country rows; coordinates {events['coord_anchor'].value_counts(dropna=False).to_dict()}")
    return events


def cascade_candidates(emdat: pd.DataFrame) -> pd.MultiIndex:
    """The EM-DAT records (disasterno, iso3) a storm may cascade with: the storms, and every record of CASCADE_TYPES in a storm's
    country whose start date (Jaeger et al.: a missing day is the 1st) lies within LAG_DAYS of the storm's (run3's find_cascades() tests their areas)."""
    recs = emdat[emdat["hazard"].notna() | emdat["disaster_type"].isin(CASCADE_TYPES)].copy()
    recs["start"] = event_dates(recs["year"], recs["start_month"], recs["start_day"])[0]
    storms = recs[recs["hazard"].notna()]
    pairs = storms[["iso3", "start"]].merge(recs[["disasterno", "iso3", "start"]], on="iso3", suffixes=("_storm", ""))
    near = (pairs["start"] - pairs["start_storm"]).abs() <= pd.Timedelta(days=LAG_DAYS)
    return pd.MultiIndex.from_frame(pd.concat([storms[["disasterno", "iso3"]], pairs.loc[near, ["disasterno", "iso3"]]]).drop_duplicates())


def with_status(df: pd.DataFrame, gaps: dict) -> pd.DataFrame:
    """Each field's <f>_gap (its reason, only where empty) and <f>_status, from {field: its gap reasons}."""
    for f, g in gaps.items():
        df[f"{f}_gap"] = g.where(df[f].isna())
        df[f"{f}_status"] = status(df[f], df[f"{f}_gap"])
    return df


def extract_table(rows: pd.DataFrame, columns: list, applies, path) -> pd.DataFrame:
    """An extract's whole table, one row per storm x country in rows, indexed by disasterno and iso3: every <f>_gap not_applicable where the
    extract does not read the subtype, else not_extracted until read; rows already read in path kept, except those with an error, a layer
    not yet available or a field new to the extract (read again)."""
    table = pd.DataFrame(index=pd.MultiIndex.from_frame(rows[["disasterno", "iso3"]].astype(str)), columns=columns[2:], dtype=object)
    gaps = [c for c in table.columns if c.endswith("_gap")]
    table[gaps] = np.repeat(np.where(applies, "not_extracted", "not_applicable")[:, None], len(gaps), axis=1)
    if path is not None and path.exists():
        old = pd.read_csv(path, dtype={"disasterno": str, "iso3": str}).set_index(["disasterno", "iso3"])
        old = old.reindex(columns=table.columns).fillna({c: "not_extracted" for c in gaps if c not in old.columns})
        old = old[~old[gaps].isin(["not_extracted", "error", "not_yet_available"]).any(axis=1)]
        done = old.index.intersection(table.index)
        table.loc[done] = old.loc[done].to_numpy()
    return table


def save_extract(table: pd.DataFrame, path: Path):
    """An extract's table with every field's <f>_gap and <f>_status."""
    df = table.reset_index()
    with_status(df, {c[:-4]: df[c] for c in df.columns if c.endswith("_gap")}).to_csv(path, index=False)


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


def run_steps(script: str, steps: list):
    """Run (name, step) in order, each step's failure reported and the next run; everything printed also goes to logs/<script>_<timestamp>.log;
    exits 1 when a step failed."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{script}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    print(f"[LOG] Writing to {log_path}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n Steps: {[n for n, _ in steps]}")
    results = []
    for name, step in steps:
        print(f"\n{'=' * 60}\n[{datetime.now().strftime('%H:%M:%S')}] {name}\n{'=' * 60}")
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
    fail = [r["name"] for r in results if r["error"]]
    print(f"\n[DONE] {len(results) - len(fail)}/{len(results)} OK" + (f" | FAILED: {', '.join(fail)}" if fail else ""))
    for r in results:
        print(f"  {r['name']:18s} {r['out'] or '-'}  {r['error'] or ''}")
    print(f"\n[LOG] Full output and tracebacks: {log_path}")
    if fail:
        sys.exit(1)
