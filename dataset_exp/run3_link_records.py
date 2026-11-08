"""Link the cleaned sources to EM-DAT's storms: event areas, IDMC and IFRC matches, cascades and best tracks -> datasets/linked/ and
datasets/clean/. Run after run2: python run3_link_records.py"""

import geopandas as gpd
import numpy as np
import pandas as pd

from lib_cleans import (AREAS, BASE_DIR, CASCADE_TYPES, EVENT_POINTS, LAG_DAYS, SCHEMA, event_dates, extract_table, run_steps,
                        save_extract, storm_events)

CLEAN = BASE_DIR / "datasets" / "clean"
TRACK_OUTPUT = BASE_DIR / "datasets" / "linked" / "track_features.csv"
# The IBTrACS points of the assigned tracks: the track table the database holds (the cleaned IBTrACS is only this script's input)
TRACK_POINTS = BASE_DIR / "datasets" / "linked" / "track_points.csv"
IBTRACS_CLEAN = CLEAN / "ibtracs.csv"
IBTRACS_NC = BASE_DIR / "datasets" / "raw" / "ibtracs" / "IBTrACS.since1980.v04r01.nc"
# IDMC / IFRC rows matched to EM-DAT storms: source -> start-date window in days (source minus EM-DAT): FLODIS's start dates at most 30
# days apart, the closest kept among several (Mester et al. 2023)
MATCH_SOURCES = {"idmc": (-30, 30), "ifrc": (-30, 30)}
# Cascades (Jaeger et al. 2025): EM-DAT records of CASCADE_TYPES in the same country whose Geo-Disasters areas overlap by at least OVERLAP_MIN
# (the higher of the two fractions of each area covered by their intersection) and whose start dates are at most LAG_DAYS apart
OVERLAP_MIN = 0.5
# EM-DAT associated hazard -> disaster types (Jaeger et al. 2025, Table 2); a partner whose type the storm's record lists is named by EM-DAT
ASSOCIATED = {"Flood": ["Flood"], "Storm": ["Storm"], "Slide (land, mud, snow, rock)": ["Mass movement (wet)", "Mass movement (dry)"],
              "Avalanche (Snow, Debris)": ["Mass movement (wet)", "Mass movement (dry)"], "Cold wave": ["Extreme temperature"],
              "Heat wave": ["Extreme temperature"], "Drought": ["Drought"], "Earthquake": ["Earthquake"], "Tsunami/Tidal wave": ["Earthquake"],
              "Volcanic activity": ["Volcanic activity"]}
# Subtypes matched to IBTrACS best tracks (a tornado or a European windstorm is never a tropical cyclone track)
TRACK_SUBTYPES = ["Tropical cyclone", "Storm surge", "Storm (General)"]
# Best tracks (Eberenz et al. 2021; CLIMADA's assign_hazard_to_emdat, high certainty): (dt1, dt2) windows in days between a storm's
# start and EM-DAT's, tried in turn; assigned where the storm is the only one whose wind field reaches the area in both
TRACK_LEVELS = [(10, 50), (15, 50), (25, 50), (10, 25), (15, 25)]
# Track fields with a gap (one reason for all: no_event_date, no_track_match, ambiguous_track); IBTrACS units: kt, mb, nautical miles, degrees
TRACK_FIELDS = ["track_wind_kt", "track_closest_km", "track_closest_wind_kt", "track_closest_rmw_nmi", "track_closest_roci_nmi", "track_closest_r34_max_nmi",
                "track_closest_speed_kt", "track_closest_dir_deg"]
TRACK_COLUMNS = ["disasterno", "iso3", "track_sid", "track_name", "track_basin", "track_level", "track_method", "track_confidence", "track_closest_time",
                 "track_closest_lat", "track_closest_lon", *TRACK_FIELDS, *[f"{f}_gap" for f in TRACK_FIELDS]]
# IBTrACS agency -> (scale, shift) of its reported wind against 1-minute sustained wind (Knapp & Kruk 2010, Table 1; as CLIMADA rescales
# IBTrACS for Eberenz et al. 2021); the US agencies report 1-minute wind
WIND_1MIN = {"tokyo": (0.60, 23.3), "reunion": (0.88, 0.0), "bom": (0.88, 0.0), "nadi": (0.88, 0.0), "wellington": (0.88, 0.0),
             "cma": (0.871, 0.0), "hko": (0.9, 0.0), "neumann": (0.88, 0.0)}


def match_source_events(events: pd.DataFrame, src: pd.DataFrame, window: tuple) -> dict:
    """Event rows of IDMC / IFRC -> EM-DAT storms: GLIDE first (GLIDE names the event), else same iso3 and
    hazard with the source start date in the window (an event without a start day is matched against its whole month, method suffix
    _month). GLIDE matches drop date matches."""
    src = src[(src["granularity"] == "event") & src["hazard"].notna()].assign(start_date=lambda d: pd.to_datetime(d["start_date"]))
    by_glide = {g: dn for dn, gl in zip(events["disasterno"], events["glide"]) if pd.notna(gl) for g in gl.split("|")}
    by_place = dict(list(events.dropna(subset=["date_lo"]).groupby(["iso3", "hazard"])))
    matches = {}
    for _, r in src.iterrows():
        dn = next((by_glide[g] for g in (r["glide"].split("|") if pd.notna(r.get("glide")) else []) if g in by_glide), None)
        method = "glide" if dn else None
        if dn is None and (r["iso3"], r["hazard"]) in by_place and pd.notna(r["start_date"]):
            cand = by_place[(r["iso3"], r["hazard"])]
            gap_min, gap_max = (r["start_date"] - cand["date_hi"]).dt.days, (r["start_date"] - cand["date_lo"]).dt.days
            ok = gap_min.clip(lower=window[0]).where((gap_max >= window[0]) & (gap_min <= window[1])).dropna()
            if not ok.empty:
                best = ok.abs().idxmin()
                month = cand.loc[best, "date_lo"] != cand.loc[best, "date_hi"]
                dn, method = cand.loc[best, "disasterno"], f"date_window_{window[0]}_{window[1]}d" + ("_month" if month else "")
        if dn:
            matches.setdefault(dn, []).append((r, method))
    return {dn: [x for x in rows if x[1] == "glide"] or rows for dn, rows in matches.items()}

def find_cascades() -> pd.DataFrame:
    """Influences between EM-DAT records with a storm at one end, as Jaeger et al. (2025) find them in their code (step1_2a, step1_2b,
    step1_3): records of the same country whose Geo-Disasters areas overlap by at least OVERLAP_MIN (intersection over the smaller area,
    areas in Web Mercator, EPSG:3857) and whose start dates (a missing day is the 1st) are at most LAG_DAYS apart, from the record that
    started on or before the other (both ways for a tie); their chained events are the closure of these edges. Records without units have no
    area, so no link."""
    import geopandas as gpd
    emdat = pd.read_csv(BASE_DIR / "datasets" / "clean" / "emdat.csv", low_memory=False)
    recs = emdat[emdat["disaster_type"].isin(CASCADE_TYPES)].copy()
    recs["start"], recs["date_hi"] = event_dates(recs["year"], recs["start_month"], recs["start_day"])
    cols = ["disasterno", "iso3", "disaster_type", "disaster_subtype", "hazard", "associated_types", "start", "date_hi"]
    recs = gpd.read_file(AREAS)[["disasterno", "iso3", "geometry"]].merge(recs[cols].dropna(subset=["start"]), on=["disasterno", "iso3"]).reset_index(drop=True)
    recs = gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)
    recs["geometry"] = recs.geometry.to_crs(3857).make_valid()
    recs["area"] = recs.geometry.area
    # Cheap filters first (same country, start dates within the lag, bounding boxes), the exact intersection only on what is left
    box = recs.geometry.bounds
    p = recs[["iso3"]].reset_index(names="a").merge(recs[["iso3"]].reset_index(names="b"), on="iso3")
    p = p[p["a"] < p["b"]]
    start = recs["start"]
    p = p[np.abs(start[p["b"]].to_numpy() - start[p["a"]].to_numpy()) <= np.timedelta64(LAG_DAYS, "D")]
    ba, bb = box.loc[p["a"]].to_numpy(), box.loc[p["b"]].to_numpy()
    p = p[(ba[:, 0] <= bb[:, 2]) & (bb[:, 0] <= ba[:, 2]) & (ba[:, 1] <= bb[:, 3]) & (bb[:, 1] <= ba[:, 3])][["a", "b"]]
    inter = [recs.geometry[a].intersection(recs.geometry[b]).area for a, b in zip(p["a"], p["b"])]
    p = p.assign(overlap=np.array(inter) / np.minimum(recs["area"][p["a"]].to_numpy(), recs["area"][p["b"]].to_numpy()))
    p = p[p["overlap"] >= OVERLAP_MIN]
    # Jaeger's influence (get_influencing_records): an overlapping record started on or before the other
    p = pd.concat([p, p.rename(columns={"a": "b", "b": "a"})])
    p = p[start[p["a"]].to_numpy() <= start[p["b"]].to_numpy()]
    named = lambda storm, other: (pd.notna(recs.at[storm, "hazard"]) and pd.notna(recs.at[storm, "associated_types"])
                                  and recs.at[other, "disaster_type"] in {x for h in recs.at[storm, "associated_types"].split("|") for x in ASSOCIATED.get(h, [])})
    day = recs["start"] == recs["date_hi"]
    cascades = pd.DataFrame({
        "from_disasterno": recs["disasterno"][p["a"]].to_numpy(), "to_disasterno": recs["disasterno"][p["b"]].to_numpy(), "iso3": recs["iso3"][p["a"]].to_numpy(),
        "days": (start[p["b"]].to_numpy() - start[p["a"]].to_numpy()) // np.timedelta64(1, "D"), "overlap_pct": (100 * p["overlap"]).round(1).to_numpy(),
        "date_precision": np.where(day[p["a"]].to_numpy() & day[p["b"]].to_numpy(), "day", "month"),
        "named_by_emdat": [named(x, y) or named(y, x) for x, y in zip(p["a"], p["b"])]})
    # Storm-centred: only the influences with a storm at one end
    storm = set(recs.loc[recs["hazard"].notna(), "disasterno"])
    cascades = cascades[cascades["from_disasterno"].isin(storm) | cascades["to_disasterno"].isin(storm)].reset_index(drop=True)
    cascades["method"] = f"overlap_{int(OVERLAP_MIN * 100)}pct_lag_{LAG_DAYS}d"
    cascades["confidence"] = np.select([cascades["named_by_emdat"], cascades["date_precision"] == "day"], ["high", "medium"], "low")
    print(f"[CASCADE] {len(recs):,} records with an area; {len(cascades):,} influences, {int(cascades['named_by_emdat'].sum())} named by EM-DAT, "
          f"date precision {cascades['date_precision'].value_counts().to_dict()}")
    return cascades

def link_areas() -> pd.DataFrame:
    """area_source and area_quality added to every storm x country row of EVENT_POINTS: units where the record has Geo-Disasters units that do
    not cross the date line, with the weakest grade of its units; point_buffer where it has only a coordinate (EM-DAT's point, or units across
    the date line), no grade; empty without a coordinate. run4 reads the area so defined."""
    units = gpd.read_file(AREAS).set_index(["disasterno", "iso3"])
    whole = units[units.bounds["maxx"] - units.bounds["minx"] <= 180]
    points = pd.read_csv(EVENT_POINTS)
    key = pd.MultiIndex.from_frame(points[["disasterno", "iso3"]].astype(str))
    has_units = key.isin(whole.index)
    points["area_source"] = np.select([has_units, points["coord_source"].notna()], ["units", "point_buffer"], None)
    points["area_quality"] = pd.Series(whole["geocoding_q"].reindex(key).to_numpy()).where(has_units).astype("Int64")
    points.to_csv(EVENT_POINTS, index=False)
    print(f"[AREAS] {points['area_source'].value_counts(dropna=False).to_dict()}")
    return points


def matched(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """The source's rows with the EM-DAT storm each was matched to (match_source_events): disasterno, match_method, <source>_confidence (GLIDE
    high, a date in the window medium, a storm known only to its month low). A row is linked only to a storm of its own country, so its
    disasterno and iso3 are always an EM-DAT row; the rest stay unlinked."""
    events = storm_events()
    links = {r.name: (dn, m) for dn, found in match_source_events(events, df, MATCH_SOURCES[source]).items() for r, m in found}
    rows = pd.DataFrame.from_dict(links, orient="index", columns=["disasterno", "match_method"])
    rows = rows[pd.MultiIndex.from_arrays([rows["disasterno"], df.loc[rows.index, "iso3"]]).isin(pd.MultiIndex.from_frame(events[["disasterno", "iso3"]]))]
    df = df.join(rows)
    df[f"{source}_confidence"] = np.select([df["match_method"].eq("glide"), df["match_method"].str.endswith("_month", na=False), df["match_method"].notna()],
                                           ["high", "low", "medium"], None)
    print(f"[{source.upper()}] {df['disasterno'].notna().sum():,} rows matched to {df['disasterno'].nunique():,} storms; "
          f"{len(links) - len(rows):,} matched to a storm EM-DAT does not list in their country, left unlinked")
    return df


def link_sources() -> pd.DataFrame:
    """IDMC and IFRC rows matched to storms (matched()), written back to their cleaned tables; run2 must have written them
    afresh (keep_storm_rows() cuts them to the matched rows)."""
    out = []
    for source in MATCH_SOURCES:
        path = CLEAN / f"{source}.csv"
        df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
        assert "disasterno" not in df.columns, f"{path.name} is already linked: run run2_clean_datasets.py first"
        df = matched(df, source)
        df.to_csv(path, index=False)
        out.append(df)
    return pd.concat(out)


def link_cascades() -> pd.DataFrame:
    """Cascades (find_cascades) -> linked/cascades.csv."""
    df = find_cascades()
    df.to_csv(BASE_DIR / "datasets" / SCHEMA["cascades"][0], index=False)
    return df


def keep_storm_rows() -> pd.DataFrame:
    """Every cleaned table cut to the rows the database holds: EM-DAT's storms and their cascade partners, the Geo-Disasters units of these
    records, the IDMC / IFRC rows matched to a storm, the country-years with a storm (the full tables served matching and the cascades)."""
    clean = CLEAN
    emdat, cascades = pd.read_csv(clean / "emdat.csv", low_memory=False), pd.read_csv(BASE_DIR / "datasets" / SCHEMA["cascades"][0])
    storms = emdat[emdat["hazard"].notna()]
    records = set(storms["disasterno"]) | set(cascades["from_disasterno"]) | set(cascades["to_disasterno"])
    storm_years = pd.MultiIndex.from_frame(storms[["iso3", "year"]])
    # EM-DAT's own point serves only event_points (clean_geodisasters()), so it leaves the table here
    keep = {"emdat": lambda df: df["disasterno"].isin(records), "geodisasters": lambda df: df["disasterno"].isin(records),
            "idmc": lambda df: df["disasterno"].notna(), "ifrc": lambda df: df["disasterno"].notna(),
            **{n: lambda df: pd.MultiIndex.from_frame(df[["iso3", "year"]]).isin(storm_years) for n in ("hdi", "wdi", "wgi")}}
    rows = []
    for name, rule in keep.items():
        df = pd.read_csv(clean / f"{name}.csv", keep_default_na=False, na_values=[""], low_memory=False)
        kept = rule(df)
        df[kept].drop(columns=["lat", "lon"] if name == "emdat" else []).to_csv(clean / f"{name}.csv", index=False)
        rows.append({"table": name, "cleaned": len(df), "kept": int(kept.sum())})
        print(f"[KEEP] {name:14s} {len(df):>8,} -> {int(kept.sum()):>7,} rows")
    return pd.DataFrame(rows)

def match_tracks(events: pd.DataFrame):
    """Best tracks of cyclone-type events (TRACK_SUBTYPES) as Eberenz et al. (2021) match them with CLIMADA: each IBTrACS storm's wind field
    (TropCyclone.from_tracks, Holland 2008, tracks at 1 h) on 300 arc-second centroids inside the event's area; a storm hits when its wind
    reaches 17.5 m/s there; a hitting storm started 0 to dt days before EM-DAT's start (for every possible start day) is assigned at the
    first level of TRACK_LEVELS where it is the only candidate for both windows and not yet assigned to another event of the country.
    Intensity: the assigned storm's strongest modelled wind over the area; beside it IBTrACS's reported values at the track point closest
    to the area and over the storm's life. Written to TRACK_OUTPUT."""
    import shapely
    from climada.hazard import Centroids, TCTracks, TropCyclone
    tracks = TCTracks.from_ibtracs_netcdf(year_range=(1990, 2030), estimate_missing=True, file_name=str(IBTRACS_NC))
    tracks.equal_timestep(time_step_h=1)
    by_sid = {t.sid: t for t in tracks.data}
    genesis = pd.Series({sid: pd.Timestamp(t.time.values[0]) for sid, t in by_sid.items()})
    areas = gpd.read_file(AREAS).set_index(["disasterno", "iso3"]).geometry
    rows = events[events["coord_source"].notna()]
    table = extract_table(rows, TRACK_COLUMNS, rows["disaster_subtype"].isin(TRACK_SUBTYPES).to_numpy(), None)
    gap_cols = [f"{f}_gap" for f in TRACK_FIELDS]
    cands, shapes = {}, {}
    for r in rows[rows["disaster_subtype"].isin(TRACK_SUBTYPES)].itertuples():
        key = (str(r.disasterno), str(r.iso3))
        shapes[key] = shape = areas.get((r.disasterno, r.iso3), shapely.Point(r.coord_lon, r.coord_lat).buffer(0.5))
        if pd.isna(r.date_lo):
            table.loc[key, gap_cols] = "no_event_date"
            continue
        # Centroids every 300 arc-seconds inside the area (Eberenz et al.'s resolution), else the point inside it
        x0, y0, x1, y1 = shape.bounds
        lon, lat = (a.ravel() for a in np.meshgrid(np.arange(x0, x1, 1 / 12) + 1 / 24, np.arange(y0, y1, 1 / 12) + 1 / 24))
        keep = shapely.contains_xy(shape, lon, lat)
        lon, lat = (lon[keep], lat[keep]) if keep.any() else (np.array([shape.representative_point().x]), np.array([shape.representative_point().y]))
        sids = genesis[(genesis <= r.date_lo) & (genesis > r.date_hi - pd.Timedelta(days=max(TRACK_LEVELS[0])))].index
        wind = (TropCyclone.from_tracks(TCTracks([by_sid[s] for s in sids]), centroids=Centroids(lat=lat, lon=lon), ignore_distance_to_coast=True)
                .intensity.max(axis=1).toarray().ravel() if len(sids) else np.array([]))
        hit = np.array(sids)[wind > 0]
        cands[key] = (hit, r.date_hi - genesis[hit], dict(zip(sids, wind)))
        table.loc[key, gap_cols] = "ambiguous_track" if len(hit) else "no_track_match"
    taken = set()
    for level, (dt1, dt2) in enumerate(TRACK_LEVELS, 1):
        for key, (hit, age, wind) in cands.items():
            if pd.notna(table.at[key, "track_sid"]):
                continue
            in1, in2 = hit[age < pd.Timedelta(days=dt1)], hit[age < pd.Timedelta(days=dt2)]
            if len(in1) == 1 and len(in2) == 1 and in1[0] == in2[0] and (in1[0], key[1]) not in taken:
                taken.add((in1[0], key[1]))
                table.loc[key, ["track_sid", "track_name", "track_wind_kt", "track_level", "track_method", "track_confidence"]] = [
                    in1[0], by_sid[in1[0]].name, round(wind[in1[0]] * 1.944, 1), level, f"eberenz_level_{level}_{dt1}_{dt2}d", "high" if level <= 3 else "medium"]
                table.loc[key, gap_cols] = None
    # IBTrACS as reported: the point closest to the area (distance in an equidistant projection centred on the area, 0 inside; the strongest
    # of the points over it), the storm's peak
    points = pd.read_csv(IBTRACS_CLEAN, parse_dates=["time"], keep_default_na=False, na_values=[""],
                         low_memory=False)
    # 1-minute wind: WMO's rescaled from its agency's averaging (WIND_1MIN), else the US agencies'
    scale, shift = (points["wmo_agency"].map(lambda a: WIND_1MIN.get(a, (1.0, 0.0))[i] if pd.notna(a) else (1.0, 0.0)[i]) for i in (0, 1))
    points["wind_kt"] = ((points["wmo_wind_kt"] - shift) / scale).round(1).fillna(points["usa_wind_kt"])
    points = points.groupby("sid")
    for key in table.index[table["track_sid"].notna()]:
        pts, centre = points.get_group(table.at[key, "track_sid"]), shapes[key].representative_point()
        aeqd = f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +units=m"
        xy = gpd.GeoSeries(gpd.points_from_xy((pts["lon"] + 180) % 360 - 180, pts["lat"]), crs=4326).to_crs(aeqd)
        dist = xy.distance(gpd.GeoSeries([shapes[key]], crs=4326).to_crs(aeqd).iloc[0]).to_numpy() / 1000
        closest = np.flatnonzero(dist == dist.min())
        p = pts.iloc[closest[np.argmax(pts["wind_kt"].iloc[closest].fillna(-1).to_numpy())]]
        table.loc[key, ["track_basin", "track_closest_time", "track_closest_lat", "track_closest_lon", "track_closest_km", "track_closest_wind_kt",
                        "track_closest_rmw_nmi", "track_closest_roci_nmi", "track_closest_r34_max_nmi", "track_closest_speed_kt", "track_closest_dir_deg"]] = [
            pts["basin"].mode().iat[0] if pts["basin"].notna().any() else None, p["time"].strftime("%Y-%m-%d %H:%M"), p["lat"], p["lon"],
            round(float(dist.min()), 1), p["wind_kt"], p["usa_rmw_nmi"], p["usa_roci_nmi"],
            p[["usa_r34_ne_nmi", "usa_r34_se_nmi", "usa_r34_sw_nmi", "usa_r34_nw_nmi"]].max(), p["storm_speed_kt"], p["storm_dir_deg"]]
    save_extract(table, TRACK_OUTPUT)
    print(f"[TRACK] {table['track_sid'].notna().sum():,} of {len(cands):,} cyclone-type rows assigned "
             f"(levels {table['track_level'].value_counts().sort_index().to_dict()}); gaps {table['track_wind_kt_gap'].value_counts().to_dict()}")


def write_track_points():
    """The IBTrACS points of the tracks assigned in TRACK_OUTPUT -> TRACK_POINTS, as IBTrACS gives them."""
    tracks = set(pd.read_csv(TRACK_OUTPUT, usecols=["track_sid"])["track_sid"].dropna())
    points = pd.read_csv(IBTRACS_CLEAN, keep_default_na=False, na_values=[""], low_memory=False)
    points[points["sid"].isin(tracks)].to_csv(TRACK_POINTS, index=False)
    print(f"[TRACK] {points['sid'].isin(tracks).sum():,} points of {len(tracks):,} assigned tracks -> {TRACK_POINTS.name}")


def link_tracks():
    """Best tracks (match_tracks()) only when TRACK_OUTPUT is missing or older than its inputs (CLIMADA's tracks take several GB and
    minutes), then their points."""
    inputs = [IBTRACS_NC, AREAS, CLEAN / "emdat.csv", IBTRACS_CLEAN]
    if not TRACK_OUTPUT.exists() or TRACK_OUTPUT.stat().st_mtime < max(p.stat().st_mtime for p in inputs):
        match_tracks(storm_events())
    write_track_points()


# (name, link) in run order: IDMC / IFRC and the cascades need the full cleaned tables, so the cut to the storms' rows follows them
STEPS = [
    ("Event areas", link_areas),
    ("IDMC, IFRC", link_sources),
    ("Cascades", link_cascades),
    ("Storm rows", keep_storm_rows),
    ("Best tracks", link_tracks),
]


if __name__ == "__main__":
    run_steps("run3", STEPS)
