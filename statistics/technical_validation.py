"""Technical validation: our values against independent records and our links against known answers; a check without its input is
skipped -> statistics/csv/. Run after run5: python statistics/technical_validation.py"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "csv"
OUT.mkdir(exist_ok=True)
LINKED = BASE / "datasets" / "linked"
sys.path.insert(0, str(BASE / "dataset_exp"))
from lib_cleans import storm_events  # noqa: E402
from run3_link_records import MATCH_SOURCES, match_source_events  # noqa: E402

# Words in EM-DAT's event names that are not a storm's name
GENERIC = {"CYCLONE", "CYCLONIC", "TYPHOON", "HURRICANE", "TROPICAL", "STORM", "DEPRESSION", "SEVERE", "SUPER", "VERY", "DEEP", "SUPERTYPHOON",
           "AND", "THE", "OF", "SS", "TS", "TC", "TD", "CAT", "CATEGORY", "REMNANTS", "POST", "EX", "FORMER", "WINTER", "STORMS", "RAIN", "RAINS",
           "FLOOD", "FLOODS", "FLOODING", "LANDSLIDE", "LANDSLIDES", "SURGE", "WIND", "WINDS", "HEAVY", "MONSOON", "LOW", "PRESSURE"}
SAFFIR = [(0, "below 64 kt"), (64, "cat 1-2"), (96, "cat 3"), (113, "cat 4-5")]
# Two names are the same storm's at this similarity or more (difflib ratio: Lilli / LILI 0.89, Beatrix / BEATRIZ 0.86)
NAME_SIMILARITY = 0.8


def read(name: str) -> pd.DataFrame:
    """A linked CSV of ours, only empty cells missing (a literal NA, the North Atlantic, stays text)."""
    return pd.read_csv(LINKED / name, keep_default_na=False, na_values=[""], low_memory=False)


def summary(check: str, group: str, values: pd.Series, **extra) -> dict:
    """One summary row: count, median and interquartile range of a ratio or a share."""
    return {"check": check, "group": group, "n": int(values.notna().sum()), "median": values.median(), "q25": values.quantile(0.25),
            "q75": values.quantile(0.75), "mean": values.mean(), **extra}


def era5_against_tracks(gee: pd.DataFrame, tracks: pd.DataFrame) -> list:
    """6a: ERA5's strongest 10 m wind over the area against IBTrACS's 1-minute wind at the closest point, linked rows read by both."""
    both = tracks[tracks["track_sid"].notna()].merge(gee[["disasterno", "iso3", "era5_wind_max_ms"]], on=["disasterno", "iso3"])
    both = both[both["era5_wind_max_ms"].notna() & (both["track_closest_wind_kt"] > 0)]
    both["ratio"] = both["era5_wind_max_ms"] * 1.944 / both["track_closest_wind_kt"]
    both["class"] = pd.cut(both["track_closest_wind_kt"], [k for k, _ in SAFFIR] + [np.inf], labels=[n for _, n in SAFFIR], right=False)
    rows = [summary("6a era5 / track wind", "all", both["ratio"], spearman=both["era5_wind_max_ms"].corr(both["track_closest_wind_kt"], method="spearman"))]
    rows += [summary("6a era5 / track wind", f"basin {b}", g["ratio"]) for b, g in both.groupby("track_basin")]
    rows += [summary("6a era5 / track wind", f"track {c}", g["ratio"]) for c, g in both.groupby("class", observed=True)]
    return rows



def names(text) -> list:
    """The words of a name in capitals, letters only (Kai-Tak -> KAITAK, Chata'an -> CHATAAN), without the generic ones, and each two
    neighbouring words joined (Son Tinh -> SONTINH); empty for a missing name."""
    words = [] if pd.isna(text) else [re.sub("[^A-Z]", "", w) for w in re.split(r"[\s/&(),]+", str(text).upper())]
    words = [w for w in words if len(w) > 1 and w not in GENERIC]
    return words + [a + b for a, b in zip(words, words[1:])]


def track_names(tracks: pd.DataFrame, emdat: pd.DataFrame) -> tuple:
    """6e: every linked row's IBTrACS name against EM-DAT's event name: confirmed (a name similar at NAME_SIMILARITY or more, for spellings
    such as Lilli / LILI), disagreement, or no name on one side."""
    rows = tracks[tracks["track_sid"].notna()].merge(emdat[["disasterno", "iso3", "event_name"]], on=["disasterno", "iso3"], how="left")
    track = rows["track_name"].map(lambda n: [] if pd.isna(n) or str(n).upper() in {"UNNAMED", "NOT_NAMED", "NOT NAMED"}
                                   else [re.sub("[^A-Z]", "", x) for x in str(n).upper().split(":")])
    emdat_words = rows["event_name"].map(names)
    letters = rows["event_name"].map(lambda n: "" if pd.isna(n) else re.sub("[^A-Z]", "", str(n).upper()))
    same = [any(a in s or SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY for a in t for b in e) for t, e, s in zip(track, emdat_words, letters)]
    rows["name_check"] = np.select([track.map(len).eq(0), emdat_words.map(len).eq(0), same], ["track_unnamed", "emdat_unnamed", "confirmed"], "disagreement")
    out = []
    for group, g in [("all", rows), *[(f"level {k}", g) for k, g in rows.groupby("track_level")]]:
        named = g[g["name_check"].isin(["confirmed", "disagreement"])]
        out.append({"check": "6e track names", "group": group, "n": len(g), "n_both_named": len(named),
                    "share": float(named["name_check"].eq("confirmed").mean()) if len(named) else np.nan,
                    **g["name_check"].value_counts().to_dict()})
    return out, rows[["disasterno", "iso3", "event_name", "track_sid", "track_name", "track_level", "track_confidence", "name_check"]]


def glide_against_dates(events: pd.DataFrame) -> tuple:
    """6f: source rows matched by GLIDE, matched again by dates only; the share landing on the same storm, per source."""
    out, detail = [], []
    for source, window in MATCH_SOURCES.items():
        src = pd.read_csv(BASE / "datasets" / "clean" / f"{source}.csv", low_memory=False)
        by_glide = {r.name: dn for dn, found in match_source_events(events, src, window).items() for r, m in found if m == "glide"}
        by_date = {r.name: dn for dn, found in match_source_events(events.assign(glide=np.nan), src.assign(glide=np.nan), window).items() for r, _ in found}
        d = pd.DataFrame({"source": source, "source_row": list(by_glide), "glide_disasterno": list(by_glide.values())})
        d["date_disasterno"] = d["source_row"].map(by_date)
        d["result"] = np.select([d["date_disasterno"].isna(), d["date_disasterno"] == d["glide_disasterno"]], ["no_date_match", "same"], "different")
        counts = d["result"].value_counts()
        out.append({"check": "6f glide vs dates", "group": source, "n": len(d), "share": float(counts.get("same", 0) / len(d)) if len(d) else np.nan,
                    **counts.to_dict()})
        detail.append(d)
    return out, pd.concat(detail)


def main():
    events = storm_events()
    emdat = pd.read_csv(BASE / "datasets" / "clean" / "emdat.csv", low_memory=False)
    rows, skipped = [], []
    if (LINKED / "track_features.csv").exists():
        tracks = read("track_features.csv")
        named, detail = track_names(tracks, emdat)
        rows += named
        detail.to_csv(OUT / "validation_track_names.csv", index=False)
    else:
        skipped.append("6a, 6e: track_features.csv (run3_link_records.py)")
    if (LINKED / "exposure_features.csv").exists():
        gee = read("event_points.csv").merge(read("exposure_features.csv"), on=["disasterno", "iso3"]).merge(read("weather_features.csv"), on=["disasterno", "iso3"])
        print(f"[VALIDATION] Earth Engine tables: {gee['gee_date'].notna().sum():,} of {len(gee):,} rows read so far")
        rows += era5_against_tracks(gee, tracks) if (LINKED / "track_features.csv").exists() else []
    else:
        skipped.append("6a: the Earth Engine tables (run4_extract_measures.py)")
    glide, detail = glide_against_dates(events)
    rows += glide
    detail.to_csv(OUT / "validation_glide_dates.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "validation_summary.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"[VALIDATION] skipped: {skipped}" if skipped else "[VALIDATION] every check ran")


if __name__ == "__main__":
    main()
