"""Build the outputs from datasets/clean/ and datasets/linked/: schema.yaml, the SQLite database, the RDF knowledge graph (with its
checks) and the public release without EM-DAT (datasets/outputs/release/). Run after run4: python run5_build_outputs.py"""

import gzip
import sqlite3
import sys
from urllib.parse import quote

import pandas as pd
import yaml
from pathlib import Path
from rdflib import RDF, XSD, Graph, Literal, URIRef

sys.path.append(str(Path(__file__).parent.parent / "dataset_download"))
from lib_cleans import SCHEMA  # noqa: E402

BASE = Path(__file__).parent.parent / "datasets"
# The extracts of run3 and run4 that the release needs
RELEASE_FILES = ["exposure_features.csv", "weather_features.csv", "convection_features.csv", "wave_features.csv",
                                     "snow_features.csv", "track_features.csv", "track_points.csv", "gtsm_features.csv", "dust_features.csv"]
RELEASE, WIKIDATA = BASE / "outputs" / "release", BASE / "raw" / "wikidata" / "countries.csv"
# Every column of every table, its kind, unit and meaning (every _gap and _status column too), and each status and gap value (table *)
DICTIONARY = Path(__file__).parent.parent / "docs" / "data_dictionary.csv"
SUPPLEMENT = Path(__file__).parent.parent.parent / "manuscript_dataset_storms" / "latex_big_earth_data" / "supplement.pdf"
SQL_TYPES = {"Int64": "INTEGER", "boolean": "INTEGER", "Float64": "REAL"}


def read_csv(path: Path) -> pd.DataFrame:
    """A CSV of ours: only an empty cell is missing (a literal NA, the North Atlantic basin, stays text); whole numbers with gaps stay whole
    (Int64, not float), so the database, the graph and the release type them as integers."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False).convert_dtypes()


def write_schema():
    """datasets/schema.yaml: every CSV's primary and foreign keys (lib_cleans.SCHEMA), the extracts included."""
    schema = {name: {"csv": csv, "key": key, "references": [{"columns": cols, "table": t} for cols, t in refs]} for name, (csv, key, refs) in SCHEMA.items()}
    (BASE / "schema.yaml").write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    print(f"[SCHEMA] {len(schema)} tables -> {BASE / 'schema.yaml'}")


def load(folder: Path) -> tuple:
    """A folder's schema.yaml and its tables whose CSV exists (an extract not run yet is left out, named)."""
    schema = yaml.safe_load((folder / "schema.yaml").read_text(encoding="utf-8"))
    missing = [name for name, t in schema.items() if not (folder / t["csv"]).exists()]
    print(f"[TABLES] {folder.name}: not run yet, left out: {missing}" if missing else f"[TABLES] {folder.name}: every table present")
    schema = {name: t for name, t in schema.items() if name not in missing}
    return schema, {name: read_csv(folder / t["csv"]) for name, t in schema.items()}


def check_dictionary(frames: dict):
    """Every column of every table described in the data dictionary, and nothing described that no table has."""
    d = pd.read_csv(DICTIONARY)
    described = set(zip(d["table"], d["column"]))
    missing = [(n, c) for n, df in frames.items() for c in df.columns if (n, c) not in described]
    extra = [(t, c) for t, c in described if t in frames and c not in frames[t].columns]
    assert not missing and not extra, f"data dictionary: undescribed {missing}, described but absent {extra}"
    print(f"[DICTIONARY] {len(d)} rows describe every column of {len(frames)} tables")


def check_keys(schema: dict, frames: dict):
    """Every primary key unique and complete, every foreign key pointing at a row of its table (an empty foreign key, a source row matched
    to no storm, points nowhere by design)."""
    for name, t in schema.items():
        df = frames[name]
        if t["key"]:
            assert df[t["key"]].notna().all().all() and not df.duplicated(t["key"]).any(), f"{name}: primary key {t['key']} empty or repeated"
        for r in t["references"]:
            if r["table"] in frames:
                target = pd.MultiIndex.from_frame(frames[r["table"]][schema[r["table"]]["key"]].astype(str))
                broken = ~pd.MultiIndex.from_frame(df[r["columns"]].dropna().astype(str)).isin(target)
                assert not broken.any(), f"{name}: {broken.sum():,} rows whose {r['columns']} point nowhere in {r['table']}"
    print(f"[KEYS] {len(schema)} tables: primary keys unique, foreign keys point to rows")


def build_database(schema: dict, frames: dict):
    """Every table into datasets/outputs/storms.sqlite with its primary and foreign keys, then SQLite's own foreign-key check."""
    path = BASE / "outputs" / "storms.sqlite"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    for name, t in schema.items():
        df = frames[name]
        cols = [f'"{c}" {SQL_TYPES.get(str(df[c].dtype), "TEXT")}' for c in df.columns]
        cols += [f"PRIMARY KEY ({', '.join(t['key'])})"] if t["key"] else []
        cols += [f"FOREIGN KEY ({', '.join(r['columns'])}) REFERENCES {r['table']} ({', '.join(schema[r['table']]['key'])})" for r in t["references"]]
        con.execute(f'CREATE TABLE "{name}" ({", ".join(cols)})')
        con.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(df.columns))})', df.astype(object).where(df.notna(), None).itertuples(index=False, name=None))
    con.commit()
    assert not con.execute("PRAGMA foreign_key_check").fetchall(), "SQLite found foreign keys that point nowhere"
    con.close()
    print(f"[SQLITE] {len(schema)} tables -> {path}")


# === The knowledge graph ===
# Names: rows <NS{table}/{key}>, things <NS{kind}/{id}> (event, country, track), vocabulary <NS vocab/...>; NS is to be registered at
# w3id.org. A table's class is CLASSES[table]; the things its key columns name are linked by LINKS; FACTS are the relations between things.
NS = "https://w3id.org/storms/"
V, GEO = NS + "vocab/", "http://www.opengis.net/ont/geosparql#"
CLASSES = {"emdat": "DisasterRecord", "geodisasters": "AffectedUnit", "track_points": "TrackPoint", "idmc": "DisplacementRecord", "ifrc": "AidAppeal",
           "hdi": "HumanDevelopment", "wdi": "Economy", "wgi": "Governance", "event_points": "EventLocation",
           "exposure_features": "EventExposure", "weather_features": "EventWeather",
           "convection_features": "EventConvection", "wave_features": "EventWaves", "snow_features": "EventSnow", "dust_features": "EventDust", "track_features": "EventTrackIntensity",
           "gtsm_features": "EventStormSurge", "cascades": "EventCascade"}
# Key column -> (thing it names, predicate from the row to it)
LINKS = {"disasterno": ("event", "event"), "from_disasterno": ("event", "fromEvent"), "to_disasterno": ("event", "toEvent"), "iso3": ("country", "country"),
         "sid": ("track", "track"), "track_sid": ("track", "track")}
KINDS = {"event": "Event", "country": "Country", "track": "Track"}
# The main classes, grouped by the components of disaster risk, and the classes under each (rdfs:subClassOf)
MAIN_CLASSES = {
    "Event": (["Storm"], "A disaster event of EM-DAT; its storms are also of class Storm."),
    "Entity": (["Country", "Track"], "A thing the records name: a country (owl:sameAs Wikidata), an IBTrACS track."),
    "SourceRecord": (["DisasterRecord", "AffectedUnit", "TrackPoint", "DisplacementRecord", "AidAppeal"], "A row of a source, as the source gives it."),
    "Location": (["EventLocation"], "Where a storm struck a country: its coordinate and its area."),
    "Hazard": (["EventTrackIntensity", "EventWeather", "EventConvection", "EventWaves", "EventStormSurge", "EventSnow", "EventDust"],
               "The storm's physical hazard over its area and days."),
    "Exposure": (["EventExposure"], "The population and assets in the storm's area."),
    "Vulnerability": (["HumanDevelopment", "Economy", "Governance"], "The national development, economy and governance in the storm's year."),
    "Cascade": (["EventCascade"], "An influence of one disaster on a later one in the same area."),
}
# (table, subject column, relation, object column): facts between things
FACTS = [("emdat", "disasterno", "hitCountry", "iso3"), ("event_points", "disasterno", "hitCountry", "iso3"),
         ("cascades", "from_disasterno", "influences", "to_disasterno"), ("track_features", "disasterno", "matchedTrack", "track_sid")]
POINTS = [("lat", "lon"), ("gtsm_lat", "gtsm_lon"), ("track_closest_lat", "track_closest_lon")]
# A column's unit by its name's ending, as a QUDT unit (US dollars, nominal, as a QUDT currency)
QUDT = "http://qudt.org/vocab/unit/"
UNITS = {"_ms": "M-PER-SEC", "_kt": "KN", "_mb": "HectoPA", "_hpa": "HectoPA", "_mm": "MilliM", "_mmh": "MilliM-PER-HR", "_m": "M", "_km": "KiloM",
         "_km2": "KiloM2", "_nmi": "MI_N", "_jkg": "J-PER-KiloGM", "_c": "DEG_C", "_k": "K", "_deg": "DEG", "_usd": "http://qudt.org/vocab/currency/USD",
         "_pct": "PERCENT", "_share": "FRACTION", "_ugm3": "MicroGM-PER-M3"}
LABEL, TYPE, SAME = "<http://www.w3.org/2000/01/rdf-schema#label>", RDF.type.n3(), "<http://www.w3.org/2002/07/owl#sameAs>"


assert sorted(c for subs, _ in MAIN_CLASSES.values() for c in subs) == sorted({*CLASSES.values(), *KINDS.values(), "Storm"} - {"Event"})


def iri(*parts) -> str:
    """An IRI under NS from path parts, each escaped."""
    return URIRef(NS + "/".join(quote(str(int(p) if isinstance(p, float) and p.is_integer() else p), safe="") for p in parts)).n3()


def lit(column: str, v) -> str:
    """A value as an RDF term: a date or time typed, anything else typed by Python (a status is its text)."""
    if column.endswith("_date") or column in ("time", "track_closest_time"):
        s = str(v).replace(" ", "T")
        return Literal(s if len(s) != 16 else s + ":00", datatype=XSD.date if len(s) == 10 else XSD.dateTime).n3()
    return Literal(v).n3()


def row_key(name: str, t: dict, rec: dict, i: int) -> list:
    """A row's key values: its primary key, else the columns that make it unique (an IBTrACS point: storm and time), else its position."""
    return [rec[k] for k in t["key"]] if t["key"] else [rec["sid"], rec["time"]] if name == "track_points" else [i]


def graph_lines(schema: dict, frames: dict):
    """Every triple, as N-Triples lines (valid Turtle): rows typed by their table's class, their values as properties, links to the things
    their keys name, foreign keys as edges to the rows they name; then the things, typed, labelled and linked (FACTS, Wikidata)."""
    things = {}
    for name, t in schema.items():
        df, cls = frames[name], f"<{V}{CLASSES[name]}>"
        props = {c: f"<{V}{name}/{c}>" for c in df.columns}
        for i, row in enumerate(df.astype(object).where(df.notna(), None).itertuples(index=False)):
            rec = dict(zip(df.columns, row))
            s = iri(name, *row_key(name, t, rec, i))
            yield f"{s} {TYPE} {cls} ."
            yield f"{s} {LABEL} {Literal(CLASSES[name] + ' ' + ' '.join(map(str, row_key(name, t, rec, i)))).n3()} ."
            for c, v in rec.items():
                if v is not None:
                    yield f"{s} {props[c]} {lit(c, v)} ."
                    if c in LINKS:
                        kind, pred = LINKS[c]
                        things.setdefault((kind, v), None)
                        yield f"{s} <{V}{pred}> {iri(kind, v)} ."
            for a, b in POINTS:
                if rec.get(a) is not None and rec.get(b) is not None:
                    geom = s[:-1] + "/geometry>"
                    yield f"{s} <{GEO}hasGeometry> {geom} ."
                    yield f'{geom} <{GEO}asWKT> "POINT({rec[b]} {rec[a]})"^^<{GEO}wktLiteral> .'
            for r in t["references"]:
                if r["table"] in frames and all(rec[c] is not None for c in r["columns"]):
                    pred = "about" if r["columns"][0] == "disasterno" else r["columns"][0].replace("_disasterno", "Record")
                    yield f"{s} <{V}{pred}> {iri(r['table'], *[rec[c] for c in r['columns']])} ."
    facts = {(iri(LINKS[a][0], x), rel, iri(LINKS[b][0], y)) for name, a, rel, b in FACTS if name in frames
             for x, y in frames[name][[a, b]].dropna().itertuples(index=False)}
    yield from (f"{x} <{V}{rel}> {y} ." for x, rel, y in sorted(facts))
    # Labels of things: from EM-DAT (in the full graph) and IBTrACS; storms are the events of event_points
    labels = {}
    if "emdat" in frames:
        e = frames["emdat"]
        labels.update({("event", d): f"{n if isinstance(n, str) else s} ({d})" for d, n, s in e[["disasterno", "event_name", "disaster_subtype"]].itertuples(index=False)})
        labels.update({("country", c): n for c, n in e[["iso3", "country_name"]].drop_duplicates("iso3").itertuples(index=False)})
    if "track_points" in frames:
        labels.update({("track", s): f"{n} {y}" for s, n, y in frames["track_points"][["sid", "name", "year"]].drop_duplicates("sid").itertuples(index=False)})
    storms = set(frames["event_points"]["disasterno"]) if "event_points" in frames else set()
    wikidata = pd.read_csv(WIKIDATA).set_index("iso3")["wikidata"].to_dict()
    for kind, v in things:
        s = iri(kind, v)
        yield f"{s} {TYPE} <{V}{KINDS[kind]}> ."
        yield f"{s} {LABEL} {Literal(str(labels.get((kind, v), v))).n3()} ."
        if kind == "event" and v in storms:
            yield f"{s} {TYPE} <{V}Storm> ."
        if kind == "country" and v in wikidata:
            yield f"{s} {SAME} <{wikidata[v]}> ."


def unit(column: str):
    """A column's QUDT unit IRI from its name's ending, if any."""
    u = next((u for end, u in sorted(UNITS.items(), key=lambda x: -len(x[0])) if column.endswith(end)), None)
    return u if u is None or u.startswith("http") else QUDT + u


def vocabulary(schema: dict, frames: dict) -> str:
    """The vocabulary as Turtle: every class with its main class, every property with its label, its table and its QUDT unit."""
    lines = [f"@prefix v: <{V}> .", "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", "@prefix qudt: <http://qudt.org/schema/qudt/> ."]
    lines += [f'v:{c} a rdfs:Class ; rdfs:label "{c}" ; rdfs:comment "One row of the table {t}." .' for t, c in CLASSES.items()]
    lines += [f'v:{k} a rdfs:Class ; rdfs:label "{k}" .' for k in [*KINDS.values(), "Storm"] if k not in CLASSES.values() and k not in MAIN_CLASSES]
    lines += [f'v:{m} a rdfs:Class ; rdfs:label "{m}" ; rdfs:comment "{about}" .' for m, (_, about) in MAIN_CLASSES.items()]
    lines += [f'v:{c} rdfs:subClassOf v:{m} .' for m, (subs, _) in MAIN_CLASSES.items() for c in subs]
    lines += [f'v:{p} a rdf:Property ; rdfs:label "{p}" .'.replace("rdf:Property", f"<{RDF}Property>") for p in
              sorted({p for _, p in LINKS.values()} | {r for _, _, r, _ in FACTS} | {"about", "fromRecord", "toRecord"})]
    for name in schema:
        for c in frames[name].columns:
            u = unit(c)
            lines.append(f'<{V}{name}/{c}> a <{RDF}Property> ; rdfs:label "{c}" ; rdfs:domain v:{CLASSES[name]}' + (f" ; qudt:unit <{u}>" if u else "") + " .")
    return "\n".join(lines) + "\n"


def shapes(schema: dict, frames: dict) -> str:
    """SHACL shapes from schema.yaml: each table's class, its key columns present once, every column at most once, numbers typed, links
    pointing at things of the right kind, foreign keys at rows of the right class."""
    xsd = {"Int64": "integer", "Float64": "double", "boolean": "boolean"}
    lines = ["@prefix sh: <http://www.w3.org/ns/shacl#> .", f"@prefix v: <{V}> .", "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ."]
    for name, t in schema.items():
        df, props = frames[name], []
        for c in df.columns:
            p = f"sh:path <{V}{name}/{c}> ; sh:maxCount 1"
            p += " ; sh:minCount 1" if c in t["key"] else ""
            p += f" ; sh:datatype xsd:{xsd[str(df[c].dtype)]}" if str(df[c].dtype) in xsd else ""
            props.append(f"[ {p} ]")
        props += [f"[ sh:path v:{LINKS[c][1]} ; sh:class v:{KINDS[LINKS[c][0]]} ]" for c in df.columns if c in LINKS]
        lines.append(f"<{NS}shapes/{name}> a sh:NodeShape ; sh:targetClass v:{CLASSES[name]} ; sh:property " + " , ".join(props) + " .")
    return "\n".join(lines) + "\n"


def void(path: Path, counts: dict, public: bool) -> str:
    """The dataset's VoID description: its size, its classes and properties with their counts, its dump, its vocabulary, its Wikidata links;
    its provenance in PROV: the build that generated it."""
    d = f"<{NS}dataset/{'public' if public else 'full'}>"
    lines = ["@prefix void: <http://rdfs.org/ns/void#> .", "@prefix dct: <http://purl.org/dc/terms/> .", "@prefix prov: <http://www.w3.org/ns/prov#> .",
             f'{d} a void:Dataset ; dct:title "Storm disaster knowledge graph{" (keys only, without EM-DAT)" if public else ""}" ; '
             f'void:triples {counts["triples"]} ; void:dataDump <{path.name}> ; void:vocabulary <{V}> ; void:uriSpace "{NS}" .']
    lines += [f'{d} void:classPartition [ void:class <{c}> ; void:entities {n} ] .' for c, n in sorted(counts["classes"].items())]
    lines += [f'{d} void:propertyPartition [ void:property <{p}> ; void:triples {n} ] .' for p, n in sorted(counts["properties"].items())]
    lines += [f'{d} void:subset [ a void:Linkset ; void:linkPredicate <http://www.w3.org/2002/07/owl#sameAs> ; void:target <http://www.wikidata.org/> ; '
              f'void:triples {counts["properties"].get("http://www.w3.org/2002/07/owl#sameAs", 0)} ] .']
    build = f"<{NS}build/{'public' if public else 'full'}>"
    lines.append(f"{d} prov:wasGeneratedBy {build} . {build} a prov:Activity ; dct:description \"run1 to run5 of the dataset code\" .")
    return "\n".join(lines) + "\n"


def build_graph(schema: dict, frames: dict, path: Path, public: bool = False):
    """The graph -> path (gzipped Turtle, line by line so memory stays flat), with its VoID description, vocabulary and SHACL shapes beside it."""
    counts = {"triples": 0, "classes": {}, "properties": {}}
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for line in graph_lines(schema, frames):
            out.write(line + "\n")
            s, p, o = line.split(" ", 2)
            counts["triples"] += 1
            counts["properties"][p[1:-1]] = counts["properties"].get(p[1:-1], 0) + 1
            if p == TYPE:
                counts["classes"][o[1:-3]] = counts["classes"].get(o[1:-3], 0) + 1
    # The graph and the tables match: one node of a table's class per row of its CSV
    rows = {CLASSES[name]: len(frames[name]) for name in schema}
    nodes = {c: counts["classes"].get(V + c, 0) for c in rows}
    assert nodes == rows, {c: (nodes[c], rows[c]) for c in rows if nodes[c] != rows[c]}
    print(f"[GRAPH] one node per row in each of {len(rows)} tables")
    stem = path.name.split(".")[0]
    (path.parent / f"{stem}_void.ttl").write_text(void(path, counts, public), encoding="utf-8")
    (path.parent / f"{stem}_vocab.ttl").write_text(vocabulary(schema, frames), encoding="utf-8")
    (path.parent / f"{stem}_shapes.ttl").write_text(shapes(schema, frames), encoding="utf-8")
    print(f"[GRAPH] {counts['triples']:,} triples, {sum(counts['classes'].values()):,} typed nodes -> {path}")


def check_graph(schema: dict, frames: dict, path: Path, n_storms: int = 150):
    """SHACL on the graph of a random sample of storms (every row naming them, the rows their keys point at, their tracks' points and their
    countries' years), as the whole graph does not fit rdflib's memory; then each competency question on the whole graph (Oxigraph)."""
    from pyshacl import validate
    sample = set(frames["event_points"]["disasterno"].drop_duplicates().sample(n_storms, random_state=7))
    sample |= set(frames["cascades"].loc[frames["cascades"]["from_disasterno"].isin(sample) | frames["cascades"]["to_disasterno"].isin(sample),
                                          ["from_disasterno", "to_disasterno"]].stack())
    countries = set(frames["emdat"].loc[frames["emdat"]["disasterno"].isin(sample), "iso3"])
    tracks = set(frames["track_features"].loc[frames["track_features"]["disasterno"].isin(sample), "track_sid"].dropna())
    # Each table's rows of the sample: by the storm they name, else the track, else the country; a table naming none is whole
    by = [("disasterno", sample), ("from_disasterno", sample), ("sid", tracks), ("iso3", countries)]
    small = {name: next((df[df[c].isin(v)] for c, v in by if c in df), df) for name, df in frames.items()}
    data = Graph().parse(data="\n".join(graph_lines(schema, small)), format="nt")
    ok, _, report = validate(data, shacl_graph=Graph().parse(data=shapes(schema, frames), format="turtle"))
    assert ok, report[:3000]
    print(f"[SHACL] {len(data):,} triples of {n_storms} storms conform to the shapes")
    run_queries(path)


def run_queries(path: Path):
    """Each competency question on a whole graph (Oxigraph), with its first answer."""
    from pyoxigraph import RdfFormat, Store
    store = Store()
    with gzip.open(path, "rb") as f:
        store.bulk_load(input=f, format=RdfFormat.N_TRIPLES)
    for question, query in QUERIES.items():
        rows = list(store.query(PREFIXES + query))
        print(f"[QUERY] {question}: {len(rows)} rows; first: {[str(x) for x in rows[0]] if rows else '-'}")


PREFIXES = f"PREFIX v: <{V}> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX owl: <http://www.w3.org/2002/07/owl#> "
# Competency questions: what the graph must answer, the full graph and the deposited one alike (a storm found by its IBTrACS track, as the
# deposited graph has no EM-DAT names); deposited as query1.rq to query6.rq
QUERIES = {
    "countries Typhoon Haiyan hit, with their Wikidata items": """SELECT DISTINCT ?country ?wikidata WHERE { ?t rdfs:label "HAIYAN 2013" .
        ?x a v:EventTrackIntensity ; v:track ?t ; v:event ?e . ?e v:hitCountry ?c . ?c rdfs:label ?country OPTIONAL { ?c owl:sameAs ?wikidata } }""",
    "storms that influenced a later disaster in the same area, with the lag and overlap": """SELECT ?storm ?later ?days ?overlap WHERE {
        ?k a v:EventCascade ; v:fromEvent ?s ; v:toEvent ?f ; <%scascades/days> ?days ; <%scascades/overlap_pct> ?overlap . ?s a v:Storm ; rdfs:label ?storm .
        ?f rdfs:label ?later } ORDER BY DESC(?overlap) LIMIT 1000""" % (V, V),
    "strongest matched cyclones where they struck, and their tracks": """SELECT ?storm ?track ?peak WHERE { ?x a v:EventTrackIntensity ; v:event ?e ; v:track ?t ;
        <%strack_features/track_closest_wind_kt> ?peak . ?e rdfs:label ?storm . ?t rdfs:label ?track } ORDER BY DESC(?peak) LIMIT 5""" % V,
    "a storm's IDMC displacement rows in each country": """SELECT ?storm ?country ?displaced WHERE { ?d a v:DisplacementRecord ; v:event ?e ; v:country ?c ;
        <%sidmc/displacement> ?displaced . ?e rdfs:label ?storm . ?c rdfs:label ?country } ORDER BY DESC(?displaced) LIMIT 5""" % V,
    "the HDI of each country Typhoon Haiyan hit, in the storm's year": """SELECT DISTINCT ?country ?hdi WHERE { ?t rdfs:label "HAIYAN 2013" .
        ?x a v:EventTrackIntensity ; v:track ?t ; v:country ?c . ?p a v:TrackPoint ; v:track ?t ; <%strack_points/year> ?y .
        ?h a v:HumanDevelopment ; v:country ?c ; <%shdi/year> ?y ; <%shdi/hdi> ?hdi . ?c rdfs:label ?country }""" % (V, V, V),
    "why values are empty: rows per status": """SELECT ?status (COUNT(*) AS ?n) WHERE { ?row ?p ?status FILTER(STRENDS(STR(?p), "_status")) } GROUP BY ?status ORDER BY DESC(?n)""",
}


def build_release(schema: dict, frames: dict):
    """The public tables: every table but emdat, without its foreign keys to emdat (EM-DAT's own points blanked before);
    no other table holds an EM-DAT value. With the data dictionary and the supplement (how every column was read).
    Files of an earlier release are removed first."""
    missing = [name for name in RELEASE_FILES if not (BASE / "linked" / name).exists()]
    assert not missing, f"[RELEASE] not run yet: {missing} (run3_link_records.py, run4_extract_measures.py); the release needs every extract"
    RELEASE.mkdir(parents=True, exist_ok=True)
    for old in RELEASE.iterdir():
        old.unlink()
    public = {name: {**t, "csv": Path(t["csv"]).name, "references": [r for r in t["references"] if r["table"] != "emdat"]}
              for name, t in schema.items() if name != "emdat"}
    (RELEASE / DICTIONARY.name).write_bytes(DICTIONARY.read_bytes())
    (RELEASE / "supplementary_information.pdf").write_bytes(SUPPLEMENT.read_bytes())
    for name in public:
        frames[name].to_csv(RELEASE / public[name]["csv"], index=False)
    (RELEASE / "schema.yaml").write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")
    for i, (question, query) in enumerate(QUERIES.items(), 1):
        (RELEASE / f"query{i}.rq").write_text(f"# {question}\n" + PREFIXES.replace(" PREFIX", "\nPREFIX") + "\n" + query + "\n", encoding="utf-8")
    print(f"[RELEASE] {len(public)} tables without EM-DAT -> {RELEASE}")


if __name__ == "__main__":
    write_schema()
    schema, frames = load(BASE)
    check_keys(schema, frames)
    check_dictionary(frames)
    build_database(schema, frames)
    graphed = {n: t for n, t in schema.items() if n in CLASSES}
    build_graph(graphed, frames, BASE / "outputs" / "storms.ttl.gz")
    check_graph(graphed, frames, BASE / "outputs" / "storms.ttl.gz")
    # The release and the public graph: every table without EM-DAT (its table, its foreign keys, its own points: the row stays, coord_source
    # emdat says where the point is)
    p = frames["event_points"]
    p.loc[p["coord_source"] == "emdat", ["lat", "lon"]] = None
    build_release(schema, frames)
    public = {n: {**t, "references": [r for r in t["references"] if r["table"] != "emdat"]} for n, t in graphed.items() if n != "emdat"}
    build_graph(public, {n: frames[n] for n in public}, RELEASE / "storms_public.ttl.gz", public=True)
    run_queries(RELEASE / "storms_public.ttl.gz")
