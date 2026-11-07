import argparse
import glob
import webbrowser
from pathlib import Path

import geopandas as gpd
import pandas as pd
from src.bias.representation import load_config
from src.features.sensor_match import load_sensor_locations

REVIEW = "data/processed/features/melbourne/sensor_match_review.csv"
EDGES = "data/interim/osm/melbourne/graph/edges.parquet"
SENSOR_MAP = "data/**/melbourne/**/sensor_segment_map.parquet"
EDGE_ID = "segment_id"
OUT_DIR = Path("review_maps")
SHOW = ["rank", "segment_id", "distance_m", "highway", "name", "bearing_deg",
        "footpath_class", "sidewalk_state", "score", "chosen_segment_id"]


def load_review():
    df = pd.read_csv(REVIEW)
    is_indoor = df.groupby("location_id")["review_reasons"].apply(
        lambda s: s.astype(str).str.contains("indoor", case=False).any()
    )
    ids = sorted(is_indoor[~is_indoor].index)
    return df[df["location_id"].isin(ids)], ids


def find_sensor_map():
    files = glob.glob(SENSOR_MAP, recursive=True)
    if not files:
        raise SystemExit("could not find sensor_segment_map.parquet")
    return files[0]


def load_sensors():
    sensors = load_sensor_locations(load_config()).to_crs(4326)
    sensors["key"] = sensors["location_id"].astype(str)
    return sensors.drop_duplicates("key").set_index("key")


def inspect(args):
    print("review columns:", list(pd.read_csv(REVIEW, nrows=1).columns))
    print("edges columns:", list(gpd.read_parquet(EDGES).columns))
    path = find_sensor_map()
    print("sensor map:", path)
    print("sensor map columns:", list(pd.read_parquet(path).columns))


def list_sensors(args):
    df, ids = load_review()
    print(len(ids), "outdoor sensors")
    if args.who == "shubham":
        mine = ids[:26]
    else:
        mine = ids[26:]
    print(args.who, "has", len(mine), "sensors")
    for lid in mine:
        reasons = sorted(set(df.loc[df.location_id == lid, "review_reasons"].astype(str)))
        print(lid, ":", ", ".join(reasons))


def axis_ok(bearing, d1, d2):
    dirs = {str(d1).strip().lower(), str(d2).strip().lower()}
    b = bearing % 180
    if dirs & {"north", "south"}:
        return min(b, 180 - b) <= 45
    if dirs & {"east", "west"}:
        return abs(b - 90) <= 45
    return None


def is_footpath(value):
    return str(value).lower() == "true"


def pick_segment(cands, d1, d2):
    cands = cands.sort_values("distance_m")
    checks = cands["bearing_deg"].apply(lambda b: axis_ok(b, d1, d2))
    if checks.isna().all():
        return cands.iloc[0], ["no direction labels"]
    ok = cands[checks == True]
    if ok.empty:
        return cands.iloc[0], ["no candidate matches direction"]
    foot = ok[ok["footpath_class"].apply(is_footpath)]
    pick = ok.iloc[0]
    if not foot.empty and foot["distance_m"].iloc[0] <= pick["distance_m"] + 5:
        pick = foot.iloc[0]
    notes = []
    if not is_footpath(pick["footpath_class"]):
        notes.append("not a footpath")
    pool = foot if is_footpath(pick["footpath_class"]) else ok
    others = pool[pool["segment_id"] != pick["segment_id"]]
    if not others.empty and others["distance_m"].iloc[0] - pick["distance_m"] < 2:
        notes.append("another matching path within 2 m")
    return pick, notes


def suggest(args):
    df, _ = load_review()
    sensors = load_sensors()
    yaml_lines = []
    for lid in args.location_ids:
        cands = df[df.location_id == lid]
        if cands.empty:
            print(lid, "not in review list")
            continue
        s = sensors.loc[str(lid)] if str(lid) in sensors.index else None
        d1 = s.get("direction_1") if s is not None else None
        d2 = s.get("direction_2") if s is not None else None
        desc = s.get("sensor_description", "") if s is not None else ""
        source = s.get("location_source", "") if s is not None else ""
        pick, notes = pick_segment(cands, d1, d2)
        if str(source).startswith("wayback"):
            notes.append("2022 location, check street")
        status = "look" if notes else "easy"
        print(f"{lid:>4}  {status:<4}  {pick['segment_id']:<14} {str(pick['highway']):<12} "
              f"{pick['distance_m']:5.1f} m  {pick['bearing_deg']:4.0f}  {d1}/{d2}  {desc}")
        for n in notes:
            print("            ", n)
        yaml_lines.append(f"  {lid}: {pick['segment_id']}")
    print()
    print("\n".join(yaml_lines))


def make_map(lid, df, edges, sensors):
    cands = df[df.location_id == lid]
    if cands.empty:
        print(lid, "not in review list")
        return
    print()
    print("sensor", lid)
    print(cands[[c for c in SHOW if c in cands.columns]].to_string(index=False))

    seg_ids = cands["segment_id"].astype(str)
    segs = edges[edges[EDGE_ID].astype(str).isin(seg_ids)].copy()
    segs["label"] = segs[EDGE_ID].astype(str)

    if str(lid) not in sensors.index:
        print("no location found for sensor", lid)
        return
    pt = sensors.loc[[str(lid)]]
    info = [c for c in pt.columns
            if "direction" in c.lower() or c in ("sensor_description", "location_source")]
    print(pt[info].to_string(index=False))

    tips = [c for c in ["label", "highway", "name"] if c in segs.columns]
    m = segs.explore(column="label", tiles="Esri.WorldImagery",
                     style_kwds={"weight": 6}, tooltip=tips)
    pt.explore(m=m, color="red", marker_kwds={"radius": 8}, tooltip=info)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"sensor_{lid}.html"
    m.save(out)
    webbrowser.open(out.resolve().as_uri())


def maps(args):
    df, _ = load_review()
    edges = gpd.read_parquet(EDGES).to_crs(4326)
    sensors = load_sensors()
    for lid in args.location_ids:
        make_map(lid, df, edges, sensors)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(required=True)

p = sub.add_parser("inspect")
p.set_defaults(func=inspect)

p = sub.add_parser("list")
p.add_argument("--who", choices=["shubham", "bimba"], required=True)
p.set_defaults(func=list_sensors)

p = sub.add_parser("suggest")
p.add_argument("location_ids", type=int, nargs="+")
p.set_defaults(func=suggest)

p = sub.add_parser("map")
p.add_argument("location_ids", type=int, nargs="+")
p.set_defaults(func=maps)

args = parser.parse_args()
args.func(args)
