from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Sequence, Iterable

import numpy as np
import pandas as pd

# --- plotting / GIS ---
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import unary_union
import osmnx as ox
import contextily as ctx
import geodatasets

# ---- your manual registry ----
from Substations_AT import SUBSTATION_LOOKUP

# =============================================================================
# PIPELINE IMPORT (Main vs Main_New)
# =============================================================================

from Main import CFG, INSTRUMENTS  # only knobs, no paths from main

from Parameters_Updated import (
    load_demand_data,
    load_exchange_data,
)

# =============================================================================
# LOCAL path helpers (ALL PATHS IN THIS SCRIPT)
# =============================================================================

def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def stage1_base_system_results_path(base_dir: Path) -> Path:
    return base_dir / "full_model_results_stage1_base.csv"


def stage1_instr_system_results_path(base_dir: Path, instrument: str) -> Path:
    slug = instrument_slug(instrument)
    return base_dir / f"full_model_results_stage1_instrument__{slug}.csv"


# =============================================================================
# CONFIG
# =============================================================================

CRS_WGS84 = "EPSG:4326"

HOURS_PER_WEEK = 168
START_WEEK = 1
WEEKS_PER_STEP = 2

# Base date defines the YEAR used when snapshot string has no year
DEFAULT_BASE_DATE = "2030-01-01"
DATE_FMT_FULL = "%Y-%m-%d %H:%M"   # full: 2030-01-01 13:00
DATE_FMT_NOYEAR = "%m-%d %H:%M"   # no-year: 01-01 13:00

BOTTLENECK_UTILIZATION = 0.70
MIN_CAPACITY_MW = 1e-6

# Country code -> OSM geocoding query
COUNTRY_NAME: Dict[str, str] = {
    "AT": "Austria",
    "DE": "Germany",
    "NL": "Netherlands",
    "BE": "Belgium",
    "LU": "Luxembourg",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "SI": "Slovenia",
    "PL": "Poland",
    "SK": "Slovakia",
    "HU": "Hungary",
    "IT": "Italy",
    "FR": "France",
}
KNOWN_COUNTRY_CODES = set(COUNTRY_NAME.keys())

# Europe clip box for geocoding results
EUROPE_BBOX = box(-15.0, 34.0, 35.0, 72.0)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False

_UMLAUT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "AE", "Ö": "OE", "Ü": "UE",
})


# =============================================================================
# PART 1 — GRID NODES + LINES FROM Data_Updated.xlsx / Exchange_Data
# =============================================================================

def _sleep(s: float = 0.25) -> None:
    time.sleep(s)


def polygon_centroid(place: str) -> Tuple[float, float]:
    """
    Uses OSMnx geocode_to_gdf(place) and clips to EUROPE_BBOX.
    Returns (lat, lon).
    """
    g = ox.geocode_to_gdf(place)
    g = g[g.geometry.notna()].copy().to_crs(CRS_WGS84)

    g["geom_eu"] = g.geometry.apply(lambda geom: geom.intersection(EUROPE_BBOX))
    g = g[g["geom_eu"].notna()]
    g = g[~g["geom_eu"].is_empty]

    if g.empty:
        raise RuntimeError(f"No European geometry found for {place}")

    geom = unary_union(g["geom_eu"].values)
    c = geom.centroid
    return float(c.y), float(c.x)


def normalize_substation_name_like_registry(x: str) -> str:
    s = str(x).strip().translate(_UMLAUT_MAP)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)  # remove isolated digits
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


def parse_country_code_if_known(x: str) -> Optional[str]:
    x = str(x).strip().upper()
    m = re.match(r"^([A-Z]{2})", x)
    if not m:
        return None
    cc = m.group(1)
    return cc if cc in KNOWN_COUNTRY_CODES else None


def strip_trailing_index(token: str) -> str:
    s = str(token).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+$", "", s).strip()
    return s


def parse_endpoint_to_node(token: str) -> str:
    raw = str(token).strip()

    cc = parse_country_code_if_known(raw)
    if cc is not None:
        return cc

    base = strip_trailing_index(raw)
    cand = normalize_substation_name_like_registry(base)

    if cand in SUBSTATION_LOOKUP:
        return cand

    return cand


def parse_line_endpoints(line_id: str) -> Tuple[str, str]:
    if "-" not in line_id:
        raise ValueError(f"Line id does not contain '-': {line_id!r}")
    a, b = line_id.split("-", 1)
    return parse_endpoint_to_node(a), parse_endpoint_to_node(b)


def load_exchange_lines_from_data_updated(xlsx_path: Path, exchange_sheet: str = "Exchange_Data") -> pd.DataFrame:
    """
    Reads Exchange_Data from Data_Updated.xlsx and returns aggregated unique lines with canonicalized direction.
    Output columns: line_id, from_node, to_node (and maybe NTC columns if present)
    """
    df = pd.read_excel(xlsx_path, sheet_name=exchange_sheet)
    if "Line" not in df.columns:
        raise ValueError(f"Sheet {exchange_sheet!r} must contain a 'Line' column. Found: {list(df.columns)}")

    df = df[df["Line"].notna()].copy()
    df["Line"] = df["Line"].astype(str).str.strip()
    df = df[df["Line"] != ""].copy()

    endpoints = df["Line"].apply(parse_line_endpoints)
    df["from_node"] = [a for a, _ in endpoints]
    df["to_node"] = [b for _, b in endpoints]

    df["a"] = df[["from_node", "to_node"]].min(axis=1)
    df["b"] = df[["from_node", "to_node"]].max(axis=1)
    df["line_id"] = df["a"] + "-" + df["b"]

    sum_cols = [c for c in ["NTC A to B [MW]", "NTC B to A [MW]"] if c in df.columns]
    if sum_cols:
        agg = df.groupby(["a", "b", "line_id"], as_index=False)[sum_cols].sum()
    else:
        agg = df.groupby(["a", "b", "line_id"], as_index=False).size().drop(columns=["size"])

    agg["from_node"] = agg["a"]
    agg["to_node"] = agg["b"]

    return agg.drop(columns=["a", "b"]).reset_index(drop=True)


def build_nodes_with_coords(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds nodes table with lat/lon:
      - AT substations: SUBSTATION_LOOKUP
      - Countries: OSM centroid (via polygon_centroid)
      - Unknown: fallback to Austria centroid (so script runs)
    """
    nodes = sorted(set(lines_df["from_node"]).union(set(lines_df["to_node"])))
    rows: List[Dict] = []

    for node_id in nodes:
        node_id = str(node_id).strip()

        if node_id in SUBSTATION_LOOKUP:
            meta = SUBSTATION_LOOKUP[node_id]
            rows.append(dict(
                node=node_id,
                lat=float(meta["lat"]),
                lon=float(meta["lon"]),
                method="manual_substation_lookup",
                state=meta.get("state", None),
            ))
            continue

        if node_id in COUNTRY_NAME:
            place = COUNTRY_NAME.get(node_id, node_id)
            lat, lon = polygon_centroid(place)
            rows.append(dict(
                node=node_id,
                lat=lat,
                lon=lon,
                method="country_centroid",
                state=None,
            ))
            _sleep(0.2)
            continue

        # fallback: Austria centroid
        try:
            lat, lon = polygon_centroid("Austria")
            rows.append(dict(
                node=node_id,
                lat=lat,
                lon=lon,
                method="fallback_to_austria_centroid",
                state=None,
            ))
        except Exception:
            rows.append(dict(node=node_id, lat=np.nan, lon=np.nan, method="missing", state=None))

    nodes_df = pd.DataFrame(rows).sort_values("node").reset_index(drop=True)

    missing = nodes_df[nodes_df["lat"].isna() | nodes_df["lon"].isna()]
    if not missing.empty:
        print("\n[WARN] Nodes with missing coordinates:")
        print(missing[["node", "method"]].to_string(index=False))

    fallback = nodes_df[nodes_df["method"] == "fallback_to_austria_centroid"]
    if not fallback.empty:
        print("\n[WARN] AT substations NOT found in SUBSTATION_LOOKUP (fell back to Austria centroid):")
        print(", ".join(fallback["node"].tolist()))

    return nodes_df


def build_lines_gdf_from_nodes_and_lines(nodes_df: pd.DataFrame, lines_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Create one LineString geometry per line_id, using endpoints lat/lon.
    """
    xy = nodes_df.set_index("node")[["lat", "lon"]].to_dict(orient="index")

    geoms = []
    ids = []
    a_nodes = []
    b_nodes = []

    for _, r in lines_df.iterrows():
        line_id = str(r["line_id"])
        a = str(r["from_node"])
        b = str(r["to_node"])

        if a not in xy or b not in xy:
            continue
        if pd.isna(xy[a]["lat"]) or pd.isna(xy[a]["lon"]) or pd.isna(xy[b]["lat"]) or pd.isna(xy[b]["lon"]):
            continue

        geom = LineString([(float(xy[a]["lon"]), float(xy[a]["lat"])),
                           (float(xy[b]["lon"]), float(xy[b]["lat"]))])
        geoms.append(geom)
        ids.append(line_id)
        a_nodes.append(a)
        b_nodes.append(b)

    return gpd.GeoDataFrame(
        {"line_id": ids, "from_node": a_nodes, "to_node": b_nodes},
        geometry=geoms,
        crs="EPSG:4326",
    )


# =============================================================================
# PART 2 — TIME + RESULTS HELPERS
# =============================================================================

def parse_user_datetime(dt: str, *, base_date: str = DEFAULT_BASE_DATE) -> pd.Timestamp:
    """
    Accepts either:
      - 'YYYY-MM-DD HH:MM'  (full)
      - 'MM-DD HH:MM'       (no year; year is taken from base_date)
    """
    s = str(dt).strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$", s):
        return pd.to_datetime(s, format=DATE_FMT_FULL)

    if re.match(r"^\d{2}-\d{2}\s+\d{2}:\d{2}$", s):
        year = pd.to_datetime(str(base_date).strip() + " 00:00", format=DATE_FMT_FULL).year
        return pd.to_datetime(f"{year}-{s}", format=DATE_FMT_FULL)

    raise ValueError(f"Unsupported datetime format: {dt!r}. Use 'YYYY-MM-DD HH:MM' or 'MM-DD HH:MM'.")


def date_to_abs_hour(dt: str, base_date: str = DEFAULT_BASE_DATE) -> int:
    t0_str = str(base_date).strip() + " 00:00"
    t0 = pd.to_datetime(t0_str, format=DATE_FMT_FULL)
    t = parse_user_datetime(dt, base_date=base_date)
    diff_h = (t - t0).total_seconds() / 3600.0

    print(f"[DEBUG-date_to_abs_hour] base_date={base_date!r} -> t0={t0} | t={t} | diff_h={diff_h:.1f}")

    return int(diff_h // 1) + 1


def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust abs_hour reconstruction.

    Priority:
      1) If 'window' exists: treat it as the rolling-window index and compute
         abs_hour from window + within-window timestep using WINDOW_HOURS.
      2) Else fall back to start_week-based reconstruction.

    This avoids the start_week ambiguity that is currently collapsing 4704 hours to 2520.
    """
    if df is None or df.empty:
        return df
    if "abs_hour" in df.columns:
        return df

    df = df.copy()

    # pick a timestep column
    if "timestep" in df.columns:
        tcol = "timestep"
    elif "t" in df.columns:
        tcol = "t"
    elif "index_1" in df.columns:
        df["timestep"] = pd.to_numeric(df["index_1"], errors="coerce").fillna(0).astype(int)
        tcol = "timestep"
    else:
        return df

    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype(int)

    # detect whether within-window timestep is 0-based or 1-based
    tmin = int(df[tcol].min()) if len(df) else 0
    offset = 1 if tmin == 0 else 0

    # --- Preferred path: window-based (most robust for rolling horizon exports) ---
    if "window" in df.columns:
        w = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int)
        window_hours = int(HOURS_PER_WEEK) * int(WEEKS_PER_STEP)
        df["abs_hour"] = w * window_hours + df[tcol] + offset
        return df

    # --- Fallback: start_week-based ---
    if "start_week" not in df.columns:
        return df

    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(1).astype(int)
    df["abs_hour"] = df[tcol] + offset + HOURS_PER_WEEK * (df["start_week"] - 1)
    return df


def read_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)

    if "component" in df.columns and "variable" not in df.columns:
        df = df.rename(columns={"component": "variable"})

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "start_week" not in df.columns and "window" in df.columns:
        df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1

    df = ensure_abs_hour(df)
    return df


def extract_flows_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["connection", "abs_hour", "flow"])

    var = df.get("variable", pd.Series([""] * len(df))).astype(str)
    flows = df[var == "flow"].copy()
    if flows.empty:
        return pd.DataFrame(columns=["connection", "abs_hour", "flow"])

    if "index_0" in flows.columns:
        flows["connection"] = flows["index_0"].astype(str)
    elif "connection" in flows.columns:
        flows["connection"] = flows["connection"].astype(str)
    else:
        flows["connection"] = flows.get("p", "").astype(str)

    flows = ensure_abs_hour(flows)

    flows["abs_hour"] = pd.to_numeric(flows.get("abs_hour", np.nan), errors="coerce")
    flows = flows.dropna(subset=["abs_hour"]).copy()
    flows["abs_hour"] = flows["abs_hour"].astype(int)

    flows["flow"] = pd.to_numeric(flows.get("value", 0.0), errors="coerce").fillna(0.0)
    return flows[["connection", "abs_hour", "flow"]]


# =============================================================================
# PART 2B — CORRIDOR NORMALIZATION + METRICS
# =============================================================================

def _normalize_name(x: str) -> str:
    s = str(x).strip().translate(_UMLAUT_MAP)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


def _strip_trailing_digits(x: str) -> str:
    return re.sub(r"\d+$", "", str(x).strip()).strip()


def _endpoint_to_node(token: str) -> str:
    tok = _normalize_name(token)
    tok_nodig = _normalize_name(_strip_trailing_digits(tok))
    if len(tok_nodig) == 2 and tok_nodig in KNOWN_COUNTRY_CODES:
        return tok_nodig
    return tok_nodig


def _is_country(node: str) -> bool:
    node = str(node).strip().upper()
    return len(node) == 2 and node in KNOWN_COUNTRY_CODES


def build_at_alias_map(nodes: Iterable[str]) -> Dict[str, str]:
    """
    Map every Austrian (non-country) node to AT1, AT2, ...
    Deterministic: sorted by node name.
    """
    nodes = [str(n).strip() for n in nodes if str(n).strip()]
    at_nodes = sorted([n for n in nodes if not _is_country(n)])
    return {n: f"AT{i+1}" for i, n in enumerate(at_nodes)}


def print_at_alias_map(at_alias: Dict[str, str]) -> None:
    if not at_alias:
        print("\n[INFO] No Austrian nodes to alias.")
        return
    print("\n" + "-" * 100)
    print("AT node alias mapping (plots/outputs use AT1/AT2/..., terminal shows mapping)")
    print("-" * 100)
    for orig, alias in sorted(at_alias.items(), key=lambda kv: kv[1]):
        print(f"{alias:>5}  <-  {orig}")
    print("-" * 100 + "\n")


def normalize_corridor(connection: str, at_alias: Optional[Dict[str, str]] = None) -> str:
    parts = str(connection).split("-")
    if len(parts) != 2:
        return str(connection)
    a = _endpoint_to_node(parts[0])
    b = _endpoint_to_node(parts[1])

    # Apply alias ONLY for Austrian nodes (non-country tokens)
    if at_alias is not None:
        a = at_alias.get(a, a)
        b = at_alias.get(b, b)

    a, b = sorted([a, b])
    return f"{a}-{b}"


def build_capacity_frame(capacity_pos, capacity_neg) -> pd.DataFrame:
    def _to_conn_series(obj, name: str) -> pd.Series:
        if isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                s = obj.iloc[:, 0]
            else:
                col = name if name in obj.columns else obj.columns[0]
                s = obj[col]
        else:
            s = pd.Series(obj)
        s = s.copy()
        s.index = s.index.astype(str)
        s = pd.to_numeric(s, errors="coerce")
        s.name = name
        return s

    cap_pos = _to_conn_series(capacity_pos, "capacity_pos")
    cap_neg = _to_conn_series(capacity_neg, "capacity_neg")

    cap_df = pd.concat([cap_pos, cap_neg], axis=1)
    cap_df["capacity_pos"] = cap_df["capacity_pos"].fillna(0.0).abs()
    cap_df["capacity_neg"] = cap_df["capacity_neg"].fillna(0.0).abs()
    cap_df["capacity_abs"] = cap_df[["capacity_pos", "capacity_neg"]].max(axis=1)

    cap_df = cap_df.reset_index().rename(columns={"index": "connection"})
    cap_df["connection"] = cap_df["connection"].astype(str)
    return cap_df[["connection", "capacity_abs"]]


def snapshot_utilization_compare(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    capacity_pos,
    capacity_neg,
    *,
    date_time: str,
    base_date: str = DEFAULT_BASE_DATE,
    aggregate_corridors: bool = True,
    min_capacity: float = MIN_CAPACITY_MW,
    at_alias: Optional[Dict[str, str]] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Snapshot utilization (% of capacity) for BASE and INSTR at a single hour.

    Debug mode prints exactly where rows disappear.
    """
    abs_hour = date_to_abs_hour(date_time, base_date=base_date)

    # capacities
    caps = build_capacity_frame(capacity_pos, capacity_neg).copy()
    caps["connection"] = caps["connection"].astype(str)
    caps["capacity_abs"] = pd.to_numeric(caps["capacity_abs"], errors="coerce").fillna(0.0).abs()

    # flows (all hours)
    f_base_all = extract_flows_long(stage1_base)
    f_instr_all = extract_flows_long(stage1_instr)

    if debug:
        print("\n" + "-" * 100)
        print(f"[DEBUG] Snapshot request: date_time={date_time!r} -> abs_hour={abs_hour}")
        for tag, f in [("BASE(all)", f_base_all), ("INSTR(all)", f_instr_all)]:
            if f.empty:
                print(f"[DEBUG] {tag}: EMPTY")
            else:
                ah_min, ah_max = int(f["abs_hour"].min()), int(f["abs_hour"].max())
                print(f"[DEBUG] {tag}: rows={len(f)} | abs_hour range=[{ah_min}, {ah_max}] | "
                      f"unique hours={f['abs_hour'].nunique()} | unique conns={f['connection'].nunique()}")
        print(f"[DEBUG] caps: rows={len(caps)} | unique conns={caps['connection'].nunique()} | "
              f"cap min/max=({caps['capacity_abs'].min():.3g},{caps['capacity_abs'].max():.3g})")

    # filter to snapshot hour
    f_base = f_base_all[f_base_all["abs_hour"] == abs_hour].copy()
    f_instr = f_instr_all[f_instr_all["abs_hour"] == abs_hour].copy()

    if debug:
        print(f"[DEBUG] After hour filter: BASE rows={len(f_base)} | INSTR rows={len(f_instr)}")

        # If no rows, show closest available abs_hour(s)
        if f_base.empty and not f_base_all.empty:
            avail = np.array(sorted(f_base_all["abs_hour"].unique()), dtype=int)
            nearest = int(avail[np.argmin(np.abs(avail - abs_hour))])
            print(f"[DEBUG] No BASE flows at abs_hour={abs_hour}. Nearest available abs_hour={nearest} "
                  f"(delta={nearest-abs_hour}).")
        if f_instr.empty and not f_instr_all.empty:
            avail = np.array(sorted(f_instr_all["abs_hour"].unique()), dtype=int)
            nearest = int(avail[np.argmin(np.abs(avail - abs_hour))])
            print(f"[DEBUG] No INSTR flows at abs_hour={abs_hour}. Nearest available abs_hour={nearest} "
                  f"(delta={nearest-abs_hour}).")

    # merge capacities on connection level
    f_base = f_base.merge(caps, on="connection", how="left")
    f_instr = f_instr.merge(caps, on="connection", how="left")

    for df in (f_base, f_instr):
        df["capacity_abs"] = pd.to_numeric(df.get("capacity_abs", 0.0), errors="coerce")
        df["flow"] = pd.to_numeric(df.get("flow", 0.0), errors="coerce").fillna(0.0)
        df["abs_flow"] = df["flow"].abs()

    if debug:
        def cap_stats(tag, df):
            if df.empty:
                print(f"[DEBUG] {tag}: EMPTY after merge")
                return
            n_missing = int(df["capacity_abs"].isna().sum())
            n_zero = int((df["capacity_abs"].fillna(0.0) <= float(min_capacity)).sum())
            print(f"[DEBUG] {tag}: after merge rows={len(df)} | missing cap={n_missing} | "
                  f"cap<=min_capacity={n_zero}")
            if n_missing > 0:
                bad = df[df["capacity_abs"].isna()][["connection"]].drop_duplicates().head(12)
                print(f"[DEBUG] {tag}: example connections with missing capacity:\n{bad.to_string(index=False)}")
        cap_stats("BASE(snapshot)", f_base)
        cap_stats("INSTR(snapshot)", f_instr)

    # drop near-zero or missing capacity to avoid nonsense %
    f_base = f_base[pd.to_numeric(f_base["capacity_abs"], errors="coerce").fillna(0.0) > float(min_capacity)].copy()
    f_instr = f_instr[pd.to_numeric(f_instr["capacity_abs"], errors="coerce").fillna(0.0) > float(min_capacity)].copy()

    if debug:
        print(f"[DEBUG] After capacity filter (> {min_capacity}): BASE rows={len(f_base)} | INSTR rows={len(f_instr)}")

    if aggregate_corridors:
        f_base["line_id"] = f_base["connection"].map(lambda x: normalize_corridor(x, at_alias))
        f_instr["line_id"] = f_instr["connection"].map(lambda x: normalize_corridor(x, at_alias))

        base_agg = (
            f_base.groupby("line_id", as_index=False)
            .agg(capacity_mw=("capacity_abs", "sum"), abs_flow_base=("abs_flow", "sum"))
        )
        instr_agg = (
            f_instr.groupby("line_id", as_index=False)
            .agg(capacity_mw=("capacity_abs", "sum"), abs_flow_instr=("abs_flow", "sum"))
        )
    else:
        base_agg = f_base.rename(columns={"connection": "line_id", "capacity_abs": "capacity_mw"})[
            ["line_id", "capacity_mw", "abs_flow"]
        ].rename(columns={"abs_flow": "abs_flow_base"})
        instr_agg = f_instr.rename(columns={"connection": "line_id", "capacity_abs": "capacity_mw"})[
            ["line_id", "capacity_mw", "abs_flow"]
        ].rename(columns={"abs_flow": "abs_flow_instr"})

    comp = base_agg.merge(instr_agg, on="line_id", how="outer", suffixes=("_base", "_instr"))

    cap_cols = [c for c in comp.columns if c.startswith("capacity_mw")]
    if len(cap_cols) == 2:
        comp["capacity_mw"] = pd.concat([comp[cap_cols[0]], comp[cap_cols[1]]], axis=1).max(axis=1)
        comp = comp.drop(columns=cap_cols)
    elif len(cap_cols) == 1:
        comp = comp.rename(columns={cap_cols[0]: "capacity_mw"})
    else:
        comp["capacity_mw"] = np.nan

    comp["capacity_mw"] = pd.to_numeric(comp["capacity_mw"], errors="coerce").fillna(0.0)
    comp["abs_flow_base"] = pd.to_numeric(comp.get("abs_flow_base", 0.0), errors="coerce").fillna(0.0)
    comp["abs_flow_instr"] = pd.to_numeric(comp.get("abs_flow_instr", 0.0), errors="coerce").fillna(0.0)

    comp = comp[comp["capacity_mw"] > float(min_capacity)].copy()

    comp["util_base_pct"] = 100.0 * comp["abs_flow_base"] / comp["capacity_mw"]
    comp["util_instr_pct"] = 100.0 * comp["abs_flow_instr"] / comp["capacity_mw"]
    comp["delta_util_pct"] = comp["util_base_pct"] - comp["util_instr_pct"]

    comp["datetime"] = date_time
    comp = comp.sort_values(["util_base_pct", "line_id"], ascending=[False, True]).reset_index(drop=True)

    if debug:
        print(f"[DEBUG] Final comp rows={len(comp)}")
        print("-" * 100 + "\n")

    return comp


def snapshot_flow_compare(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    *,
    date_time: str,
    base_date: str = DEFAULT_BASE_DATE,
    aggregate_corridors: bool = True,
    at_alias: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    abs_hour = date_to_abs_hour(date_time, base_date=base_date)

    f_base = extract_flows_long(stage1_base)
    f_instr = extract_flows_long(stage1_instr)

    f_base = f_base[f_base["abs_hour"] == abs_hour].rename(columns={"flow": "flow_base"})
    f_instr = f_instr[f_instr["abs_hour"] == abs_hour].rename(columns={"flow": "flow_instr"})

    comp = pd.merge(f_base, f_instr, on=["connection", "abs_hour"], how="outer")
    comp["flow_base"] = pd.to_numeric(comp.get("flow_base", 0.0), errors="coerce").fillna(0.0)
    comp["flow_instr"] = pd.to_numeric(comp.get("flow_instr", 0.0), errors="coerce").fillna(0.0)

    comp["abs_base"] = comp["flow_base"].abs()
    comp["abs_instr"] = comp["flow_instr"].abs()
    comp["abs_delta_flow"] = comp["abs_base"] - comp["abs_instr"]  # + means reduction in |flow|
    comp["datetime"] = date_time

    if aggregate_corridors:
        comp["line_id"] = comp["connection"].map(lambda x: normalize_corridor(x, at_alias))
        comp = (
            comp.groupby("line_id", as_index=False)
            .agg(
                flow_base=("flow_base", "sum"),
                flow_instr=("flow_instr", "sum"),
                abs_base=("abs_base", "sum"),
                abs_instr=("abs_instr", "sum"),
            )
        )
        comp["abs_delta_flow"] = comp["abs_base"] - comp["abs_instr"]
    else:
        comp = comp.rename(columns={"connection": "line_id"})

    comp = comp.sort_values(["abs_delta_flow", "line_id"], ascending=[False, True]).reset_index(drop=True)
    return comp


def bottleneck_counts(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    capacity_pos,
    capacity_neg,
    *,
    utilization_threshold: float = BOTTLENECK_UTILIZATION,
    min_capacity: float = MIN_CAPACITY_MW,
    aggregate_corridors: bool = True,
    at_alias: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    caps = build_capacity_frame(capacity_pos, capacity_neg)
    caps["capacity_abs"] = pd.to_numeric(caps["capacity_abs"], errors="coerce").fillna(0.0)

    flows_base = extract_flows_long(stage1_base).rename(columns={"flow": "flow_base"})
    flows_instr = extract_flows_long(stage1_instr).rename(columns={"flow": "flow_instr"})

    flows = flows_base.merge(flows_instr, on=["connection", "abs_hour"], how="outer")
    flows["connection"] = flows["connection"].astype(str)

    flows["flow_base"] = pd.to_numeric(flows.get("flow_base", 0.0), errors="coerce").fillna(0.0)
    flows["flow_instr"] = pd.to_numeric(flows.get("flow_instr", 0.0), errors="coerce").fillna(0.0)

    flows = flows.merge(caps, on="connection", how="left")
    flows["capacity_abs"] = pd.to_numeric(flows.get("capacity_abs", 0.0), errors="coerce").fillna(0.0)

    flows = flows[flows["capacity_abs"] > float(min_capacity)].copy()
    if flows.empty:
        return pd.DataFrame(columns=["line_id", "n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"])

    thr = float(utilization_threshold)
    flows["is_bneck_base"] = flows["flow_base"].abs() >= (thr * flows["capacity_abs"])
    flows["is_bneck_instr"] = flows["flow_instr"].abs() >= (thr * flows["capacity_abs"])

    flows["line_id"] = (
        flows["connection"].map(lambda x: normalize_corridor(x, at_alias))
        if aggregate_corridors else flows["connection"]
    )

    stats = (
        flows.groupby("line_id", as_index=False)
        .agg(
            n_bottlenecks_base=("is_bneck_base", "sum"),
            n_bottlenecks_instr=("is_bneck_instr", "sum"),
        )
    )
    stats["n_bottlenecks_base"] = stats["n_bottlenecks_base"].astype(int)
    stats["n_bottlenecks_instr"] = stats["n_bottlenecks_instr"].astype(int)
    stats["delta_bottlenecks"] = stats["n_bottlenecks_base"] - stats["n_bottlenecks_instr"]  # + means improvement
    stats = stats.sort_values(["delta_bottlenecks", "line_id"], ascending=[False, True]).reset_index(drop=True)
    return stats


# =============================================================================
# PART 3 — DURATION CURVES (MERGED IN)
# =============================================================================

def _split_corridor(corridor: str) -> Tuple[str, str]:
    parts = str(corridor).split("-")
    if len(parts) != 2:
        return str(corridor), ""
    return parts[0].strip(), parts[1].strip()

def list_at_border_corridors_from_flows(
    flows_long: pd.DataFrame,
    at_alias: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Return sorted unique corridors where:
      - exactly one side is a country code
      - that country is NOT 'AT'
      - the other side is not a country code (treated as AT substation)
    Output corridor IDs will use AT1/AT2/... if at_alias is provided.
    """
    if flows_long is None or flows_long.empty:
        return []

    uniq_conn = flows_long["connection"].astype(str).dropna().unique().tolist()
    corridors = pd.Series(uniq_conn).map(lambda x: normalize_corridor(x, at_alias)).dropna().unique().tolist()

    keep: List[str] = []
    for c in corridors:
        a, b = _split_corridor(c)
        a_is = _is_country(a)
        b_is = _is_country(b)

        if a_is ^ b_is:
            country = a if a_is else b
            other = b if a_is else a
            if country != "AT" and not _is_country(other):
                keep.append(c)

    return sorted(set(keep))

def list_interconnector_corridors_from_flows(
    flows_long: pd.DataFrame,
    *,
    at_alias: Optional[Dict[str, str]] = None,
    include_country_country: bool = True,
    include_country_to_node: bool = True,
) -> List[str]:
    """
    Interconnectors = corridors involving country nodes.

    - country-country: DE-CH, IT-SI, ...
    - country-to-node: AT3-DE, AT7-CZ, ... (country ↔ Austrian node)

    Returns sorted unique corridor IDs (normalized; AT alias applied if provided).
    """
    if flows_long is None or flows_long.empty or "connection" not in flows_long.columns:
        return []

    uniq_conn = flows_long["connection"].astype(str).dropna().unique().tolist()
    corridors = pd.Series(uniq_conn).map(lambda x: normalize_corridor(x, at_alias)).dropna().unique().tolist()

    keep: List[str] = []
    for c in corridors:
        a, b = _split_corridor(c)
        if not a or not b or a == b:
            continue

        a_is = _is_country(a)
        b_is = _is_country(b)

        if include_country_country and a_is and b_is:
            keep.append(c)
            continue

        if include_country_to_node and (a_is ^ b_is):
            keep.append(c)
            continue

    return sorted(set(keep))

def build_corridor_capacity_map(
    capacity_pos,
    capacity_neg,
    *,
    at_alias: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Returns: { corridor_id -> capacity_mw }

    Corridor id is normalize_corridor(connection, at_alias).
    Capacity is sum of capacity_abs for all links in that corridor.
    """
    caps = build_capacity_frame(capacity_pos, capacity_neg).copy()
    caps["connection"] = caps["connection"].astype(str)
    caps["capacity_abs"] = pd.to_numeric(caps["capacity_abs"], errors="coerce").fillna(0.0).abs()

    caps["corridor_id"] = caps["connection"].map(lambda x: normalize_corridor(x, at_alias))
    cap_map = (
        caps.groupby("corridor_id", as_index=True)["capacity_abs"]
        .sum()
        .to_dict()
    )
    # ensure plain floats
    return {str(k): float(v) for k, v in cap_map.items()}

def plot_duration_multi_instruments_side_by_side(
    base_curve: np.ndarray,
    instr_curves: Dict[str, np.ndarray],
    *,
    corridor_id: str,
    ylabel: str,
    use_absolute: bool,
    capacity_mw: Optional[float] = None,   # <-- NEW: corridor capacity in MW
    show_utilization_lines: bool = False,  # <-- optional: add horiz. lines at peak utilizations
    out_png: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot BASE vs each instrument curve in separate subplots (side-by-side),
    grouped by corridor_id.

    Annotates per subplot:
      - Δ peak in MW (rank 1)
      - Δ peak as % of capacity (if capacity_mw provided)
      - optional: BASE/INSTR peak utilization % (if capacity_mw provided)
    """
    if base_curve.size == 0 and all(v.size == 0 for v in instr_curves.values()):
        print(f"⚠️ Nothing to plot for {corridor_id} (all empty).")
        return

    names = list(instr_curves.keys())
    ncols = max(1, len(names))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(5.8 * ncols, 4.8),
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    abs_tag = "ABS" if use_absolute else "SIGNED"
    fig.suptitle(f"Flow duration curves ({abs_tag}) — {corridor_id}", y=1.03)

    cap = None
    if capacity_mw is not None:
        try:
            cap = float(capacity_mw)
        except Exception:
            cap = None
        if cap is not None and cap <= 0:
            cap = None

    for ax, name in zip(axes, names):
        instr_curve = instr_curves.get(name, np.array([], dtype=float))
        n = min(len(base_curve), len(instr_curve))

        if n == 0:
            ax.set_title(name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Ranked hours")
            ax.grid(True, alpha=0.25)
            continue

        x = np.arange(1, n + 1)
        ax.plot(x, base_curve[:n], label="BASE")
        ax.plot(x, instr_curve[:n], label=name)

        # Peaks (rank 1 of the displayed curve)
        peak_base = float(base_curve[0])
        peak_instr = float(instr_curve[0])
        peak_delta_mw = peak_base - peak_instr  # signed difference of displayed peaks

        # Build annotation text
        lines = [f"Δ peak = {peak_delta_mw:.2f} MW"]

        if cap is not None:
            delta_pct = 100.0 * peak_delta_mw / cap
            util_base_pct = 100.0 * peak_base / cap
            util_instr_pct = 100.0 * peak_instr / cap
            lines.append(f"Δ peak = {delta_pct:.2f}% cap")
            lines.append(f"BASE peak = {util_base_pct:.1f}% | {name} peak = {util_instr_pct:.1f}%")

            if show_utilization_lines:
                ax.axhline(peak_base, linewidth=1.0, alpha=0.25)
                ax.axhline(peak_instr, linewidth=1.0, alpha=0.25)

        ax.text(
            0.98, 0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        ax.set_title(name)
        ax.set_xlabel("Ranked hours")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")

    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_duration_curve(
    flows_long: pd.DataFrame,
    *,
    line_id: str,
    use_absolute: bool,
    aggregate_corridors: bool,
    keep_hours: Optional[int] = None,
    at_alias: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Build a (descending) duration curve for a single corridor/connection.

    Parameters
    ----------
    flows_long : DataFrame
        Must contain columns: ['connection','abs_hour','flow'] (at least).
    line_id : str
        Corridor/connection id to select (already normalized if aggregate_corridors=True).
    use_absolute : bool
        If True, use |flow| before sorting.
    aggregate_corridors : bool
        If True, normalize corridor ids (and aggregate parallel links per hour).
    keep_hours : Optional[int]
        If set, keep only top N ranked hours.
    at_alias : Optional[Dict[str,str]]
        Mapping for Austrian node aliases (AT1, AT2, ...) used in normalize_corridor.

    Returns
    -------
    np.ndarray
        Sorted values, highest first.
    """
    if flows_long is None or flows_long.empty:
        return np.array([], dtype=float)

    f = flows_long.copy()
    f["connection"] = f["connection"].astype(str)

    if aggregate_corridors:
        f["connection"] = f["connection"].map(lambda x: normalize_corridor(x, at_alias))

        if use_absolute:
            per_hour = (
                f.assign(flow_abs=f["flow"].abs())
                .groupby(["connection", "abs_hour"], as_index=False)["flow_abs"].sum()
                .rename(columns={"flow_abs": "flow"})
            )
        else:
            per_hour = f.groupby(["connection", "abs_hour"], as_index=False)["flow"].sum()

        f = per_hour

    f_line = f[f["connection"] == str(line_id)].copy()
    if f_line.empty:
        return np.array([], dtype=float)

    vals = pd.to_numeric(f_line["flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if use_absolute:
        vals = np.abs(vals)

    vals_sorted = np.sort(vals)[::-1]
    if keep_hours is not None:
        vals_sorted = vals_sorted[: int(keep_hours)]
    return vals_sorted


def plot_duration_multi_instruments_side_by_side(
    base_curve: np.ndarray,
    instr_curves: Dict[str, np.ndarray],
    *,
    corridor_id: str,
    ylabel: str,
    use_absolute: bool,
    capacity_mw: Optional[float] = None,   # <-- NEW: corridor capacity in MW
    show_utilization_lines: bool = False,  # <-- optional: add horiz. lines at peak utilizations
    out_png: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot BASE vs each instrument curve in separate subplots (side-by-side),
    grouped by corridor_id.

    Annotates per subplot:
      - Δ peak in MW (rank 1)
      - Δ peak as % of capacity (if capacity_mw provided)
      - optional: BASE/INSTR peak utilization % (if capacity_mw provided)
    """
    if base_curve.size == 0 and all(v.size == 0 for v in instr_curves.values()):
        print(f"⚠️ Nothing to plot for {corridor_id} (all empty).")
        return

    names = list(instr_curves.keys())
    ncols = max(1, len(names))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(5.8 * ncols, 4.8),
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    abs_tag = "ABS" if use_absolute else "SIGNED"
    fig.suptitle(f"Flow duration curves ({abs_tag}) — {corridor_id}", y=1.03)

    cap = None
    if capacity_mw is not None:
        try:
            cap = float(capacity_mw)
        except Exception:
            cap = None
        if cap is not None and cap <= 0:
            cap = None

    for ax, name in zip(axes, names):
        instr_curve = instr_curves.get(name, np.array([], dtype=float))
        n = min(len(base_curve), len(instr_curve))

        if n == 0:
            ax.set_title(name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Ranked hours")
            ax.grid(True, alpha=0.25)
            continue

        x = np.arange(1, n + 1)
        ax.plot(x, base_curve[:n], label="BASE")
        ax.plot(x, instr_curve[:n], label=name)

        # Peaks (rank 1 of the displayed curve)
        peak_base = float(base_curve[0])
        peak_instr = float(instr_curve[0])
        peak_delta_mw = peak_base - peak_instr  # signed difference of displayed peaks

        # Build annotation text
        lines = [f"Δ peak = {peak_delta_mw:.2f} MW"]

        if cap is not None:
            delta_pct = 100.0 * peak_delta_mw / cap
            util_base_pct = 100.0 * peak_base / cap
            util_instr_pct = 100.0 * peak_instr / cap
            lines.append(f"Δ peak = {delta_pct:.2f}% cap")
            lines.append(f"BASE peak = {util_base_pct:.1f}% | {name} peak = {util_instr_pct:.1f}%")

            if show_utilization_lines:
                ax.axhline(peak_base, linewidth=1.0, alpha=0.25)
                ax.axhline(peak_instr, linewidth=1.0, alpha=0.25)

        ax.text(
            0.98, 0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        ax.set_title(name)
        ax.set_xlabel("Ranked hours")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")

    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# PART 4 — PLOTTING MAPS
# =============================================================================

def plot_metric_map_tiles_with_borders(
    lines_gdf: gpd.GeoDataFrame,
    metric_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    nodes_df: Optional[pd.DataFrame] = None,
    highlight_top_n: Optional[int] = None,  # default: all
    rank_abs: bool = True,
    zoom_pad_m: float = 120_000,
    basemap=ctx.providers.CartoDB.PositronNoLabels,
    draw_countries: bool = True,
    draw_regions: bool = True,
    # line styling
    context_lw: float = 1.2,
    context_alpha: float = 0.25,
    hi_lw: float = 4.2,
    hi_alpha: float = 0.98,
    # borders styling
    country_lw: float = 1.3,
    country_alpha: float = 0.75,
    region_lw: float = 0.8,
    region_alpha: float = 0.35,
    # country names
    label_countries: bool = True,
    country_label_min_area_km2: float = 1_000,
    country_label_fs: int = 11,
    # colorbar
    cbar_label: Optional[str] = "MW",
    out_path: Optional[Path] = None,
    # zero handling
    zero_eps: float = 1e-9,
    zero_color: str = "lightgray",
) -> None:
    import matplotlib.patheffects as pe
    import matplotlib.colors as mcolors

    # ---- merge metric onto lines ----
    gdf = lines_gdf.merge(metric_df[["line_id", metric_col]], on="line_id", how="left")
    gdf[metric_col] = pd.to_numeric(gdf[metric_col], errors="coerce").fillna(0.0)

    # ---- nodes to gdf ----
    nodes_gdf = None
    if nodes_df is not None and {"lat", "lon"}.issubset(nodes_df.columns):
        nd = nodes_df.copy()
        nd["lat"] = pd.to_numeric(nd["lat"], errors="coerce")
        nd["lon"] = pd.to_numeric(nd["lon"], errors="coerce")
        nd = nd.dropna(subset=["lat", "lon"])
        if not nd.empty:
            nodes_gdf = gpd.GeoDataFrame(
                nd,
                geometry=gpd.points_from_xy(nd["lon"], nd["lat"]),
                crs="EPSG:4326",
            )

    # ---- project to 3857 for tiles ----
    gdf_3857 = gdf.to_crs(epsg=3857)
    nodes_3857 = nodes_gdf.to_crs(epsg=3857) if nodes_gdf is not None else None

    # ---- extent ----
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    if nodes_3857 is not None and not nodes_3857.empty:
        nminx, nminy, nmaxx, nmaxy = nodes_3857.total_bounds
        minx, miny = min(minx, nminx), min(miny, nminy)
        maxx, maxy = max(maxx, nmaxx), max(maxy, nmaxy)

    minx -= zoom_pad_m
    miny -= zoom_pad_m
    maxx += zoom_pad_m
    maxy += zoom_pad_m

    bbox_3857_geom = box(minx, miny, maxx, maxy)
    bbox_4326_geom = gpd.GeoSeries([bbox_3857_geom], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    # ---- borders ----
    countries_3857 = None
    regions_3857 = None

    if draw_countries:
        try:
            countries = gpd.read_file(geodatasets.get_path("naturalearth.countries")).to_crs("EPSG:4326")
        except Exception:
            countries = None

        if countries is not None and not countries.empty:
            countries = countries[countries.intersects(bbox_4326_geom)].copy()
            countries_3857 = countries.to_crs(epsg=3857)

    if draw_regions:
        try:
            regions = gpd.read_file(geodatasets.get_path("naturalearth.admin_1_states_provinces")).to_crs("EPSG:4326")
        except Exception:
            regions = None

        if regions is not None and not regions.empty:
            regions = regions[regions.intersects(bbox_4326_geom)].copy()
            regions_3857 = regions.to_crs(epsg=3857)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ctx.add_basemap(ax, source=basemap, attribution_size=6)

    if regions_3857 is not None and not regions_3857.empty:
        regions_3857.boundary.plot(ax=ax, color="black", linewidth=region_lw, alpha=region_alpha, zorder=2)

    if countries_3857 is not None and not countries_3857.empty:
        countries_3857.boundary.plot(ax=ax, color="black", linewidth=country_lw, alpha=country_alpha, zorder=3)

        if label_countries:
            c = countries_3857.copy()
            c["_area_km2"] = c.geometry.area / 1e6
            c = c[c["_area_km2"] >= float(country_label_min_area_km2)].copy()

            candidate_cols = [
                "name", "NAME", "ADMIN", "NAME_EN", "NAME_LONG", "SOVEREIGNT",
                "NAME_SORT", "BRK_NAME", "FORMAL_EN"
            ]
            name_col = next((col for col in candidate_cols if col in c.columns), None)

            if name_col:
                c["_pt"] = c.representative_point()
                for _, r in c.iterrows():
                    x, y = r["_pt"].x, r["_pt"].y
                    ax.text(
                        x, y, str(r[name_col]),
                        fontsize=country_label_fs,
                        color="dimgray",
                        ha="center", va="center",
                        zorder=20,
                        alpha=0.9,
                        path_effects=[pe.withStroke(linewidth=3.5, foreground="white", alpha=0.9)],
                    )

    # context: all lines
    gdf_3857.plot(ax=ax, color="black", linewidth=context_lw + 1.2, alpha=0.25, zorder=4)
    gdf_3857.plot(ax=ax, color="lightgray", linewidth=context_lw, alpha=context_alpha, zorder=5)

    # highlight set
    hi = gdf_3857
    if highlight_top_n is not None and highlight_top_n > 0:
        r = hi[metric_col].abs() if rank_abs else hi[metric_col]
        hi = hi.loc[r.nlargest(int(highlight_top_n)).index].copy()
    else:
        hi = hi.copy()

    # halo
    hi.plot(ax=ax, color="black", linewidth=hi_lw + 3.0, alpha=0.85, zorder=9)

    # split zeros vs non-zeros
    vals = pd.to_numeric(hi[metric_col], errors="coerce").fillna(0.0)
    is_zero = vals.abs() <= float(zero_eps)
    hi_zero = hi[is_zero].copy()
    hi_nz = hi[~is_zero].copy()

    legend_kw = {"label": (cbar_label if cbar_label is not None else metric_col), "shrink": 0.75}

    # non-zero: diverging around 0
    if not hi_nz.empty:
        vmin = float(pd.to_numeric(hi_nz[metric_col], errors="coerce").min())
        vmax = float(pd.to_numeric(hi_nz[metric_col], errors="coerce").max())
        m = max(abs(vmin), abs(vmax))
        norm = mcolors.TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)

        hi_nz.plot(
            ax=ax,
            column=metric_col,
            linewidth=hi_lw,
            alpha=hi_alpha,
            legend=True,
            cmap="RdYlGn",
            norm=norm,
            legend_kwds=legend_kw,
            zorder=10,
        )

    # zero: always grey on top
    if not hi_zero.empty:
        hi_zero.plot(ax=ax, color=zero_color, linewidth=hi_lw, alpha=hi_alpha, zorder=11)

    # nodes
    if nodes_3857 is not None and not nodes_3857.empty:
        nodes_3857.plot(ax=ax, color="white", markersize=70, alpha=0.95, zorder=12)
        nodes_3857.plot(ax=ax, color="black", markersize=18, alpha=0.98, zorder=13)

    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved figure: {out_path}")

    plt.show()
    plt.close()


def plot_metric_maps_tiles_side_by_side_with_borders(
    lines_gdf: gpd.GeoDataFrame,
    metric_dfs: Sequence[pd.DataFrame],
    *,
    metric_col: str,
    titles: Sequence[str],
    nodes_df: Optional[pd.DataFrame] = None,
    zoom_pad_m: float = 120_000,
    basemap=ctx.providers.CartoDB.PositronNoLabels,
    cbar_label: Optional[str] = None,
    out_path: Optional[Path] = None,
):
    """
    Symmetric layout.
    Each panel has its own colorbar.
    Green (low utilization) → Red (high utilization).
    Thick halo-highlighted lines.
    """
    import matplotlib.colors as mcolors

    # ------------------------------------------------------------------
    # Merge metrics
    # ------------------------------------------------------------------
    merged_list = []

    for metric_df in metric_dfs:
        gdf = lines_gdf.merge(
            metric_df[["line_id", metric_col]],
            on="line_id",
            how="left",
        )
        gdf[metric_col] = pd.to_numeric(gdf[metric_col], errors="coerce").fillna(0.0)
        merged_list.append(gdf)

    # Gather all plotted values (across panels) to pick a consistent norm
    all_vals = []
    for gdf in merged_list:
        v = pd.to_numeric(gdf[metric_col], errors="coerce").to_numpy()
        all_vals.append(v)
    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        all_vals = np.array([0.0])
    
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    
    # AUTO mode:
    # - diverging if values cross 0 (both negative and positive)
    # - sequential (0..max) otherwise
    if vmin < 0.0 and vmax > 0.0:
        # robust symmetric range (ignore extreme outliers)
        q = 0.98  # try 0.98 or 0.95
        m = float(np.quantile(np.abs(all_vals), q))
        m = max(m, 1e-9)
    
        norm = mcolors.TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
        cmap = "RdYlGn"      # negative=red, positive=green
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
        cmap = "RdYlGn_r"    # low=green, high=red (e.g. utilization, |flow|)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    nodes_gdf = None
    if nodes_df is not None and {"lat", "lon"}.issubset(nodes_df.columns):
        nd = nodes_df.dropna(subset=["lat", "lon"]).copy()
        if not nd.empty:
            nodes_gdf = gpd.GeoDataFrame(
                nd,
                geometry=gpd.points_from_xy(nd["lon"], nd["lat"]),
                crs="EPSG:4326",
            )

    base_3857 = lines_gdf.to_crs(epsg=3857)
    nodes_3857 = nodes_gdf.to_crs(epsg=3857) if nodes_gdf is not None else None

    minx, miny, maxx, maxy = base_3857.total_bounds
    minx -= zoom_pad_m
    miny -= zoom_pad_m
    maxx += zoom_pad_m
    maxy += zoom_pad_m

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    n = len(merged_list)
    fig_width = 18 * n
    fig_height = 12

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.02},   # minimal spacing
    )

    if n == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02)

    for ax, gdf, title in zip(axes, merged_list, titles):
        gdf_3857 = gdf.to_crs(epsg=3857)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.margins(0)

        ctx.add_basemap(ax, source=basemap, attribution_size=6)

        # background
        gdf_3857.plot(ax=ax, color="lightgray", linewidth=1.2, alpha=0.25, zorder=1)

        # halo
        gdf_3857.plot(ax=ax, color="black", linewidth=7.0, alpha=0.85, zorder=3)

        # colored overlay
        gdf_3857.plot(
            ax=ax,
            column=metric_col,
            cmap=cmap,
            norm=norm,
            linewidth=4.8,
            alpha=0.97,
            zorder=5,
        )

        # nodes
        if nodes_3857 is not None and not nodes_3857.empty:
            nodes_3857.plot(ax=ax, color="white", markersize=90, zorder=10)
            nodes_3857.plot(ax=ax, color="black", markersize=24, zorder=11)

        ax.set_axis_off()
        ax.set_title(title, fontsize=26, pad=18)

        # individual colorbar per panel
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.04,
            pad=0.02
        )
        cbar.set_label(cbar_label if cbar_label else metric_col, fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=320, bbox_inches="tight")
        print(f"✅ Saved figure: {out_path}")

    plt.show()
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def print_block(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_top(df: pd.DataFrame, cols: List[str], n: int = 30) -> None:
    if df is None or df.empty:
        print("⚠️ Empty.")
        return
    print(df.loc[:, cols].head(n).to_string(index=False))


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS
    # -------------------------------------------------------------------------
    PROJECT_ROOT = Path(r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1")
    INPUT_XLSX = PROJECT_ROOT / "Data_Updated.xlsx"
    EXCHANGE_SHEET = "Exchange_Data"

    # Snapshot accepts "MM-DD HH:MM" (no year)
    SNAPSHOT_DATE_TIME = "07-01 13:00"
    AGGREGATE_CORRIDORS = True
    TOP_N_PRINT = 30

    # Always plot both: flows + bottlenecks
    MAKE_FIGURES = True
    FIG_DIR = PROJECT_ROOT / "Figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Duration curves
    PLOT_DURATION_CURVES_AT_BORDERS = True
    DURATION_AGGREGATE_CORRIDORS = True   # must be True for corridor list
    DURATION_USE_ABSOLUTE = True
    DURATION_KEEP_HOURS = None
    SAVE_DURATION_PNGS = True
    SHOW_DURATION_PLOTS = True

    # Exact instruments (as you wrote them)
    INSTRUMENT_SELECTION: Sequence[str] = ["RTP", "Peak-Shaving", "Capacity-Tariff"]

    # -------------------------------------------------------------------------
    # Resolve instruments exactly (and fail loudly if spelling mismatch)
    # -------------------------------------------------------------------------
    instruments: List[str] = [inst for inst in INSTRUMENTS if inst in INSTRUMENT_SELECTION]
    if len(instruments) != len(INSTRUMENT_SELECTION):
        missing = set(INSTRUMENT_SELECTION) - set(instruments)
        raise RuntimeError(
            f"These instruments were not found in INSTRUMENTS: {missing}\n"
            f"Available INSTRUMENTS: {INSTRUMENTS}"
        )

    # -------------------------------------------------------------------------
    # Build geometries
    # -------------------------------------------------------------------------
    print_block("1) BUILDING LINE GEOMETRY FROM Data_Updated.xlsx")
    lines_df = load_exchange_lines_from_data_updated(INPUT_XLSX, exchange_sheet=EXCHANGE_SHEET)
    nodes_df = build_nodes_with_coords(lines_df)

    # -------------------------------------------------------------------------
    # Build AT alias map (AT substations -> AT1, AT2, ...) and APPLY it
    # IMPORTANT: This changes *node names* used for plotting/printing/IDs.
    # -------------------------------------------------------------------------
    at_alias = build_at_alias_map(nodes_df["node"].tolist())
    print_at_alias_map(at_alias)

    # Apply alias to nodes_df
    nodes_df = nodes_df.copy()
    nodes_df["node"] = nodes_df["node"].map(lambda x: at_alias.get(str(x).strip(), str(x).strip()))

    # Apply alias to lines_df and recompute canonical line_id
    lines_df = lines_df.copy()
    lines_df["from_node"] = lines_df["from_node"].map(lambda x: at_alias.get(str(x).strip(), str(x).strip()))
    lines_df["to_node"] = lines_df["to_node"].map(lambda x: at_alias.get(str(x).strip(), str(x).strip()))
    lines_df["a"] = lines_df[["from_node", "to_node"]].min(axis=1)
    lines_df["b"] = lines_df[["from_node", "to_node"]].max(axis=1)
    lines_df["line_id"] = lines_df["a"] + "-" + lines_df["b"]
    lines_df = lines_df.drop(columns=["a", "b"], errors="ignore")

    # Build geometry with aliased names
    lines_gdf = build_lines_gdf_from_nodes_and_lines(nodes_df, lines_df)

    print(f"Lines (unique): {len(lines_df)}")
    print(f"Nodes (unique): {len(nodes_df)}")
    print(f"Plottable geometries: {len(lines_gdf)}")

    # -------------------------------------------------------------------------
    # Load shared pipeline inputs (capacities, paths, base)
    # -------------------------------------------------------------------------
    print_block("2) LOADING PIPELINE INPUTS (capacities) + BASE RESULTS")

    DATA_XLSX = str(INPUT_XLSX)  # keep Parameters_Updated calls happy

    demand_data = load_demand_data(DATA_XLSX, "Demand_Profiles")
    try:
        _, _, _, _, node_to_control_area = demand_data
    except Exception:
        node_to_control_area = demand_data[-1]

    ptdf_csv_path = str(PROJECT_ROOT / "PTDF_Synchronized.csv")

    exch = load_exchange_data(
        DATA_XLSX,
        "Exchange_Data",
        ptdf_csv_path=ptdf_csv_path,
        slack_node=None,
        verbose=False,
    )
    incidence_matrix = exch[2]  # kept for completeness
    capacity_pos = exch[3]
    capacity_neg = exch[4]

    base_results_path = stage1_base_system_results_path(PROJECT_ROOT)
    stage1_base = read_results(str(base_results_path))
    base_flows = extract_flows_long(stage1_base)
    print(f"BASE results: {base_results_path}")

    # -------------------------------------------------------------------------
    # Preload instrument results once (so duration curves can use all instruments)
    # -------------------------------------------------------------------------
    print_block("3) LOADING INSTRUMENT RESULTS")
    instr_stage1: Dict[str, pd.DataFrame] = {}
    instr_flows: Dict[str, pd.DataFrame] = {}
    instr_slug: Dict[str, str] = {}

    for instrument in instruments:
        stage1_instr_path = stage1_instr_system_results_path(PROJECT_ROOT, instrument)
        if not stage1_instr_path.exists():
            raise FileNotFoundError(f"Missing instrument results file: {stage1_instr_path}")

        df_instr = read_results(str(stage1_instr_path))
        instr_stage1[instrument] = df_instr
        instr_flows[instrument] = extract_flows_long(df_instr)
        instr_slug[instrument] = instrument_slug(instrument)

        print(f"✅ {instrument}: {stage1_instr_path}")

    # -------------------------------------------------------------------------
    # Determine AT-border corridors for duration curves (with AT alias)
    # -------------------------------------------------------------------------
    at_border_corridors: List[str] = []
    if PLOT_DURATION_CURVES_AT_BORDERS:
        at_border_corridors = list_at_border_corridors_from_flows(base_flows, at_alias=at_alias)
        if not at_border_corridors:
            print("⚠️ No AT-border corridors found in BASE flows after normalize_corridor().")
        else:
            print(f"✅ Found {len(at_border_corridors)} AT-border corridors (ATi↔foreign country).")

    # -------------------------------------------------------------------------
    # Collect bottleneck maps to plot ALL instruments side-by-side at the end
    # -------------------------------------------------------------------------
    bottleneck_metric_frames: List[pd.DataFrame] = []
    bottleneck_titles: List[str] = []

    # -------------------------------------------------------------------------
    # Per-instrument analysis: print + map plots (snapshot utilization + bottlenecks)
    # -------------------------------------------------------------------------
    for instrument in instruments:
        print_block(f"4) INSTRUMENT: {instrument}")

        stage1_instr = instr_stage1[instrument]
        slug = instr_slug[instrument]

        # (A) Snapshot utilization (% of capacity) — BASE vs INSTR
        snap_u = snapshot_utilization_compare(
            stage1_base=stage1_base,
            stage1_instr=stage1_instr,
            capacity_pos=capacity_pos,
            capacity_neg=capacity_neg,
            date_time=SNAPSHOT_DATE_TIME,
            aggregate_corridors=AGGREGATE_CORRIDORS,
            min_capacity=MIN_CAPACITY_MW,
            at_alias=at_alias,
        )

        print(
            f"Snapshot UTILIZATION @ {SNAPSHOT_DATE_TIME} | rows={len(snap_u)} | "
            f"aggregate_corridors={AGGREGATE_CORRIDORS}"
        )

        print("\nTop BASE utilization (%):")
        snap_u_base = snap_u.sort_values("util_base_pct", ascending=False).reset_index(drop=True)
        print_top(snap_u_base, ["line_id", "capacity_mw", "abs_flow_base", "util_base_pct"], n=TOP_N_PRINT)

        print("\nTop INSTRUMENT utilization (%):")
        snap_u_instr = snap_u.sort_values("util_instr_pct", ascending=False).reset_index(drop=True)
        print_top(snap_u_instr, ["line_id", "capacity_mw", "abs_flow_instr", "util_instr_pct"], n=TOP_N_PRINT)

        print("\nLargest utilization REDUCTIONS (positive delta_util_pct):")
        snap_u_improve = snap_u.sort_values("delta_util_pct", ascending=False).reset_index(drop=True)
        print_top(snap_u_improve, ["line_id", "util_base_pct", "util_instr_pct", "delta_util_pct"], n=TOP_N_PRINT)

        print("\nLargest utilization INCREASES (most negative delta_util_pct):")
        snap_u_worse = snap_u.sort_values("delta_util_pct", ascending=True).reset_index(drop=True)
        print_top(snap_u_worse, ["line_id", "util_base_pct", "util_instr_pct", "delta_util_pct"], n=TOP_N_PRINT)

        # (B) Bottleneck counts (across all hours)
        bneck = bottleneck_counts(
            stage1_base=stage1_base,
            stage1_instr=stage1_instr,
            capacity_pos=capacity_pos,
            capacity_neg=capacity_neg,
            utilization_threshold=BOTTLENECK_UTILIZATION,
            min_capacity=MIN_CAPACITY_MW,
            aggregate_corridors=AGGREGATE_CORRIDORS,
            at_alias=at_alias,
        )

        print(f"\nBottlenecks | util>={BOTTLENECK_UTILIZATION:.2f} | rows={len(bneck)}")
        print("\nTop IMPROVEMENTS (positive delta_bottlenecks = fewer bottlenecks):")
        print_top(bneck, ["line_id", "n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"], n=TOP_N_PRINT)

        print("\nTop WORSENING (negative delta_bottlenecks):")
        bneck_w = bneck.sort_values("delta_bottlenecks", ascending=True).reset_index(drop=True)
        print_top(bneck_w, ["line_id", "n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"], n=TOP_N_PRINT)

        # collect bottleneck metrics for the end-of-loop multi-panel plot
        bneck_metric = bneck[["line_id", "delta_bottlenecks"]].copy()
        bneck_metric = bneck_metric.rename(columns={"delta_bottlenecks": "delta_bneck"})
        bottleneck_metric_frames.append(bneck_metric)
        bottleneck_titles.append(f"Δ Bottlenecks — {instrument}")

        # (C) Figures
        if MAKE_FIGURES:
            util_base_metric = snap_u[["line_id", "util_base_pct"]].rename(columns={"util_base_pct": "util_pct"})
            util_instr_metric = snap_u[["line_id", "util_instr_pct"]].rename(columns={"util_instr_pct": "util_pct"})

            plot_metric_maps_tiles_side_by_side_with_borders(
                lines_gdf,
                [util_base_metric, util_instr_metric],
                metric_col="util_pct",
                titles=[
                    f"Snapshot Utilization — BASE ({SNAPSHOT_DATE_TIME})",
                    f"Snapshot Utilization — {instrument} ({SNAPSHOT_DATE_TIME})",
                ],
                nodes_df=nodes_df,
                cbar_label="% of capacity",
                out_path=FIG_DIR / f"Map__snapshot_util__BASE_vs__{slug}.png",
            )

    # -------------------------------------------------------------------------
    # Bottleneck maps: ALL instruments side-by-side
    # -------------------------------------------------------------------------
    if MAKE_FIGURES and bottleneck_metric_frames:
        plot_metric_maps_tiles_side_by_side_with_borders(
            lines_gdf,
            bottleneck_metric_frames,
            metric_col="delta_bneck",
            titles=bottleneck_titles,
            nodes_df=nodes_df,
            cbar_label="count",
            out_path=FIG_DIR / "Map__delta_bottlenecks__ALL_instruments.png",
        )

    # -------------------------------------------------------------------------
    # Duration curves: BASE vs instruments (side-by-side) — ALL INTERCONNECTORS
    # -------------------------------------------------------------------------
    PLOT_DURATION_CURVES_INTERCONNECTORS = True
    DURATION_AGGREGATE_CORRIDORS = True
    DURATION_USE_ABSOLUTE = True
    DURATION_KEEP_HOURS = None
    SAVE_DURATION_PNGS = True
    SHOW_DURATION_PLOTS = True
    
    if PLOT_DURATION_CURVES_INTERCONNECTORS:
        print_block("5) DURATION CURVES — ALL INTERCONNECTORS (BASE vs ALL INSTRUMENTS)")
    
        # List all interconnector corridors from BASE flows (normalized + AT alias)
        interconnectors = list_interconnector_corridors_from_flows(
            base_flows,
            at_alias=at_alias,
            include_country_country=True,
            include_country_to_node=True,
        )
    
        if not interconnectors:
            print("⚠️ No interconnector corridors found.")
        else:
            print(f"✅ Found {len(interconnectors)} interconnector corridors.")
    
        # Corridor capacities (MW) aggregated consistently with normalize_corridor()
        corridor_cap = build_corridor_capacity_map(
            capacity_pos,
            capacity_neg,
            at_alias=at_alias,
        )
    
        out_dir = FIG_DIR / "Duration_Curves" / "Interconnectors"
        if SAVE_DURATION_PNGS:
            out_dir.mkdir(parents=True, exist_ok=True)
    
        for corridor_id in interconnectors:
            base_curve = build_duration_curve(
                base_flows,
                line_id=corridor_id,
                use_absolute=DURATION_USE_ABSOLUTE,
                aggregate_corridors=DURATION_AGGREGATE_CORRIDORS,
                keep_hours=DURATION_KEEP_HOURS,
                at_alias=at_alias,
            )
    
            instr_curves: Dict[str, np.ndarray] = {}
            for instrument in instruments:
                instr_curves[instrument] = build_duration_curve(
                    instr_flows[instrument],
                    line_id=corridor_id,
                    use_absolute=DURATION_USE_ABSOLUTE,
                    aggregate_corridors=DURATION_AGGREGATE_CORRIDORS,
                    keep_hours=DURATION_KEEP_HOURS,
                    at_alias=at_alias,
                )
    
            if base_curve.size == 0 and all(v.size == 0 for v in instr_curves.values()):
                continue
    
            cap_mw = corridor_cap.get(str(corridor_id), None)
    
            out_png = None
            if SAVE_DURATION_PNGS:
                safe_corr = re.sub(r"[^\w\-]+", "_", corridor_id)[:200]
                out_png = out_dir / f"DurationCurve__{safe_corr}__BASE_vs_ALL.png"
    
            plot_duration_multi_instruments_side_by_side(
                base_curve=base_curve,
                instr_curves=instr_curves,
                corridor_id=corridor_id,
                ylabel="|Flow|" if DURATION_USE_ABSOLUTE else "Flow",
                use_absolute=DURATION_USE_ABSOLUTE,
                capacity_mw=cap_mw,          # <-- enables MW + %cap annotation
                out_png=out_png,
                show=SHOW_DURATION_PLOTS,
            )
    
        print_block("DONE")


if __name__ == "__main__":
    main()