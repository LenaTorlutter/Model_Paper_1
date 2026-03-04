from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Reuse Main_New pipeline paths + instrument naming
# -----------------------------
from Main_New import (
    CFG,
    INSTRUMENTS,
    build_paths,
    instrument_full_paths,
    instrument_slug,
)

from Parameters_Updated import (
    load_demand_data,
    load_exchange_data,
)

# -----------------------------
# Constants
# -----------------------------
HOURS_PER_WEEK = 168
DEFAULT_BASE_DATE = "2030-01-01"
DATE_FMT = "%Y-%m-%d %H:%M"


# ============================================================
# Time helpers
# ============================================================

def parse_user_datetime(dt: str) -> pd.Timestamp:
    return pd.to_datetime(dt, format=DATE_FMT)


def date_to_abs_hour(dt: str, base_date: str = DEFAULT_BASE_DATE) -> int:
    """
    abs_hour = 1 at base_date 00:00
    abs_hour = 2 at base_date 01:00
    ...
    """
    t0 = pd.to_datetime(str(base_date).strip() + " 00:00", format=DATE_FMT)
    t = parse_user_datetime(dt)
    return int((t - t0).total_seconds() // 3600) + 1


def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build abs_hour from (start_week, timestep/t) if needed.
    """
    if "abs_hour" in df.columns:
        return df
    if "start_week" not in df.columns:
        return df

    if "timestep" in df.columns:
        time_col = "timestep"
    elif "t" in df.columns:
        time_col = "t"
    else:
        return df

    df = df.copy()
    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(0).astype(int)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(int)
    df["abs_hour"] = df[time_col] + HOURS_PER_WEEK * (df["start_week"] - 1)
    return df


def read_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "component" in df.columns and "variable" not in df.columns:
        df = df.rename(columns={"component": "variable"})

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "start_week" not in df.columns and "window" in df.columns:
        df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1

    df = ensure_abs_hour(df)
    return df


# ============================================================
# Flow extraction
# ============================================================

def extract_flows_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract line flows in long form with (connection, abs_hour, flow).
    """
    flows = df[df["variable"] == "flow"].copy()
    if flows.empty:
        return pd.DataFrame(columns=["connection", "abs_hour", "flow"])

    if {"index_0", "index_1"}.issubset(flows.columns):
        flows["connection"] = flows["index_0"].astype(str)
        flows["timestep"] = pd.to_numeric(flows["index_1"], errors="coerce").fillna(0).astype(int)
        flows["start_week"] = pd.to_numeric(flows.get("start_week", 1), errors="coerce").fillna(1).astype(int)
        flows = ensure_abs_hour(flows)
    else:
        flows["connection"] = flows.get("p", flows.columns[0]).astype(str)
        flows["abs_hour"] = pd.to_numeric(flows.get("abs_hour", np.nan), errors="coerce")
        # if abs_hour is missing here, try reconstructing via start_week + t
        flows = ensure_abs_hour(flows)

    flows = flows.rename(columns={"value": "flow"})
    flows["flow"] = pd.to_numeric(flows["flow"], errors="coerce").fillna(0.0)
    flows["abs_hour"] = pd.to_numeric(flows["abs_hour"], errors="coerce").fillna(-1).astype(int)

    return flows[["connection", "abs_hour", "flow"]]


# ============================================================
# Incidence -> endpoints metadata for ALL lines
# ============================================================

def melt_incidence_to_long(incidence_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    incidence_matrix: rows = nodes, cols = connections
    Returns (node, connection, coeff) only for non-zero entries.
    """
    long_map = (
        incidence_matrix.stack()
        .rename("coeff")
        .reset_index()
        .rename(columns={"level_0": "node", "level_1": "connection"})
    )
    long_map["node"] = long_map["node"].astype(str)
    long_map["connection"] = long_map["connection"].astype(str)
    long_map["coeff"] = pd.to_numeric(long_map["coeff"], errors="coerce").fillna(0.0)
    return long_map[long_map["coeff"] != 0.0]


def build_connection_metadata(
    incidence_matrix: pd.DataFrame,
    node_to_control_area: Dict[str, str],
) -> pd.DataFrame:
    """
    Build one metadata row per transmission line (connection)
    derived from the incidence matrix.

    Returns columns:
      - connection
      - node_from, node_to
      - ca_from, ca_to
      - relation_type  ('INTERNAL' or 'BORDER')
      - ca_pair        (directionless, e.g. 'AT-DE')
    """

    # Long form: one row per (node, connection) with non-zero incidence
    long_map = melt_incidence_to_long(incidence_matrix).copy()
    long_map["node"] = long_map["node"].astype(str)
    long_map["connection"] = long_map["connection"].astype(str)
    long_map["coeff"] = pd.to_numeric(long_map["coeff"], errors="coerce").fillna(0.0)

    # Attach control areas
    long_map["control_area"] = long_map["node"].map(
        lambda n: str(node_to_control_area.get(n, "UNKNOWN"))
    )

    rows = []

    # Process each line exactly once
    for conn, g in long_map.groupby("connection", sort=False):
        # Standard DC incidence: one negative, one positive
        neg = g[g["coeff"] < 0]
        pos = g[g["coeff"] > 0]

        if not neg.empty and not pos.empty:
            node_from = neg["node"].iloc[0]
            node_to   = pos["node"].iloc[0]
            ca_from   = neg["control_area"].iloc[0]
            ca_to     = pos["control_area"].iloc[0]
        else:
            # Fallback for irregular data (transformers, malformed lines)
            g2 = g.sort_values("coeff")
            nodes = g2["node"].tolist()
            cas   = g2["control_area"].tolist()

            node_from = nodes[0] if len(nodes) > 0 else None
            node_to   = nodes[1] if len(nodes) > 1 else None
            ca_from   = cas[0] if len(cas) > 0 else "UNKNOWN"
            ca_to     = cas[1] if len(cas) > 1 else "UNKNOWN"

        relation_type = "INTERNAL" if ca_from == ca_to else "BORDER"
        ca_pair = "-".join(sorted([ca_from, ca_to]))

        rows.append(
            {
                "connection": conn,
                "node_from": node_from,
                "node_to": node_to,
                "ca_from": ca_from,
                "ca_to": ca_to,
                "relation_type": relation_type,
                "ca_pair": ca_pair,
            }
        )

    meta = pd.DataFrame(rows)

    # Hard safety: guarantee uniqueness and clean schema
    meta = meta.drop_duplicates(subset=["connection"], keep="first").reset_index(drop=True)
    meta = meta.loc[:, ~meta.columns.duplicated()].copy()

    return meta



# ============================================================
# Optional corridor aggregation (AT1-DE1 + AT2-DE2 -> AT-DE)
# ============================================================

_LEADING_ALPHA = re.compile(r"^([A-Za-z]+)")

def _corridor_endpoint(token: str) -> str:
    token = str(token).strip()
    m = _LEADING_ALPHA.match(token)
    return m.group(1).upper() if m else token.upper()

def normalize_corridor(connection: str) -> str:
    parts = str(connection).split("-")
    if len(parts) != 2:
        return str(connection)
    a = _corridor_endpoint(parts[0])
    b = _corridor_endpoint(parts[1])
    a, b = sorted([a, b])
    return f"{a}-{b}"

def _mode_or_first(s: pd.Series):
    s = s.dropna().astype(str)
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


# ============================================================
# Core comparison for ALL lines
# ============================================================

def compare_all_line_flows(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    node_to_control_area: Dict[str, str],
    DATE_TIME: str,
    base_date: str = DEFAULT_BASE_DATE,
    *,
    aggregate_corridors: bool = False,
) -> pd.DataFrame:
    """
    Compare flows for ALL connections at one datetime.

    Output includes:
      connection, abs_hour, datetime,
      flow_stage1_base, flow_stage1_instr,
      abs_stage1_base, abs_stage1_instr,
      abs_diff = |base| - |instr|,
      change,
      node_from/node_to, ca_from/ca_to, relation_type, ca_pair

    If aggregate_corridors=True:
      - group by corridor key (AT-DE) and sum |flow| across parallel ties
      - recompute abs_diff at corridor level
      - keep metadata as best-effort (mode for categorical fields)
    """
    abs_hour = date_to_abs_hour(DATE_TIME, base_date=base_date)

    f_base = extract_flows_long(stage1_base)
    f_instr = extract_flows_long(stage1_instr)

    f_base = f_base[f_base["abs_hour"] == abs_hour].rename(columns={"flow": "flow_stage1_base"})
    f_instr = f_instr[f_instr["abs_hour"] == abs_hour].rename(columns={"flow": "flow_stage1_instr"})

    comp = pd.merge(
        f_base,
        f_instr,
        on=["connection", "abs_hour"],
        how="outer",
    )

    comp["datetime"] = DATE_TIME
    comp["flow_stage1_base"] = pd.to_numeric(comp.get("flow_stage1_base", 0.0), errors="coerce").fillna(0.0)
    comp["flow_stage1_instr"] = pd.to_numeric(comp.get("flow_stage1_instr", 0.0), errors="coerce").fillna(0.0)

    comp["abs_stage1_base"] = comp["flow_stage1_base"].abs()
    comp["abs_stage1_instr"] = comp["flow_stage1_instr"].abs()
    comp["abs_diff"] = comp["abs_stage1_base"] - comp["abs_stage1_instr"]
    comp["change"] = np.where(
        comp["abs_diff"] > 0, "DECREASE",
        np.where(comp["abs_diff"] < 0, "INCREASE", "NO CHANGE")
    )

    # attach endpoints + CA metadata
    meta = build_connection_metadata(incidence_matrix, node_to_control_area)
    comp = comp.merge(meta, on="connection", how="left")

    # Optional corridor aggregation
    if aggregate_corridors:
        comp["corridor"] = comp["connection"].map(normalize_corridor)

        agg = (
            comp.groupby(["corridor"], as_index=False)
            .agg(
                abs_stage1_base=("abs_stage1_base", "sum"),
                abs_stage1_instr=("abs_stage1_instr", "sum"),
                # keep some useful metadata in aggregated view
                relation_type=("relation_type", _mode_or_first),
                ca_pair=("ca_pair", _mode_or_first),
                ca_from=("ca_from", _mode_or_first),
                ca_to=("ca_to", _mode_or_first),
                node_from=("node_from", _mode_or_first),
                node_to=("node_to", _mode_or_first),
            )
        )

        agg["abs_diff"] = agg["abs_stage1_base"] - agg["abs_stage1_instr"]
        agg["change"] = np.where(
            agg["abs_diff"] > 0, "DECREASE",
            np.where(agg["abs_diff"] < 0, "INCREASE", "NO CHANGE")
        )
        agg["abs_hour"] = abs_hour
        agg["datetime"] = DATE_TIME

        # rename corridor -> connection so downstream export stays unchanged
        comp = agg.rename(columns={"corridor": "connection"}).copy()

    # stable sort: largest reductions first
    comp = comp.sort_values(["abs_diff", "connection"], ascending=[False, True]).reset_index(drop=True)
    return comp


# ============================================================
# Export (Tableau-friendly)
# ============================================================

def write_flow_analysis_csv(out_df: pd.DataFrame, out_csv: str) -> None:
    """
    Tableau-safe export:
      - only: line_id + abs_diff
      - numeric-safe formatting
    """
    if "line_id" not in out_df.columns:
        raise ValueError("Expected column 'line_id' not found in output DataFrame.")

    export = out_df[["line_id", "abs_diff"]].copy()
    export["abs_diff"] = pd.to_numeric(export["abs_diff"], errors="coerce").fillna(0.0)

    export.to_csv(
        out_csv,
        index=False,
        sep=";",
        decimal=".",
        float_format="%.6f",
        encoding="utf-8-sig",
    )


# ============================================================
# Main: run BASE vs ALL instruments
# ============================================================

def main():
    # --- config switches ---
    DATE_TIME = "2030-07-01 13:00"
    AGGREGATE_CORRIDORS = True  # True -> AT-DE instead of AT1-DE1, AT2-DE2, ...

    paths = build_paths(CFG)

    # --- load mapping + incidence (same data as Main_New) ---
    demand_data = load_demand_data(paths.data_path, "Demand_Profiles")
    try:
        _, _, _, _, node_to_control_area = demand_data
    except Exception:
        node_to_control_area = demand_data[-1]

    base_dir = os.path.dirname(paths.data_path)
    ptdf_csv_path = os.path.join(base_dir, "PTDF_Synchronized.csv")

    exch = load_exchange_data(
        paths.data_path,
        "Exchange_Data",
        ptdf_csv_path=ptdf_csv_path,
        slack_node=None,
        verbose=False,
    )
    incidence_matrix = exch[2]

    # --- read BASE once ---
    if not os.path.exists(paths.full_stage1_base):
        raise FileNotFoundError(f"Missing BASE results: {paths.full_stage1_base}")
    stage1_base = read_results(paths.full_stage1_base)

    for instrument in INSTRUMENTS:
        _, stage1_instr_path = instrument_full_paths(paths, instrument)
        if not os.path.exists(stage1_instr_path):
            print(f"⚠️ Missing instrument results file, skipping: {stage1_instr_path}")
            continue

        stage1_instr = read_results(stage1_instr_path)

        comp = compare_all_line_flows(
            stage1_base=stage1_base,
            stage1_instr=stage1_instr,
            incidence_matrix=incidence_matrix,
            node_to_control_area=node_to_control_area,
            DATE_TIME=DATE_TIME,
            aggregate_corridors=AGGREGATE_CORRIDORS,
        )

        if comp.empty:
            print(f"⚠️ Empty comparison for {instrument} at {DATE_TIME} (no flow rows). Skipping export.")
            continue

        # ✅ rename for output
        out_df = comp.rename(columns={"connection": "line_id"}).copy()

        slug = instrument_slug(instrument)
        out_path = os.path.join(paths.base_dir, f"Flow_Analysis__{slug}.csv")
        write_flow_analysis_csv(out_df, out_path)  # <-- write renamed df
        print(f"✅ Wrote {out_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()

