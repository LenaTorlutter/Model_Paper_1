from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================
# Uses your Main_New pipeline structure
# ============================================================
from Main_New import (
    CFG,
    INSTRUMENTS,
    build_paths,
    instrument_full_paths,
    instrument_slug,
)

from Parameters_Updated import load_demand_data, load_exchange_data


# ============================================================
# Constants
# ============================================================

HOURS_PER_WEEK = 168
DEFAULT_BASE_DATE = "2030-01-01"
DATE_FMT = "%Y-%m-%d %H:%M"

# Bottleneck criterion: only count if >= 70% of capacity
BOTTLENECK_UTILIZATION = 0.70

# Optional: treat tiny capacities as missing
MIN_CAPACITY_MW = 1e-6


# ============================================================
# Time helpers
# ============================================================

def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build abs_hour from (start_week, timestep/t/index_1) if needed.

    Robust to 0-based or 1-based timestep conventions:
      - if min(timestep)==0 -> abs_hour = timestep + 1 + 168*(start_week-1)
      - else                -> abs_hour = timestep + 168*(start_week-1)
    """
    if df is None or df.empty:
        return df
    if "abs_hour" in df.columns:
        return df

    df = df.copy()

    # start_week source (also handle rolling-horizon 'window')
    if "start_week" not in df.columns:
        if "window" in df.columns:
            df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1
        else:
            return df

    # timestep source
    if "timestep" in df.columns:
        tcol = "timestep"
    elif "t" in df.columns:
        tcol = "t"
    elif "index_1" in df.columns:
        df["timestep"] = pd.to_numeric(df["index_1"], errors="coerce").fillna(0).astype(int)
        tcol = "timestep"
    else:
        return df

    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(1).astype(int)
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype(int)

    # 0-based vs 1-based detection
    tmin = int(df[tcol].min()) if len(df) else 0
    offset = 1 if tmin == 0 else 0

    df["abs_hour"] = df[tcol] + offset + HOURS_PER_WEEK * (df["start_week"] - 1)
    return df


def read_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    # normalize schema (same logic as your plotting script)
    if "component" in df.columns and "variable" not in df.columns:
        df = df.rename(columns={"component": "variable"})

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "start_week" not in df.columns and "window" in df.columns:
        df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1

    df = ensure_abs_hour(df)
    return df


# ============================================================
# Extractors
# ============================================================

def extract_flows_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract flows in long form:
      connection, abs_hour, flow
    """
    var = df.get("variable", pd.Series([""] * len(df))).astype(str)
    flows = df[var == "flow"].copy()
    if flows.empty:
        return pd.DataFrame(columns=["connection", "abs_hour", "flow"])

    # connection id
    if "index_0" in flows.columns:
        flows["connection"] = flows["index_0"].astype(str)
    elif "connection" in flows.columns:
        flows["connection"] = flows["connection"].astype(str)
    else:
        # last resort: try 'p'
        flows["connection"] = flows.get("p", "").astype(str)

    flows = ensure_abs_hour(flows)

    flows["abs_hour"] = pd.to_numeric(flows.get("abs_hour", np.nan), errors="coerce")
    flows = flows.dropna(subset=["abs_hour"]).copy()
    flows["abs_hour"] = flows["abs_hour"].astype(int)

    flows["flow"] = pd.to_numeric(flows.get("value", 0.0), errors="coerce").fillna(0.0)
    return flows[["connection", "abs_hour", "flow"]]


# ============================================================
# Capacity normalization
# ============================================================

def _to_conn_series(obj, name: str) -> pd.Series:
    """
    Normalize capacity_pos/capacity_neg into a Series indexed by connection.
    """
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


def build_capacity_frame(capacity_pos, capacity_neg) -> pd.DataFrame:
    """
    Return DataFrame:
      connection, capacity_abs
    where capacity_abs = max(|cap_pos|, |cap_neg|)
    """
    cap_pos = _to_conn_series(capacity_pos, "capacity_pos")
    cap_neg = _to_conn_series(capacity_neg, "capacity_neg")

    cap_df = pd.concat([cap_pos, cap_neg], axis=1)
    cap_df["capacity_pos"] = cap_df["capacity_pos"].fillna(0.0).abs()
    cap_df["capacity_neg"] = cap_df["capacity_neg"].fillna(0.0).abs()
    cap_df["capacity_abs"] = cap_df[["capacity_pos", "capacity_neg"]].max(axis=1)

    cap_df = cap_df.reset_index().rename(columns={"index": "connection"})
    cap_df["connection"] = cap_df["connection"].astype(str)

    return cap_df[["connection", "capacity_abs"]]


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


# ============================================================
# Bottleneck counting logic
# ============================================================

def bottleneck_counts_all_lines(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    capacity_pos,
    capacity_neg,
    *,
    utilization_threshold: float = BOTTLENECK_UTILIZATION,
    min_capacity: float = MIN_CAPACITY_MW,
    aggregate_corridors: bool = False,
) -> pd.DataFrame:
    """
    Count bottlenecks across ALL lines over the full horizon.

    Bottleneck definition:
        |flow[c,t]| >= utilization_threshold * capacity_abs[c]

    Returns per line (or per corridor if aggregate_corridors=True):
      line_id, n_bottlenecks_base, n_bottlenecks_instr, delta_bottlenecks
    """
    caps = build_capacity_frame(capacity_pos, capacity_neg)
    caps["capacity_abs"] = pd.to_numeric(caps["capacity_abs"], errors="coerce").fillna(0.0)

    flows_base = extract_flows_long(stage1_base).rename(columns={"flow": "flow_base"})
    flows_instr = extract_flows_long(stage1_instr).rename(columns={"flow": "flow_instr"})

    flows = flows_base.merge(
        flows_instr,
        on=["connection", "abs_hour"],
        how="outer",
    )

    flows["connection"] = flows["connection"].astype(str)
    flows["flow_base"] = pd.to_numeric(flows.get("flow_base", 0.0), errors="coerce").fillna(0.0)
    flows["flow_instr"] = pd.to_numeric(flows.get("flow_instr", 0.0), errors="coerce").fillna(0.0)

    # attach capacities
    flows = flows.merge(caps, on="connection", how="left")
    flows["capacity_abs"] = pd.to_numeric(flows.get("capacity_abs", 0.0), errors="coerce").fillna(0.0)

    # drop lines with no/zero capacity
    flows = flows[flows["capacity_abs"] > float(min_capacity)].copy()
    if flows.empty:
        return pd.DataFrame(columns=["line_id", "n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"])

    thr = float(utilization_threshold)
    flows["is_bneck_base"] = flows["flow_base"].abs() >= (thr * flows["capacity_abs"])
    flows["is_bneck_instr"] = flows["flow_instr"].abs() >= (thr * flows["capacity_abs"])

    if aggregate_corridors:
        flows["line_id"] = flows["connection"].map(normalize_corridor)
    else:
        flows["line_id"] = flows["connection"]

    stats = (
        flows.groupby("line_id", as_index=False)
        .agg(
            n_bottlenecks_base=("is_bneck_base", "sum"),
            n_bottlenecks_instr=("is_bneck_instr", "sum"),
        )
    )

    stats["n_bottlenecks_base"] = stats["n_bottlenecks_base"].astype(int)
    stats["n_bottlenecks_instr"] = stats["n_bottlenecks_instr"].astype(int)
    stats["delta_bottlenecks"] = stats["n_bottlenecks_base"] - stats["n_bottlenecks_instr"]

    # sort: biggest decrease first (positive delta)
    stats = stats.sort_values(
        ["delta_bottlenecks", "line_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return stats[["line_id", "n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"]]

# ============================================================
# Export (Tableau-friendly)
# ============================================================

def write_bottleneck_analysis_csv(out_df: pd.DataFrame, out_csv: str) -> None:
    """
    Tableau-friendly export (semicolon-separated).
    """
    export = out_df.copy()
    export["line_id"] = export["line_id"].astype(str)
    for c in ["n_bottlenecks_base", "n_bottlenecks_instr", "delta_bottlenecks"]:
        export[c] = pd.to_numeric(export[c], errors="coerce").fillna(0).astype(int)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    export.to_csv(out_csv, index=False, sep=";", encoding="utf-8-sig")
    print(f"✅ Wrote: {out_csv} (rows={len(export)})")


# ============================================================
# Main
# ============================================================

def main():
    # --- switches ---
    AGGREGATE_CORRIDORS = True  # True -> AT-DE instead of AT1-DE1, AT2-DE2, ...
    UTIL = BOTTLENECK_UTILIZATION  # 0.70

    paths = build_paths(CFG)

    # --- load mapping + exchange data (for capacities/incidence) ---
    # node_to_control_area not strictly needed for bottleneck counting, but keep it consistent with your pipeline
    _ = load_demand_data(paths.data_path, "Demand_Profiles")

    base_dir = os.path.dirname(paths.data_path)
    ptdf_csv_path = os.path.join(base_dir, "PTDF_Synchronized.csv")

    exch = load_exchange_data(
        paths.data_path,
        "Exchange_Data",
        ptdf_csv_path=ptdf_csv_path,
        slack_node=None,
        verbose=False,
    )

    # expected tuple: _, connections, incidence_matrix, capacity_pos, capacity_neg, xborder, ..., ...
    capacity_pos = exch[3]
    capacity_neg = exch[4]

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

        out = bottleneck_counts_all_lines(
            stage1_base=stage1_base,
            stage1_instr=stage1_instr,
            capacity_pos=capacity_pos,
            capacity_neg=capacity_neg,
            utilization_threshold=UTIL,
            aggregate_corridors=AGGREGATE_CORRIDORS,
        )

        if out.empty:
            print(f"⚠️ Empty bottleneck output for {instrument}. (No flows or no capacities?)")
            continue

        slug = instrument_slug(instrument)
        out_path = os.path.join(
            paths.base_dir,
            f"Bottleneck_Analysis__{slug}.csv"
        )
        write_bottleneck_analysis_csv(out, out_path)

if __name__ == "__main__":
    main()
