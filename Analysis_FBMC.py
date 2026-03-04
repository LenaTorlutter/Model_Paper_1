from __future__ import annotations
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths and constants 
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
STAGE0_PATH = os.path.join(BASE_DIR, "full_model_results_stage0_test.csv")
STAGE1_PATH_BASE = os.path.join(BASE_DIR, "full_model_results_stage1_base_test.csv")
STAGE1_PATH_INSTRUMENT = os.path.join(BASE_DIR, "full_model_results_stage1_instrument_test.csv")
FULL_PRICE_FOLLOWER_PATH_BASE = os.path.join(BASE_DIR, 'full_price_follower_values_base_test.csv')
FULL_PRICE_FOLLOWER_PATH_INSTRUMENT = os.path.join(BASE_DIR, 'full_price_follower_values_instrument_test.csv')
DATA_XLSX = os.path.join(BASE_DIR, "Data.xlsx")

HOURS_PER_WEEK = 168
DEFAULT_BASE_DATE = "2030-01-01"

RESULTS_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Data loaders from your parameters module
# ============================================================

from Parameters import (
    load_general_data,
    load_demand_data,
    load_thermal_power_plant_data,
    load_phs_power_plant_data,
    load_phs_inflow_data,
    load_phs_storage_profile_data,
    load_renewable_power_plant_data,
    load_res_profile_data,
    load_flexibility_data,
    load_exchange_data,
)

# ============================================================
# Small utilities
# ============================================================

def to_float(x) -> float:
    import numpy as _np
    import pandas as _pd
    if isinstance(x, _pd.DataFrame):
        return float(_np.asarray(x.values, dtype="object").reshape(-1)[0])
    if isinstance(x, (_pd.Series, list, tuple, _np.ndarray)):
        return float(_np.asarray(x, dtype="object").reshape(-1)[0])
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)

# ============================================================
# Date parsing helpers
# ============================================================

BASE_DATE_FMT = "%Y-%m-%d" 
DATE_FMT = "%Y-%m-%d %H:%M"

def coerce_base_date(base_date: str) -> str:
    """
    Very simple: expect 'YYYY-MM-DD'. Strip whitespace and return.
    No flexible parsing, no guessing.
    """
    return str(base_date).strip()


def parse_user_datetime(dt: str, base_date: str = DEFAULT_BASE_DATE) -> pd.Timestamp:
    """
    Parse a user-provided datetime string.

    Only accepted format (strict):
        'YYYY-MM-DD HH:MM'
    Example:
        '2030-01-15 12:00'

    The 'base_date' parameter is kept only for API compatibility
    and is not used here.
    """
    return pd.to_datetime(dt, format=DATE_FMT)


def parse_user_range(
    start: str | None,
    end: str | None,
    base_date: str = DEFAULT_BASE_DATE,
) -> tuple[str, str] | None:
    """
    Parse a user-provided [start, end] datetime range.

    Both 'start' and 'end' must be either:
        - both None   -> return None
        - both strings in 'YYYY-MM-DD HH:MM' format

    Returns normalized strings in the same 'YYYY-MM-DD HH:MM' format.
    """
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("Both start and end must be provided when specifying a range.")

    ts0 = parse_user_datetime(start)
    ts1 = parse_user_datetime(end)

    if ts1 < ts0:
        raise ValueError("End datetime must be >= start datetime.")

    return ts0.strftime(DATE_FMT), ts1.strftime(DATE_FMT)


def date_to_abs_hour(ts: str, base_date: str = DEFAULT_BASE_DATE) -> int:
    """
    Map a datetime string 'YYYY-MM-DD HH:MM' to an absolute model hour (1-based)
    relative to a base date 'YYYY-MM-DD'.

    Definition:
        abs_hour = 1 at base_date 00:00
        abs_hour = 2 at base_date 01:00
        ...

    Assumes that each timestep corresponds to exactly one real hour.
    """
    base_iso = coerce_base_date(base_date)
    # Base timestamp at midnight of base_date
    t0 = pd.to_datetime(base_iso + " 00:00", format=DATE_FMT)
    t = parse_user_datetime(ts, base_date)
    return int((t - t0).total_seconds() // 3600) + 1


def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "abs_hour" in df.columns:
        return df

    if "start_week" not in df.columns:
        return df

    # Accept ANY time column: "timestep" (stage) or "t" (PF)
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

# ============================================================
# Reading and shaping input CSVs
# ============================================================

def read_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "start_week" in df.columns:
        df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = ensure_abs_hour(df)
    return df

def extract_dual_prices_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract nodal dual prices in long form with (start_week, node, timestep, abs_hour).

    Returns columns:
        start_week, node, timestep, abs_hour, dual_price
    """
    duals = df[df["variable"] == "dual:demand_equilibrium_constraint"].copy()
    if duals.empty:
        return pd.DataFrame(columns=["start_week", "node", "timestep", "abs_hour", "dual_price"])

    # start_week
    duals["start_week"] = pd.to_numeric(duals.get("start_week", 1), errors="coerce").fillna(1).astype(int)

    # timestep (model time index)
    if {"index_0", "index_1"}.issubset(duals.columns):
        duals["node"] = duals["index_0"].astype(str)
        duals["timestep"] = pd.to_numeric(duals["index_1"], errors="coerce").astype(int)
    else:
        duals["node"] = duals.get("p", duals.columns[0]).astype(str)
        duals["timestep"] = pd.to_numeric(duals.get("t", 1), errors="coerce").astype(int)

    duals = duals[["start_week", "node", "timestep", "value"]].rename(columns={"value": "dual_price"})
    duals["dual_price"] = -pd.to_numeric(duals["dual_price"], errors="coerce").fillna(0.0)

    # add abs_hour from (start_week, timestep)
    duals = ensure_abs_hour(duals)

    return duals[["start_week", "node", "timestep", "abs_hour", "dual_price"]]


def extract_flows_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract line flows in long form with (start_week, connection, timestep, abs_hour).

    Returns columns:
        start_week, connection, timestep, abs_hour, flow
    """
    flows = df[df["variable"] == "flow"].copy()
    if flows.empty:
        return pd.DataFrame(columns=["start_week", "connection", "timestep", "abs_hour", "flow"])

    # start_week
    flows["start_week"] = pd.to_numeric(flows.get("start_week", 1), errors="coerce").fillna(1).astype(int)

    # timestep (model time index)
    if {"index_0", "index_1"}.issubset(flows.columns):
        flows["connection"] = flows["index_0"].astype(str)
        flows["timestep"] = pd.to_numeric(flows["index_1"], errors="coerce").astype(int)
    else:
        flows["connection"] = flows.get("p", flows.columns[0]).astype(str)
        flows["timestep"] = pd.to_numeric(flows.get("t", 1), errors="coerce").astype(int)

    flows = flows[["start_week", "connection", "timestep", "value"]].rename(columns={"value": "flow"})
    flows["flow"] = pd.to_numeric(flows["flow"], errors="coerce").fillna(0.0)

    # add abs_hour from (start_week, timestep)
    flows = ensure_abs_hour(flows)  # uses HOURS_PER_WEEK

    return flows[["start_week", "connection", "timestep", "abs_hour", "flow"]]


def melt_incidence_to_long(incidence_matrix: pd.DataFrame) -> pd.DataFrame:
    long_map = (
        incidence_matrix.stack()
        .rename("coeff")
        .reset_index()
        .rename(columns={"level_0": "node", "level_1": "connection"})
    )
    long_map["node"] = long_map["node"].astype(str)
    long_map["connection"] = long_map["connection"].astype(str)
    long_map["coeff"] = pd.to_numeric(long_map["coeff"], errors="coerce").fillna(0.0)
    long_map = long_map[long_map["coeff"] != 0.0]
    return long_map

# ============================================================
# Flow comparison
# ============================================================

def compute_control_area_flows(
    stage_df: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    node_to_control_area: Dict[str, str],
    control_area: str,
    DATE_TIME: str,
    base_date: str = DEFAULT_BASE_DATE,
) -> pd.DataFrame:
    """
    For a given stage (model result DF), incidence matrix and node→control_area map,
    compute all line flows that *touch* `control_area`
    (internal + border lines) at the given datetime 'YYYY-MM-DD HH:MM'.

    Returns one row per connection with:
       connection, datetime, abs_hour,
       flow, node_ca, node_other,
       other_control_area,
       relation_type  ("INTERNAL" if all endpoints in control_area,
                       "BORDER" otherwise),
       sign_for_ca    (incidence coeff for the chosen node in control_area, if unique)
    """
    abs_hour = date_to_abs_hour(DATE_TIME, base_date=base_date)

    flows_all = extract_flows_long(stage_df)
    flows = flows_all[flows_all["abs_hour"] == abs_hour].copy()
    if flows.empty:
        return pd.DataFrame(columns=[
            "connection", "datetime", "abs_hour", "flow",
            "node_ca", "node_other", "other_control_area",
            "relation_type", "sign_for_ca"
        ])

    long_map = melt_incidence_to_long(incidence_matrix)

    merged = flows.merge(long_map, on="connection", how="inner")
    merged["node"] = merged["node"].astype(str)
    merged["control_area"] = merged["node"].map(lambda n: node_to_control_area.get(n, "UNKNOWN"))

    ca_sets = (
        merged.groupby("connection")["control_area"]
        .agg(lambda s: set(s))
        .reset_index()
        .rename(columns={"control_area": "ca_set"})
    )

    def touches_target(s: set) -> bool:
        return control_area in s

    touching_conns = ca_sets[ca_sets["ca_set"].apply(touches_target)]
    touching = merged[merged["connection"].isin(touching_conns["connection"])].copy()

    if touching.empty:
        return pd.DataFrame(columns=[
            "connection", "datetime", "abs_hour", "flow",
            "node_ca", "node_other", "other_control_area",
            "relation_type", "sign_for_ca"
        ])

    ca_set_map = touching_conns.set_index("connection")["ca_set"].to_dict()

    def agg_connection(g: pd.DataFrame) -> pd.Series:
        flow_val = g["flow"].iloc[0]
        conn = g["connection"].iloc[0]
        ca_set = ca_set_map.get(conn, set())

        if ca_set == {control_area}:
            relation_type = "INTERNAL"
        else:
            relation_type = "BORDER"

        g_ca = g[g["control_area"] == control_area]
        node_ca = g_ca["node"].iloc[0] if not g_ca.empty else None

        if len(g_ca) == 1:
            sign_for_ca = g_ca["coeff"].iloc[0]
        else:
            sign_for_ca = np.nan

        g_other = g[g["node"] != node_ca]
        if not g_other.empty:
            node_other = g_other["node"].iloc[0]
            other_ca = g_other["control_area"].iloc[0]
        else:
            node_other = None
            other_ca = control_area

        return pd.Series({
            "flow": flow_val,
            "node_ca": node_ca,
            "node_other": node_other,
            "other_control_area": other_ca,
            "relation_type": relation_type,
            "sign_for_ca": sign_for_ca,
        })

    ca_agg = (
        touching
        .groupby("connection", as_index=False)
        .apply(agg_connection, include_groups=True)
        .reset_index(drop=True)
    )

    ca_agg["datetime"] = DATE_TIME
    ca_agg["abs_hour"] = abs_hour

    return ca_agg[
        ["connection", "datetime", "abs_hour",
         "flow", "node_ca", "node_other",
         "other_control_area", "relation_type",
         "sign_for_ca"]
    ]


def compare_control_area_flows(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    node_to_control_area: Dict[str, str],
    control_area: str,
    DATE_TIME: str,
    base_date: str = DEFAULT_BASE_DATE,
) -> pd.DataFrame:

    abs_hour = date_to_abs_hour(DATE_TIME, base_date=base_date)

    f_base = compute_control_area_flows(
        stage_df=stage1_base,
        incidence_matrix=incidence_matrix,
        node_to_control_area=node_to_control_area,
        control_area=control_area,
        DATE_TIME=DATE_TIME,
        base_date=base_date,
    ).rename(columns={"flow": "flow_stage1_base"})

    f_instr = compute_control_area_flows(
        stage_df=stage1_instr,
        incidence_matrix=incidence_matrix,
        node_to_control_area=node_to_control_area,
        control_area=control_area,
        DATE_TIME=DATE_TIME,
        base_date=base_date,
    ).rename(columns={"flow": "flow_stage1_instr"})

    comp = pd.merge(
        f_base,
        f_instr,
        on=["connection", "datetime", "abs_hour"],
        how="outer",
        suffixes=("_base", "_instr")
    )

    def coalesce(col1: str, col2: str):
        if col1 not in comp.columns:
            comp[col1] = np.nan
        if col2 not in comp.columns:
            comp[col2] = np.nan
        return np.where(comp[col1].notna(), comp[col1], comp[col2])

    comp["node_ca"] = coalesce("node_ca_base", "node_ca_instr")
    comp["node_other"] = coalesce("node_other_base", "node_other_instr")
    comp["other_control_area"] = coalesce("other_control_area_base", "other_control_area_instr")
    comp["relation_type"] = coalesce("relation_type_base", "relation_type_instr")

    comp["flow_stage1_base"] = pd.to_numeric(comp.get("flow_stage1_base", 0.0), errors="coerce").fillna(0.0)
    comp["flow_stage1_instr"] = pd.to_numeric(comp.get("flow_stage1_instr", 0.0), errors="coerce").fillna(0.0)

    comp["abs_stage1_base"] = comp["flow_stage1_base"].abs()
    comp["abs_stage1_instr"] = comp["flow_stage1_instr"].abs()
    comp["abs_diff"] = comp["abs_stage1_base"] - comp["abs_stage1_instr"]
    comp["change"] = np.where(
        comp["abs_diff"] > 0, "DECREASE",
        np.where(comp["abs_diff"] < 0, "INCREASE", "NO CHANGE")
    )

    duals_base = extract_dual_prices_long(stage1_base)
    duals_base_tw = duals_base[duals_base["abs_hour"] == abs_hour]
    dual_map_base = duals_base_tw.set_index("node")["dual_price"].to_dict()

    duals_instr = extract_dual_prices_long(stage1_instr)
    duals_instr_tw = duals_instr[duals_instr["abs_hour"] == abs_hour]
    dual_map_instr = duals_instr_tw.set_index("node")["dual_price"].to_dict()

    comp["dual_node_ca_stage1_base"] = comp["node_ca"].map(lambda n: dual_map_base.get(str(n), np.nan))
    comp["dual_node_other_stage1_base"] = comp["node_other"].map(lambda n: dual_map_base.get(str(n), np.nan))

    comp["dual_node_ca_stage1_instr"] = comp["node_ca"].map(lambda n: dual_map_instr.get(str(n), np.nan))
    comp["dual_node_other_stage1_instr"] = comp["node_other"].map(lambda n: dual_map_instr.get(str(n), np.nan))

    comp["relation_type"] = pd.Categorical(
        comp["relation_type"],
        categories=["INTERNAL", "BORDER"],
        ordered=True
    )

    cols_out = [
        "connection",
        "datetime", "abs_hour",
        "node_ca", "node_other",
        "other_control_area", "relation_type",
        "flow_stage1_base", "flow_stage1_instr",
        "abs_stage1_base", "abs_stage1_instr", "abs_diff", "change",
        "dual_node_ca_stage1_base",    "dual_node_ca_stage1_instr",
        "dual_node_other_stage1_base", "dual_node_other_stage1_instr",
    ]

    comp = comp[cols_out].sort_values(
        by=["relation_type", "abs_diff"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # ★★★ TERMINAL OUTPUT ★★★
    comp_print = comp.copy()

    # no formatting, just keep plain text
    internal = comp_print[comp_print["relation_type"] == "INTERNAL"]
    border = comp_print[comp_print["relation_type"] == "BORDER"]

    print("\n================ CONTROL AREA FLOW COMPARISON ================")
    print(f"Control Area: {control_area}")
    print(f"Datetime:     {DATE_TIME}   (abs_hour={abs_hour})")
    print("==============================================================\n")

    print(">>> INTERNAL FLOWS:")
    if internal.empty:
        print("(none)\n")
    else:
        print(internal.to_string(index=False))
        print()

    print(">>> BORDER FLOWS:")
    if border.empty:
        print("(none)\n")
    else:
        print(border.to_string(index=False))
        print()

    print("==============================================================\n")

    return comp

def find_binding_flows_only_in_one_case(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    capacity_pos: pd.Series | pd.DataFrame,
    capacity_neg: pd.Series | pd.DataFrame,
    xborder: pd.Series | pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    *,
    active_case: str = "base",      # "base" or "instr"
    tolerance: float = 1e-3,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,       # optional filter: start date (e.g. "2019-06-01")
    end: str | None = None,         # optional filter: end date (e.g. "2019-08-31")
    only_xborder: bool = True,
    print_summary: bool = True,     # kept for backwards compatibility, not used anymore
) -> pd.DataFrame:
    """
    Find (start_week, timestep, connection) where:
        - In the *active* case the flow hits its max capacity (within `tolerance`)
        - In the *other* case the same connection at the same time is NOT at max capacity
        - Optionally restricted to connections with xborder == 1
        - Optionally restricted to a calendar date range [start, end]

    Parameters
    ----------
    active_case : {"base", "instr"}
        - "base": look for bottlenecks in BASE that are not bottlenecks in INSTRUMENT
        - "instr": look for bottlenecks in INSTRUMENT that are not bottlenecks in BASE
    start : str or None
        Start date (inclusive) for filtering, e.g. "2019-06-01". If None, no lower bound.
    end : str or None
        End date (inclusive) for filtering, e.g. "2019-08-31". If None, no upper bound.

    Returns
    -------
    DataFrame with columns:
        date, time, datetime, start_week, timestep, abs_hour,
        connection,
        flow_stage1_base, flow_stage1_instr,
        capacity_pos, capacity_neg, capacity_abs,
        dual_min_stage1_base, dual_max_stage1_base,
        dual_min_stage1_instr, dual_max_stage1_instr
    """

    # ------------------------------------------------------------------
    # 0) Validate active_case & set which case is "reference"
    # ------------------------------------------------------------------
    active_case = active_case.lower()
    if active_case not in ("base", "instr"):
        raise ValueError("active_case must be 'base' or 'instr'")

    ref_case = "instr" if active_case == "base" else "base"

    # ------------------------------------------------------------------
    # 1) Normalize capacities and xborder to Series indexed by connection
    # ------------------------------------------------------------------
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
    xb      = _to_conn_series(xborder,      "xborder")

    # capacities + xborder in one frame (index = connection)
    cap_df = pd.concat([cap_pos, cap_neg, xb], axis=1)

    cap_df["capacity_pos"] = cap_df["capacity_pos"].fillna(0.0).abs()
    cap_df["capacity_neg"] = cap_df["capacity_neg"].fillna(0.0).abs()
    cap_df["capacity_abs"] = cap_df[["capacity_pos", "capacity_neg"]].max(axis=1)

    cap_df = cap_df.reset_index().rename(columns={"index": "connection"})
    cap_df["connection"] = cap_df["connection"].astype(str)

    # ------------------------------------------------------------------
    # 2) Long flows in both cases
    # ------------------------------------------------------------------
    flows_base  = extract_flows_long(stage1_base).rename(columns={"flow": "flow_stage1_base"})
    flows_instr = extract_flows_long(stage1_instr).rename(columns={"flow": "flow_stage1_instr"})

    if flows_base.empty and flows_instr.empty:
        return pd.DataFrame()

    flows = flows_base.merge(
        flows_instr,
        on=["start_week", "connection", "timestep", "abs_hour"],
        how="outer",
    )

    flows["connection"] = flows["connection"].astype(str)
    flows["flow_stage1_base"]  = pd.to_numeric(flows.get("flow_stage1_base", 0.0),  errors="coerce").fillna(0.0)
    flows["flow_stage1_instr"] = pd.to_numeric(flows.get("flow_stage1_instr", 0.0), errors="coerce").fillna(0.0)

    # Attach capacities + xborder
    flows = flows.merge(
        cap_df,  # has: connection, capacity_pos, capacity_neg, capacity_abs, xborder
        on="connection",
        how="left",
    )

    # Only rows with positive capacity
    flows = flows[flows["capacity_abs"] > 0.0].copy()
    if flows.empty:
        return pd.DataFrame()

    # Restrict to cross-border connections if requested
    if only_xborder:
        flows = flows[pd.to_numeric(flows["xborder"], errors="coerce").fillna(0.0) >= 0.5]
        if flows.empty:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3) Binding tests depending on active_case / ref_case
    # ------------------------------------------------------------------
    # Dynamic column names
    flow_active_col = f"flow_stage1_{active_case}"
    flow_ref_col    = f"flow_stage1_{ref_case}"

    flows["abs_flow_active"] = flows[flow_active_col].abs()
    flows["abs_flow_ref"]    = flows[flow_ref_col].abs()

    flows["binding_active"] = flows["abs_flow_active"] >= (flows["capacity_abs"] - float(tolerance))
    flows["binding_ref"]    = flows["abs_flow_ref"]    >= (flows["capacity_abs"] - float(tolerance))

    # active hits capacity, ref does NOT
    mask = flows["binding_active"] & (~flows["binding_ref"])
    binding = flows[mask].copy()

    if binding.empty:
        return pd.DataFrame(columns=[
            "date", "time", "datetime",
            "start_week", "timestep", "abs_hour",
            "connection",
            "flow_stage1_base", "flow_stage1_instr",
            "capacity_pos", "capacity_neg", "capacity_abs",
            "dual_min_stage1_base", "dual_max_stage1_base",
            "dual_min_stage1_instr", "dual_max_stage1_instr",
        ])

    # ------------------------------------------------------------------
    # 4) Incidence matrix & nodal duals
    # ------------------------------------------------------------------
    long_map = melt_incidence_to_long(incidence_matrix)  # node, connection, coeff != 0
    long_map = long_map[["node", "connection"]].drop_duplicates()
    long_map["connection"] = long_map["connection"].astype(str)
    long_map["node"] = long_map["node"].astype(str)

    duals_base  = extract_dual_prices_long(stage1_base)
    duals_instr = extract_dual_prices_long(stage1_instr)

    def _conn_duals(duals: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if duals.empty:
            return pd.DataFrame(
                columns=[
                    "start_week", "connection", "timestep", "abs_hour",
                    f"dual_min_stage1_{suffix}", f"dual_max_stage1_{suffix}",
                ]
            )

        tmp = long_map.merge(duals, on="node", how="inner")
        grp = tmp.groupby(
            ["start_week", "connection", "timestep", "abs_hour"], as_index=False
        )["dual_price"].agg(
            dual_min=lambda s: float(np.min(s)),
            dual_max=lambda s: float(np.max(s)),
        )
        grp = grp.rename(
            columns={
                "dual_min": f"dual_min_stage1_{suffix}",
                "dual_max": f"dual_max_stage1_{suffix}",
            }
        )
        return grp

    dual_conn_base  = _conn_duals(duals_base,  "base")
    dual_conn_instr = _conn_duals(duals_instr, "instr")

    binding = binding.merge(
        dual_conn_base,
        on=["start_week", "connection", "timestep", "abs_hour"],
        how="left",
    )
    binding = binding.merge(
        dual_conn_instr,
        on=["start_week", "connection", "timestep", "abs_hour"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 5) Add real-world datetime, date, time
    # ------------------------------------------------------------------
    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso + " 00:00")

    binding["datetime"] = [
        base_ts + pd.Timedelta(hours=int(h - 1))
        for h in binding["abs_hour"].astype(int).values
    ]
    binding["date"] = binding["datetime"].dt.date.astype(str)
    binding["time"] = binding["datetime"].dt.time.astype(str)

    # Optional date filtering via [start, end]
    if start is not None or end is not None:
        if start is not None:
            start_ts = pd.Timestamp(start)
            binding = binding[binding["datetime"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            binding = binding[binding["datetime"] <= end_ts]

        if binding.empty:
            return pd.DataFrame(columns=[
                "date", "time", "datetime",
                "start_week", "timestep", "abs_hour",
                "connection",
                "flow_stage1_base", "flow_stage1_instr",
                "capacity_pos", "capacity_neg", "capacity_abs",
                "dual_min_stage1_base", "dual_max_stage1_base",
                "dual_min_stage1_instr", "dual_max_stage1_instr",
            ])

    # ------------------------------------------------------------------
    # 6) Order & return nicely
    # ------------------------------------------------------------------
    cols_out = [
        "date", "time", "datetime",
        "start_week", "timestep", "abs_hour",
        "connection",
        "flow_stage1_base", "flow_stage1_instr",
        "capacity_pos", "capacity_neg", "capacity_abs",
        "dual_min_stage1_base", "dual_max_stage1_base",
        "dual_min_stage1_instr", "dual_max_stage1_instr",
    ]
    for c in cols_out:
        if c not in binding.columns:
            binding[c] = np.nan

    binding = binding[cols_out].sort_values(
        by=["abs_hour", "connection"]
    ).reset_index(drop=True)

    return binding


def compute_bottleneck_stats_per_connection(
    stage1_base: pd.DataFrame,
    stage1_instr: pd.DataFrame,
    capacity_pos: pd.Series | pd.DataFrame,
    capacity_neg: pd.Series | pd.DataFrame,
    xborder: pd.Series | pd.DataFrame,
    *,
    tolerance: float = 1e-3,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,      # e.g. "2023-06-01"
    end: str | None = None,        # e.g. "2023-08-31"
    only_xborder: bool = True,
    print_summary: bool = True,
) -> pd.DataFrame:
    """
    For every connection (optionally only xborder == 1), within an optional time window,
    count how many timesteps are bottlenecks (i.e. |flow| >= capacity_abs - tolerance) in

        - the base case, and
        - the instrument case,

    and show whether the instrument increases or decreases the number of
    bottlenecks relative to the base case.

    Parameters
    ----------
    base_date : str
        Start date (YYYY-MM-DD) corresponding to abs_hour == 1.
    start, end : str or None
        Calendar dates (YYYY-MM-DD) used to restrict the analysis window.
        If None, no lower/upper bound is applied respectively.

    Returns
    -------
    DataFrame with one row per connection:
        connection
        n_bottlenecks_base
        n_bottlenecks_instr
        delta_instr_minus_base
        direction       ("increase" / "decrease" / "no_change")
        rel_change_pct  (relative change in %, NaN if base has 0)
    """

    # --------------------------------------------------------
    # 0) Normalize capacities and xborder to Series per connection
    # --------------------------------------------------------
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
    xb      = _to_conn_series(xborder,      "xborder")

    caps = pd.concat([cap_pos, cap_neg, xb], axis=1)
    caps["capacity_pos"] = caps["capacity_pos"].fillna(0.0).abs()
    caps["capacity_neg"] = caps["capacity_neg"].fillna(0.0).abs()
    caps["capacity_abs"] = caps[["capacity_pos", "capacity_neg"]].max(axis=1)

    caps = caps.reset_index().rename(columns={"index": "connection"})
    caps["connection"] = caps["connection"].astype(str)

    # --------------------------------------------------------
    # 1) Long flows in both cases
    # --------------------------------------------------------
    flows_base  = extract_flows_long(stage1_base).rename(columns={"flow": "flow_stage1_base"})
    flows_instr = extract_flows_long(stage1_instr).rename(columns={"flow": "flow_stage1_instr"})

    if flows_base.empty and flows_instr.empty:
        if print_summary:
            print("No flows in either case.")
        return pd.DataFrame(columns=[
            "connection",
            "n_bottlenecks_base",
            "n_bottlenecks_instr",
            "delta_instr_minus_base",
            "direction",
            "rel_change_pct",
        ])

    flows = flows_base.merge(
        flows_instr,
        on=["start_week", "connection", "timestep", "abs_hour"],
        how="outer",
    )

    flows["connection"] = flows["connection"].astype(str)
    flows["flow_stage1_base"]  = pd.to_numeric(flows.get("flow_stage1_base", 0.0),  errors="coerce").fillna(0.0)
    flows["flow_stage1_instr"] = pd.to_numeric(flows.get("flow_stage1_instr", 0.0), errors="coerce").fillna(0.0)

    # Attach capacities + xborder
    flows = flows.merge(caps, on="connection", how="left")

    # Keep only connections with positive capacity
    flows = flows[flows["capacity_abs"] > 0.0].copy()
    if flows.empty:
        if print_summary:
            print("No connections with positive capacity found.")
        return pd.DataFrame(columns=[
            "connection",
            "n_bottlenecks_base",
            "n_bottlenecks_instr",
            "delta_instr_minus_base",
            "direction",
            "rel_change_pct",
        ])

    # Optionally restrict to xborder == 1
    if only_xborder:
        flows = flows[pd.to_numeric(flows["xborder"], errors="coerce").fillna(0.0) >= 0.5]
        if flows.empty:
            if print_summary:
                print("No xborder == 1 connections found with flows.")
            return pd.DataFrame(columns=[
                "connection",
                "n_bottlenecks_base",
                "n_bottlenecks_instr",
                "delta_instr_minus_base",
                "direction",
                "rel_change_pct",
            ])

    # --------------------------------------------------------
    # 2) Add real-world datetime & optional time filtering
    # --------------------------------------------------------
    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso + " 00:00")

    flows["datetime"] = base_ts + pd.to_timedelta(
        flows["abs_hour"].astype(int) - 1, unit="h"
    )

    if start is not None or end is not None:
        start_ts = pd.to_datetime(start) if start is not None else flows["datetime"].min()
        end_ts   = pd.to_datetime(end)   if end is not None   else flows["datetime"].max()

        flows = flows[(flows["datetime"] >= start_ts) & (flows["datetime"] <= end_ts)]
        if flows.empty:
            if print_summary:
                print(
                    f"No flows in the selected time window "
                    f"({start_ts.date()} to {end_ts.date()})."
                )
            return pd.DataFrame(columns=[
                "connection",
                "n_bottlenecks_base",
                "n_bottlenecks_instr",
                "delta_instr_minus_base",
                "direction",
                "rel_change_pct",
            ])

    # --------------------------------------------------------
    # 3) Bottleneck flags in both cases
    # --------------------------------------------------------
    flows["abs_flow_base"]  = flows["flow_stage1_base"].abs()
    flows["abs_flow_instr"] = flows["flow_stage1_instr"].abs()

    flows["bottleneck_base"]  = flows["abs_flow_base"]  >= (flows["capacity_abs"] - float(tolerance))
    flows["bottleneck_instr"] = flows["abs_flow_instr"] >= (flows["capacity_abs"] - float(tolerance))

    # --------------------------------------------------------
    # 4) Aggregate per connection
    # --------------------------------------------------------
    gb = flows.groupby("connection", as_index=False)

    stats = gb.agg(
        n_bottlenecks_base=("bottleneck_base", "sum"),
        n_bottlenecks_instr=("bottleneck_instr", "sum"),
    )

    # convert from numpy/int64 to plain int
    stats["n_bottlenecks_base"]  = stats["n_bottlenecks_base"].astype(int)
    stats["n_bottlenecks_instr"] = stats["n_bottlenecks_instr"].astype(int)

    stats["delta_instr_minus_base"] = (
        stats["n_bottlenecks_instr"] - stats["n_bottlenecks_base"]
    )

    # direction: increase / decrease / no_change
    stats["direction"] = np.select(
        [
            stats["delta_instr_minus_base"] > 0,
            stats["delta_instr_minus_base"] < 0,
        ],
        [
            "increase",
            "decrease",
        ],
        default="no_change",
    )

    # relative change in % (instrument vs base)
    stats["rel_change_pct"] = np.where(
        stats["n_bottlenecks_base"] > 0,
        100.0 * stats["delta_instr_minus_base"] / stats["n_bottlenecks_base"],
        np.nan,
    )

    # --------------------------------------------------------
    # 5) Optional summary printing
    # --------------------------------------------------------
    if print_summary:
        total_base  = int(stats["n_bottlenecks_base"].sum())
        total_instr = int(stats["n_bottlenecks_instr"].sum())
        delta_total = total_instr - total_base

        n_increase  = int((stats["direction"] == "increase").sum())
        n_decrease  = int((stats["direction"] == "decrease").sum())
        n_no_change = int((stats["direction"] == "no_change").sum())

        if start is not None or end is not None:
            start_ts = pd.to_datetime(start) if start is not None else flows["datetime"].min()
            end_ts   = pd.to_datetime(end)   if end is not None   else flows["datetime"].max()
            print(
                f"=== Bottleneck statistics per xborder connection "
                f"({start_ts.date()} to {end_ts.date()}) ==="
            )
        else:
            print("=== Bottleneck statistics per xborder connection (full horizon) ===")

        print(f"Total bottlenecks BASE:       {total_base}")
        print(f"Total bottlenecks INSTRUMENT: {total_instr}")
        print(f"Δ (instr - base):             {delta_total}")
        print()
        print(f"Connections with increase in bottlenecks: {n_increase}")
        print(f"Connections with decrease in bottlenecks: {n_decrease}")
        print(f"Connections with no change:               {n_no_change}")

    # Sort: worst increases first
    stats = stats.sort_values(
        by=["delta_instr_minus_base", "connection"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return stats

# ============================================================
# Plot functions 
# ============================================================

def plot_stage0_prices_for_node(
    stage0_df: pd.DataFrame,
    node: str | int,
    base_date: str = DEFAULT_BASE_DATE,
    stage1_base_df: pd.DataFrame | None = None,
    stage1_instr_df: pd.DataFrame | None = None,
) -> str:
    """
    Plot nodal prices (dual of demand equilibrium constraint) for a single node
    over the entire model horizon.

    By default, only Stage 0 prices are shown. If `stage1_base_df` and/or
    `stage1_instr_df` are provided, their prices are added as extra lines.

    Parameters
    ----------
    stage0_df : pd.DataFrame
        Full result DataFrame of Stage 0 (already read via `read_results`).
    node : str | int
        Node ID for which prices should be shown (e.g. "VBG").
    base_date : str, optional
        Base date "YYYY-MM-DD" used to translate abs_hour into real time.
    stage1_base_df : pd.DataFrame, optional
        Full result DataFrame of Stage 1 (BASE case). If None, it is skipped.
    stage1_instr_df : pd.DataFrame, optional
        Full result DataFrame of Stage 1 (INSTRUMENT case). If None, it is skipped.
    
    Returns
    -------
    str
        Path to the saved PNG file.
    """

    import warnings

    node_str = str(node)
    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso)

    def _get_node_series(
        df: pd.DataFrame | None,
        label_prefix: str,
    ) -> pd.DataFrame | None:
        """Use extract_dual_prices_long, then filter and build datetime index."""
        if df is None:
            return None

        duals = extract_dual_prices_long(df)
        sub = duals[duals["node"].astype(str) == node_str].copy()

        if sub.empty:
            warnings.warn(
                f"No {label_prefix} price entries found for node {node_str!r}."
            )
            return None

        sort_cols = [c for c in ["abs_hour", "start_week", "timestep"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(by=sort_cols)

        if "abs_hour" not in sub.columns:
            raise ValueError(
                f"{label_prefix} dual result has no 'abs_hour' column – cannot build "
                "continuous time index for price plot."
            )

        sub["datetime"] = [
            base_ts + pd.Timedelta(hours=int(h - 1))
            for h in sub["abs_hour"].values
        ]
        sub = sub.set_index("datetime")
        return sub

    # Stage 0 is required
    sub0 = _get_node_series(stage0_df, "Stage 0")
    if sub0 is None or sub0.empty:
        raise ValueError(
            f"No Stage 0 price entries found for node {node_str!r}."
        )

    # Optional Stage 1 series
    sub1_base = _get_node_series(stage1_base_df, "Stage 1 BASE")
    sub1_instr = _get_node_series(stage1_instr_df, "Stage 1 INSTRUMENT")

    # Plot
    plt.figure(figsize=(12, 5))

    plt.plot(
        sub0.index,
        sub0["dual_price"],
        linewidth=1.6,
        color="black",
        label=f"Stage 0 @ node {node_str}",
    )

    if sub1_base is not None and not sub1_base.empty:
        plt.plot(
            sub1_base.index,
            sub1_base["dual_price"],
            linewidth=1.3,
            color="tab:blue",
            label=f"Stage 1 BASE @ node {node_str}",
        )

    if sub1_instr is not None and not sub1_instr.empty:
        plt.plot(
            sub1_instr.index,
            sub1_instr["dual_price"],
            linewidth=1.3,
            color="tab:red",
            label=f"Stage 1 INSTR @ node {node_str}",
        )

    plt.grid(True, alpha=0.3)
    plt.xlabel("Time")
    plt.ylabel("Price [€/MWh]")

    stages_present = ["Stage 0"]
    if sub1_base is not None and not sub1_base.empty:
        stages_present.append("Stage 1 BASE")
    if sub1_instr is not None and not sub1_instr.empty:
        stages_present.append("Stage 1 INSTR")

    stages_str = ", ".join(stages_present)
    plt.title(f"Nodal prices — node {node_str} ({stages_str})")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Filename
    safe_node = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in node_str)
    if (sub1_base is None or sub1_base.empty) and (sub1_instr is None or sub1_instr.empty):
        filename = f"stage0_prices_node_{safe_node}.png"
    else:
        filename = f"stage0_stage1_prices_node_{safe_node}.png"

    out_png = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out_png, dpi=150)
    plt.show()

    return out_png



import warnings

def plot_total_values(
    stage_df_with: pd.DataFrame,
    stage_df_without: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    demand_data: pd.DataFrame,
    node_to_control_area: dict,
    *,
    # scope selection (exactly one of these)
    control_area: str | None = None,
    node: str | None = None,
    # price-follower data (optional)
    pf_nodewise_with: pd.DataFrame | None = None,
    pf_nodewise_without: pd.DataFrame | None = None,
    # model structure
    thermal_node_idx: Dict[str, list],
    phs_node_idx: Dict[str, list],
    renewable_node_idx: Dict[str, list],
    dsr_node_idx: Dict[str, list],
    battery_node_idx: Dict[str, list],
    base_date: str = DEFAULT_BASE_DATE,
    start: str,
    end: str,
) -> str:
    """
    Build and plot generation/consumption stacks for either:
      - a control area (control_area='APG', node=None), or
      - a single node (node='VBG', control_area=None).

    It:
      1. builds two panels (with and without instrument),
      2. slices them to the explicit [start, end] time window,
      3. plots a side-by-side mirrored stackplot,
      4. returns the PNG path.
    """

    # --------------------------------------------------------
    # 1) Determine set of nodes for the chosen scope
    # --------------------------------------------------------
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    if node is not None:
        # Single-node scope
        nodes = [str(node)]
        scope_label = f"Node_{node}"
    else:
        # Control-area scope
        nodes = [n for n, ca in node_to_control_area.items()
                 if str(ca) == str(control_area)]
        if not nodes:
            raise ValueError(
                f"No nodes found for control_area = {control_area}. "
                f"Available control areas = {sorted(set(map(str, node_to_control_area.values())))}"
            )
        scope_label = f"Area_{control_area}"

    nodes_set = set(map(str, nodes))

    # --------------------------------------------------------
    # 2) Inner helper: build panel for a given stage_df + pf_nodewise
    # --------------------------------------------------------
    def build_panel(stage_df: pd.DataFrame,
                    pf_nodewise: pd.DataFrame | None) -> pd.DataFrame:

        def agg_by_nodes(varname: str,
                         plant_to_nodes: Dict[str, list] | None,
                         is_node_indexed: bool = False,
                         signed: float = +1.0) -> pd.DataFrame:
            # Filter by variable name
            part = stage_df[stage_df["variable"] == varname].copy()
            if part.empty:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])

            # Normalize start_week
            part["start_week"] = pd.to_numeric(
                part.get("start_week", 1),
                errors="coerce"
            ).fillna(1).astype(int)

            # Map index_0: either node itself or plant -> node mapping
            if is_node_indexed:
                # index_0 is a node id
                part["node"] = part["index_0"].astype(str)
            else:
                if not plant_to_nodes:
                    return pd.DataFrame(columns=["start_week", "timestep", "value"])
                # plant_to_nodes: node -> [plants...]
                m = [(str(p), str(n)) for n, plants in plant_to_nodes.items() for p in plants]
                map_df = pd.DataFrame(m, columns=["plant", "node"])
                map_df["plant"] = map_df["plant"].astype(str)
                map_df["node"] = map_df["node"].astype(str)
                part["plant"] = part["index_0"].astype(str)
                part = part.merge(map_df, on="plant", how="inner")
                if part.empty:
                    return pd.DataFrame(columns=["start_week", "timestep", "value"])

            # timestep from index_1 (if present)
            if "index_1" in part.columns:
                part["timestep"] = pd.to_numeric(
                    part["index_1"],
                    errors="coerce"
                ).fillna(0).astype(int)
            else:
                part["timestep"] = 0

            # restrict to chosen nodes
            part["node"] = part["node"].astype(str)
            part = part[part["node"].isin(nodes_set)]
            if part.empty:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])

            # numeric value * sign
            part["value"] = pd.to_numeric(part["value"], errors="coerce").fillna(0.0) * float(signed)

            out = part.groupby(["start_week", "timestep"], as_index=False)["value"].sum()
            return out

        # generation / down components
        thermal = agg_by_nodes("thermal_generation", thermal_node_idx)
        phs_gen = agg_by_nodes("phs_turbine_generation", phs_node_idx)
        res_gen = agg_by_nodes("renewable_generation",renewable_node_idx)
        dsr_down = agg_by_nodes("dsr_down", dsr_node_idx)
        bat_out = agg_by_nodes("battery_out", battery_node_idx)

        # consumption / up components
        phs_pump = agg_by_nodes("phs_pump_consumption", phs_node_idx)
        dsr_up = agg_by_nodes("dsr_up", dsr_node_idx)
        bat_in = agg_by_nodes("battery_in", battery_node_idx)

        # non-served energy (node-indexed)
        nse = agg_by_nodes("nse", None, is_node_indexed=True)
        
        # demand from model: expr:demand (node-indexed)
        demand_expr = agg_by_nodes("expr:demand", None, is_node_indexed=True)

        # ----------------------------------------------------
        # Inline exchange calculation 
        # ----------------------------------------------------
        flows = extract_flows_long(stage_df)
        if flows.empty:
            exchange_scope = pd.DataFrame(columns=["start_week", "timestep", "exchange"])
        else:
            long_map = melt_incidence_to_long(incidence_matrix)
            merged = flows.merge(long_map, on="connection", how="inner")
            merged["contrib"] = merged["coeff"] * merged["flow"]
            # node-level exchange
            ts = (
                merged.groupby(["start_week", "node", "timestep"], as_index=False)["contrib"]
                .sum()
                .rename(columns={"contrib": "exchange"})
            )
            # restrict to scope and aggregate over nodes
            exch_scope = (
                ts[ts["node"].astype(str).isin(nodes_set)]
                .groupby(["start_week", "timestep"], as_index=False)["exchange"]
                .sum()
            )
            exchange_scope = exch_scope

        # helper to merge components
        def m2(left: pd.DataFrame, comp: pd.DataFrame, name: str) -> pd.DataFrame:
            if left.empty and comp.empty:
                return pd.DataFrame(columns=["start_week", "timestep", name])
            base = comp[["start_week", "timestep"]].drop_duplicates().copy() if left.empty else left
            if comp is None or comp.empty:
                base[name] = 0.0
                return base
            return (
                base.merge(
                    comp.rename(columns={"value": name}),
                    on=["start_week", "timestep"], how="left"
                )
                .fillna({name: 0.0})
            )

        # Build base panel with generation/consumption components
        if not thermal.empty:
            base = thermal.rename(columns={"value": "thermal"})
        else:
            base = pd.DataFrame(columns=["start_week", "timestep", "thermal"])

        for comp, nm in [
            (phs_gen, "phs_gen"),
            (res_gen, "res"),
            (dsr_down, "dsr_out"),
            (bat_out, "battery_out"),
            (phs_pump, "phs_cons"),
            (dsr_up, "dsr_in"),
            (bat_in, "battery_in"),
        ]:
            base = m2(base, comp, nm)

        # Merge NSE or early-out if everything is empty
        if base.empty and not nse.empty:
            base = nse.rename(columns={"value": "nse"})
        elif base.empty and nse.empty:
            # nothing at all
            return pd.DataFrame(
                columns=[
                    "start_week", "timestep", "abs_hour",
                    "thermal", "phs_gen", "res", "dsr_out", "battery_out",
                    "phs_cons", "dsr_in", "battery_in", "nse", "exchange",
                    "demand", "pf_exports_price_follower", "pf_imports_price_follower",
                ]
            )
        else:
            base = base.merge(
                nse.rename(columns={"value": "nse"})
                if not nse.empty
                else pd.DataFrame(columns=["start_week", "timestep", "nse"]),
                on=["start_week", "timestep"], how="left"
            ).fillna({"nse": 0.0})

        # exchange
        if exchange_scope is None or exchange_scope.empty:
            base["exchange"] = 0.0
        else:
            base = base.merge(exchange_scope, on=["start_week", "timestep"], how="left").fillna({"exchange": 0.0})

        # absolute hour index
        base["abs_hour"] = base["timestep"] + HOURS_PER_WEEK * (base["start_week"] - 1)

        # 🔹 demand from model expression expr:demand (already aggregated over nodes)
        if demand_expr is None or demand_expr.empty:
            base["demand"] = 0.0
        else:
            base = base.merge(
                demand_expr.rename(columns={"value": "demand"}),
                on=["start_week", "timestep"],
                how="left"
            )
            base["demand"] = base["demand"].fillna(0.0)

        # ----------------------------------------------------
        # price-follower overlay with explicit warnings
        # ----------------------------------------------------
        
        if pf_nodewise is not None and not pf_nodewise.empty:
            pf_scope = pf_nodewise[pf_nodewise["node"].astype(str).isin(nodes_set)].copy()

            if pf_scope.empty:
                warnings.warn(
                    f"No price-follower rows for scope {scope_label} "
                    f"(nodes={sorted(nodes_set)}). Setting PF exports/imports to 0."
                )
                base["pf_exports_price_follower"] = 0.0
                base["pf_imports_price_follower"] = 0.0
            else:
                # compute PF exports/imports
                pf_scope["exports_pf"] = (
                    pd.to_numeric(pf_scope["pv_feed_to_system"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(pf_scope["battery_to_system"], errors="coerce").fillna(0.0)
                )
                pf_scope["imports_pf"] = (
                    pd.to_numeric(pf_scope["imports_to_demand"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(pf_scope["imports_to_battery"], errors="coerce").fillna(0.0)
                )

                if (
                    float(pf_scope["exports_pf"].abs().sum()) < 1e-9
                    and float(pf_scope["imports_pf"].abs().sum()) < 1e-9
                ):
                    warnings.warn(
                        f"Price-follower data present for {scope_label} "
                        "but exports and imports are numerically zero. "
                        "PF overlays will be zero."
                    )

                pf_scope = pf_scope.groupby(["start_week", "t", "abs_hour"], as_index=False).agg(
                    pf_exports_price_follower=("exports_pf", "sum"),
                    pf_imports_price_follower=("imports_pf", "sum"),
                )

                # Merge by abs_hour only (start_week, t used only in aggregation)
                base = base.drop(columns=["pf_exports_price_follower", "pf_imports_price_follower"], errors="ignore")
                base = base.merge(
                    pf_scope[["abs_hour", "pf_exports_price_follower", "pf_imports_price_follower"]],
                    on="abs_hour", how="left"
                )
                base["pf_exports_price_follower"] = base["pf_exports_price_follower"].fillna(0.0)
                base["pf_imports_price_follower"] = base["pf_imports_price_follower"].fillna(0.0)
        else:
            # PF data not provided at all → assume user knows; no warning
            base["pf_exports_price_follower"] = 0.0
            base["pf_imports_price_follower"] = 0.0
            
        if "demand" in base.columns:
            base["demand"] = (
                base["demand"]
                - base["pf_imports_price_follower"]
                + base["pf_exports_price_follower"]
            )

        return base

    # build both panels
    panel_with = build_panel(stage_df_with, pf_nodewise_with)
    panel_without = build_panel(stage_df_without, pf_nodewise_without)

    # --------------------------------------------------------
    # 3) Time window: must be explicit
    # --------------------------------------------------------
    use_start, use_end = parse_user_range(start, end, base_date)

    start_ts = parse_user_datetime(use_start, base_date)
    end_ts   = parse_user_datetime(use_end, base_date)

    # --------------------------------------------------------
    # 4) Slice panels in abs_hour
    # --------------------------------------------------------
    ah0 = date_to_abs_hour(use_start, base_date)
    ah1 = date_to_abs_hour(use_end, base_date)

    w_with = panel_with[(panel_with["abs_hour"] >= ah0) & (panel_with["abs_hour"] <= ah1)].copy()
    w_without = panel_without[(panel_without["abs_hour"] >= ah0) & (panel_without["abs_hour"] <= ah1)].copy()

    if w_with.empty or w_without.empty:
        raise ValueError(
            f"Chosen window [{use_start} → {use_end}] produced an empty slice "
            "in at least one of the panels."
        )

    # --------------------------------------------------------
    # 5) Prepare for plotting
    # --------------------------------------------------------
    internal_cols = [
        "pf_exports_price_follower", "pf_imports_price_follower",
        "thermal", "phs_gen", "res", "dsr_out", "battery_out",
        "phs_cons", "dsr_in", "battery_in", "exchange", "nse", "demand",
    ]
    for c in internal_cols:
        if c not in w_with.columns:
            w_with[c] = 0.0
        if c not in w_without.columns:
            w_without[c] = 0.0

    idx = np.unique(w_with["abs_hour"].values)
    base_iso = coerce_base_date(base_date)
    base_ts = pd.to_datetime(base_iso + " 00:00", format=DATE_FMT)
    time_index = pd.to_datetime([base_ts + pd.Timedelta(hours=int(h - 1)) for h in idx])

    colors = {
        # Generation (GEN)
        "Thermal": "#E60026",
        "PHS (gen)": "#005BBB",
        "RES": "#2A9D8F",
        "DSR (down)": "#F4A300",
        "Battery (out)": "#8E44AD",
    
        # Price-follower exports
        "Exports Price Follower": "#C837AB",
    
        # Imports/Exports
        "Imports": "#00C8FF",
        "Exports": "#FFD500",
    
        # Consumption / load shifting
        "PHS (pump)": "#003F88",
        "DSR (up)": "#E85D04",
        "Battery (in)": "#6A4C93",
        "Imports Price Follower": "#FF66FF",
    
        # Lines
        "NSE": "#FF0099",
        "Demand (served)": "#000000"
    }

    def series_select(dfw: pd.DataFrame, col: str) -> pd.Series:
        s = dfw.set_index("abs_hour")[col]
        return s.reindex(idx).fillna(0.0)

    def build_stackframes(dfw: pd.DataFrame):
        thermal = series_select(dfw, "thermal")
        phs_gen = series_select(dfw, "phs_gen")
        res_gen = series_select(dfw, "res")
        dsr_dn  = series_select(dfw, "dsr_out")
        batt_out = series_select(dfw, "battery_out")
        pf_exp = series_select(dfw, "pf_exports_price_follower")
        demand = series_select(dfw, "demand")

        phs_pmp = series_select(dfw, "phs_cons")
        dsr_up  = series_select(dfw, "dsr_in")
        batt_in = series_select(dfw, "battery_in")
        pf_imp  = series_select(dfw, "pf_imports_price_follower")

        exch = series_select(dfw, "exchange")
        nse  = series_select(dfw, "nse")

        imports = exch.clip(lower=0.0)
        exports = (-exch).clip(lower=0.0)

        gen_df = pd.DataFrame(
            {
                "Thermal": thermal.values,
                "PHS (gen)": phs_gen.values,
                "RES": res_gen.values,
                "DSR (down)": dsr_dn.values,
                "Battery (out)": batt_out.values,
                "Imports": imports.values,
                "Exports Price Follower": pf_exp.values,
                
            },
            index=time_index,
        ).clip(lower=0.0)

        cons_df = pd.DataFrame(
            {
                "PHS (pump)": phs_pmp.values,
                "DSR (up)": dsr_up.values,
                "Battery (in)": batt_in.values,
                "Exports": exports.values,
                "Imports Price Follower": pf_imp.values,
            },
            index=time_index,
        ).clip(lower=0.0)

        demand_line = pd.Series(demand.values, index=time_index, name="Demand (served)")
        nse_line = pd.Series(nse.values, index=time_index, name="NSE")
        return gen_df, cons_df, demand_line, nse_line

    gen_with, cons_with, demand_with, nse_with = build_stackframes(w_with)
    gen_without, cons_without, demand_without, nse_without = build_stackframes(w_without)

    # --------------------------------------------------------
    # 6) Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)

    def plot_panel(ax, gen_df, cons_df, demand_line, nse_line, title):
        gen_cols = [
            "Thermal", "PHS (gen)", "RES", "DSR (down)",
            "Battery (out)", "Exports Price Follower", "Imports",
        ]
        cons_cols = [
            "PHS (pump)", "DSR (up)", "Battery (in)", "Exports", "Imports Price Follower",
        ]

        ax.stackplot(
            gen_df.index,
            [gen_df[c].values for c in gen_cols],
            labels=gen_cols,
            colors=[colors[c] for c in gen_cols],
            alpha=0.9,
        )
        ax.stackplot(
            cons_df.index,
            [-cons_df[c].values for c in cons_cols],
            labels=[f"-{c}" for c in cons_cols],
            colors=[colors[c] for c in cons_cols],
            alpha=0.9,
        )

        ax.plot(
            demand_line.index,
            demand_line.values,
            color="black",
            linewidth=2.0,
            label="Demand (served)",
            zorder=6,
        )
        if float(nse_line.sum()) > 1e-6:
            ax.plot(
                nse_line.index,
                nse_line.values,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label="NSE",
                zorder=6,
            )

        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Power [MW]")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

    plot_panel(
        axes[0],
        gen_with, cons_with, demand_with, nse_with,
        f"{scope_label}  {start_ts.date()} → {end_ts.date()}  (BASE CASE)",
    )
    plot_panel(
        axes[1],
        gen_without, cons_without, demand_without, nse_without,
        f"{scope_label}  {start_ts.date()} → {end_ts.date()}  (INSTRUMENT CASE)",
    )

    plt.tight_layout()
    out_png = os.path.join(
        RESULTS_DIR,
        f"gen_cons_{scope_label}_"
        f"{start_ts.strftime('%Y-%m-%d_%H%M')}_to_{end_ts.strftime('%Y-%m-%d_%H%M')}.png",
    )
    plt.savefig(out_png, dpi=150)
    plt.show()
    return out_png


def print_pf_values(
    pf_nodewise: pd.DataFrame,
    node_to_control_area: dict,
    *,
    control_area: str | None = None,
    node: str | int | None = None,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,
    end: str | None = None,
    round_digits: int = 3,
    n_rows: int = 10,
) -> pd.DataFrame:
    """
    Summarize key price-follower variables for

      * a single node (node='VBG', control_area=None), or
      * an entire control area (control_area='APG', node=None; summed over nodes),
      * all nodes (control_area=None, node=None).

    For control areas, all quantities are summed across nodes,
    while price is averaged across nodes (not summed).
    """

    # disallow conflicting scope definitions
    if control_area is not None and node is not None:
        raise ValueError("Specify at most one of `control_area` or `node`.")

    cols_interest = [
        "price",
        "local_demand",
        "pv_availability", "pv_to_demand", "pv_to_battery",
        "pv_feed_to_system", "pv_curtailment",
        "imports_to_demand", "imports_to_battery",
        "battery_in", "battery_to_demand",
        "battery_to_system", "battery_out",
        "battery_storage",
        "revenue", "pv_used_cost", "import_energy_costs",
        "fees", "objective",
    ]

    # work on a copy and make sure abs_hour is present
    pf_df = ensure_abs_hour(pf_nodewise.copy())

    # node / control-area tagging
    pf_df["node"] = pf_df["node"].astype(str)
    pf_df["control_area"] = pf_df["node"].map(node_to_control_area).astype(str)

    # --------------------------------------------------------
    # Scope filter: node / control_area / all nodes
    # --------------------------------------------------------
    if node is not None:
        node_str = str(node)
        sub = pf_df[pf_df["node"] == node_str]
        print(f"\n🔎 Showing entries for node {node_str}")
    elif control_area is not None:
        ca_str = str(control_area)
        sub = pf_df[pf_df["control_area"] == ca_str]
        print(f"\n🔎 Showing entries for control area {ca_str} (summed over nodes)")
    else:
        sub = pf_df
        print("\n🔎 Showing entries for all nodes")

    if sub.empty:
        print("⚠️ No matching PF data found after node/control_area filter.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # Time window: EXACTLY like plot_pf_import_export_and_prices
    # --------------------------------------------------------
    if start is not None or end is not None:
        try:
            use_start, use_end = parse_user_range(start, end, base_date)
        except ValueError as e:
            print(f"⚠️ Could not parse date range: {e}")
            return pd.DataFrame()

        s_abs = date_to_abs_hour(use_start, base_date)
        e_abs = date_to_abs_hour(use_end, base_date)
        print(f"⏱️ Filtering by abs_hour range: {s_abs}–{e_abs}")

        if "abs_hour" not in sub.columns:
            print("⚠️ No 'abs_hour' column found for time filtering.")
            return pd.DataFrame()

        sub = sub[(sub["abs_hour"] >= s_abs) & (sub["abs_hour"] <= e_abs)]

        if sub.empty:
            print("⚠️ No matching PF data found after date/time filtering.")
            return pd.DataFrame()

    # --------------------------------------------------------
    # Select & clean columns
    # --------------------------------------------------------
    cols_present = [c for c in cols_interest if c in sub.columns]
    id_cols = [
        c for c in ["start_week", "abs_hour", "node",
                    "control_area", "t"]
        if c in sub.columns
    ]

    sub = sub[id_cols + cols_present].copy()

    for c in cols_present:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)

    # --------------------------------------------------------
    # Aggregate over nodes if a control area (and no single node)
    #   -> quantities: sum over nodes
    #   -> price:     average over nodes
    # --------------------------------------------------------
    if control_area is not None and node is None:
        group_cols = [c for c in id_cols if c != "node"]

        agg_dict = {c: "sum" for c in cols_present}
        if "price" in agg_dict:
            agg_dict["price"] = "mean"

        sub = sub.groupby(group_cols, as_index=False).agg(agg_dict)

    # rounding
    for c in cols_present:
        sub[c] = sub[c].round(round_digits)

    # sort & row limiting
    sort_cols = [
        c for c in ["start_week", "abs_hour", "t",
                    "control_area", "node"]
        if c in sub.columns
    ]
    if sort_cols:
        sub = sub.sort_values(by=sort_cols, ascending=True)

    if start is None and end is None:
        sub = sub.head(n_rows)
        print(f"(Showing first {len(sub)} rows)")
    else:
        print(f"(Showing all {len(sub)} rows in requested date range)")

    print(sub.to_string(index=False))
    return sub


def plot_pf_import_export_and_prices(
    pf_nodewise: pd.DataFrame,
    node_to_control_area: dict,
    *,
    control_area: str | None = None,
    node: str | int | None = None,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,
    end: str | None = None,
    show_flows: bool = True,
) -> str:
    """
    Plot PF exports/imports and PF input prices either for
      - a control area (control_area='APG', node=None), or
      - a single node (node='VBG', control_area=None; CA inferred from mapping).

    Exports (MW) are plotted above zero, imports (MW) below zero.
    Price (€/MWh) is plotted on a right y-axis if show_flows=True, otherwise
    as a single line on the main axis.

    Parameters
    ----------
    pf_nodewise : pd.DataFrame
        Concatenated price-follower result (all weeks), with at least
        columns: ['node', 'start_week', 't', 'abs_hour', 'price',
                  'pv_feed_to_system', 'battery_to_system',
                  'imports_to_demand', 'imports_to_battery'].
    node_to_control_area : dict
        Mapping node -> control area.
    control_area : str, optional
        If given (and node=None), aggregate over all PF nodes in this CA.
    node : str | int, optional
        If given (and control_area=None), restrict to this single node.
    base_date : str
        Base date "YYYY-MM-DD" used for abs_hour -> timestamp.
    start, end : str, optional
        Optional time window in user format understood by parse_user_range.
    show_flows : bool, default True
        If False, only the PF input price is plotted (no exports/imports).

    Returns
    -------
    str
        Path to saved PNG.
    """

    # --------------------------------------------------------
    # 1) Determine scope: area vs node
    # --------------------------------------------------------
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    if node is not None:
        # Node scope
        node_str = str(node)

        ca_of_node = str(
            node_to_control_area.get(node_str, node_to_control_area.get(node, "UNKNOWN"))
        )
        if ca_of_node == "UNKNOWN":
            raise ValueError(
                f"Node {node_str!r} not found in node_to_control_area mapping."
            )

        control_area = ca_of_node  # normalize
        if show_flows:
            scope_label = f"{control_area} — Node {node_str} (PF exports/imports & price)"
        else:
            scope_label = f"{control_area} — Node {node_str} (PF input price)"

        # PF rows for this node
        pf_scope = pf_nodewise[pf_nodewise["node"].astype(str) == node_str].copy()
        if pf_scope.empty:
            raise ValueError(
                f"No PF data for node {node_str} in control area {control_area}"
            )

        # Prices: node-specific
        price_agg = pf_scope[["abs_hour", "price"]].copy()

        safe_node = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in node_str)
        out_name_suffix = f"{control_area}_node_{safe_node}"

    else:
        # Control-area scope
        ca = str(control_area)
        if show_flows:
            scope_label = f"{ca} — PF Exports/Imports & PF Input Prices"
        else:
            scope_label = f"{ca} — PF Input Prices"

        # Collect nodes in CA
        nodes = [n for n, ca_map in node_to_control_area.items() if str(ca_map) == ca]
        node_set = {str(n) for n in nodes}
        if not node_set:
            raise ValueError(
                f"No nodes found for control_area={ca}. "
                f"Available CAs = {sorted(set(map(str, node_to_control_area.values())))}"
            )

        pf_scope = pf_nodewise[pf_nodewise["node"].astype(str).isin(node_set)].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for control area {ca}")

        # Area-average price per abs_hour
        price_agg = (
            pf_scope.groupby("abs_hour", as_index=False)["price"]
            .mean()
        )

        out_name_suffix = f"{ca}"

    # --------------------------------------------------------
    # 2) Compute exports / imports per abs_hour
    # --------------------------------------------------------
    pf_scope["exports_pf"] = (
        pd.to_numeric(pf_scope["pv_feed_to_system"], errors="coerce").fillna(0.0)
        + pd.to_numeric(pf_scope["battery_to_system"], errors="coerce").fillna(0.0)
    )
    pf_scope["imports_pf"] = (
        pd.to_numeric(pf_scope["imports_to_demand"], errors="coerce").fillna(0.0)
        + pd.to_numeric(pf_scope["imports_to_battery"], errors="coerce").fillna(0.0)
    )

    # Sum in case there are multiple rows per abs_hour
    pf_agg = (
        pf_scope.groupby("abs_hour", as_index=False)[["exports_pf", "imports_pf"]]
        .sum()
    )

    # Merge prices
    agg = pf_agg.merge(price_agg, on="abs_hour", how="left")
    if "price" not in agg:
        agg["price"] = 0.0

    # --------------------------------------------------------
    # 3) Build absolute time index & optional time window
    # --------------------------------------------------------
    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso)

    def abs_to_ts(abs_hours: np.ndarray) -> pd.DatetimeIndex:
        return pd.to_datetime(
            [base_ts + pd.Timedelta(hours=int(h - 1)) for h in abs_hours]
        )

    idx = np.sort(agg["abs_hour"].unique())

    # optional [start, end] window
    if start is not None or end is not None:
        use_start, use_end = parse_user_range(start, end, base_date)
        s_abs = date_to_abs_hour(use_start, base_date)
        e_abs = date_to_abs_hour(use_end, base_date)
        agg = agg[(agg["abs_hour"] >= s_abs) & (agg["abs_hour"] <= e_abs)].copy()
        if agg.empty:
            raise ValueError(
                f"Chosen window [{use_start} → {use_end}] produced an empty slice."
            )
        idx = np.sort(agg["abs_hour"].unique())

    time_index = abs_to_ts(idx)
    agg = agg.set_index(time_index)

    # --------------------------------------------------------
    # 4) Plotting
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(scope_label)

    if show_flows:
        # PF exports/imports area plot on primary axis
        ax1.fill_between(
            agg.index,
            0,
            agg["exports_pf"].clip(lower=0.0),
            color="#0066FF",  # bright blue
            alpha=0.9,
            label="Exports (PF)",
        )
        ax1.fill_between(
            agg.index,
            0,
            -agg["imports_pf"].clip(lower=0.0),
            color="#FF8800",  # bright orange
            alpha=0.9,
            label="Imports (PF)",
        )
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_ylabel("Power [MW]")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Price on right axis
        ax2 = ax1.twinx()
        price_ax = ax2
        price_ax.set_ylabel("PF Input Price [€/MWh]")
        price_ax.tick_params(axis="y", labelcolor="black")

    else:
        # No flows: just price on main axis
        ax1.set_ylabel("PF Input Price [€/MWh]")
        ax1.grid(True, alpha=0.3)
        price_ax = ax1  # use primary axis

    # Price line
    price_ax.plot(
        agg.index,
        agg["price"],
        linestyle="--",
        linewidth=1.8,
        color="#000000",
        label="PF Input Price [€/MWh]",
    )

    # Legend for price if using main axis or if we want separate legends
    if show_flows:
        price_ax.legend(loc="upper right")
    else:
        price_ax.legend(loc="upper right")

    fig.tight_layout()

    prefix = "pf_exports_imports_prices" if show_flows else "pf_prices"
    out_png = os.path.join(
        RESULTS_DIR,
        f"{prefix}_{out_name_suffix}.png",
    )
    plt.savefig(out_png, dpi=150)
    plt.show()

    return out_png


def plot_pf_battery_and_prices(
    pf_nodewise: pd.DataFrame,
    node_to_control_area: dict,
    *,
    control_area: str | None = None,
    node: str | int | None = None,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,
    end: str | None = None,
    show_flows: bool = True,
) -> str:
    """
    Plot PF battery behavior (storage, in, out) and PF input prices either for
      - a control area (control_area='APG', node=None), or
      - a single node (node='VBG', control_area=None; CA inferred from mapping).

    battery_in / battery_out are in MW (per time step),
    battery_storage is in MWh (or consistent energy unit),
    price in €/MWh on the right axis.

    Parameters
    ----------
    pf_nodewise : pd.DataFrame
        Concatenated price-follower result (all weeks), with at least
        columns: ['node', 'start_week', 't', 'abs_hour', 'price',
                  'battery_in', 'battery_out', 'battery_storage'].
    node_to_control_area : dict
        Mapping node -> control area.
    control_area : str, optional
        If given (and node=None), aggregate over all PF nodes in this CA.
    node : str | int, optional
        If given (and control_area=None), restrict to this single node.
    base_date : str
        Base date "YYYY-MM-DD" used for abs_hour -> timestamp.
    start, end : str, optional
        Optional time window in user format understood by parse_user_range.
    show_flows : bool, default True
        If False, only the PF input price is plotted.

    Returns
    -------
    str
        Path to saved PNG.
    """

    # --------------------------------------------------------
    # 1) Determine scope: area vs node
    # --------------------------------------------------------
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    if node is not None:
        # Node scope
        node_str = str(node)

        ca_of_node = str(
            node_to_control_area.get(node_str, node_to_control_area.get(node, "UNKNOWN"))
        )
        if ca_of_node == "UNKNOWN":
            raise ValueError(
                f"Node {node_str!r} not found in node_to_control_area mapping."
            )

        control_area = ca_of_node  # normalize
        if show_flows:
            scope_label = f"{control_area} — Node {node_str} (Battery & PF Input Price)"
        else:
            scope_label = f"{control_area} — Node {node_str} (PF Input Price)"

        pf_scope = pf_nodewise[pf_nodewise["node"].astype(str) == node_str].copy()
        if pf_scope.empty:
            raise ValueError(
                f"No PF data for node {node_str} in control area {control_area}"
            )

        # Node-specific price
        price_agg = pf_scope[["abs_hour", "price"]].copy()

        safe_node = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in node_str)
        out_name_suffix = f"{control_area}_node_{safe_node}"

    else:
        # Control-area scope
        ca = str(control_area)
        if show_flows:
            scope_label = f"{ca} — Battery Storage / In / Out & PF Input Prices"
        else:
            scope_label = f"{ca} — PF Input Prices"

        # Collect nodes in CA
        nodes = [n for n, ca_map in node_to_control_area.items() if str(ca_map) == ca]
        node_set = {str(n) for n in nodes}
        if not node_set:
            raise ValueError(
                f"No nodes found for control_area={ca}. "
                f"Available CAs = {sorted(set(map(str, node_to_control_area.values())))}"
            )

        pf_scope = pf_nodewise[pf_nodewise["node"].astype(str).isin(node_set)].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for control area {ca}")

        # Area-average price per abs_hour
        price_agg = (
            pf_scope.groupby("abs_hour", as_index=False)["price"]
            .mean()
        )

        out_name_suffix = f"{ca}"

    # --------------------------------------------------------
    # 2) Compute battery aggregates per abs_hour
    # --------------------------------------------------------
    pf_scope["battery_in_pf"] = pd.to_numeric(
        pf_scope.get("battery_in", 0.0), errors="coerce"
    ).fillna(0.0)

    pf_scope["battery_out_pf"] = pd.to_numeric(
        pf_scope.get("battery_out", 0.0), errors="coerce"
    ).fillna(0.0)

    pf_scope["battery_storage_pf"] = pd.to_numeric(
        pf_scope.get("battery_storage", 0.0), errors="coerce"
    ).fillna(0.0)

    # Sum over nodes if control-area scope, or just aggregate in case of multi-rows
    pf_agg = (
        pf_scope.groupby("abs_hour", as_index=False)[
            ["battery_in_pf", "battery_out_pf", "battery_storage_pf"]
        ]
        .sum()
    )

    # Merge prices
    agg = pf_agg.merge(price_agg, on="abs_hour", how="left")
    if "price" not in agg:
        agg["price"] = 0.0

    # --------------------------------------------------------
    # 3) Build absolute time index & optional time window
    # --------------------------------------------------------
    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso)

    def abs_to_ts(abs_hours: np.ndarray) -> pd.DatetimeIndex:
        return pd.to_datetime(
            [base_ts + pd.Timedelta(hours=int(h - 1)) for h in abs_hours]
        )

    idx = np.sort(agg["abs_hour"].unique())

    # optional [start, end] window
    if start is not None or end is not None:
        use_start, use_end = parse_user_range(start, end, base_date)
        s_abs = date_to_abs_hour(use_start, base_date)
        e_abs = date_to_abs_hour(use_end, base_date)
        agg = agg[(agg["abs_hour"] >= s_abs) & (agg["abs_hour"] <= e_abs)].copy()
        if agg.empty:
            raise ValueError(
                f"Chosen window [{use_start} → {use_end}] produced an empty slice."
            )
        idx = np.sort(agg["abs_hour"].unique())

    time_index = abs_to_ts(idx)
    agg = agg.set_index(time_index)

    # --------------------------------------------------------
    # 4) Plotting
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(scope_label)

    if show_flows:
        # Plot battery_in (charging) and -battery_out (discharging) as lines
        ax1.plot(
            agg.index,
            agg["battery_in_pf"],
            linewidth=1.8,
            label="Battery In [MW]",
        )
        ax1.plot(
            agg.index,
            -agg["battery_out_pf"],
            linewidth=1.8,
            linestyle="--",
            label="Battery Out (negative) [MW]",
        )

        # Plot battery_storage as a separate line on the same axis
        ax1.plot(
            agg.index,
            agg["battery_storage_pf"],
            linewidth=1.4,
            linestyle="-.",
            label="Battery Storage [MWh]",
        )

        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_ylabel("Battery Flows / Storage [MW / MWh]")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Price on right axis
        ax2 = ax1.twinx()
        price_ax = ax2
        price_ax.set_ylabel("PF Input Price [€/MWh]")
        price_ax.tick_params(axis="y", labelcolor="black")

    else:
        # No battery curves: only price on main axis
        ax1.set_ylabel("PF Input Price [€/MWh]")
        ax1.grid(True, alpha=0.3)
        price_ax = ax1

    # Price line
    price_ax.plot(
        agg.index,
        agg["price"],
        linestyle="--",
        linewidth=1.8,
        color="#000000",
        label="PF Input Price [€/MWh]",
    )

    # Legend for price
    price_ax.legend(loc="upper right")

    fig.tight_layout()

    prefix = "pf_battery_prices" if show_flows else "pf_prices_only"
    out_png = os.path.join(
        RESULTS_DIR,
        f"{prefix}_{out_name_suffix}.png",
    )
    plt.savefig(out_png, dpi=150)
    plt.show()

    return out_png

############
### Main ###
############

def main():
    stage0 = read_results(STAGE0_PATH)
    stage1_base = read_results(STAGE1_PATH_BASE)
    stage1_instr = read_results(STAGE1_PATH_INSTRUMENT)
    pf_nodewise_base = read_results(FULL_PRICE_FOLLOWER_PATH_BASE)
    pf_nodewise_instr = read_results(FULL_PRICE_FOLLOWER_PATH_INSTRUMENT)

    co2_price, voll_raw, wind_pv_cost, phs_cost, dsr_cost,flow_cost = load_general_data(DATA_XLSX, "General_Data")
    voll = to_float(voll_raw)

    _, node_demand_data, control_areas, nodes, node_to_control_area = load_demand_data(
        DATA_XLSX, "Total_Demand", "Demand_Shares", "Demand_Profiles"
    )
    thermal_specific, thermal_node_idx = load_thermal_power_plant_data(
        DATA_XLSX, "Thermal_Power_Data", "Thermal_Power_Specific_Data", "RES_Shares"
    )
    phs_specific, _, phs_node_idx = load_phs_power_plant_data(
        DATA_XLSX, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data"
    )
    renewable_specific, renewable_node_idx = load_renewable_power_plant_data(
        DATA_XLSX, "RES_Power_Data", "RES_Power_Specific_Data", "RES_Shares"
    )
    dsr_specific, battery_specific, dsr_node_idx, battery_node_idx = load_flexibility_data(
        DATA_XLSX, "Flexibility_Data", "Flexibility_Specific_Data", "RES_Shares"
    )
    _, connections, incidence_matrix, capacity_pos, capacity_neg, xborder, conductance_series, ptdf_results = load_exchange_data(
        DATA_XLSX, 'Exchange_Data', slack_node=None, verbose=False
    )
    
    '''
    compare_control_area_flows(
        stage1_base=stage1_base,
        stage1_instr=stage1_instr,
        incidence_matrix=incidence_matrix,
        node_to_control_area=node_to_control_area,
        control_area="APG",
        DATE_TIME="2030-01-22 13:00"
    )

    DATE_TIME_START = "2030-01-01 00:00"
    DATE_TIME_END = "2030-01-07 23:00"
    
    bottleneck_stats = compute_bottleneck_stats_per_connection(
        stage1_base=stage1_base,
        stage1_instr=stage1_instr,
        capacity_pos=capacity_pos,
        capacity_neg=capacity_neg,
        xborder=xborder,
        tolerance=1e-3,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
        only_xborder=True,
        print_summary=True,
    )
    
    print("\nPer-connection bottleneck comparison (xborder only):")
    print(bottleneck_stats.to_string(index=False))
    
    DATE_TIME_START1 = "2030-01-01 00:00"
    DATE_TIME_END1 = "2030-01-07 23:00"
    
    bottleneck_stats1 = compute_bottleneck_stats_per_connection(
        stage1_base=stage1_base,
        stage1_instr=stage1_instr,
        capacity_pos=capacity_pos,
        capacity_neg=capacity_neg,
        xborder=xborder,
        tolerance=1e-3,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START1,
        end=DATE_TIME_END1,
        only_xborder=True,
        print_summary=True,
    )
    
    print("\nPer-connection bottleneck comparison (xborder only):")
    print(bottleneck_stats1.to_string(index=False))
    

    binding_only_base_xborder = find_binding_flows_only_in_one_case(
        stage1_base=stage1_base,
        stage1_instr=stage1_instr,
        capacity_pos=capacity_pos,
        capacity_neg=capacity_neg,
        xborder=xborder,
        incidence_matrix=incidence_matrix,
        active_case="base",          # base vs instr
        tolerance=1e-3,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
        only_xborder=True,
    )
    
    print("\n==== Flows at max capacity in STAGE 1 BASE only (not in instrument) ====\n")
    if binding_only_base_xborder.empty:
        print("No such hours found.")
    else:
        # show first few rows
        print(binding_only_base_xborder.head(20).to_string(index=False))
        
    binding_only_instr_xborder = find_binding_flows_only_in_one_case(
        stage1_base=stage1_base,
        stage1_instr=stage1_instr,
        capacity_pos=capacity_pos,
        capacity_neg=capacity_neg,
        xborder=xborder,
        incidence_matrix=incidence_matrix,
        active_case="instr",         # instr vs base
        tolerance=1e-3,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
        only_xborder=True,
    )
    
    print("\n==== Flows at max capacity in STAGE 1 INSTRUMENT only (not in instrument) ====\n")
    if binding_only_instr_xborder.empty:
        print("No such hours found.")
    else:
        # show first few rows
        print(binding_only_instr_xborder.head(20).to_string(index=False))
     
    
    plot_stage0_prices_for_node(
        stage0_df=stage0,
        stage1_base_df=stage1_base,      # <-- new
        stage1_instr_df=stage1_instr,    # <-- new
        node="VBG",
        base_date=DEFAULT_BASE_DATE,
    )
    '''
    
    DATE_TIME_START = "2030-01-01 00:00"
    DATE_TIME_END = "2030-01-01 23:00"
    
    plot_total_values(
    stage_df_with=stage1_base,
    stage_df_without=stage1_instr,
    incidence_matrix=incidence_matrix,
    demand_data=node_demand_data,
    node_to_control_area=node_to_control_area,
    control_area="APG",
    pf_nodewise_with=pf_nodewise_base,
    pf_nodewise_without=pf_nodewise_instr,
    thermal_node_idx=thermal_node_idx,
    phs_node_idx=phs_node_idx,
    renewable_node_idx=renewable_node_idx,
    dsr_node_idx=dsr_node_idx,
    battery_node_idx=battery_node_idx,
    base_date=DEFAULT_BASE_DATE,
    start=DATE_TIME_START,
    end=DATE_TIME_END
    )
    
    '''
    plot_total_values(
    stage_df_with=stage1_base,
    stage_df_without=stage1_instr,
    incidence_matrix=incidence_matrix,
    demand_data=node_demand_data,
    node_to_control_area=node_to_control_area,
    node="VBG",
    pf_nodewise_with=pf_nodewise_base,
    pf_nodewise_without=pf_nodewise_instr,
    thermal_node_idx=thermal_node_idx,
    phs_node_idx=phs_node_idx,
    renewable_node_idx=renewable_node_idx,
    dsr_node_idx=dsr_node_idx,
    battery_node_idx=battery_node_idx,
    base_date=DEFAULT_BASE_DATE,
    start=DATE_TIME_START,
    end=DATE_TIME_END
    )
    '''
    '''  
    print_pf_values(
        pf_nodewise=pf_nodewise_base,
        node_to_control_area=node_to_control_area,
        node="VBG",
        start=DATE_TIME_START,
        end=DATE_TIME_END
    )
    print_pf_values(
        pf_nodewise=pf_nodewise_instr,
        node_to_control_area=node_to_control_area,
        node="VBG",
        start=DATE_TIME_START,
        end=DATE_TIME_END
    )
    '''
    
    print_pf_values(
        pf_nodewise=pf_nodewise_base,
        node_to_control_area=node_to_control_area,
        control_area="APG",
        start=DATE_TIME_START,
        end=DATE_TIME_END
    )
    
    print_pf_values(
        pf_nodewise=pf_nodewise_instr,
        node_to_control_area=node_to_control_area,
        control_area="APG",
        start=DATE_TIME_START,
        end=DATE_TIME_END
    )

    '''
    DATE_TIME_START1 = "2030-01-01 08:00"
    DATE_TIME_END1 = "2030-01-01 08:00"

    nodes_in_apg = [
        n for n, ca in node_to_control_area.items()
        if str(ca) == "APG"
    ]
    
    for n in nodes_in_apg:
        print(f"\n===== Node {n} =====")
        
        print_pf_values(
            pf_nodewise=pf_nodewise_base,
            node_to_control_area=node_to_control_area,
            node=n,
            start=DATE_TIME_START1,
            end=DATE_TIME_END1
        )
        print_pf_values(
            pf_nodewise=pf_nodewise_instr,
            node_to_control_area=node_to_control_area,
            node=n,
            start=DATE_TIME_START1,
            end=DATE_TIME_END1
        )
    '''

    plot_pf_import_export_and_prices(
        pf_nodewise=pf_nodewise_base,
        control_area="APG",
        node_to_control_area=node_to_control_area,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
        show_flows=True
    )

    plot_pf_import_export_and_prices(
        pf_nodewise=pf_nodewise_instr,
        control_area="APG",
        node_to_control_area=node_to_control_area,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
        show_flows=True
    )
    
    '''
    plot_pf_import_export_and_prices(
        pf_nodewise=pf_nodewise_base,
        node="VBG",
        node_to_control_area=node_to_control_area,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
    )
    
    plot_pf_import_export_and_prices(
        pf_nodewise=pf_nodewise_instr,
        node="VBG",
        node_to_control_area=node_to_control_area,
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
    )
    
    plot_pf_battery_and_prices(
        pf_nodewise=pf_nodewise_base,
        node_to_control_area=node_to_control_area,
        node="VBG",
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
    )
    
    plot_pf_battery_and_prices(
        pf_nodewise=pf_nodewise_instr,
        node_to_control_area=node_to_control_area,
        node="VBG",
        base_date=DEFAULT_BASE_DATE,
        start=DATE_TIME_START,
        end=DATE_TIME_END,
    
    )
    '''

    
if __name__ == "__main__":
    main()
