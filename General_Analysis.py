from __future__ import annotations

import os
import re
import warnings
import inspect
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_FILE = "Data_Updated.xlsx"
DATA_XLSX = os.path.join(BASE_DIR, DATA_FILE)

# Stage-0 extractions written by your rolling-horizon pipeline:
STAGE0_PRICES_CSV = os.path.join(BASE_DIR, "full_stage0_prices.csv")            # node,start_week,t,price
STAGE0_PV_CSV     = os.path.join(BASE_DIR, "full_stage0_pv_quantities.csv")     # start_week,t,pv_generation

DEFAULT_BASE_DATE = "2030-01-01"
HOURS_PER_WEEK = 168
WEEKS_PER_STEP = 2

# ---- SCOPE ----
CONTROL_AREA = "AT"   # <- now works (uses node_to_control_area from load_demand_data)
NODE = None           # set e.g. "DE" and set CONTROL_AREA=None if you want node scope

INSTRUMENTS = ["RTP", "Peak-Shaving", "Capacity-Tariff"]

START = "2030-07-01 00:00"
END = "2030-07-02 00:00"

SHOW_PF_FLOWS = True

# ---- Requested switches ----
SHOW_EXCHANGE_TABLES = False   # "comment out" Exchange tables (TABLE 2A + 2B)


# ============================================================
# Helpers: pipeline naming
# ============================================================

def stage1_base_path(base_dir: str) -> str:
    return os.path.join(base_dir, "full_model_results_stage1_base.csv")

def follower_base_path(base_dir: str) -> str:
    return os.path.join(base_dir, "full_follower_values_base.csv")

def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def stage1_instr_path(base_dir: str, instrument: str) -> str:
    slug = instrument_slug(instrument)
    return os.path.join(base_dir, f"full_model_results_stage1_instrument__{slug}.csv")

def follower_instr_path(base_dir: str, instrument: str) -> str:
    slug = instrument_slug(instrument)
    return os.path.join(base_dir, f"full_follower_values_instrument__{slug}.csv")


# ============================================================
# Import Parameters loader (prefer updated)
# ============================================================

def call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)

try:
    from Parameters_Updated import (
        load_demand_data,
        load_thermal_power_plant_data,
        load_phs_power_plant_data,
        load_renewable_power_plant_data,
        load_flexibility_data,
        load_exchange_data,
    )
    _PARAMS_FLAVOR = "Parameters_Updated"
except Exception:
    from Parameters import (  # type: ignore
        load_demand_data,
        load_thermal_power_plant_data,
        load_phs_power_plant_data,
        load_renewable_power_plant_data,
        load_flexibility_data,
        load_exchange_data,
    )
    _PARAMS_FLAVOR = "Parameters"


# ============================================================
# Date parsing helpers (STRICT)
# ============================================================

DATE_FMT = "%Y-%m-%d %H:%M"

def coerce_base_date(base_date: str) -> str:
    return str(base_date).strip()

def parse_user_datetime(dt: str, base_date: str = DEFAULT_BASE_DATE) -> pd.Timestamp:
    return pd.to_datetime(dt, format=DATE_FMT)

def parse_user_range(
    start: str | None,
    end: str | None,
    base_date: str = DEFAULT_BASE_DATE,
) -> tuple[str, str]:
    if start is None or end is None:
        raise ValueError("Both start and end must be provided.")
    ts0 = parse_user_datetime(start, base_date)
    ts1 = parse_user_datetime(end, base_date)
    if ts1 < ts0:
        raise ValueError("End datetime must be >= start datetime.")
    return ts0.strftime(DATE_FMT), ts1.strftime(DATE_FMT)

def date_to_abs_hour(ts: str, base_date: str = DEFAULT_BASE_DATE) -> int:
    base_iso = coerce_base_date(base_date)
    t0 = pd.to_datetime(base_iso + " 00:00", format=DATE_FMT)
    t = parse_user_datetime(ts, base_date)
    return int((t - t0).total_seconds() // 3600) + 1

def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust abs_hour reconstruction.

    Priority:
      1) If 'window' exists: treat it as rolling-window index and compute abs_hour
         from window + within-window timestep using WINDOW_HOURS (=168*WEEKS_PER_STEP).
      2) Else fall back to start_week-based reconstruction.

    This avoids start_week ambiguity in rolling horizon exports.
    """
    if df is None or df.empty:
        return df
    if "abs_hour" in df.columns:
        return df

    df = df.copy()

    # pick timestep column
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

    # detect 0-based vs 1-based within-window timestep
    tmin = int(df[tcol].min()) if len(df) else 0
    offset = 1 if tmin == 0 else 0

    # ---- preferred: window-based ----
    if "window" in df.columns:
        w = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int)
        window_hours = int(HOURS_PER_WEEK) * int(WEEKS_PER_STEP)   # 336 if 2-week windows
        df["abs_hour"] = w * window_hours + df[tcol] + offset
        return df

    # ---- fallback: start_week-based ----
    if "start_week" not in df.columns:
        return df

    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(1).astype(int)
    df["abs_hour"] = df[tcol] + offset + HOURS_PER_WEEK * (df["start_week"] - 1)
    return df

# ============================================================
# Reading and shaping input CSVs
# ============================================================

def read_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)

    if "component" in df.columns and "variable" not in df.columns:
        df = df.rename(columns={"component": "variable"})

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "start_week" not in df.columns and "window" in df.columns:
        df["start_week"] = (
            pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) * WEEKS_PER_STEP + 1
        )

    df = ensure_abs_hour(df)
    return df


# ============================================================
# Demand helpers (use node_demand_data + node_to_control_area)
# ============================================================

def demand_series_from_node_demand_data(
    node_demand_df: pd.DataFrame,
    *,
    node_to_control_area: dict,
    control_area: str | None,
    node: str | None,
    hours_per_week: int = HOURS_PER_WEEK,
) -> pd.DataFrame:
    """
    Uses node_demand_df returned by your load_demand_data (second return value):
      - index: Hour (1..N)
      - columns: nodes + foreign CA aggregates at end

    If node is given: demand is node_demand_df[node]
    If control_area is given: sum all columns with node_to_control_area[col] == control_area

    Returns: start_week,timestep,demand (MW)
    """
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of control_area or node for demand scope.")

    if node_demand_df is None or node_demand_df.empty:
        raise ValueError("node_demand_df is empty (from load_demand_data).")

    df = node_demand_df.copy()

    # ensure numeric index "Hour"
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[df.index.notna()].copy()
    df.index = df.index.astype(int)

    df = df[(df.index >= 1) & (df.index <= int(hours_per_week))].copy()
    if df.empty:
        raise ValueError("node_demand_df has no rows in [1..hours_per_week].")

    if node is not None:
        n = str(node).strip()
        if n not in df.columns.astype(str).tolist():
            raise ValueError(f"Node demand column '{n}' not found in node_demand_df columns.")
        s = pd.to_numeric(df[n], errors="coerce").fillna(0.0)
    else:
        ca = str(control_area).strip()
        cols = [c for c in df.columns.astype(str) if str(node_to_control_area.get(str(c), "")).strip() == ca]
        if not cols:
            # helpful diagnostics
            ca_set = sorted(set(map(str, node_to_control_area.values())))
            raise ValueError(
                f"No demand columns found for control_area='{ca}'. "
                f"Available control areas in mapping: {ca_set}"
            )
        s = pd.to_numeric(df[cols].sum(axis=1), errors="coerce").fillna(0.0)

    out = pd.DataFrame(
        {
            "start_week": 1,
            "timestep": df.index.astype(int),
            "demand": s.values.astype(float),
        }
    )
    return out[["start_week", "timestep", "demand"]]


# ============================================================
# PF bridge helper: SINGLE SOURCE OF TRUTH for plotting+printing
# ============================================================

def compute_pf_bridge_series(
    *,
    pf_nodewise: pd.DataFrame | None,
    nodes_set: set[str],
    abs_hours: np.ndarray,
    abs_to_dt: dict[int, pd.Timestamp],
) -> dict[str, pd.Series]:
    """
    Returns time-indexed series aligned to the provided abs_hours/time mapping.
    These are the *only* follower↔system boundary flows used for the system stackplot:

      exports: pv_feed_to_system, battery_to_system
      imports: imports_to_demand, imports_to_battery
    """
    abs_list = [int(h) for h in np.asarray(abs_hours).astype(int).tolist()]
    time_index = pd.to_datetime([abs_to_dt[int(h)] for h in abs_list])

    zero = pd.Series(0.0, index=time_index)

    out = {
        "pf_exports_pv": zero.copy(),
        "pf_exports_battery": zero.copy(),
        "pf_imports_to_demand": zero.copy(),
        "pf_imports_to_battery": zero.copy(),
    }

    if pf_nodewise is None or pf_nodewise.empty:
        return out

    pf = ensure_abs_hour(pf_nodewise.copy())
    if "node" not in pf.columns or "abs_hour" not in pf.columns:
        return out

    pf["node"] = pf["node"].astype(str)
    pf = pf[pf["node"].isin(set(map(str, nodes_set)))].copy()

    pf["abs_hour"] = pd.to_numeric(pf["abs_hour"], errors="coerce")
    pf = pf.dropna(subset=["abs_hour"]).copy()
    pf["abs_hour"] = pf["abs_hour"].astype(int)
    pf = pf[pf["abs_hour"].isin(set(abs_list))].copy()

    if pf.empty:
        return out

    for c in ["pv_feed_to_system", "battery_to_system", "imports_to_demand", "imports_to_battery"]:
        pf[c] = pd.to_numeric(pf.get(c, 0.0), errors="coerce").fillna(0.0)

    agg = pf.groupby("abs_hour", as_index=True).agg(
        pf_exports_pv=("pv_feed_to_system", "sum"),
        pf_exports_battery=("battery_to_system", "sum"),
        pf_imports_to_demand=("imports_to_demand", "sum"),
        pf_imports_to_battery=("imports_to_battery", "sum"),
    )

    agg = agg.reindex(abs_list, fill_value=0.0)
    agg.index = time_index

    out["pf_exports_pv"] = agg["pf_exports_pv"]
    out["pf_exports_battery"] = agg["pf_exports_battery"]
    out["pf_imports_to_demand"] = agg["pf_imports_to_demand"]
    out["pf_imports_to_battery"] = agg["pf_imports_to_battery"]
    return out


# ============================================================
# Stage-1 base price reconstruction (from Stage-0 CSVs)
# ============================================================

def pv_weighted_price_over_horizon(prices_by_t: pd.DataFrame, pv_by_t: pd.DataFrame) -> float:
    """
    Exactly mirrors your pipeline logic (PV-weighted price).
    prices_by_t: columns t, price  (use abs/global t)
    pv_by_t:     columns t, pv_generation
    """
    if prices_by_t is None or prices_by_t.empty:
        raise ValueError("prices_by_t is empty.")
    if pv_by_t is None or pv_by_t.empty:
        raise ValueError("pv_by_t is empty.")
    if not {"t", "price"}.issubset(prices_by_t.columns):
        raise ValueError("prices_by_t must contain columns ['t','price'].")
    if not {"t", "pv_generation"}.issubset(pv_by_t.columns):
        raise ValueError("pv_by_t must contain columns ['t','pv_generation'].")

    df = prices_by_t[["t", "price"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # same sign handling as your pipeline
    if (df["price"] <= 0).all():
        df["price"] = -df["price"]
    elif (df["price"] >= 0).all():
        pass
    else:
        bad = df[df["price"] < 0].head()
        raise ValueError(f"Mixed positive/negative prices detected. Examples:\n{bad}")

    pv = pv_by_t[["t", "pv_generation"]].copy()
    pv["pv_generation"] = pd.to_numeric(pv["pv_generation"], errors="coerce").fillna(0.0).clip(lower=0.0)

    merged = df.merge(pv, on="t", how="inner")
    if merged.empty:
        raise ValueError("No overlap in 't' between prices_by_t and pv_by_t.")

    pv_sum = float(merged["pv_generation"].sum())
    if pv_sum <= 0.0:
        raise ValueError("Sum of PV generation is 0; cannot compute PV-weighted price.")

    weights = merged["pv_generation"] / pv_sum
    p_base = float((merged["price"] * weights).sum())
    if not np.isfinite(p_base):
        raise ValueError("PV-weighted price is not finite.")
    return p_base


def compute_stage1_base_price_from_stage0_csvs(prices_csv: str, pv_csv: str) -> float:
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(f"Missing Stage-0 prices CSV: {prices_csv}")
    if not os.path.exists(pv_csv):
        raise FileNotFoundError(f"Missing Stage-0 PV CSV: {pv_csv}")

    prices = pd.read_csv(prices_csv, low_memory=False)
    pv = pd.read_csv(pv_csv, low_memory=False)

    req_p = {"t", "price"}
    req_v = {"t", "pv_generation"}
    if not req_p.issubset(prices.columns):
        raise ValueError(f"Stage-0 prices CSV missing columns {sorted(req_p - set(prices.columns))}")
    if not req_v.issubset(pv.columns):
        raise ValueError(f"Stage-0 PV CSV missing columns {sorted(req_v - set(pv.columns))}")

    prices_by_t = prices.groupby("t", as_index=False)["price"].mean()
    pv_by_t = pv.groupby("t", as_index=False)["pv_generation"].sum()

    return pv_weighted_price_over_horizon(prices_by_t, pv_by_t)


def nodal_prices_timeseries_from_stage0(
    stage0_prices_csv: str,
    *,
    start: str,
    end: str,
    base_date: str,
) -> pd.DataFrame:
    """
    Returns time-indexed nodal price time series:
      index = datetime
      columns = node names
      values = Stage-0 dual price
    """

    if not os.path.exists(stage0_prices_csv):
        raise FileNotFoundError(f"Stage-0 prices file not found: {stage0_prices_csv}")

    df = pd.read_csv(stage0_prices_csv)

    required = {"node", "start_week", "t", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stage-0 prices missing columns: {missing}")

    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Window slicing
    use_start, use_end = parse_user_range(start, end, base_date)
    ah0 = date_to_abs_hour(use_start, base_date)
    ah1 = date_to_abs_hour(use_end, base_date)

    df = df[(df["t"] >= ah0) & (df["t"] <= ah1)].copy()
    if df.empty:
        raise ValueError("No Stage-0 prices found in selected window.")

    # Pivot to wide format
    price_wide = df.pivot_table(
        index="t",
        columns="node",
        values="price",
        aggfunc="mean",
    )

    # Convert to datetime index
    base_ts = pd.to_datetime(base_date + " 00:00", format=DATE_FMT)
    price_wide.index = [
        base_ts + pd.Timedelta(hours=int(t - 1))
        for t in price_wide.index
    ]

    price_wide = price_wide.sort_index()

    return price_wide


# ============================================================
# Sanity checks / prints
# ============================================================

def print_stackplot_variables(
    *,
    time_index: pd.DatetimeIndex,
    abs_hours: np.ndarray,
    gen_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    dem_excel_s: pd.Series,
    exchange_signed_s: pd.Series,
    stage_df: pd.DataFrame | None = None,
    pf_nodewise: pd.DataFrame | None = None,
    nodes_in_scope: set[str] | None = None,
    pf_bridge: dict[str, pd.Series] | None = None,
    n: int = 10,
    decimals: int = 2,
    flow_top_k: int = 12,
    show_flow_pos_neg: bool = False,
    title: str | None = None,
) -> None:
    """
    TABLE 1: Stackplot variables (system) + NET follower imports/exports (from pf_bridge)
    TABLE 2A/2B: Exchange (OPTIONAL; toggled by SHOW_EXCHANGE_TABLES)
    TABLE 3: Explicit follower vars (your list) + price cols
    """

    def _print_df(df: pd.DataFrame, header: str) -> None:
        print("\n" + "-" * 90)
        print(header)
        print("-" * 90)
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 260,
            "display.max_rows", None,
        ):
            print(df.round(int(decimals)).head(int(n)))

    def _reindex_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(index=time_index)
        return df.reindex(time_index).fillna(0.0)

    def _reindex_s(s: pd.Series) -> pd.Series:
        if s is None or getattr(s, "empty", False):
            return pd.Series(0.0, index=time_index)
        return s.reindex(time_index).fillna(0.0)

    def _ensure_abs_hour_like(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "abs_hour" in df.columns:
            return df

        df = df.copy()
        if "start_week" not in df.columns and "window" in df.columns:
            df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1
        if "start_week" not in df.columns:
            df["start_week"] = 1
        df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(1).astype(int)

        if "timestep" in df.columns:
            tcol = "timestep"
        elif "t" in df.columns:
            tcol = "t"
        elif "index_1" in df.columns:
            tcol = "index_1"
        else:
            return df

        df[tcol] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype(int)
        df["abs_hour"] = df[tcol] + HOURS_PER_WEEK * (df["start_week"] - 1)
        return df

    if len(time_index) != len(abs_hours):
        raise ValueError("time_index and abs_hours must have the same length.")

    abs_hours = np.asarray(abs_hours).astype(int)
    abs_set = set(abs_hours.tolist())
    abs_to_dt = {int(h): time_index[i] for i, h in enumerate(abs_hours.tolist())}

    gen = _reindex_df(gen_df).copy()
    cons = _reindex_df(cons_df).copy()
    dem_excel_s = _reindex_s(dem_excel_s)
    ex = _reindex_s(exchange_signed_s)

    if title:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    # =========================================================
    # PF bridge: use the SAME series as the plot
    # =========================================================
    if pf_bridge is None:
        if nodes_in_scope is None:
            nodes_in_scope = set()
        pf_bridge = compute_pf_bridge_series(
            pf_nodewise=pf_nodewise,
            nodes_set=set(map(str, nodes_in_scope)),
            abs_hours=abs_hours,
            abs_to_dt=abs_to_dt,
        )

    pf_exp_pv = pf_bridge.get("pf_exports_pv", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_exp_ba = pf_bridge.get("pf_exports_battery", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_imp_de = pf_bridge.get("pf_imports_to_demand", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_imp_ba = pf_bridge.get("pf_imports_to_battery", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)

    pf_exports_s = (pf_exp_pv + pf_exp_ba).reindex(time_index).fillna(0.0)
    pf_imports_s = (pf_imp_de + pf_imp_ba).reindex(time_index).fillna(0.0)

    # =========================================================
    # TABLE 1 — Stackplot variables (NET follower effect only)
    # =========================================================
    
    drop_gen_cols = [
        "Follower exports (from PV)",
        "Follower exports (from battery)",
    ]
    drop_cons_cols = [
        "Follower imports (to demand)",
        "Follower imports (to battery)",
    ]
    gen_clean = gen.drop(columns=drop_gen_cols, errors="ignore")
    cons_clean = cons.drop(columns=drop_cons_cols, errors="ignore")

    out1 = pd.concat(
        [
            gen_clean,
            cons_clean.add_prefix("-"),
            pd.DataFrame(
                {
                    "Follower exports (PV+battery)": pf_exports_s.values,
                    "-Follower imports (demand+battery)": pf_imports_s.values,
                    "Demand (from Excel)": dem_excel_s.values,
                },
                index=time_index,
            ),
        ],
        axis=1,
    )
    _print_df(out1, "TABLE 1 — Stackplot variables (net follower imports/exports only)")

    # =========================================================
    # TABLE 2A / 2B — Exchange tables (OPTIONAL)
    # =========================================================
    
    if SHOW_EXCHANGE_TABLES:
        # TABLE 2A — Exchange + imports/exports
        t2a = pd.DataFrame(
            {
                "Exchange (signed)": ex.values,
                "Imports (>=0)": ex.clip(lower=0.0).values,
                "Exports (>=0)": (-ex).clip(lower=0.0).values,
            },
            index=time_index,
        )
        _print_df(t2a, "TABLE 2A — Exchange (scope)")

        # TABLE 2B — System model flows (connection indexed)
        if stage_df is None or stage_df.empty:
            _print_df(
                pd.DataFrame({"(no stage_df provided)": [np.nan] * len(time_index)}, index=time_index),
                "TABLE 2B — System model flows (missing stage_df)",
            )
        else:
            varcol = "variable" if "variable" in stage_df.columns else ("component" if "component" in stage_df.columns else None)
            if varcol is None:
                _print_df(
                    pd.DataFrame({"(stage_df has no variable/component col)": [np.nan] * len(time_index)}, index=time_index),
                    "TABLE 2B — System model flows (cannot locate variable names)",
                )
            else:
                st = stage_df.copy()
                st[varcol] = st[varcol].astype(str)
                flows = st[st[varcol] == "flow"].copy()
                flows = _ensure_abs_hour_like(flows)

                if flows.empty or "index_0" not in flows.columns:
                    vars_present = sorted(set(st[varcol].unique().tolist()))
                    diag = pd.DataFrame(
                        {
                            "note": ["No system variable == 'flow' rows found (or missing index_0)."],
                            "some_variables": [", ".join(vars_present[:40]) + (" ..." if len(vars_present) > 40 else "")],
                        },
                        index=[time_index[0]],
                    )
                    _print_df(diag, "TABLE 2B — System model flows: NOT FOUND (diagnostic)")
                else:
                    flows["connection"] = flows["index_0"].astype(str)
                    flows["abs_hour"] = pd.to_numeric(flows["abs_hour"], errors="coerce").fillna(-999).astype(int)
                    flows["flow"] = pd.to_numeric(flows.get("value", 0.0), errors="coerce").fillna(0.0)
                    flows = flows[flows["abs_hour"].isin(abs_set)].copy()

                    if flows.empty:
                        _print_df(
                            pd.DataFrame({"(flow exists but empty in window)": [np.nan] * len(time_index)}, index=time_index),
                            "TABLE 2B — System model flows (empty in window)",
                        )
                    else:
                        score = flows.groupby("connection")["flow"].apply(lambda s: float(np.abs(s).sum()))
                        top_conns = score.sort_values(ascending=False).head(int(flow_top_k)).index.tolist()
                        flows = flows[flows["connection"].isin(top_conns)].copy()

                        flow_wide = flows.pivot_table(
                            index="abs_hour",
                            columns="connection",
                            values="flow",
                            aggfunc="sum",
                            fill_value=0.0,
                        )

                        abs_head = [int(h) for h in abs_hours[: int(n)]]
                        flow_wide = flow_wide.reindex(abs_head, fill_value=0.0)
                        flow_wide.index = [abs_to_dt[h] for h in abs_head]
                        flow_wide.columns = [f"flow[{c}]" for c in flow_wide.columns]

                        _print_df(flow_wide, f"TABLE 2B — System line flows (flow[c,t]) — top {int(flow_top_k)} connections")

                        if show_flow_pos_neg:
                            for vn, lab in [("flow_pos", "flow_pos[c,t]"), ("flow_neg", "flow_neg[c,t]")]:
                                part = st[st[varcol] == vn].copy()
                                part = _ensure_abs_hour_like(part)
                                if part.empty or "index_0" not in part.columns:
                                    continue
                                part["connection"] = part["index_0"].astype(str)
                                part["abs_hour"] = pd.to_numeric(part["abs_hour"], errors="coerce").fillna(-999).astype(int)
                                part["val"] = pd.to_numeric(part.get("value", 0.0), errors="coerce").fillna(0.0)
                                part = part[part["abs_hour"].isin(set(abs_head)) & part["connection"].isin(top_conns)].copy()
                                if part.empty:
                                    continue
                                wide = part.pivot_table(index="abs_hour", columns="connection", values="val", aggfunc="sum", fill_value=0.0)
                                wide = wide.reindex(abs_head, fill_value=0.0)
                                wide.index = [abs_to_dt[h] for h in abs_head]
                                wide.columns = [f"{vn}[{c}]" for c in wide.columns]
                                _print_df(wide, f"TABLE 2C — {lab} (same top-k connections)")

    # =========================================================
    # TABLE 3 — explicit follower vars + follower price col(s)
    # =========================================================
    
    if pf_nodewise is None or pf_nodewise.empty:
        _print_df(
            pd.DataFrame({"(no pf_nodewise provided)": [np.nan] * len(time_index)}, index=time_index),
            "TABLE 3 — PF follower vars (missing pf_nodewise)",
        )
        return

    if nodes_in_scope is None:
        nodes_in_scope = set()

    pf = ensure_abs_hour(pf_nodewise.copy())
    if "node" not in pf.columns or "abs_hour" not in pf.columns:
        _print_df(
            pd.DataFrame({"(pf_nodewise missing 'node'/'abs_hour')": [np.nan] * len(time_index)}, index=time_index),
            "TABLE 3 — PF follower vars (invalid pf_nodewise format)",
        )
        return

    pf["node"] = pf["node"].astype(str)
    if nodes_in_scope:
        pf = pf[pf["node"].isin(set(map(str, nodes_in_scope)))].copy()

    pf["abs_hour"] = pd.to_numeric(pf["abs_hour"], errors="coerce").fillna(-999).astype(int)
    pf = pf[pf["abs_hour"].isin(abs_set)].copy()

    if pf.empty:
        _print_df(
            pd.DataFrame({"(no PF rows in scope/window)": [np.nan] * len(time_index)}, index=time_index),
            "TABLE 3 — PF follower vars (scope/window empty)",
        )
        return

    follower_flow_vars = [
        "pv_to_demand",
        "pv_to_battery",
        "pv_feed_to_system",
        "pv_curtailment",
        "imports_to_demand",
        "imports_to_battery",
        "battery_to_demand",
        "battery_to_system",
        "battery_storage",
    ]
    for c in follower_flow_vars:
        pf[c] = pd.to_numeric(pf.get(c, 0.0), errors="coerce").fillna(0.0)

    sum_agg = pf.groupby("abs_hour", as_index=True)[follower_flow_vars].sum()

    # detect price-like columns in follower output
    candidate_cols = [c for c in pf.columns if c not in {"node", "start_week", "window", "timestep", "t", "index_0", "index_1", "abs_hour"}]
    price_cols = [c for c in candidate_cols if "price" in c.lower() or any(k in c.lower() for k in ["tariff", "fee", "fees", "markup", "spread", "lmp", "lambda"])]
    if "price" in pf.columns and "price" not in price_cols:
        price_cols.append("price")
    price_cols = [c for c in dict.fromkeys(price_cols) if c in pf.columns]

    if price_cols:
        tmp = pf[["abs_hour"] + price_cols].copy()
        for c in price_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
        price_agg = tmp.groupby("abs_hour", as_index=True)[price_cols].mean()
        t3 = pd.concat([price_agg, sum_agg], axis=1)
    else:
        t3 = sum_agg.copy()

    abs_head = [int(h) for h in abs_hours[: int(n)]]
    t3 = t3.reindex(abs_head, fill_value=0.0)
    t3.index = [abs_to_dt[h] for h in abs_head]

    rename_map = {}
    if "price" in t3.columns:
        rename_map["price"] = "PF input price [€/MWh]"
    if "battery_storage" in t3.columns:
        rename_map["battery_storage"] = "Battery SOC (sum) [MWh]"
    t3 = t3.rename(columns=rename_map)

    _print_df(t3, "TABLE 3 — PF follower vars (sum) + follower price cols (mean)")


# ============================================================
# Plot: instrument-only stacks
# ============================================================

# ============================================================
# SUPPLY SIDE (Generation & Inflows)
# ============================================================

COLORS_SUPPLY_PUB = {
    "Thermal":                         "#D55E00",  # vermillion (strong, distinct warm tone)

    "PHS (gen)":                       "#0072B2",  # deep blue
    "Battery (out)":                   "#E69F00",  # orange (distinct from Thermal)
    
    "RES":                             "#009E73",  # bluish green (keep this fixed across paper)
    "DSR (down)":                      "#44AA99",  # teal (clearly distinct from RES)

    "Imports":                         "#56B4E9",  # sky blue (lighter than PHS gen)

    # Follower flows → purple/magenta family (own visual block)
    "Follower exports (from PV)":      "#CC79A7",  # magenta
    "Follower exports (from battery)": "#AA4499",  # deep magenta

    "NSE":                             "#000000",  # black (must remain strongest contrast)
}


# ============================================================
# CONSUMPTION SIDE (Load & Outflows)
# ============================================================

COLORS_CONS_PUB = {
    "PHS (pump)":                      "#004488",  # dark navy (not same as PHS gen)
    "Battery (in)":                    "#EE7733",  # darker orange (paired with Battery out)

    "DSR (up)":                        "#999999",  # neutral grey (demand flexibility)

    "Exports":                         "#4D4D4D",  # dark grey (distinct from NSE black)

    # Follower imports → red/pink family (NOT same as Thermal)
    "Follower imports (to demand)":    "#BB5566",  # muted red
    "Follower imports (to battery)":   "#EE99AA",  # light rose

    "Curtailment":                     "#882255",  # dark wine (clearly not magenta export)
}


def plot_total_values_instrument_only(
    stage_df: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
    node_to_control_area: dict,
    *,
    case_name: str,
    control_area: str | None = None,
    node: str | None = None,
    pf_nodewise: pd.DataFrame | None = None,
    thermal_node_idx: Dict[str, list],
    phs_node_idx: Dict[str, list],
    renewable_node_idx: Dict[str, list],
    dsr_node_idx: Dict[str, list],
    battery_node_idx: Dict[str, list],
    node_demand_df: pd.DataFrame | None = None,
    base_date: str = DEFAULT_BASE_DATE,
    start: str = None,  # type: ignore
    end: str = None,    # type: ignore
    demand_fraction: float = 0.0,
    apply_demand_fraction_to_excel: bool = True,
    show_model_demand_line: bool = True,
    show_balance_debug: bool = True,

    # --- NEW: publication controls ---
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    title_prefix: str | None = None,   # e.g. "AT" instead of "Area_AT"
) -> None:
    
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    # -----------------------------
    # Scope: nodes in CA or single node
    # -----------------------------
    if node is not None:
        nodes = [str(node)]
        scope_label = f"Node_{node}"
        demand_scope_control_area = None
        demand_scope_node = str(node)
    else:
        ca_use = str(control_area)
        nodes = [n for n, ca in node_to_control_area.items() if str(ca) == ca_use]
        if not nodes:
            raise ValueError(
                f"No nodes found for control_area = {control_area}. "
                f"Available control areas = {sorted(set(map(str, node_to_control_area.values())))}"
            )
        scope_label = f"Area_{control_area}"
        demand_scope_control_area = ca_use
        demand_scope_node = None

    nodes_set = set(map(str, nodes))

    # -----------------------------
    # Aggregation helper
    # -----------------------------
    def agg_by_nodes(
        varname: str,
        plant_to_nodes: Dict[str, list] | None,
        *,
        is_node_indexed: bool = False,
    ) -> pd.DataFrame:
        part = stage_df[stage_df["variable"] == varname].copy()
        if part.empty:
            return pd.DataFrame(columns=["start_week", "timestep", "value"])

        part["start_week"] = pd.to_numeric(part.get("start_week", 1), errors="coerce").fillna(1).astype(int)

        if is_node_indexed:
            if "index_0" not in part.columns:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])
            part["node"] = part["index_0"].astype(str)
        else:
            if not plant_to_nodes or "index_0" not in part.columns:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])
            m = [(str(p), str(n)) for n, plants in plant_to_nodes.items() for p in plants]
            map_df = pd.DataFrame(m, columns=["plant", "node"])
            part["plant"] = part["index_0"].astype(str)
            part = part.merge(map_df, on="plant", how="inner")
            if part.empty:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])

        # timestep parsing
        if "index_1" in part.columns:
            part["timestep"] = pd.to_numeric(part["index_1"], errors="coerce").fillna(0).astype(int)
        elif "timestep" in part.columns:
            part["timestep"] = pd.to_numeric(part["timestep"], errors="coerce").fillna(0).astype(int)
        else:
            return pd.DataFrame(columns=["start_week", "timestep", "value"])

        part["node"] = part["node"].astype(str)
        part = part[part["node"].isin(nodes_set)]
        if part.empty:
            return pd.DataFrame(columns=["start_week", "timestep", "value"])

        part["value"] = pd.to_numeric(part["value"], errors="coerce").fillna(0.0)
        return part.groupby(["start_week", "timestep"], as_index=False)["value"].sum()

    # -----------------------------
    # Supply components (system)
    # -----------------------------
    thermal  = agg_by_nodes("thermal_generation", thermal_node_idx)
    phs_gen  = agg_by_nodes("phs_turbine_generation", phs_node_idx)
    res_gen  = agg_by_nodes("renewable_generation", renewable_node_idx)
    dsr_down = agg_by_nodes("dsr_down", dsr_node_idx)
    bat_out  = agg_by_nodes("battery_out", battery_node_idx)

    # -----------------------------
    # Consumption components (system)
    # -----------------------------
    phs_pump = agg_by_nodes("phs_pump_consumption", phs_node_idx)
    dsr_up   = agg_by_nodes("dsr_up", dsr_node_idx)
    bat_in   = agg_by_nodes("battery_in", battery_node_idx)

    # -----------------------------
    # Balance terms from system
    # -----------------------------
    nse = agg_by_nodes("nse", None, is_node_indexed=True)
    curtail = agg_by_nodes("curtailment", None, is_node_indexed=True)

    exchange_ex = agg_by_nodes("exchange", None, is_node_indexed=True)
    if exchange_ex.empty:
        exchange_ex = agg_by_nodes("expr:exchange", None, is_node_indexed=True)
    if exchange_ex.empty:
        exchange_ex = pd.DataFrame(columns=["start_week", "timestep", "value"])

    # -----------------------------
    # Merge onto a common base table
    # -----------------------------
    def m2(left: pd.DataFrame, comp: pd.DataFrame, name: str) -> pd.DataFrame:
        if left.empty and (comp is None or comp.empty):
            return pd.DataFrame(columns=["start_week", "timestep", name])
        base0 = comp[["start_week", "timestep"]].drop_duplicates().copy() if left.empty else left
        if comp is None or comp.empty:
            base0[name] = 0.0
            return base0
        return (
            base0.merge(comp.rename(columns={"value": name}), on=["start_week", "timestep"], how="left")
            .fillna({name: 0.0})
        )

    base = (
        thermal.rename(columns={"value": "thermal"})
        if not thermal.empty
        else pd.DataFrame(columns=["start_week", "timestep", "thermal"])
    )
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

    base = base.merge(
        (nse.rename(columns={"value": "nse"}) if not nse.empty else pd.DataFrame(columns=["start_week", "timestep", "nse"])),
        on=["start_week", "timestep"], how="left"
    ).fillna({"nse": 0.0})

    base = base.merge(
        (curtail.rename(columns={"value": "curtailment"}) if not curtail.empty else pd.DataFrame(columns=["start_week", "timestep", "curtailment"])),
        on=["start_week", "timestep"], how="left"
    ).fillna({"curtailment": 0.0})

    base = base.merge(
        (exchange_ex.rename(columns={"value": "exchange"}) if not exchange_ex.empty else pd.DataFrame(columns=["start_week", "timestep", "exchange"])),
        on=["start_week", "timestep"], how="left"
    ).fillna({"exchange": 0.0})

    base["start_week"] = pd.to_numeric(base.get("start_week", 1), errors="coerce").fillna(1).astype(int)
    base["timestep"] = pd.to_numeric(base.get("timestep", 0), errors="coerce").fillna(0).astype(int)
    base["abs_hour"] = base["timestep"] + HOURS_PER_WEEK * (base["start_week"] - 1)

    # ============================================================
    # Demand line (from node_demand_df)
    # ============================================================
    if node_demand_df is None or node_demand_df.empty:
        raise ValueError("node_demand_df (from load_demand_data) is required to plot demand, especially for AT.")

    demand_excel = demand_series_from_node_demand_data(
        node_demand_df,
        node_to_control_area=node_to_control_area,
        control_area=demand_scope_control_area,
        node=demand_scope_node,
        hours_per_week=HOURS_PER_WEEK,
    ).copy()
    demand_excel["abs_hour"] = (
        demand_excel["timestep"].astype(int)
        + HOURS_PER_WEEK * (demand_excel["start_week"].astype(int) - 1)
    )

    # Optional alignment with system model demand_fraction logic (approx for CA-scope)
    if apply_demand_fraction_to_excel and demand_fraction > 0.0 and pf_nodewise is not None and not pf_nodewise.empty:
        pf_tmp = ensure_abs_hour(pf_nodewise.copy())
        if "node" in pf_tmp.columns:
            pf_tmp["node"] = pf_tmp["node"].astype(str)
            pf_tmp = pf_tmp[pf_tmp["node"].isin(nodes_set)]
            follower_nodes_present = set(pf_tmp["node"].unique().tolist())
        else:
            follower_nodes_present = set()

        if node is not None and str(node) in follower_nodes_present:
            demand_excel["demand"] = demand_excel["demand"] * float(demand_fraction)
        elif node is None and len(nodes_set) > 0:
            frac = (len(follower_nodes_present) / float(len(nodes_set))) if len(nodes_set) else 0.0
            demand_excel["demand"] = demand_excel["demand"] * float(demand_fraction) * float(frac)

    base = base.merge(demand_excel[["abs_hour", "demand"]], on="abs_hour", how="left")
    base["demand"] = pd.to_numeric(base["demand"], errors="coerce").fillna(0.0)

    # -----------------------------
    # Window slice
    # -----------------------------
    use_start, use_end = parse_user_range(start, end, base_date)
    ah0 = date_to_abs_hour(use_start, base_date)
    ah1 = date_to_abs_hour(use_end, base_date)

    w = base[(base["abs_hour"] >= ah0) & (base["abs_hour"] <= ah1)].copy()
    if w.empty:
        raise ValueError(f"Window [{use_start} → {use_end}] produced an empty slice.")

    idx = np.unique(w["abs_hour"].values)
    base_ts = pd.to_datetime(coerce_base_date(base_date) + " 00:00", format=DATE_FMT)
    time_index = pd.to_datetime([base_ts + pd.Timedelta(hours=int(h - 1)) for h in idx])

    abs_to_dt = {int(h): time_index[i] for i, h in enumerate(idx.tolist())}

    def series_select(col: str) -> pd.Series:
        s = w.set_index("abs_hour")[col]
        return s.reindex(idx).fillna(0.0)

    # system stacks
    thermal_s = series_select("thermal")
    phs_gen_s = series_select("phs_gen")
    res_s     = series_select("res")
    dsr_dn_s  = series_select("dsr_out")
    batt_o_s  = series_select("battery_out")

    phs_p_s   = series_select("phs_cons")
    dsr_up_s  = series_select("dsr_in")
    batt_i_s  = series_select("battery_in")

    exch_s    = series_select("exchange")
    nse_s     = series_select("nse")
    curt_s    = series_select("curtailment")

    for s in [thermal_s, phs_gen_s, res_s, dsr_dn_s, batt_o_s,
              phs_p_s, dsr_up_s, batt_i_s, exch_s, nse_s, curt_s]:
        s.index = time_index

    # --- Demand aligned to the plotted absolute hours (repeat weekly profile) ---
    how = ((idx - 1) % HOURS_PER_WEEK) + 1  # hour-of-week in 1..168
    
    if node is not None:
        n = str(node).strip()
        if n not in node_demand_df.columns.astype(str).tolist():
            raise ValueError(f"Node demand column '{n}' not found in node_demand_df.")
        dem_vals = pd.to_numeric(node_demand_df.loc[how, n], errors="coerce").fillna(0.0).to_numpy()
    else:
        ca = str(control_area).strip()
        cols = [c for c in node_demand_df.columns.astype(str) if str(node_to_control_area.get(str(c), "")).strip() == ca]
        if not cols:
            raise ValueError(f"No demand columns found for control_area='{ca}'.")
        dem_vals = pd.to_numeric(node_demand_df.loc[how, cols].sum(axis=1), errors="coerce").fillna(0.0).to_numpy()
    
    dem_excel_s = pd.Series(dem_vals, index=time_index)
    
    # keep your existing demand_fraction logic (apply after building dem_excel_s)
    if apply_demand_fraction_to_excel and demand_fraction > 0.0 and pf_nodewise is not None and not pf_nodewise.empty:
        pf_tmp = ensure_abs_hour(pf_nodewise.copy())
        follower_nodes_present = set()
        if "node" in pf_tmp.columns:
            pf_tmp["node"] = pf_tmp["node"].astype(str)
            pf_tmp = pf_tmp[pf_tmp["node"].isin(nodes_set)]
            follower_nodes_present = set(pf_tmp["node"].unique().tolist())
    
        if node is not None and str(node) in follower_nodes_present:
            dem_excel_s *= float(demand_fraction)
        elif node is None and len(nodes_set) > 0:
            frac = (len(follower_nodes_present) / float(len(nodes_set))) if len(nodes_set) else 0.0
            dem_excel_s *= float(demand_fraction) * float(frac)

    imports = exch_s.clip(lower=0.0)
    exports = (-exch_s).clip(lower=0.0)

    # ============================================================
    # PF bridge series: SINGLE SOURCE used for BOTH plot + print
    # ============================================================
    pf_bridge = compute_pf_bridge_series(
        pf_nodewise=pf_nodewise,
        nodes_set=nodes_set,
        abs_hours=idx,
        abs_to_dt=abs_to_dt,
    )
    pf_exp_pv_s  = pf_bridge["pf_exports_pv"].reindex(time_index).fillna(0.0)
    pf_exp_bat_s = pf_bridge["pf_exports_battery"].reindex(time_index).fillna(0.0)
    pf_imp_dem_s = pf_bridge["pf_imports_to_demand"].reindex(time_index).fillna(0.0)
    pf_imp_bat_s = pf_bridge["pf_imports_to_battery"].reindex(time_index).fillna(0.0)

    # -----------------------------
    # STACKS
    # -----------------------------
    gen_df = pd.DataFrame(
        {
            "Thermal": thermal_s.values,
            "PHS (gen)": phs_gen_s.values,
            "RES": res_s.values,
            "DSR (down)": dsr_dn_s.values,
            "Battery (out)": batt_o_s.values,
            "Imports": imports.values,
            "Follower exports (from PV)": pf_exp_pv_s.values,
            "Follower exports (from battery)": pf_exp_bat_s.values,
            "NSE": nse_s.values,
        },
        index=time_index,
    ).clip(lower=0.0)

    cons_df = pd.DataFrame(
        {
            "PHS (pump)": phs_p_s.values,
            "DSR (up)": dsr_up_s.values,
            "Battery (in)": batt_i_s.values,
            "Exports": exports.values,
            "Follower imports (to demand)": pf_imp_dem_s.values,
            "Follower imports (to battery)": pf_imp_bat_s.values,
            "Curtailment": curt_s.values,
        },
        index=time_index,
    ).clip(lower=0.0)

    # -----------------------------
    # Print first timesteps (tables)
    # -----------------------------
    print_stackplot_variables(
        time_index=time_index,
        abs_hours=idx,
        gen_df=gen_df,
        cons_df=cons_df,
        dem_excel_s=dem_excel_s,
        exchange_signed_s=exch_s,
        stage_df=stage_df,
        pf_nodewise=pf_nodewise,
        nodes_in_scope=nodes_set,
        pf_bridge=pf_bridge,  # pass SAME PF series used for plot
        n=24,
        decimals=2,
        flow_top_k=12,
        show_flow_pos_neg=False,
        title=f"{scope_label} ({case_name}) — first 24 timesteps: output tables",
    )

    # -----------------------------
    # Scope label (FIXED)
    # -----------------------------
    if node is not None:
        scope_short = str(node)
    else:
        scope_short = str(control_area)

    # if you still need the old internal label, keep it, but for titles use scope_short
    # scope_label = f"Node_{node}" if node is not None else f"Area_{control_area}"

    ...
    # -----------------------------
    # Plot (UPDATED: use provided ax; consistent colors; publication title)
    # -----------------------------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 7))
    else:
        fig = ax.figure

    gen_cols  = list(gen_df.columns)
    cons_cols = list(cons_df.columns)

    ax.stackplot(
        gen_df.index,
        [gen_df[c].values for c in gen_cols],
        labels=gen_cols,
        colors=[COLORS_SUPPLY_PUB.get(c, "#333333") for c in gen_cols],
        alpha=0.85,
        linewidth=0.0,
    )
    ax.stackplot(
        cons_df.index,
        [-cons_df[c].values for c in cons_cols],
        labels=[f"-{c}" for c in cons_cols],
        colors=[COLORS_CONS_PUB.get(c, "#999999") for c in cons_cols],
        alpha=0.85,
        linewidth=0.0,
    )

    # demand line
    ax.plot(
        time_index,
        dem_excel_s.values,
        color="black",
        linewidth=2.2,
        label="Demand",
        zorder=6,
    )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Power [MW]")
    ax.grid(True, alpha=0.25)

    # Title: "AT (Capacity-Tariff)"
    prefix = title_prefix if title_prefix is not None else scope_short
    ax.set_title(f"{prefix} ({case_name})")

    if show_legend:
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

    if ax is None:
        ax.set_xlabel("Time")
        plt.tight_layout()
        plt.show()
        
def plot_total_values_comparison_figure(
    *,
    cases: list[tuple[str, pd.DataFrame, pd.DataFrame]],  # [(case_name, stage_df, pf_df), ...]
    incidence_matrix: pd.DataFrame,
    node_to_control_area: dict,
    control_area: str | None,
    node: str | None,
    thermal_node_idx: Dict[str, list],
    phs_node_idx: Dict[str, list],
    renewable_node_idx: Dict[str, list],
    dsr_node_idx: Dict[str, list],
    battery_node_idx: Dict[str, list],
    node_demand_df: pd.DataFrame,
    base_date: str,
    start: str,
    end: str,
) -> None:
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of control_area or node.")

    scope_short = str(node) if node is not None else str(control_area)

    # Publication-ish defaults (feel free to tweak)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

    n = len(cases)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(18, 3.4 * n),
        sharex=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    for i, (case_name, stage_df, pf_df) in enumerate(cases):
        ax = axes[i]

        # show legend only once (top axis)
        show_legend = (i == 0)

        plot_total_values_instrument_only(
            stage_df=stage_df,
            incidence_matrix=incidence_matrix,
            node_to_control_area=node_to_control_area,
            case_name=case_name,
            control_area=control_area,
            node=node,
            pf_nodewise=pf_df,
            thermal_node_idx=thermal_node_idx,
            phs_node_idx=phs_node_idx,
            renewable_node_idx=renewable_node_idx,
            dsr_node_idx=dsr_node_idx,
            battery_node_idx=battery_node_idx,
            node_demand_df=node_demand_df,
            base_date=base_date,
            start=start,
            end=end,
            demand_fraction=0.95,
            apply_demand_fraction_to_excel=True,
            show_model_demand_line=True,
            show_balance_debug=False,

            ax=ax,
            show_legend=show_legend,
            title_prefix=scope_short,
        )

        # keep y-label compact
        ax.set_ylabel("MW")

    axes[-1].set_xlabel("Time")
    
    import matplotlib.dates as mdates
    # --- X-axis: show ONLY date (no time) ---
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())              # one tick per day
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))  # e.g. 01.01
    
    plt.show()
    #fig.suptitle(f"{scope_short} — system balance decomposition (Base + instruments)", y=1.02)
    plt.show()

# ============================================================
# PF plot: instrument-only
# ============================================================

def plot_pf_import_export_and_prices(
    pf_nodewise: pd.DataFrame,
    node_to_control_area: dict,
    *,
    case_name: str,
    control_area: str | None = None,
    node: str | int | None = None,
    base_date: str = DEFAULT_BASE_DATE,
    start: str | None = None,
    end: str | None = None,
    show_flows: bool = True,
) -> None:
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    pf_nodewise = ensure_abs_hour(pf_nodewise.copy())
    pf_nodewise["node"] = pf_nodewise["node"].astype(str)

    if node is not None:
        node_str = str(node)
        ca_of_node = str(node_to_control_area.get(node_str, node_to_control_area.get(node, "UNKNOWN")))
        if ca_of_node == "UNKNOWN":
            raise ValueError(f"Node {node_str!r} not found in node_to_control_area mapping.")
        control_area = ca_of_node
        scope_label = f"{control_area} — Node {node_str} (PF flows & price, {case_name})" if show_flows else f"{control_area} — Node {node_str} (PF price, {case_name})"
        pf_scope = pf_nodewise[pf_nodewise["node"] == node_str].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for node {node_str} in control area {control_area}")
        price_agg = pf_scope[["abs_hour", "price"]].copy() if "price" in pf_scope.columns else pf_scope[["abs_hour"]].assign(price=np.nan)
    else:
        ca = str(control_area)
        scope_label = f"{ca} — PF flows & price ({case_name})" if show_flows else f"{ca} — PF price ({case_name})"
        nodes = [n for n, ca_map in node_to_control_area.items() if str(ca_map) == ca]
        node_set = {str(n) for n in nodes}
        if not node_set:
            raise ValueError(
                f"No nodes found for control_area={ca}. "
                f"Available CAs = {sorted(set(map(str, node_to_control_area.values())))}"
            )
        pf_scope = pf_nodewise[pf_nodewise["node"].isin(node_set)].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for control area {ca}")
        if "price" in pf_scope.columns:
            price_agg = pf_scope.groupby("abs_hour", as_index=False)["price"].mean()
        else:
            price_agg = pf_scope.groupby("abs_hour", as_index=False).size().rename(columns={"size": "price"}).assign(price=np.nan)[["abs_hour", "price"]]

    pf_scope["exports_pf"] = (
        pd.to_numeric(pf_scope.get("pv_feed_to_system", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(pf_scope.get("battery_to_system", 0.0), errors="coerce").fillna(0.0)
    )
    pf_scope["imports_pf"] = (
        pd.to_numeric(pf_scope.get("imports_to_demand", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(pf_scope.get("imports_to_battery", 0.0), errors="coerce").fillna(0.0)
    )

    pf_agg = pf_scope.groupby("abs_hour", as_index=False)[["exports_pf", "imports_pf"]].sum()
    agg = pf_agg.merge(price_agg, on="abs_hour", how="left")
    if "price" not in agg.columns:
        agg["price"] = np.nan

    if start is not None or end is not None:
        use_start, use_end = parse_user_range(start, end, base_date)
        s_abs = date_to_abs_hour(use_start, base_date)
        e_abs = date_to_abs_hour(use_end, base_date)
        agg = agg[(agg["abs_hour"] >= s_abs) & (agg["abs_hour"] <= e_abs)].copy()
        if agg.empty:
            raise ValueError(f"Chosen window [{use_start} → {use_end}] produced an empty slice.")

    base_iso = coerce_base_date(base_date)
    base_ts = pd.Timestamp(base_iso)
    time_index = pd.to_datetime([base_ts + pd.Timedelta(hours=int(h - 1)) for h in agg["abs_hour"].astype(int).values])
    agg = agg.set_index(time_index)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(scope_label)

    if show_flows:
        ax1.fill_between(agg.index, 0, agg["exports_pf"].clip(lower=0.0), alpha=0.9, label="Exports (PF)")
        ax1.fill_between(agg.index, 0, -agg["imports_pf"].clip(lower=0.0), alpha=0.9, label="Imports (PF)")
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_ylabel("Power [MW]")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        price_ax = ax2
        price_ax.set_ylabel("PF Input Price [€/MWh]")
    else:
        ax1.set_ylabel("PF Input Price [€/MWh]")
        ax1.grid(True, alpha=0.3)
        price_ax = ax1

    price_ax.plot(agg.index, agg["price"], linestyle="--", linewidth=1.8, color="black", label="PF Input Price [€/MWh]")
    price_ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"Using parameter module: {_PARAMS_FLAVOR}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_XLSX: {DATA_XLSX}")

    # --- Load indices/mappings from Excel via load_demand_data ---
    demand_out = call_with_supported_kwargs(load_demand_data, DATA_XLSX, "Demand_Profiles")

    normalized_profiles_df = None
    node_demand_data = None
    node_to_control_area = None

    # With your provided implementation, the return is:
    # normalized_profiles_df, node_demand_df, control_areas, nodes, node_to_control_area
    if isinstance(demand_out, tuple) and len(demand_out) >= 5:
        normalized_profiles_df = demand_out[0] if isinstance(demand_out[0], pd.DataFrame) else None
        node_demand_data = demand_out[1] if isinstance(demand_out[1], pd.DataFrame) else None
        node_to_control_area = demand_out[4] if isinstance(demand_out[4], dict) else None
    else:
        # fallback (older style): search tuple for df + dict
        if isinstance(demand_out, tuple):
            for obj in demand_out[::-1]:
                if isinstance(obj, dict):
                    node_to_control_area = obj
                    break
            for obj in demand_out:
                if isinstance(obj, pd.DataFrame):
                    node_demand_data = obj
                    break

    if node_demand_data is None:
        node_demand_data = pd.DataFrame()
    if node_to_control_area is None:
        raise ValueError("Could not obtain node_to_control_area from load_demand_data output.")
    if node_demand_data.empty:
        raise ValueError("Could not obtain node_demand_df (node_demand_data) from load_demand_data output.")

    # --- Load power plant indices/mappings ---
    thermal_out = call_with_supported_kwargs(load_thermal_power_plant_data, DATA_XLSX, "Thermal_Power_Data", "Thermal_Power_Specific_Data")
    thermal_node_idx = None
    if isinstance(thermal_out, tuple):
        for obj in thermal_out:
            if isinstance(obj, dict):
                thermal_node_idx = obj
    if thermal_node_idx is None:
        raise ValueError("Could not obtain thermal_node_idx from load_thermal_power_plant_data.")

    phs_out = call_with_supported_kwargs(load_phs_power_plant_data, DATA_XLSX, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data")
    phs_node_idx = None
    if isinstance(phs_out, tuple):
        for obj in phs_out:
            if isinstance(obj, dict):
                phs_node_idx = obj
    if phs_node_idx is None:
        raise ValueError("Could not obtain phs_node_idx from load_phs_power_plant_data.")

    res_out = call_with_supported_kwargs(load_renewable_power_plant_data, DATA_XLSX, "RES_Power_Data", "RES_Power_Specific_Data")
    renewable_node_idx = None
    if isinstance(res_out, tuple):
        for obj in res_out:
            if isinstance(obj, dict):
                renewable_node_idx = obj
    if renewable_node_idx is None:
        raise ValueError("Could not obtain renewable_node_idx from load_renewable_power_plant_data.")

    flex_out = call_with_supported_kwargs(load_flexibility_data, DATA_XLSX, "Flexibility_Data", "Flexibility_Specific_Data")
    dsr_node_idx = None
    battery_node_idx = None
    if isinstance(flex_out, tuple):
        dicts = [obj for obj in flex_out if isinstance(obj, dict)]
        if len(dicts) >= 2:
            dsr_node_idx = dicts[-2]
            battery_node_idx = dicts[-1]
    if dsr_node_idx is None or battery_node_idx is None:
        raise ValueError("Could not obtain dsr_node_idx / battery_node_idx from load_flexibility_data.")

    # --- Exchange data (kept for signature compatibility) ---
    exchange_out = call_with_supported_kwargs(
        load_exchange_data,
        DATA_XLSX,
        "Exchange_Data",
        ptdf_csv_path=os.path.join(BASE_DIR, "PTDF_Synchronized.csv"),
        slack_node=None,
        verbose=False,
    )
    incidence_matrix = None
    if isinstance(exchange_out, tuple):
        for obj in exchange_out:
            if isinstance(obj, pd.DataFrame):
                if obj.shape[0] > 0 and obj.shape[1] > 0:
                    incidence_matrix = obj
                    break
    if incidence_matrix is None:
        try:
            incidence_matrix = exchange_out[2]  # type: ignore[index]
        except Exception:
            pass
    if incidence_matrix is None or not isinstance(incidence_matrix, pd.DataFrame):
        raise ValueError("Could not obtain incidence_matrix from load_exchange_data.")

    '''
    # ============================================================
    # PRICE TABLE (Stage-0 dual prices) — per timestep (wide)
    # ============================================================
    try:
        stage1_base_price = compute_stage1_base_price_from_stage0_csvs(STAGE0_PRICES_CSV, STAGE0_PV_CSV)
    except Exception as e:
        stage1_base_price = float("nan")
        print(f"\n⚠️ Could not compute Stage-1 BASE price from Stage-0 CSVs: {e}")

    print("\n" + "=" * 90)
    print("PRICE TABLE (Stage-0 dual prices) — per timestep")
    print(f"Window: {START} → {END}")
    print("=" * 90)
    if np.isfinite(stage1_base_price):
        print(f"Stage-1 BASE price used (PV-weighted): {stage1_base_price:.6f} €/MWh\n")
    else:
        print("Stage-1 BASE price used (PV-weighted): (not available)\n")

    try:
        price_ts = nodal_prices_timeseries_from_stage0(
            STAGE0_PRICES_CSV,
            start=START,
            end=END,
            base_date=DEFAULT_BASE_DATE,
        )

        # Optional: enforce deterministic column order
        price_ts = price_ts.reindex(sorted(price_ts.columns.astype(str)), axis=1)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 250,
        ):
            print(price_ts.round(6))
    except Exception as e:
        print(f"⚠️ Could not print Stage-0 nodal price time series: {e}")
    '''
        
    # --- Read all cases first ---
    cases = [("Base Case", stage1_base_path(BASE_DIR), follower_base_path(BASE_DIR))]
    cases += [(inst, stage1_instr_path(BASE_DIR, inst), follower_instr_path(BASE_DIR, inst)) for inst in INSTRUMENTS]

    loaded_cases: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for case_name, sys_path, fol_path in cases:
        stage_df = read_results(sys_path)
        pf_df = read_results(fol_path)
        loaded_cases.append((case_name, stage_df, pf_df))

    # --- ONE publication-ready comparison figure ---
    plot_total_values_comparison_figure(
        cases=loaded_cases,
        incidence_matrix=incidence_matrix,
        node_to_control_area=node_to_control_area,
        control_area=CONTROL_AREA if NODE is None else None,
        node=NODE,
        thermal_node_idx=thermal_node_idx,
        phs_node_idx=phs_node_idx,
        renewable_node_idx=renewable_node_idx,
        dsr_node_idx=dsr_node_idx,
        battery_node_idx=battery_node_idx,
        node_demand_df=node_demand_data,
        base_date=DEFAULT_BASE_DATE,
        start=START,
        end=END,
    )

    '''
    # --- keep PF plots if you want (optional) ---
    for case_name, stage_df, pf_df in loaded_cases:
        plot_pf_import_export_and_prices(
            pf_nodewise=pf_df,
            node_to_control_area=node_to_control_area,
            case_name=case_name,
            control_area=CONTROL_AREA if NODE is None else None,
            node=NODE,
            base_date=DEFAULT_BASE_DATE,
            start=START,
            end=END,
            show_flows=SHOW_PF_FLOWS,
        )
    '''

if __name__ == "__main__":
    main()
