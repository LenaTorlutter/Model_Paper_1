from __future__ import annotations

import os
import re
import warnings
import inspect
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_FILE = "Data_Updated.xlsx"
DATA_XLSX = os.path.join(BASE_DIR, DATA_FILE)

# Stage-0 extractions written by your rolling-horizon pipeline:
STAGE0_PRICES_CSV = os.path.join(BASE_DIR, "full_stage0_prices.csv")            # node,start_week,t,price
STAGE0_PV_CSV     = os.path.join(BASE_DIR, "full_stage0_pv_quantities.csv")     # start_week,t,pv_generation

# Output figures
FIGURES_DIR = os.path.join(BASE_DIR, "Figures", "General_Analysis")

DEFAULT_BASE_DATE = "2030-01-01"
HOURS_PER_WEEK = 168
WEEKS_PER_STEP = 2

# ---- Requested: do NOT import control areas from main ----
FOLLOWER_CONTROL_AREAS = (
    "AT", "BE", "CH", "CZ", "DE", "FR", "LU", "HU", "IT", "NL", "PL", "SI", "SK"
)

INSTRUMENTS = ["RTP", "Peak-Shaving", "Capacity-Tariff"]

# Plot window
START = "2030-07-01 00:00"
END   = "2030-07-02 00:00"

# Demand fraction used in the plotter (same behavior as your current script)
DEMAND_FRACTION_FOR_PLOT = 0.95

# Exchange tables (TABLE 2A + 2B)
SHOW_EXCHANGE_TABLES = False

# ---- Requested: Table 1 should always be displayed ----
PRINT_TABLE1_ALWAYS = True
PRINT_TABLE1_NROWS = 24  # rows of Table 1 to print per case & CA

# ---- Requested: possibility to plot follower imports/exports and print values ----
ENABLE_FOLLOWER_IO_PLOT = False      # create PF IO plot per case & CA
ENABLE_FOLLOWER_IO_PRINT = True      # include detailed PF IO columns in Table 1
SHOW_PF_PRICE_ON_IO_PLOT = True      # overlay PF price if available
FOLLOWER_IO_PLOT_SHOW = False        # show PF IO plot windows interactively
FOLLOWER_IO_PLOT_SAVE = True         # save PF IO plot PNGs

# ---- Comparison figure behavior ----
SHOW_COMPARISON_PLOTS = True         # show each CA comparison figure interactively
SAVE_COMPARISON_PLOTS = True         # save PNG for each CA comparison figure


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

def hour_of_week_from_abs_hour(abs_hours: np.ndarray, demand_index: pd.Index) -> np.ndarray:
    """
    Fallback mapping ONLY (used when demand profile is weekly).
    """
    idx = pd.to_numeric(pd.Index(demand_index), errors="coerce")
    idx = idx[idx.notna()]
    if len(idx) == 0:
        return ((abs_hours - 1) % HOURS_PER_WEEK) + 1

    idx_min = int(idx.min())
    if idx_min == 0:
        return (abs_hours - 1) % HOURS_PER_WEEK          # 0..167
    else:
        return ((abs_hours - 1) % HOURS_PER_WEEK) + 1     # 1..168


# ============================================================
# Robust abs_hour reconstruction (USE THIS date assignment)
# ============================================================

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

    # unify naming if needed
    if "component" in df.columns and "variable" not in df.columns:
        df = df.rename(columns={"component": "variable"})

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # keep start_week if present; still reconstruct abs_hour robustly using ensure_abs_hour
    if "start_week" not in df.columns and "window" in df.columns:
        # not strictly required, but helpful for older downstream code
        df["start_week"] = (
            pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) * WEEKS_PER_STEP + 1
        )

    df = ensure_abs_hour(df)
    return df


# ============================================================
# Demand helpers (use node_demand_data + node_to_control_area)
# ============================================================

def demand_series_from_abs_hour_index(
    abs_hour_index: pd.Index,
    node_demand_df: pd.DataFrame,
    node_to_control_area: dict,
    *,
    control_area: str,
    node: str | None = None,
) -> pd.Series:
    """
    Build demand series aligned to the given abs_hour_index (the *true* hours you plot).

    If node_demand_df has >= max(abs_hour) rows (e.g. 8760), we use it DIRECTLY by abs_hour.
    Otherwise we fall back to repeating a weekly profile.
    """
    abs_hours = pd.to_numeric(pd.Index(abs_hour_index), errors="coerce").astype("Int64")
    if abs_hours.isna().any():
        raise ValueError("abs_hour_index contains non-numeric / NaN values; cannot compute demand.")

    prof = node_demand_df.copy()

    # normalize index to int if possible
    prof.index = pd.to_numeric(prof.index, errors="coerce")
    prof = prof.loc[prof.index.notna()].copy()
    prof.index = prof.index.astype(int)

    max_h = int(abs_hours.max())

    # ---- Select demand columns (node or control-area sum) ----
    if node is not None:
        n = str(node).strip()
        if n not in prof.columns.astype(str).tolist():
            raise ValueError(f"Node demand column '{n}' not found in Demand sheet.")
        get_vals = lambda df: pd.to_numeric(df[n], errors="coerce").fillna(0.0)
    else:
        ca = str(control_area).strip()
        cols = [
            c for c in prof.columns.astype(str)
            if str(node_to_control_area.get(str(c), "")).strip() == ca
        ]
        if not cols:
            raise ValueError(f"No demand columns found for control_area='{ca}'.")
        get_vals = lambda df: pd.to_numeric(df[cols].sum(axis=1), errors="coerce").fillna(0.0)

    # ---- Case A: FULL-YEAR (or long) profile -> use abs_hour directly ----
    # We consider it "long" if it covers the needed max hour.
    # Typical: index 1..8760 (or 0..8759).
    if len(prof.index) >= max_h:
        idx_min = int(prof.index.min())
        # if the sheet is 0-based, shift abs_hour to 0-based
        if idx_min == 0:
            lookup = abs_hours.to_numpy(dtype=int) - 1
        else:
            lookup = abs_hours.to_numpy(dtype=int)

        s = get_vals(prof.reindex(lookup))
        return pd.Series(s.to_numpy(dtype=float), index=abs_hour_index, name="demand")

    # ---- Case B: weekly profile -> repeat ----
    how = hour_of_week_from_abs_hour(abs_hours.to_numpy(dtype=int), prof.index).astype(int)
    s = get_vals(prof.reindex(how))
    return pd.Series(s.to_numpy(dtype=float), index=abs_hour_index, name="demand")

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
    These are the follower↔system boundary flows used for the system stackplot:

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
        "pf_price_mean": pd.Series(np.nan, index=time_index),
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

    if "price" in pf.columns:
        tmp = pf[["abs_hour", "price"]].copy()
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
        price_agg = tmp.groupby("abs_hour", as_index=True)["price"].mean().reindex(abs_list)
        out["pf_price_mean"] = pd.Series(price_agg.values, index=time_index)

    return out


# ============================================================
# Stage-1 base price reconstruction (from Stage-0 CSVs)
# ============================================================

def pv_weighted_price_over_horizon(prices_by_t: pd.DataFrame, pv_by_t: pd.DataFrame) -> float:
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

    # mirror your pipeline sign logic
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


# ============================================================
# Tables: Table 1 always + optional exchange tables
# ============================================================

def print_stackplot_variables(
    *,
    time_index: pd.DatetimeIndex,
    abs_hours: np.ndarray,
    gen_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    dem_excel_s: pd.Series,
    exchange_signed_s: pd.Series,
    pf_bridge: dict[str, pd.Series] | None = None,
    n: int = 24,
    decimals: int = 2,
    title: str | None = None,
) -> None:
    def _print_df(df: pd.DataFrame, header: str) -> None:
        print("\n" + "-" * 110)
        print(header)
        print("-" * 110)
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

    if len(time_index) != len(abs_hours):
        raise ValueError("time_index and abs_hours must have the same length.")

    gen = _reindex_df(gen_df).copy()
    cons = _reindex_df(cons_df).copy()
    dem_excel_s = _reindex_s(dem_excel_s)
    ex = _reindex_s(exchange_signed_s)

    if title:
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)

    if pf_bridge is None:
        pf_bridge = {}

    pf_exp_pv = pf_bridge.get("pf_exports_pv", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_exp_ba = pf_bridge.get("pf_exports_battery", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_imp_de = pf_bridge.get("pf_imports_to_demand", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)
    pf_imp_ba = pf_bridge.get("pf_imports_to_battery", pd.Series(0.0, index=time_index)).reindex(time_index).fillna(0.0)

    pf_exports_s = (pf_exp_pv + pf_exp_ba).reindex(time_index).fillna(0.0)
    pf_imports_s = (pf_imp_de + pf_imp_ba).reindex(time_index).fillna(0.0)

    drop_gen_cols = ["Follower exports (from PV)", "Follower exports (from battery)"]
    drop_cons_cols = ["Follower imports (to demand)", "Follower imports (to battery)"]
    gen_clean = gen.drop(columns=drop_gen_cols, errors="ignore")
    cons_clean = cons.drop(columns=drop_cons_cols, errors="ignore")

    base_cols = {
        "Follower exports (PV+battery)": pf_exports_s.values,
        "-Follower imports (demand+battery)": pf_imports_s.values,
        "Demand (from Excel)": dem_excel_s.values,
        "Exchange (signed)": ex.values,
    }

    if ENABLE_FOLLOWER_IO_PRINT:
        base_cols.update(
            {
                "PF export: PV→system": pf_exp_pv.values,
                "PF export: battery→system": pf_exp_ba.values,
                "PF import: to demand": pf_imp_de.values,
                "PF import: to battery": pf_imp_ba.values,
            }
        )
        if "pf_price_mean" in pf_bridge:
            p = pf_bridge.get("pf_price_mean", pd.Series(np.nan, index=time_index)).reindex(time_index)
            base_cols["PF input price (mean)"] = p.values

    out1 = pd.concat(
        [
            gen_clean,
            cons_clean.add_prefix("-"),
            pd.DataFrame(base_cols, index=time_index),
        ],
        axis=1,
    )

    _print_df(out1, "TABLE 1 — Stackplot variables (incl. net follower imports/exports; optional PF detail)")

    if SHOW_EXCHANGE_TABLES:
        t2a = pd.DataFrame(
            {
                "Exchange (signed)": ex.values,
                "Imports (>=0)": ex.clip(lower=0.0).values,
                "Exports (>=0)": (-ex).clip(lower=0.0).values,
            },
            index=time_index,
        )
        _print_df(t2a, "TABLE 2A — Exchange (scope)")


# ============================================================
# Plot colors (unchanged)
# ============================================================

COLORS_SUPPLY_PUB = {
    "Thermal":                         "#D55E00",
    "PHS (gen)":                       "#0072B2",
    "Battery (out)":                   "#E69F00",
    "RES":                             "#009E73",
    "DSR (down)":                      "#44AA99",
    "Imports":                         "#56B4E9",
    "Follower exports (from PV)":      "#CC79A7",
    "Follower exports (from battery)": "#AA4499",
    "NSE":                             "#000000",
}

COLORS_CONS_PUB = {
    "PHS (pump)":                      "#004488",
    "Battery (in)":                    "#EE7733",
    "DSR (up)":                        "#999999",
    "Exports":                         "#4D4D4D",
    "Follower imports (to demand)":    "#BB5566",
    "Follower imports (to battery)":   "#EE99AA",
    "Curtailment":                     "#882255",
}


# ============================================================
# Plot: instrument-only stacks (single panel)
# ============================================================

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
    show_balance_debug_tables: bool = False,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    title_prefix: str | None = None,
) -> dict:
    """
    Returns a dict with useful aligned series, including:
      - time_index
      - pf_bridge
      - pf_io_agg (exports_pf/imports_pf + optional price) time-indexed
    """
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    # Scope: nodes in CA or single node
    if node is not None:
        nodes = [str(node)]
        scope_label = f"Node_{node}"
        demand_scope_control_area = None
        demand_scope_node = str(node)
        scope_short = str(node)
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
        scope_short = str(control_area)

    nodes_set = set(map(str, nodes))

    # Aggregation helper
    def agg_by_nodes(
        varname: str,
        plant_to_nodes: Dict[str, list] | None,
        *,
        is_node_indexed: bool = False,
    ) -> pd.DataFrame:
        part = stage_df[stage_df["variable"] == varname].copy()
        if part.empty:
            return pd.DataFrame(columns=["abs_hour", "value"])

        part = ensure_abs_hour(part)
        if "abs_hour" not in part.columns:
            return pd.DataFrame(columns=["abs_hour", "value"])

        if is_node_indexed:
            if "index_0" not in part.columns:
                return pd.DataFrame(columns=["abs_hour", "value"])
            part["node"] = part["index_0"].astype(str)
        else:
            if not plant_to_nodes or "index_0" not in part.columns:
                return pd.DataFrame(columns=["abs_hour", "value"])
            m = [(str(p), str(n)) for n, plants in plant_to_nodes.items() for p in plants]
            map_df = pd.DataFrame(m, columns=["plant", "node"])
            part["plant"] = part["index_0"].astype(str)
            part = part.merge(map_df, on="plant", how="inner")
            if part.empty:
                return pd.DataFrame(columns=["abs_hour", "value"])

        part["node"] = part["node"].astype(str)
        part = part[part["node"].isin(nodes_set)]
        if part.empty:
            return pd.DataFrame(columns=["abs_hour", "value"])

        part["abs_hour"] = pd.to_numeric(part["abs_hour"], errors="coerce")
        part = part.dropna(subset=["abs_hour"]).copy()
        part["abs_hour"] = part["abs_hour"].astype(int)

        part["value"] = pd.to_numeric(part.get("value", 0.0), errors="coerce").fillna(0.0)
        return part.groupby(["abs_hour"], as_index=False)["value"].sum()

    # Supply components (system)
    thermal  = agg_by_nodes("thermal_generation", thermal_node_idx)
    phs_gen  = agg_by_nodes("phs_turbine_generation", phs_node_idx)
    res_gen  = agg_by_nodes("renewable_generation", renewable_node_idx)
    dsr_down = agg_by_nodes("dsr_down", dsr_node_idx)
    bat_out  = agg_by_nodes("battery_out", battery_node_idx)

    # Consumption components (system)
    phs_pump = agg_by_nodes("phs_pump_consumption", phs_node_idx)
    dsr_up   = agg_by_nodes("dsr_up", dsr_node_idx)
    bat_in   = agg_by_nodes("battery_in", battery_node_idx)

    # Balance terms from system
    nse = agg_by_nodes("nse", None, is_node_indexed=True)
    curtail = agg_by_nodes("curtailment", None, is_node_indexed=True)

    exchange_ex = agg_by_nodes("exchange", None, is_node_indexed=True)
    if exchange_ex.empty:
        exchange_ex = agg_by_nodes("expr:exchange", None, is_node_indexed=True)
    if exchange_ex.empty:
        exchange_ex = pd.DataFrame(columns=["abs_hour", "value"])

    # Merge to base table on abs_hour
    def merge_abs(left: pd.DataFrame, comp: pd.DataFrame, name: str) -> pd.DataFrame:
        if left is None or left.empty:
            base0 = comp[["abs_hour"]].drop_duplicates().copy() if (comp is not None and not comp.empty) else pd.DataFrame(columns=["abs_hour"])
        else:
            base0 = left.copy()
        if comp is None or comp.empty:
            base0[name] = 0.0
            return base0
        return base0.merge(comp.rename(columns={"value": name}), on="abs_hour", how="left").fillna({name: 0.0})

    base = pd.DataFrame(columns=["abs_hour"])
    for comp, nm in [
        (thermal, "thermal"),
        (phs_gen, "phs_gen"),
        (res_gen, "res"),
        (dsr_down, "dsr_out"),
        (bat_out, "battery_out"),
        (phs_pump, "phs_cons"),
        (dsr_up, "dsr_in"),
        (bat_in, "battery_in"),
        (nse, "nse"),
        (curtail, "curtailment"),
        (exchange_ex, "exchange"),
    ]:
        base = merge_abs(base, comp, nm)

    base["abs_hour"] = pd.to_numeric(base["abs_hour"], errors="coerce").fillna(0).astype(int)

    # Demand line (from node_demand_df): repeat weekly profile onto abs_hour window
    if node_demand_df is None or node_demand_df.empty:
        raise ValueError("node_demand_df (from load_demand_data) is required to plot demand.")

    # Window slice
    use_start, use_end = parse_user_range(start, end, base_date)
    ah0 = date_to_abs_hour(use_start, base_date)
    ah1 = date_to_abs_hour(use_end, base_date)

    w = base[(base["abs_hour"] >= ah0) & (base["abs_hour"] <= ah1)].copy()
    if w.empty:
        raise ValueError(f"Window [{use_start} → {use_end}] produced an empty slice.")

    idx = np.unique(w["abs_hour"].values.astype(int))
    base_ts = pd.to_datetime(coerce_base_date(base_date) + " 00:00", format=DATE_FMT)
    time_index = pd.to_datetime([base_ts + pd.Timedelta(hours=int(h - 1)) for h in idx])
    abs_to_dt = {int(h): time_index[i] for i, h in enumerate(idx.tolist())}

    def series_select(col: str) -> pd.Series:
        s = w.set_index("abs_hour")[col]
        return s.reindex(idx).fillna(0.0)

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

    # =========================================================
    # DEMAND FIX (robust, aligned to the *actual* abs_hour index)
    # =========================================================
    # Demand aligned to plotted abs_hours:
    # - if Demand sheet is long (e.g., 8760): use abs_hour directly
    # - else: repeat weekly profile
    dem_abs = demand_series_from_abs_hour_index(
        pd.Index(idx),
        node_demand_df=node_demand_df,
        node_to_control_area=node_to_control_area,
        control_area=str(control_area) if node is None else "NA",
        node=str(node) if node is not None else None,
    )
    
    dem_excel_s = pd.Series(dem_abs.to_numpy(dtype=float), index=time_index)

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

    # PF bridge (used for plot + prints)
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

    # --- Table 1 always (requested) ---
    if PRINT_TABLE1_ALWAYS or show_balance_debug_tables:
        print_stackplot_variables(
            time_index=time_index,
            abs_hours=idx,
            gen_df=gen_df,
            cons_df=cons_df,
            dem_excel_s=dem_excel_s,
            exchange_signed_s=exch_s,
            pf_bridge=pf_bridge,
            n=int(PRINT_TABLE1_NROWS),
            decimals=2,
            title=f"{scope_label} ({case_name}) — TABLE 1 (first {int(PRINT_TABLE1_NROWS)} timesteps)",
        )

    # Plot
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

    ax.plot(time_index, dem_excel_s.values, color="black", linewidth=2.2, label="Demand", zorder=6)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Power [MW]")
    ax.grid(True, alpha=0.25)

    prefix = title_prefix if title_prefix is not None else scope_short
    ax.set_title(f"{prefix} ({case_name})")

    if show_legend:
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

    # Return aligned follower IO series as well (for optional separate plots)
    pf_io_agg = pd.DataFrame(
        {
            "exports_pf": (pf_exp_pv_s + pf_exp_bat_s).values,
            "imports_pf": (pf_imp_dem_s + pf_imp_bat_s).values,
        },
        index=time_index,
    )
    if "pf_price_mean" in pf_bridge:
        pf_io_agg["price_mean"] = pf_bridge["pf_price_mean"].reindex(time_index).values

    return {
        "time_index": time_index,
        "pf_bridge": pf_bridge,
        "pf_io_agg": pf_io_agg,
    }


# ============================================================
# Plot: comparison figure (Base + instruments) — returns fig
# ============================================================

def format_hourly_time_axis(ax: plt.Axes, *, hour_interval: int = 6) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.figure.autofmt_xdate(rotation=0, ha="center")

def plot_total_values_comparison_figure(
    *,
    cases: list[tuple[str, pd.DataFrame, pd.DataFrame]],
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
    demand_fraction_for_plot: float,
    show_balance_debug_tables: bool = False,
) -> plt.Figure:
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of control_area or node.")

    scope_short = str(node) if node is not None else str(control_area)

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
            demand_fraction=float(demand_fraction_for_plot),
            apply_demand_fraction_to_excel=True,
            show_balance_debug_tables=bool(show_balance_debug_tables),
            ax=ax,
            show_legend=show_legend,
            title_prefix=scope_short,
        )
        ax.set_ylabel("MW")

    axes[-1].set_xlabel("Time")
    format_hourly_time_axis(axes[-1], hour_interval=6)

    return fig


# ============================================================
# Optional: PF imports/exports plot (per case & CA)
# ============================================================

def plot_pf_import_export_and_prices(
    *,
    pf_io_agg: pd.DataFrame,
    scope_label: str,
    case_name: str,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    pf_io_agg index = datetime, columns: exports_pf, imports_pf, optional price_mean
    """
    if pf_io_agg is None or pf_io_agg.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f"{scope_label} — PF imports/exports ({case_name})")

    exp = pd.to_numeric(pf_io_agg.get("exports_pf", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    imp = pd.to_numeric(pf_io_agg.get("imports_pf", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    ax1.fill_between(pf_io_agg.index, 0, exp.values, alpha=0.85, label="PF exports (to system)")
    ax1.fill_between(pf_io_agg.index, 0, -imp.values, alpha=0.85, label="PF imports (from system)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Power [MW]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # optional price overlay
    if SHOW_PF_PRICE_ON_IO_PLOT and ("price_mean" in pf_io_agg.columns) and pf_io_agg["price_mean"].notna().any():
        ax2 = ax1.twinx()
        ax2.set_ylabel("PF input price [€/MWh]")
        ax2.plot(
            pf_io_agg.index,
            pd.to_numeric(pf_io_agg["price_mean"], errors="coerce"),
            linestyle="--",
            linewidth=1.8,
            color="black",
            label="PF input price (mean)",
        )
        ax2.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"Using parameter module: {_PARAMS_FLAVOR}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_XLSX: {DATA_XLSX}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load indices/mappings from Excel via load_demand_data ---
    demand_out = call_with_supported_kwargs(load_demand_data, DATA_XLSX, "Demand_Profiles")

    node_demand_data = None
    node_to_control_area = None

    # Expected: normalized_profiles_df, node_demand_df, control_areas, nodes, node_to_control_area
    if isinstance(demand_out, tuple) and len(demand_out) >= 5:
        node_demand_data = demand_out[1] if isinstance(demand_out[1], pd.DataFrame) else None
        node_to_control_area = demand_out[4] if isinstance(demand_out[4], dict) else None
    else:
        if isinstance(demand_out, tuple):
            for obj in demand_out[::-1]:
                if isinstance(obj, dict):
                    node_to_control_area = obj
                    break
            for obj in demand_out:
                if isinstance(obj, pd.DataFrame):
                    node_demand_data = obj
                    break

    if node_to_control_area is None:
        raise ValueError("Could not obtain node_to_control_area from load_demand_data output.")
    if node_demand_data is None or node_demand_data.empty:
        raise ValueError("Could not obtain node_demand_df (node_demand_data) from load_demand_data output.")

    # =========================================================
    # CRITICAL: normalize Demand_Profiles index to clean ints ONCE
    # =========================================================
    node_demand_data = node_demand_data.copy()
    node_demand_data.index = pd.to_numeric(node_demand_data.index, errors="coerce")
    node_demand_data = node_demand_data.loc[node_demand_data.index.notna()].copy()
    node_demand_data.index = node_demand_data.index.astype(int)

    # --- Load power plant indices/mappings ---
    thermal_out = call_with_supported_kwargs(
        load_thermal_power_plant_data, DATA_XLSX, "Thermal_Power_Data", "Thermal_Power_Specific_Data"
    )
    thermal_node_idx = None
    if isinstance(thermal_out, tuple):
        for obj in thermal_out:
            if isinstance(obj, dict):
                thermal_node_idx = obj
    if thermal_node_idx is None:
        raise ValueError("Could not obtain thermal_node_idx from load_thermal_power_plant_data.")

    phs_out = call_with_supported_kwargs(
        load_phs_power_plant_data, DATA_XLSX, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data"
    )
    phs_node_idx = None
    if isinstance(phs_out, tuple):
        for obj in phs_out:
            if isinstance(obj, dict):
                phs_node_idx = obj
    if phs_node_idx is None:
        raise ValueError("Could not obtain phs_node_idx from load_phs_power_plant_data.")

    res_out = call_with_supported_kwargs(
        load_renewable_power_plant_data, DATA_XLSX, "RES_Power_Data", "RES_Power_Specific_Data"
    )
    renewable_node_idx = None
    if isinstance(res_out, tuple):
        for obj in res_out:
            if isinstance(obj, dict):
                renewable_node_idx = obj
    if renewable_node_idx is None:
        raise ValueError("Could not obtain renewable_node_idx from load_renewable_power_plant_data.")

    flex_out = call_with_supported_kwargs(
        load_flexibility_data, DATA_XLSX, "Flexibility_Data", "Flexibility_Specific_Data"
    )
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
            if isinstance(obj, pd.DataFrame) and obj.shape[0] > 0 and obj.shape[1] > 0:
                incidence_matrix = obj
                break
    if incidence_matrix is None:
        try:
            incidence_matrix = exchange_out[2]  # type: ignore[index]
        except Exception:
            pass
    if incidence_matrix is None or not isinstance(incidence_matrix, pd.DataFrame):
        raise ValueError("Could not obtain incidence_matrix from load_exchange_data.")

    # --- Compute + print Stage-1 base price (optional) ---
    try:
        stage1_base_price = compute_stage1_base_price_from_stage0_csvs(STAGE0_PRICES_CSV, STAGE0_PV_CSV)
        print(f"\nStage-1 BASE price used (PV-weighted): {stage1_base_price:.6f} €/MWh\n")
    except Exception as e:
        print(f"\n⚠️ Could not compute Stage-1 BASE price from Stage-0 CSVs: {e}\n")

    # --- Read all cases once ---
    case_specs = [("Base Case", stage1_base_path(BASE_DIR), follower_base_path(BASE_DIR))]
    case_specs += [(inst, stage1_instr_path(BASE_DIR, inst), follower_instr_path(BASE_DIR, inst)) for inst in INSTRUMENTS]

    loaded_cases: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for case_name, sys_path, fol_path in case_specs:
        stage_df = read_results(sys_path)
        pf_df = read_results(fol_path)
        loaded_cases.append((case_name, stage_df, pf_df))

    # --- Produce + save comparison figure for ALL follower control areas ---
    print("\n" + "=" * 110)
    print("Creating comparison plots for all follower control areas")
    print(f"Window: {START} → {END}")
    print(f"Saving to: {FIGURES_DIR}")
    print("=" * 110)

    safe_start = START.replace(":", "-").replace(" ", "_")
    safe_end = END.replace(":", "-").replace(" ", "_")

    for ca in FOLLOWER_CONTROL_AREAS:
        try:
            fig = plot_total_values_comparison_figure(
                cases=loaded_cases,
                incidence_matrix=incidence_matrix,
                node_to_control_area=node_to_control_area,
                control_area=ca,
                node=None,
                thermal_node_idx=thermal_node_idx,
                phs_node_idx=phs_node_idx,
                renewable_node_idx=renewable_node_idx,
                dsr_node_idx=dsr_node_idx,
                battery_node_idx=battery_node_idx,
                node_demand_df=node_demand_data,
                base_date=DEFAULT_BASE_DATE,
                start=START,
                end=END,
                demand_fraction_for_plot=float(DEMAND_FRACTION_FOR_PLOT),
                show_balance_debug_tables=False,
            )

            if SAVE_COMPARISON_PLOTS:
                out_path = os.path.join(FIGURES_DIR, f"{ca}_comparison_{safe_start}__{safe_end}.png")
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"✅ Saved comparison: {out_path}")

            if SHOW_COMPARISON_PLOTS:
                plt.show()

            plt.close(fig)

            # ---- Optional: PF IO plots per case (imports/exports + optional price) ----
            if ENABLE_FOLLOWER_IO_PLOT:
                for case_name, stage_df, pf_df in loaded_cases:
                    tmp_fig, tmp_ax = plt.subplots(1, 1, figsize=(6, 3))
                    try:
                        res = plot_total_values_instrument_only(
                            stage_df=stage_df,
                            incidence_matrix=incidence_matrix,
                            node_to_control_area=node_to_control_area,
                            case_name=case_name,
                            control_area=ca,
                            node=None,
                            pf_nodewise=pf_df,
                            thermal_node_idx=thermal_node_idx,
                            phs_node_idx=phs_node_idx,
                            renewable_node_idx=renewable_node_idx,
                            dsr_node_idx=dsr_node_idx,
                            battery_node_idx=battery_node_idx,
                            node_demand_df=node_demand_data,
                            base_date=DEFAULT_BASE_DATE,
                            start=START,
                            end=END,
                            demand_fraction=float(DEMAND_FRACTION_FOR_PLOT),
                            apply_demand_fraction_to_excel=True,
                            show_balance_debug_tables=False,
                            ax=tmp_ax,
                            show_legend=False,
                            title_prefix=str(ca),
                        )
                        pf_io = res.get("pf_io_agg", pd.DataFrame())
                    finally:
                        plt.close(tmp_fig)

                    if pf_io is None or pf_io.empty:
                        continue

                    save_path = None
                    if FOLLOWER_IO_PLOT_SAVE:
                        save_path = os.path.join(
                            FIGURES_DIR,
                            f"{ca}_PF_IO_{instrument_slug(case_name)}_{safe_start}__{safe_end}.png",
                        )

                    plot_pf_import_export_and_prices(
                        pf_io_agg=pf_io,
                        scope_label=str(ca),
                        case_name=case_name,
                        save_path=save_path,
                        show=FOLLOWER_IO_PLOT_SHOW,
                    )
                    if save_path is not None:
                        print(f"✅ Saved PF IO plot: {save_path}")

        except Exception as e:
            print(f"⚠️ Skipped {ca}: {e}")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()