"""
kpi_country_tables_terminal_follow_main.py

TERMINAL-ONLY KPI reporting: one table per country/control area.

Follows the file naming + logic from Main_New.py:
  BASE:
    full_model_results_stage1_base.csv
    full_follower_values_base.csv
  INSTRUMENTS:
    full_model_results_stage1_instrument__{slug}.csv
    full_follower_values_instrument__{slug}.csv

KPIs per country (aggregated across nodes in that country):
  - Max Imp. (MW)  = max_t sum_nodes(exchange_t)+
  - Max Exp. (MW)  = max_t sum_nodes((-exchange)_t)+
  - Curt.          = sum_t sum_nodes(curtailment_t)

Follower-based profit components per country (EUR), using follower-exported price:
  - Exp.-Revenue   = sum_{n,t} price * (pv_feed_to_system + battery_to_system)
  - Imp.-Cost      = sum_{n,t} price * (imports_to_demand + imports_to_battery)
  - Fees           = sum_{n,t} (fee_pv_feed*pv_feed + fee_battery_in*imp_to_batt
                                + fee_battery_out*batt_to_system + fee_imports_to_demand*imp_to_demand)
  - Cap.-Tariff    = prefer follower 'power_charge' (EUR) if present (deduplicated per node/week),
                     else (peak_imports_new + peak_exports_new) * peak_rate_eur_per_mw (deduplicated per node/week),
                     else 0.

Percent changes vs BASE are printed for:
  - Max Imp., Max Exp., Curt., Exp.-Revenue, Imp.-Cost

NOTE:
  - Absolute Profit column is intentionally NOT printed.
  - "% Profit vs. Base" is intentionally NOT printed.

OPTIONAL:
  - You can also print INTERNAL AT-node tables (one table per AT node) via PRINT_AT_NODE_TABLES.

Run:
  python kpi_country_tables_terminal_follow_main.py
"""

from __future__ import annotations
import os
import re
import inspect
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# ============================================================
# IMPORTS (ONLY knobs, NOT paths)
# ============================================================

from Main import CFG, INSTRUMENTS

# ============================================================
# PATHS (ALL PATHS DEFINED HERE)
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_XLSX = os.path.join(BASE_DIR, "Data_Updated.xlsx")

# BASE (you can point to *_test.csv or the canonical files)
BASE_SYSTEM_CSV   = os.path.join(BASE_DIR, "full_model_results_stage1_base_test.csv")
BASE_FOLLOWER_CSV = os.path.join(BASE_DIR, "full_follower_values_base_test.csv")

# INSTRUMENT files (auto-built from BASE_DIR + slug naming)
def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def instrument_system_csv(base_dir: str, instrument: str) -> str:
    slug = instrument_slug(instrument)
    return os.path.join(base_dir, f"full_model_results_stage1_instrument__{slug}_test.csv")

def instrument_follower_csv(base_dir: str, instrument: str) -> str:
    slug = instrument_slug(instrument)
    return os.path.join(base_dir, f"full_follower_values_instrument__{slug}_test.csv")


# ============================================================
# NUMERIC KNOBS (still authoritative from main CFG)
# ============================================================

HOURS_PER_WEEK = int(getattr(getattr(CFG, "horizon", object()), "hours_per_week", 168))

FEE_PV_FEED = float(getattr(getattr(CFG, "fees", object()), "fee_pv_feed", 0.0))
FEE_BATTERY_IN = float(getattr(getattr(CFG, "fees", object()), "fee_battery_in", 100.0))
FEE_BATTERY_OUT = float(getattr(getattr(CFG, "fees", object()), "fee_battery_out", 0.0))
FEE_IMPORTS_TO_DEMAND = float(getattr(getattr(CFG, "fees", object()), "fee_imports_to_demand", 100.0))

PEAK_RATE_EUR_PER_MW = float(getattr(CFG, "peak_rate_eur_per_mw", 1000.0))

# Printing
ROUND_LEVELS = 2
ROUND_PCT = 1
SHOW_ONLY_COUNTRIES: Optional[List[str]] = None  # e.g. ["AT", "DE"] or None

# Optional internal AT-node tables (one table per AT node)
PRINT_AT_NODE_TABLES = False

# ============================================================
# Helpers
# ============================================================

def call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    return fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def pct_change_vs_base(val: float, base: float) -> float:
    if base is None:
        return np.nan
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.nan
    return 100.0 * (val - base) / base


def load_node_to_control_area(data_xlsx: str) -> Dict[str, str]:
    """
    Uses Parameters_Updated.load_demand_data(data_path, "Demand_Profiles") like Main_New.py,
    and extracts node_to_control_area dict from its output (robustly).
    """
    from Parameters_Updated import load_demand_data  # as in Main_New.py

    out = call_with_supported_kwargs(load_demand_data, data_xlsx, "Demand_Profiles")

    if isinstance(out, dict):
        return {str(k): str(v) for k, v in out.items()}

    if isinstance(out, tuple):
        for obj in reversed(out):
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}

    raise RuntimeError(
        "Could not extract node_to_control_area from Parameters_Updated.load_demand_data output."
    )


# ============================================================
# System CSV normalization
# ============================================================

def system_long_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the concatenated system CSV exported by dump_system_window/concat_weekly_csvs:
      columns typically: window, component, value, index_0, index_1, ...
    Produces:
      variable, node, t_local, start_week, abs_hour, value
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["variable", "node", "t_local", "start_week", "abs_hour", "value"])

    d = df.copy()

    if "component" in d.columns and "variable" not in d.columns:
        d = d.rename(columns={"component": "variable"})

    # start_week from window
    if "start_week" not in d.columns:
        if "window" in d.columns:
            d["start_week"] = pd.to_numeric(d["window"], errors="coerce").fillna(0).astype(int) + 1
        else:
            d["start_week"] = 1
    d["start_week"] = pd.to_numeric(d["start_week"], errors="coerce").fillna(1).astype(int)

    # local timestep in index_1
    if "index_1" in d.columns:
        d["t_local"] = pd.to_numeric(d["index_1"], errors="coerce").fillna(0).astype(int)
    elif "t" in d.columns:
        d["t_local"] = pd.to_numeric(d["t"], errors="coerce").fillna(0).astype(int)
    elif "timestep" in d.columns:
        d["t_local"] = pd.to_numeric(d["timestep"], errors="coerce").fillna(0).astype(int)
    else:
        d["t_local"] = 0

    d["abs_hour"] = d["t_local"] + HOURS_PER_WEEK * (d["start_week"] - 1)

    # node is index_0 for node-indexed vars
    if "index_0" in d.columns:
        d["node"] = d["index_0"].astype(str)
    elif "node" in d.columns:
        d["node"] = d["node"].astype(str)
    else:
        d["node"] = ""

    d["variable"] = d.get("variable", "").astype(str)
    d["value"] = pd.to_numeric(d.get("value", 0.0), errors="coerce").fillna(0.0)

    return d[["variable", "node", "t_local", "start_week", "abs_hour", "value"]]


def aggregate_system_var_by_country(
    sys_long: pd.DataFrame,
    node_to_ca: Dict[str, str],
    varname: str,
) -> pd.DataFrame:
    """
    Returns wide time series by country for a given system variable.
    Index: abs_hour, Columns: country/control_area, Values: sum over nodes in that CA.
    """
    part = sys_long[sys_long["variable"] == varname].copy()
    if part.empty:
        return pd.DataFrame()

    part = part[part["node"].isin(node_to_ca.keys())].copy()
    part["country"] = part["node"].map(lambda n: node_to_ca.get(str(n), "UNKNOWN"))

    out = (
        part.groupby(["abs_hour", "country"], as_index=False)["value"]
        .sum()
        .pivot(index="abs_hour", columns="country", values="value")
        .fillna(0.0)
        .sort_index()
    )
    return out


def aggregate_system_var_by_node(sys_long: pd.DataFrame, varname: str) -> pd.DataFrame:
    """
    Wide time series by node for a given system variable.
    Index: abs_hour, Columns: node, Values: sum over entries for that node.
    """
    part = sys_long[sys_long["variable"] == varname].copy()
    if part.empty:
        return pd.DataFrame()

    out = (
        part.groupby(["abs_hour", "node"], as_index=False)["value"]
        .sum()
        .pivot(index="abs_hour", columns="node", values="value")
        .fillna(0.0)
        .sort_index()
    )
    return out


# ============================================================
# Follower normalization + profit component aggregation
# ============================================================

FOLLOWER_FLOW_KEYS = [
    "pv_feed_to_system",
    "battery_to_system",
    "imports_to_demand",
    "imports_to_battery",
]


def follower_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes follower export (full_follower_values_*.csv)

    Required columns:
      - node, start_week, t

    Expected columns for decomposition:
      - price (€/MWh)
      - pv_feed_to_system, battery_to_system, imports_to_demand, imports_to_battery (MWh per timestep)

    Optional capacity tariff exports:
      - power_charge (EUR), OR peak_imports_new & peak_exports_new (MW)

    Produces abs_hour = t + hours_per_week*(start_week-1)
    """
    if df is None or df.empty:
        cols = ["node", "start_week", "t", "abs_hour", "price", *FOLLOWER_FLOW_KEYS]
        return pd.DataFrame(columns=cols)

    required = {"node", "start_week", "t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Follower CSV missing required columns: {sorted(missing)}")

    d = df.copy()
    d["node"] = d["node"].astype(str)
    d["start_week"] = pd.to_numeric(d["start_week"], errors="coerce").fillna(1).astype(int)
    d["t"] = pd.to_numeric(d["t"], errors="coerce").fillna(0).astype(int)
    d["abs_hour"] = d["t"] + HOURS_PER_WEEK * (d["start_week"] - 1)

    # required for decomposition
    if "price" not in d.columns:
        raise ValueError("Follower CSV must contain 'price' column for profit decomposition.")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")

    # Ensure flow keys exist (fill missing with 0)
    for k in FOLLOWER_FLOW_KEYS:
        if k not in d.columns:
            d[k] = 0.0
        d[k] = pd.to_numeric(d[k], errors="coerce").fillna(0.0)

    # Optional capacity tariff signals
    for opt in ["power_charge", "peak_imports_new", "peak_exports_new"]:
        if opt in d.columns:
            d[opt] = pd.to_numeric(d[opt], errors="coerce").fillna(0.0)

    return d


def compute_profit_components_by_country_from_follower(
    follower_df: pd.DataFrame,
    node_to_ca: Dict[str, str],
) -> pd.DataFrame:
    """
    Returns country-indexed absolute components (EUR):
      Exp.-Revenue, Imp.-Cost, Fees, Cap.-Tariff

    Profit is NOT returned (and not printed).
    """
    if follower_df is None or follower_df.empty:
        return pd.DataFrame(
            columns=["Exp.-Revenue", "Imp.-Cost", "Fees", "Cap.-Tariff"],
            index=pd.Index([], name="country"),
        )

    d = follower_df.copy()

    # keep only mapped nodes
    d = d[d["node"].isin(node_to_ca.keys())].copy()
    d["country"] = d["node"].map(lambda n: node_to_ca.get(str(n), "UNKNOWN"))

    if d["price"].isna().any():
        bad = d[d["price"].isna()][["node", "start_week", "t"]].head(10)
        raise ValueError(f"Follower price contains NaN. Examples:\n{bad}")

    # Revenue / cost (EUR)
    d["exp_revenue"] = d["price"] * (d["pv_feed_to_system"] + d["battery_to_system"])
    d["imp_cost"] = d["price"] * (d["imports_to_demand"] + d["imports_to_battery"])

    # Fees (EUR)
    d["fees"] = (
        FEE_PV_FEED * d["pv_feed_to_system"]
        + FEE_BATTERY_IN * d["imports_to_battery"]
        + FEE_BATTERY_OUT * d["battery_to_system"]
        + FEE_IMPORTS_TO_DEMAND * d["imports_to_demand"]
    )

    # Cap.-Tariff (EUR):
    if "power_charge" in d.columns:
        tmp = d[["country", "node", "start_week", "power_charge"]].copy()
        tmp["power_charge"] = pd.to_numeric(tmp["power_charge"], errors="coerce").fillna(0.0)
        tmp = tmp.drop_duplicates(subset=["country", "node", "start_week"])
        cap_by_country = tmp.groupby("country", as_index=True)["power_charge"].sum()
    elif {"peak_imports_new", "peak_exports_new"}.issubset(d.columns):
        tmp = d[["country", "node", "start_week", "peak_imports_new", "peak_exports_new"]].copy()
        tmp = tmp.drop_duplicates(subset=["country", "node", "start_week"])
        cap_by_country = (
            (tmp["peak_imports_new"] + tmp["peak_exports_new"]) * PEAK_RATE_EUR_PER_MW
        ).groupby(tmp["country"]).sum()
    else:
        cap_by_country = pd.Series(dtype=float)

    grp = d.groupby("country", as_index=True).agg(
        **{
            "Exp.-Revenue": ("exp_revenue", "sum"),
            "Imp.-Cost": ("imp_cost", "sum"),
            "Fees": ("fees", "sum"),
        }
    )

    grp["Cap.-Tariff"] = cap_by_country.reindex(grp.index).fillna(0.0)
    return grp


def compute_profit_components_by_node_from_follower(
    follower_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns node-indexed absolute components (EUR):
      Exp.-Revenue, Imp.-Cost, Fees, Cap.-Tariff
    """
    if follower_df is None or follower_df.empty:
        return pd.DataFrame(
            columns=["Exp.-Revenue", "Imp.-Cost", "Fees", "Cap.-Tariff"],
            index=pd.Index([], name="node"),
        )

    d = follower_df.copy()

    if d["price"].isna().any():
        bad = d[d["price"].isna()][["node", "start_week", "t"]].head(10)
        raise ValueError(f"Follower price contains NaN. Examples:\n{bad}")

    # Revenue / cost (EUR)
    d["exp_revenue"] = d["price"] * (d["pv_feed_to_system"] + d["battery_to_system"])
    d["imp_cost"] = d["price"] * (d["imports_to_demand"] + d["imports_to_battery"])

    # Fees (EUR)
    d["fees"] = (
        FEE_PV_FEED * d["pv_feed_to_system"]
        + FEE_BATTERY_IN * d["imports_to_battery"]
        + FEE_BATTERY_OUT * d["battery_to_system"]
        + FEE_IMPORTS_TO_DEMAND * d["imports_to_demand"]
    )

    # Cap.-Tariff (EUR)
    if "power_charge" in d.columns:
        tmp = d[["node", "start_week", "power_charge"]].copy()
        tmp["power_charge"] = pd.to_numeric(tmp["power_charge"], errors="coerce").fillna(0.0)
        tmp = tmp.drop_duplicates(subset=["node", "start_week"])
        cap_by_node = tmp.groupby("node", as_index=True)["power_charge"].sum()
    elif {"peak_imports_new", "peak_exports_new"}.issubset(d.columns):
        tmp = d[["node", "start_week", "peak_imports_new", "peak_exports_new"]].copy()
        tmp = tmp.drop_duplicates(subset=["node", "start_week"])
        cap_by_node = (
            (tmp["peak_imports_new"] + tmp["peak_exports_new"]) * PEAK_RATE_EUR_PER_MW
        ).groupby(tmp["node"]).sum()
    else:
        cap_by_node = pd.Series(dtype=float)

    grp = d.groupby("node", as_index=True).agg(
        **{
            "Exp.-Revenue": ("exp_revenue", "sum"),
            "Imp.-Cost": ("imp_cost", "sum"),
            "Fees": ("fees", "sum"),
        }
    )
    grp["Cap.-Tariff"] = cap_by_node.reindex(grp.index).fillna(0.0)
    return grp


# ============================================================
# KPI assembly (system KPIs + profit components)
# ============================================================

def compute_country_kpis(
    system_csv: str,
    follower_csv: str,
    node_to_ca: Dict[str, str],
) -> pd.DataFrame:
    """
    Returns a country-indexed KPI DataFrame with columns:
      Max Imp., Max Exp., Curt., Exp.-Revenue, Imp.-Cost, Fees, Cap.-Tariff

    Definitions (ALL follower-based):
      - Max Imp. = max_t sum_{n in CA}(imports_to_demand + imports_to_battery)
      - Max Exp. = max_t sum_{n in CA}(pv_feed_to_system + battery_to_system)
      - Curt.    = sum_t sum_{n in CA}(pv_curtailment)
    """
    # NOTE: system_csv kept in signature to stay compatible with existing calls,
    # but no longer used for Curt. when follower PV curtailment is desired.
    _ = system_csv

    fol_raw = read_csv(follower_csv)
    fol = follower_prepare(fol_raw)

    # keep only mapped nodes
    d = fol[fol["node"].isin(node_to_ca.keys())].copy()
    d["country"] = d["node"].map(lambda n: node_to_ca.get(str(n), "UNKNOWN"))

    # --- follower flows ---
    d["follower_imports"] = d["imports_to_demand"] + d["imports_to_battery"]
    d["follower_exports"] = d["pv_feed_to_system"] + d["battery_to_system"]

    imp_by_ca_t = (
        d.groupby(["abs_hour", "country"])["follower_imports"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    exp_by_ca_t = (
        d.groupby(["abs_hour", "country"])["follower_exports"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )

    max_imp = imp_by_ca_t.max(axis=0) if not imp_by_ca_t.empty else pd.Series(dtype=float)
    max_exp = exp_by_ca_t.max(axis=0) if not exp_by_ca_t.empty else pd.Series(dtype=float)

    # --- follower PV curtailment (required) ---
    if "pv_curtailment" not in d.columns:
        raise ValueError(
            "Follower CSV does not contain 'pv_curtailment'. "
            "Export m.pv_curtailment from the follower model to use follower-only curtailment KPI."
        )
    d["pv_curtailment"] = pd.to_numeric(d["pv_curtailment"], errors="coerce").fillna(0.0)

    curt_by_ca = d.groupby("country", as_index=True)["pv_curtailment"].sum()

    # Profit components (follower-based)
    comps = compute_profit_components_by_country_from_follower(fol, node_to_ca)

    # Country universe
    countries = sorted(
        set(max_imp.index.tolist())
        | set(max_exp.index.tolist())
        | set(curt_by_ca.index.tolist())
        | set(comps.index.tolist())
    )

    rows = []
    for c in countries:
        rows.append(
            {
                "country": c,
                "Max Imp.": float(max_imp.get(c, 0.0)),
                "Max Exp.": float(max_exp.get(c, 0.0)),
                "Curt.": float(curt_by_ca.get(c, 0.0)),
                "Exp.-Revenue": float(comps.loc[c, "Exp.-Revenue"]) if c in comps.index else 0.0,
                "Imp.-Cost": float(comps.loc[c, "Imp.-Cost"]) if c in comps.index else 0.0,
                "Fees": float(comps.loc[c, "Fees"]) if c in comps.index else 0.0,
                "Cap.-Tariff": float(comps.loc[c, "Cap.-Tariff"]) if c in comps.index else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("country").sort_index()


def compute_node_kpis(
    system_csv: str,
    follower_csv: str,
) -> pd.DataFrame:
    """
    Returns a node-indexed KPI DataFrame with columns:
      Max Imp., Max Exp., Curt., Exp.-Revenue, Imp.-Cost, Fees, Cap.-Tariff

    Definitions (ALL follower-based):
      - Max Imp. = max_t (imports_to_demand + imports_to_battery)
      - Max Exp. = max_t (pv_feed_to_system + battery_to_system)
      - Curt.    = sum_t (pv_curtailment)
    """
    _ = system_csv

    fol_raw = read_csv(follower_csv)
    fol = follower_prepare(fol_raw)

    d = fol.copy()

    d["follower_imports"] = d["imports_to_demand"] + d["imports_to_battery"]
    d["follower_exports"] = d["pv_feed_to_system"] + d["battery_to_system"]

    imp_by_node_t = (
        d.groupby(["abs_hour", "node"])["follower_imports"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    exp_by_node_t = (
        d.groupby(["abs_hour", "node"])["follower_exports"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )

    max_imp = imp_by_node_t.max(axis=0) if not imp_by_node_t.empty else pd.Series(dtype=float)
    max_exp = exp_by_node_t.max(axis=0) if not exp_by_node_t.empty else pd.Series(dtype=float)

    if "pv_curtailment" not in d.columns:
        raise ValueError(
            "Follower CSV does not contain 'pv_curtailment'. "
            "Export m.pv_curtailment from the follower model to use follower-only curtailment KPI."
        )
    d["pv_curtailment"] = pd.to_numeric(d["pv_curtailment"], errors="coerce").fillna(0.0)
    curt_by_node = d.groupby("node", as_index=True)["pv_curtailment"].sum()

    comps = compute_profit_components_by_node_from_follower(fol)

    nodes = sorted(
        set(max_imp.index.tolist())
        | set(max_exp.index.tolist())
        | set(curt_by_node.index.tolist())
        | set(comps.index.tolist())
    )

    rows = []
    for n in nodes:
        rows.append(
            {
                "node": n,
                "Max Imp.": float(max_imp.get(n, 0.0)),
                "Max Exp.": float(max_exp.get(n, 0.0)),
                "Curt.": float(curt_by_node.get(n, 0.0)),
                "Exp.-Revenue": float(comps.loc[n, "Exp.-Revenue"]) if n in comps.index else 0.0,
                "Imp.-Cost": float(comps.loc[n, "Imp.-Cost"]) if n in comps.index else 0.0,
                "Fees": float(comps.loc[n, "Fees"]) if n in comps.index else 0.0,
                "Cap.-Tariff": float(comps.loc[n, "Cap.-Tariff"]) if n in comps.index else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("node").sort_index()


def format_table_one_country(country: str, levels: pd.DataFrame) -> pd.DataFrame:
    """
    levels: rows=scenarios, cols=absolute metrics.

    Adds % columns vs BASE for:
      - Max Imp., Max Exp., Curt., Exp.-Revenue, Imp.-Cost

    Does NOT add Profit or % Profit columns.
    """
    base = levels.loc["BASE"]

    pct_targets = ["Max Imp.", "Max Exp.", "Curt.", "Exp.-Revenue", "Imp.-Cost"]

    out = levels.copy()
    for col in pct_targets:
        pct_col = f"% {col} vs. Base"
        out[pct_col] = [
            0.0 if scen == "BASE" else pct_change_vs_base(float(out.loc[scen, col]), float(base[col]))
            for scen in out.index
        ]

    # rounding
    abs_cols = list(levels.columns)
    pct_cols = [c for c in out.columns if c.startswith("% ") and c.endswith(" vs. Base")]

    out[abs_cols] = out[abs_cols].round(ROUND_LEVELS)
    out[pct_cols] = out[pct_cols].round(ROUND_PCT)

    # final column order
    ordered = [
        "Max Imp.", "Max Exp.", "Curt.",
        "Exp.-Revenue", "Imp.-Cost", "Fees", "Cap.-Tariff",
        "% Max Imp. vs. Base", "% Max Exp. vs. Base", "% Curt. vs. Base",
        "% Exp.-Revenue vs. Base", "% Imp.-Cost vs. Base",
    ]
    ordered = [c for c in ordered if c in out.columns]
    return out[ordered]


# ============================================================
# Main
# ============================================================

def main() -> None:
    node_to_ca = load_node_to_control_area(DATA_XLSX)

    # --- compute KPIs per scenario (COUNTRY) ---
    scenario_kpis_country: Dict[str, pd.DataFrame] = {}
    scenario_kpis_country["BASE"] = compute_country_kpis(
        BASE_SYSTEM_CSV, BASE_FOLLOWER_CSV, node_to_ca
    )

    # --- optional: compute KPIs per scenario (NODE) for AT nodes ---
    scenario_kpis_node: Dict[str, pd.DataFrame] = {}
    if PRINT_AT_NODE_TABLES:
        scenario_kpis_node["BASE"] = compute_node_kpis(BASE_SYSTEM_CSV, BASE_FOLLOWER_CSV)

    for inst in INSTRUMENTS:
        fol_path = instrument_follower_csv(BASE_DIR, inst)
        sys_path = instrument_system_csv(BASE_DIR, inst)
    
        scenario_kpis_country[inst] = compute_country_kpis(sys_path, fol_path, node_to_ca)
    
        if PRINT_AT_NODE_TABLES:
            scenario_kpis_node[inst] = compute_node_kpis(sys_path, fol_path)

    # --- countries universe ---
    all_countries = sorted(set().union(*[df.index.tolist() for df in scenario_kpis_country.values()]))

    if SHOW_ONLY_COUNTRIES is not None:
        all_countries = [c for c in all_countries if c in set(SHOW_ONLY_COUNTRIES)]

    # --- print one table per country/control area ---
    for c in all_countries:
        rows = {}
        base_cols = list(scenario_kpis_country["BASE"].columns)

        for scen, df in scenario_kpis_country.items():
            if c in df.index:
                rows[scen] = df.loc[c].to_dict()
            else:
                rows[scen] = {col: 0.0 for col in base_cols}

        levels = pd.DataFrame.from_dict(rows, orient="index")
        levels = levels.loc[["BASE"] + [i for i in INSTRUMENTS if i in levels.index]]

        table = format_table_one_country(c, levels)

        print("\n" + "=" * 110)
        print(f"CONTROL AREA: {c}")
        print("=" * 110)
        print(table.to_string())

    # --- optional: print INTERNAL AT-node tables ---
    if PRINT_AT_NODE_TABLES:
        at_nodes = sorted([n for n, ca in node_to_ca.items() if str(ca) == "AT"])
        if not at_nodes:
            print("\n[WARN] PRINT_AT_NODE_TABLES=True but no AT nodes found in node_to_control_area mapping (value == 'AT').")
            return

        for n in at_nodes:
            rows = {}
            base_cols = list(scenario_kpis_node["BASE"].columns)

            for scen, df in scenario_kpis_node.items():
                if n in df.index:
                    rows[scen] = df.loc[n].to_dict()
                else:
                    rows[scen] = {col: 0.0 for col in base_cols}

            levels = pd.DataFrame.from_dict(rows, orient="index")
            levels = levels.loc[["BASE"] + [i for i in INSTRUMENTS if i in levels.index]]

            table = format_table_one_country(f"AT::{n}", levels)

            print("\n" + "=" * 110)
            print(f"AT NODE: {n}")
            print("=" * 110)
            print(table.to_string())


if __name__ == "__main__":
    main()
