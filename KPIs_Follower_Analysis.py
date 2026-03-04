"""
KPI_ANALYSIS.py

This script computes structured KPI tables (terminal-only) for the Stage-1 +
Follower model architecture.

It compares BASE vs. policy instruments defined in Main.INSTRUMENTS and prints
one table per control area (country) and optionally per Austrian node.

───────────────────────────────────────────────────────────────────────────────
KPI DEFINITIONS (ALL FOLLOWER-BASED)
───────────────────────────────────────────────────────────────────────────────

All physical KPIs are derived exclusively from the follower model export
(full_follower_values_*.csv). No system-level dispatch values are used.

Per control area (aggregated across nodes):

1) Max Imp. (MW)
   max_t Σ_nodes (imports_to_demand + imports_to_battery) / dt

2) Max Exp. (MW)
   max_t Σ_nodes (pv_feed_to_system + battery_to_system) / dt

3) Curt. (MWh)
   Σ_t Σ_nodes pv_curtailment

4) Batt FLH (h)
   Σ_t Σ_nodes (battery_to_demand + battery_to_system)
   ----------------------------------------------------
        Σ_nodes P_battery_out_rated

   where rated battery discharge power (MW) is derived from
   Data_Updated.xlsx → sheet "Flexibility_Specific_Data":

       Turbine Capacity [GW]
         × battery_share (from Main.CFG.stage1.Shares)
         × 1000  → MW

   This ensures:
     - FLH reflects total battery discharge
     - Installed turbine power is used
     - No dependence on follower CSV telemetry like bat_out_cap_dt

Follower profit decomposition (EUR):

   Exp.-Revenue = Σ price × (pv_feed_to_system + battery_to_system)
   Imp.-Cost    = Σ price × (imports_to_demand + imports_to_battery)
   Fees         = Σ fee terms
   Cap.-Tariff  = power_charge (if exported)
                  else peak_imports_new + peak_exports_new × peak_rate

Percent changes vs BASE are printed for:
   - Max Imp.
   - Max Exp.
   - Curt.
   - Batt FLH
   - Exp.-Revenue
   - Imp.-Cost

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURE
───────────────────────────────────────────────────────────────────────────────

- Imports only CFG + INSTRUMENTS from Main
- All file paths are defined inside this script
- Instrument CSV paths are constructed via slug logic
- Rated battery power is derived from Data_Updated.xlsx
- No dependency on bat_out_cap_dt in follower CSV

Run:
    python KPI_ANALYSIS.py
"""

from __future__ import annotations

import os
import re
import inspect
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# IMPORTS (ONLY KNOBS — NOT PATHS)
# ============================================================

from Main import CFG, INSTRUMENTS


# ============================================================
# PATH CONFIGURATION (FULLY LOCAL)
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_XLSX = os.path.join(BASE_DIR, "Data_Updated.xlsx")

BASE_SYSTEM_CSV = os.path.join(BASE_DIR, "full_model_results_stage1_base_test.csv")
BASE_FOLLOWER_CSV = os.path.join(BASE_DIR, "full_follower_values_base_test.csv")


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
# CONFIG FROM CFG
# ============================================================

HOURS_PER_WEEK = int(getattr(getattr(CFG, "horizon", object()), "hours_per_week", 168))

FEE_PV_FEED = float(getattr(getattr(CFG, "fees", object()), "fee_pv_feed", 0.0))
FEE_BATTERY_IN = float(getattr(getattr(CFG, "fees", object()), "fee_battery_in", 0.0))
FEE_BATTERY_OUT = float(getattr(getattr(CFG, "fees", object()), "fee_battery_out", 0.0))
FEE_IMPORTS_TO_DEMAND = float(getattr(getattr(CFG, "fees", object()), "fee_imports_to_demand", 0.0))

PEAK_RATE_EUR_PER_MW = float(getattr(CFG, "peak_rate_eur_per_mw", 0.0))

# Stage1 shares (authoritative source for battery_share)
BATTERY_SHARE = float(getattr(getattr(CFG, "stage1", object()), "battery_share", 1.0))

ROUND_LEVELS = 2
ROUND_PCT = 1
SHOW_ONLY_COUNTRIES: Optional[List[str]] = None
PRINT_AT_NODE_TABLES = False


# ============================================================
# UTILITY HELPERS
# ============================================================

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def pct_change_vs_base(val: float, base: float) -> float:
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.nan
    return 100.0 * (val - base) / base


# ============================================================
# LOAD NODE → CONTROL AREA
# ============================================================

def load_node_to_control_area(data_xlsx: str) -> Dict[str, str]:
    from Parameters_Updated import load_demand_data
    out = load_demand_data(data_xlsx, "Demand_Profiles")
    if isinstance(out, dict):
        return {str(k): str(v) for k, v in out.items()}
    if isinstance(out, tuple):
        for obj in reversed(out):
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
    raise RuntimeError("Could not extract node_to_control_area.")


# ============================================================
# LOAD BATTERY RATED POWER FROM EXCEL
# ============================================================

def load_battery_rated_power_mw_by_node(data_xlsx: str) -> pd.Series:
    df = pd.read_excel(data_xlsx, sheet_name="Flexibility_Specific_Data")
    d = df[df["Power Plant Type"].astype(str).str.lower() == "battery"].copy()
    d["Node"] = d["Node"].astype(str)
    d["Turbine Capacity [GW]"] = pd.to_numeric(d["Turbine Capacity [GW]"], errors="coerce").fillna(0.0)

    d["rated_power_mw"] = d["Turbine Capacity [GW]"] * BATTERY_SHARE * 1000.0

    return d.groupby("Node")["rated_power_mw"].max().clip(lower=0.0)


# ============================================================
# FOLLOWER PREP
# ============================================================

def follower_prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["node"] = d["node"].astype(str)
    d["abs_hour"] = d["t"] + HOURS_PER_WEEK * (d["start_week"] - 1)

    for col in [
        "pv_feed_to_system",
        "battery_to_system",
        "battery_to_demand",
        "imports_to_demand",
        "imports_to_battery",
        "pv_curtailment",
    ]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    if "battery_out" in d.columns:
        d["battery_out"] = pd.to_numeric(d["battery_out"], errors="coerce").fillna(0.0)

    d["price"] = pd.to_numeric(d["price"], errors="coerce").fillna(0.0)

    return d


# ============================================================
# BATTERY FLH
# ============================================================

def battery_flh_by_node(fol: pd.DataFrame, rated_power_mw_by_node: pd.Series) -> pd.Series:
    if "battery_out" in fol.columns:
        discharge = fol.groupby("node")["battery_out"].sum()
    else:
        discharge = fol.groupby("node").apply(
            lambda g: (g["battery_to_demand"] + g["battery_to_system"]).sum()
        )

    rp = rated_power_mw_by_node.reindex(discharge.index).fillna(0.0)
    return discharge / rp.replace(0.0, np.nan)


def battery_flh_by_country(
    fol: pd.DataFrame,
    node_to_ca: Dict[str, str],
    rated_power_mw_by_node: pd.Series
) -> pd.Series:

    fol = fol.copy()
    fol["country"] = fol["node"].map(node_to_ca)

    if "battery_out" in fol.columns:
        fol["_dis"] = fol["battery_out"]
    else:
        fol["_dis"] = fol["battery_to_demand"] + fol["battery_to_system"]

    discharge_by_country = fol.groupby("country")["_dis"].sum()

    node_country = fol[["node", "country"]].drop_duplicates().set_index("node")["country"]
    rp = rated_power_mw_by_node.reindex(node_country.index).fillna(0.0)
    rated_by_country = rp.groupby(node_country).sum()

    return discharge_by_country / rated_by_country.replace(0.0, np.nan)


# ============================================================
# COUNTRY KPI COMPUTATION
# ============================================================

def compute_country_kpis(
    follower_csv: str,
    node_to_ca: Dict[str, str],
    rated_power_mw_by_node: pd.Series,
) -> pd.DataFrame:

    fol = follower_prepare(read_csv(follower_csv))

    fol["country"] = fol["node"].map(node_to_ca)

    dt = 1.0

    imports = (
        (fol["imports_to_demand"] + fol["imports_to_battery"]) / dt
    ).groupby([fol["abs_hour"], fol["country"]]).sum().unstack().fillna(0)

    exports = (
        (fol["pv_feed_to_system"] + fol["battery_to_system"]) / dt
    ).groupby([fol["abs_hour"], fol["country"]]).sum().unstack().fillna(0)

    max_imp = imports.max()
    max_exp = exports.max()
    curt = fol.groupby("country")["pv_curtailment"].sum()

    batt_flh = battery_flh_by_country(fol, node_to_ca, rated_power_mw_by_node)

    fol["exp_rev"] = fol["price"] * (fol["pv_feed_to_system"] + fol["battery_to_system"])
    fol["imp_cost"] = fol["price"] * (fol["imports_to_demand"] + fol["imports_to_battery"])

    fol["fees"] = (
        FEE_PV_FEED * fol["pv_feed_to_system"]
        + FEE_BATTERY_IN * fol["imports_to_battery"]
        + FEE_BATTERY_OUT * fol["battery_to_system"]
        + FEE_IMPORTS_TO_DEMAND * fol["imports_to_demand"]
    )

    grp = fol.groupby("country").agg(
        {
            "exp_rev": "sum",
            "imp_cost": "sum",
            "fees": "sum",
        }
    )

    df = pd.DataFrame({
        "Max Imp.": max_imp,
        "Max Exp.": max_exp,
        "Curt.": curt,
        "Batt FLH": batt_flh,
        "Exp.-Revenue": grp["exp_rev"],
        "Imp.-Cost": grp["imp_cost"],
        "Fees": grp["fees"],
    })

    return df.sort_index()


# ============================================================
# MAIN
# ============================================================

def main():

    node_to_ca = load_node_to_control_area(DATA_XLSX)
    rated_power = load_battery_rated_power_mw_by_node(DATA_XLSX)

    scenario_tables = {}

    scenario_tables["BASE"] = compute_country_kpis(
        BASE_FOLLOWER_CSV,
        node_to_ca,
        rated_power,
    )

    for inst in INSTRUMENTS:
        scenario_tables[inst] = compute_country_kpis(
            instrument_follower_csv(BASE_DIR, inst),
            node_to_ca,
            rated_power,
        )

    countries = scenario_tables["BASE"].index.tolist()

    for c in countries:

        rows = {}
        for scen, df in scenario_tables.items():
            rows[scen] = df.loc[c]

        levels = pd.DataFrame(rows).T
        base = levels.loc["BASE"]

        for col in ["Max Imp.", "Max Exp.", "Curt.", "Batt FLH",
                    "Exp.-Revenue", "Imp.-Cost"]:
            levels[f"% {col} vs. Base"] = [
                0.0 if s == "BASE" else pct_change_vs_base(levels.loc[s, col], base[col])
                for s in levels.index
            ]

        levels = levels.round(ROUND_LEVELS)

        print("\n" + "=" * 100)
        print(f"CONTROL AREA: {c}")
        print("=" * 100)
        print(levels.to_string())


if __name__ == "__main__":
    main()