#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data_Loaders_With_PTDF_CSV.py  (ADAPTED for synchronized PTDF export)

Loads all input tables from Data_Updated.xlsx and (for exchange) imports an
already-computed PTDF from a CSV, aligned to the Exchange_Data ordering
(connections x non-slack nodes).

CHANGE
------
- Uses the NEW synchronized PTDF CSV produced by your Data_Updated builder:
  e.g. "PTDF_Synchronized.csv" (rows=connections, cols=nodes INCLUDING slack col as 0)

Notes
-----
- Your new PTDF export (from build_data_updated) exports FULL columns (including slack column),
  with slack column = 0.0. This loader will:
    - align rows to Exchange_Data connections
    - align columns to Exchange_Data nodes
    - drop slack column to return (connections x non-slack nodes) as expected downstream
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =============================================================================
# CONFIG
# =============================================================================

FILEPATH_XLSX = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Data_Updated.xlsx"

# NEW: synchronized PTDF export path (adjust to your actual file name/path)
PTDF_CSV_PATH = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\PTDF_Synchronized.csv"


# =============================================================================
# QUICK CHECK / PRINT HELPERS
# =============================================================================

def print_general_quick_check(co2_price, voll, wind_pv_cost, phs_cost, dsr_cost, flow_cost) -> None:
    print("\n[GENERAL] QUICK CHECK")
    print("CO2 price [€/t]:", co2_price)
    print("VoLL [€/MWh]:", voll)
    print("Wind & PV cost [€/MWh]:", wind_pv_cost)
    print("Hydro cost [€/MWh]:", phs_cost)
    print("DSR cost [€/MWh]:", dsr_cost)
    print("Flow cost [€/MWh]:", flow_cost)


def print_demand_quick_check(
    normalized_profiles_df: pd.DataFrame,
    node_demand_df: pd.DataFrame,
    control_areas: List[str],
    nodes: List[str],
    node_to_control_area: Dict[str, str],
) -> None:
    print("\n[DEMAND] QUICK CHECK")
    print("\nFirst entry of node_demand_df (first row, first 5 nodes):")
    print(node_demand_df.iloc[0, :5])

    print("\ncontrol_areas:")
    print(control_areas)

    print("\nnodes (first 20):")
    print(nodes[:20])

    print("\nNormalized profile column sums (should be ~1):")
    print(normalized_profiles_df.sum(axis=0).round(6))

    print("\nNode → Control Area mapping (first 20):")
    for k in list(node_to_control_area.keys())[:20]:
        print(f"{k}: {node_to_control_area[k]}")


def print_thermal_quick_check(
    thermal_power_plant_specific_data: pd.DataFrame,
    thermal_power_plant_node_indices: Dict[str, List[int]],
) -> None:
    print("\n[THERMAL] QUICK CHECK")

    if len(thermal_power_plant_specific_data) == 0:
        print("Thermal specific data is EMPTY!")
        return

    print("\nFirst row (FULL):")
    print(thermal_power_plant_specific_data.iloc[0])

    if "Control Area" in thermal_power_plant_specific_data.columns:
        print("\nControl Areas:")
        print(sorted(thermal_power_plant_specific_data["Control Area"].astype(str).unique().tolist()))

    if "Node" in thermal_power_plant_specific_data.columns:
        print("\nNodes (first 20):")
        print(sorted(thermal_power_plant_specific_data["Node"].astype(str).unique().tolist())[:20])

    if "Power Plant Type" in thermal_power_plant_specific_data.columns:
        print("\nPower Plant Types:")
        print(sorted(thermal_power_plant_specific_data["Power Plant Type"].astype(str).unique().tolist()))

    print("\nNode-indices mapping:")
    print("n_keys:", len(thermal_power_plant_node_indices))
    sample_keys = list(thermal_power_plant_node_indices.keys())[:15]
    print("sample keys:", sample_keys)
    if sample_keys:
        k0 = sample_keys[0]
        vals = thermal_power_plant_node_indices[k0]
        print(f"example value for '{k0}': len={len(vals)}, first 10={vals[:10]}")


def print_phs_quick_check(
    phs_power_plant_specific_data: pd.DataFrame,
    phs_power_plant_control_area_indices: Dict[str, List[int]],
    phs_power_plant_node_indices: Dict[str, List[int]],
) -> None:
    print("\n[(P)HS] QUICK CHECK")

    if len(phs_power_plant_specific_data) == 0:
        print("(P)HS specific data is EMPTY!")
        return

    print("\nFirst row (FULL):")
    print(phs_power_plant_specific_data.iloc[0])

    if "Control Area" in phs_power_plant_specific_data.columns:
        print("\nControl Areas:")
        print(sorted(phs_power_plant_specific_data["Control Area"].astype(str).unique().tolist()))

    if "Node" in phs_power_plant_specific_data.columns:
        print("\nNodes (first 20):")
        print(sorted(phs_power_plant_specific_data["Node"].astype(str).unique().tolist())[:20])

    if "Power Plant Type" in phs_power_plant_specific_data.columns:
        print("\nPower Plant Types:")
        print(sorted(phs_power_plant_specific_data["Power Plant Type"].astype(str).unique().tolist()))

    print("\nControl-area indices mapping:")
    print("n_keys:", len(phs_power_plant_control_area_indices))
    print("sample keys:", list(phs_power_plant_control_area_indices.keys())[:15])

    print("\nNode indices mapping:")
    print("n_keys:", len(phs_power_plant_node_indices))
    print("sample keys:", list(phs_power_plant_node_indices.keys())[:15])


def print_inflow_quick_check(
    inflow_data: pd.DataFrame,
    hourly_inflow_data: pd.DataFrame,
    totals: Dict[str, float],
    *,
    label: str,
) -> None:
    print(f"\n[{label}] QUICK CHECK")

    if inflow_data is None or len(inflow_data) == 0:
        print("Inflow data is EMPTY!")
        return

    print("\nOriginal inflow first row (FULL):")
    print(inflow_data.iloc[0])

    print("\nHourly inflow first row (FULL):")
    print(hourly_inflow_data.iloc[0])

    print("\nTotals (first 20):")
    for k in list(totals.keys())[:20]:
        print(f"{k}: {totals[k]}")

    print("\nHourly inflow column sums (should be ~1):")
    print(hourly_inflow_data.sum(axis=0).round(6))


def print_storage_profile_quick_check(weekly_df: pd.DataFrame, interpolated_df: pd.DataFrame) -> None:
    print("\n[STORAGE PROFILES] QUICK CHECK")

    if weekly_df is None or len(weekly_df) == 0:
        print("Weekly storage profiles are EMPTY!")
        return

    # --- NEW: print FULL headers ---
    print("\nWeekly storage profile headers (ALL columns):")
    print(list(weekly_df.columns.astype(str)))

    if interpolated_df is not None and len(interpolated_df) > 0:
        print("\nInterpolated storage profile headers (ALL columns):")
        print(list(interpolated_df.columns.astype(str)))
    else:
        print("\nInterpolated storage profiles are EMPTY!")

    print("\nWeekly first row (FULL):")
    print(weekly_df.iloc[0])

    if interpolated_df is not None and len(interpolated_df) > 0:
        print("\nInterpolated first row (FULL):")
        print(interpolated_df.iloc[0])

    print("\nShapes:")
    wk_shape = weekly_df.shape if weekly_df is not None else None
    ip_shape = interpolated_df.shape if interpolated_df is not None else None
    print("weekly:", wk_shape, "interpolated:", ip_shape)


def print_res_quick_check(res_specific_data: pd.DataFrame, res_node_indices: Dict[str, List[int]]) -> None:
    print("\n[RES] QUICK CHECK")

    if len(res_specific_data) == 0:
        print("RES specific data is EMPTY!")
        return

    print("\nFirst row (FULL):")
    print(res_specific_data.iloc[0])

    if "Control Area" in res_specific_data.columns:
        print("\nControl Areas:")
        print(sorted(res_specific_data["Control Area"].astype(str).unique().tolist()))

    if "Node" in res_specific_data.columns:
        print("\nNodes (first 20):")
        print(sorted(res_specific_data["Node"].astype(str).unique().tolist())[:20])

    if "Power Plant Type" in res_specific_data.columns:
        print("\nPower Plant Types:")
        print(sorted(res_specific_data["Power Plant Type"].astype(str).unique().tolist()))

    print("\nNode-indices mapping:")
    print("n_keys:", len(res_node_indices))
    print("sample keys:", list(res_node_indices.keys())[:15])


def print_res_profile_quick_check(profile_df: pd.DataFrame, *, name: str, expected_sum: str | None = None) -> None:
    print(f"\n[{name}] QUICK CHECK")

    if profile_df is None or len(profile_df) == 0:
        print("Profile data is EMPTY!")
        return

    print("\nFirst row (FULL):")
    print(profile_df.iloc[0])

    print("\nShape:", profile_df.shape)
    print("\nColumns (first 20):", list(profile_df.columns)[:20])

    if expected_sum == "colsum1":
        print("\nColumn sums (should be ~1):")
        print(profile_df.sum(axis=0).round(6))


def print_flex_quick_check(
    dsr_specific_data: pd.DataFrame,
    battery_specific_data: pd.DataFrame,
    dsr_node_indices: Dict[str, List[int]],
    battery_node_indices: Dict[str, List[int]],
) -> None:
    print("\n[FLEXIBILITY] QUICK CHECK")

    print("\n--- DSR specific data ---")
    if len(dsr_specific_data) == 0:
        print("DSR specific data is EMPTY!")
    else:
        print("First row (FULL):")
        print(dsr_specific_data.iloc[0])
        print("Nodes (first 20):", sorted(dsr_specific_data["Node"].astype(str).unique().tolist())[:20])

    print("\n--- Battery specific data ---")
    if len(battery_specific_data) == 0:
        print("Battery specific data is EMPTY!")
    else:
        print("First row (FULL):")
        print(battery_specific_data.iloc[0])
        print("Nodes (first 20):", sorted(battery_specific_data["Node"].astype(str).unique().tolist())[:20])

    print("\nDSR node-indices mapping:")
    print("n_keys:", len(dsr_node_indices))
    print("sample keys:", list(dsr_node_indices.keys())[:15])

    print("\nBattery node-indices mapping:")
    print("n_keys:", len(battery_node_indices))
    print("sample keys:", list(battery_node_indices.keys())[:15])


def print_exchange_quick_check(
    nodes: List[str],
    connections: List[str],
    incidence_matrix: pd.DataFrame,
    ntc_pos: Dict[str, float],
    ntc_neg: Dict[str, float],
    xborder: Dict[str, float],
    conductance_series: pd.Series,
    *,
    max_show: int = 20,
) -> None:
    print("\n[EXCHANGE] QUICK CHECK")
    print("n_nodes:", len(nodes), "| n_lines:", len(connections))

    print("\nNodes (first 20):")
    print(nodes[:20])

    print("\nConnections (first 20):")
    print(connections[:20])

    print("\nIncidence matrix shape:", incidence_matrix.shape)
    if len(connections) > 0 and len(nodes) > 0:
        print("\nIncidence matrix sample (first 5 nodes x first 5 lines):")
        print(incidence_matrix.iloc[:5, :5])

    print("\nNTC_pos sample (first 10):")
    for k in list(ntc_pos.keys())[:10]:
        print(f"{k}: {ntc_pos[k]}")

    print("\nNTC_neg sample (first 10):")
    for k in list(ntc_neg.keys())[:10]:
        print(f"{k}: {ntc_neg[k]}")

    print("\nxBorder sample (first 10):")
    for k in list(xborder.keys())[:10]:
        print(f"{k}: {xborder[k]}")

    print("\nConductance sample (first 10):")
    print(conductance_series.head(10))


# =============================================================================
# LOADERS
# =============================================================================

# -----------------------------------------------------------------------------
# GENERAL PARAMETERS
# -----------------------------------------------------------------------------

def load_general_data(filepath: str, sheetname_general: str):
    general_data = pd.read_excel(filepath, sheet_name=sheetname_general)

    co2_price = general_data["CO2 Price [€/t]"].to_numpy()
    voll = general_data["VoLL [€/MWh]"].to_numpy()
    wind_pv_cost = general_data["Wind PV Cost [€/MWh]"].to_numpy()
    phs_cost = general_data["Hydro Cost [€/MWh]"].to_numpy()
    dsr_cost = general_data["DSR Cost [€/MWh]"].to_numpy()
    flow_cost = general_data["Flow Cost [€/MWh]"].to_numpy()

    return co2_price, voll, wind_pv_cost, phs_cost, dsr_cost, flow_cost


# -----------------------------------------------------------------------------
# DEMAND
# -----------------------------------------------------------------------------

def load_demand_data(filepath: str, sheetname_demand_profiles: str):
    demand_df = pd.read_excel(filepath, sheet_name=sheetname_demand_profiles)

    time_col = demand_df.columns[0]
    demand_df = demand_df.copy()

    # First col is "Hour" (1..N), not timestamps
    demand_df[time_col] = pd.to_numeric(demand_df[time_col], errors="coerce")
    if demand_df[time_col].isna().any():
        bad = demand_df.loc[demand_df[time_col].isna()].head(5)
        raise ValueError(
            f"[load_demand_data] Could not parse numeric timestep in first column '{time_col}'. Examples:\n{bad}"
        )

    demand_df = demand_df.set_index(time_col)

    # Nodes are all remaining columns
    nodes = demand_df.columns.astype(str).str.strip().tolist()

    # Ensure numeric demand
    demand_df = demand_df.apply(pd.to_numeric, errors="coerce")
    if demand_df.isna().any().any():
        bad_cols = demand_df.columns[demand_df.isna().any()].astype(str).tolist()[:20]
        raise ValueError(
            f"[load_demand_data] Non-numeric or missing demand values detected. Example columns with NaNs: {bad_cols}"
        )

    node_demand_df = demand_df.copy()

    # ------------------------------------------------------------------
    # NEW: derive control areas from the Demand_Profiles header
    # ------------------------------------------------------------------
    # These are the aggregate non-AT control-area columns at the end of the sheet.
    non_at_ca_cols = {"DE", "NL", "BE", "LU", "CH", "CZ", "SI", "PL", "SK", "HU", "IT", "FR"}

    # Sanity: which of these are actually present?
    present_non_at = [c for c in non_at_ca_cols if c in node_demand_df.columns]
    missing_non_at = [c for c in non_at_ca_cols if c not in node_demand_df.columns]

    if not present_non_at:
        raise ValueError(
            "[load_demand_data] None of the expected non-AT CA columns were found in Demand_Profiles. "
            f"Expected at least one of: {sorted(non_at_ca_cols)}. "
            f"Found columns (first 30): {list(node_demand_df.columns.astype(str))[:30]}"
        )

    # Everything that is NOT one of the foreign CA aggregates is treated as an AT node.
    node_to_control_area: Dict[str, str] = {}
    for n in node_demand_df.columns.astype(str):
        n_str = n.strip()
        if n_str in non_at_ca_cols:
            node_to_control_area[n_str] = n_str   # e.g. "DE" -> "DE"
        else:
            node_to_control_area[n_str] = "AT"    # all Austrian nodes

    # Build ordered control_areas list:
    # - use the present non-AT CA columns (sorted) plus "AT"
    # - optionally keep any other CAs if your sheet contains them (rare)
    control_areas: List[str] = ["AT"] + sorted(present_non_at)

    # OPTIONAL: warn if some expected non-AT columns are missing (not fatal)
    if missing_non_at:
        print(f"[load_demand_data] Warning: missing expected CA columns in Demand_Profiles: {missing_non_at}")

    # ------------------------------------------------------------------
    # Aggregate hourly demand per control area and normalize
    # ------------------------------------------------------------------
    ca_hourly = pd.DataFrame(index=node_demand_df.index)

    for ca in control_areas:
        ca_nodes = [n for n, ca2 in node_to_control_area.items() if ca2 == ca]
        if len(ca_nodes) == 0:
            ca_hourly[ca] = 0.0
        else:
            ca_hourly[ca] = node_demand_df[ca_nodes].sum(axis=1)

    normalized_profiles_df = ca_hourly.div(ca_hourly.sum(axis=0), axis=1).fillna(0.0)

    return normalized_profiles_df, node_demand_df, control_areas, nodes, node_to_control_area

# -----------------------------------------------------------------------------
# THERMAL POWER PLANTS
# -----------------------------------------------------------------------------

def load_thermal_power_plant_data(filepath: str, sheetname_thermal1: str, sheetname_thermal2: str):
    df_data = pd.read_excel(filepath, sheet_name=sheetname_thermal1)
    df_spec = pd.read_excel(filepath, sheet_name=sheetname_thermal2)

    df_spec["Power Plant Type"] = df_spec["Power Plant Type"].astype(str).str.strip()
    df_spec["Node"] = df_spec["Node"].astype(str).str.strip()
    if "Control Area" in df_spec.columns:
        df_spec["Control Area"] = df_spec["Control Area"].astype(str).str.strip()

    data_by_type = df_data.set_index("Power Plant Type")
    spec = df_spec.merge(data_by_type, on="Power Plant Type", how="left")

    node_indices = {node: spec[spec["Node"] == node].index.tolist() for node in spec["Node"].unique()}
    return spec, node_indices


# -----------------------------------------------------------------------------
# (P)HS POWER PLANTS + INFLOW + STORAGE PROFILES
# -----------------------------------------------------------------------------

def load_phs_power_plant_data(filepath: str, sheetname_phs1: str, sheetname_phs2: str):
    df_data = pd.read_excel(filepath, sheet_name=sheetname_phs1)
    df_spec = pd.read_excel(filepath, sheet_name=sheetname_phs2)

    df_spec["Power Plant Type"] = df_spec["Power Plant Type"].astype(str).str.strip()
    df_spec["Node"] = df_spec["Node"].astype(str).str.strip()
    if "Control Area" in df_spec.columns:
        df_spec["Control Area"] = df_spec["Control Area"].astype(str).str.strip()

    data_by_type = df_data.set_index("Power Plant Type")
    spec = df_spec.merge(data_by_type, on="Power Plant Type", how="left")

    ca_indices = {ca: spec[spec["Control Area"] == ca].index.tolist() for ca in spec["Control Area"].unique()}
    node_indices = {node: spec[spec["Node"] == node].index.tolist() for node in spec["Node"].unique()}
    return spec, ca_indices, node_indices


def load_phs_inflow_data(filepath: str, sheetname_phs_inflow: str):
    inflow_data = pd.read_excel(filepath, sheet_name=sheetname_phs_inflow)

    total_hours = 8760
    inflow_data = inflow_data.copy()
    inflow_data.rename(
        columns={inflow_data.columns[0]: "Start", inflow_data.columns[1]: "End"},
        inplace=True,
    )

    time_steps = inflow_data["End"].values
    hourly_time_steps = np.arange(1, total_hours + 1)  # 1-based indexing

    interpolated_data: Dict[str, np.ndarray] = {}
    totals: Dict[str, float] = {}

    for control_area in inflow_data.columns[2:]:
        inflow_values = inflow_data[control_area].values
        interp_func = interp1d(time_steps, inflow_values, kind="linear", fill_value="extrapolate")
        hourly_values = interp_func(hourly_time_steps)

        total = float(np.sum(hourly_values))
        totals[control_area] = total
        interpolated_data[control_area] = hourly_values / total if total != 0 else np.zeros_like(hourly_values)

    hourly_inflow_data = pd.DataFrame(interpolated_data, index=hourly_time_steps)
    hourly_inflow_data.index.name = "Hour"

    return inflow_data, hourly_inflow_data, totals


def load_phs_storage_profile_data(filepath: str, sheetname_phs_storage_profiles: str, *, mins: int):
    weekly = pd.read_excel(filepath, sheet_name=sheetname_phs_storage_profiles)
    weekly = weekly.set_index("Week")

    points_per_week = int(168 * 60 / mins)
    total_weeks = len(weekly)

    original_points = np.arange(total_weeks) * points_per_week + 1
    total_points = int(original_points[-1])
    interpolated_index = np.arange(1, total_points + 1)

    interpolated: Dict[str, np.ndarray] = {}
    for control_area in weekly.columns:
        weekly_values = weekly[control_area].values
        interp_func = interp1d(original_points, weekly_values, kind="linear", fill_value="extrapolate")
        interpolated[control_area] = interp_func(interpolated_index)

    interpolated_df = pd.DataFrame(interpolated, index=interpolated_index)
    interpolated_df.index.name = "Timestep"

    return weekly, interpolated_df


# -----------------------------------------------------------------------------
# RENEWABLES + PROFILES
# -----------------------------------------------------------------------------

def load_renewable_power_plant_data(filepath: str, sheetname_renewable1: str, sheetname_renewable2: str):
    df_data = pd.read_excel(filepath, sheet_name=sheetname_renewable1)
    df_spec = pd.read_excel(filepath, sheet_name=sheetname_renewable2)

    df_spec["Power Plant Type"] = df_spec["Power Plant Type"].astype(str).str.strip()
    df_spec["Node"] = df_spec["Node"].astype(str).str.strip()
    if "Control Area" in df_spec.columns:
        df_spec["Control Area"] = df_spec["Control Area"].astype(str).str.strip()

    data_by_type = df_data.set_index("Power Plant Type")
    spec = df_spec.merge(data_by_type, on="Power Plant Type", how="left")

    node_indices = {node: spec[spec["Node"] == node].index.tolist() for node in spec["Node"].unique()}
    return spec, node_indices


def load_res_profile_data(filepath: str, sheetname_res_profile: str, *, plant_type: str):
    profile = pd.read_excel(filepath, sheet_name=sheetname_res_profile)
    profile.set_index(profile.columns[0], inplace=True)

    if plant_type == "RoR":
        for control_area in profile.columns:
            total = profile[control_area].sum()
            profile[control_area] = profile[control_area] / total if total != 0 else 0.0

    return profile


# -----------------------------------------------------------------------------
# FLEXIBILITY
# -----------------------------------------------------------------------------

def load_flexibility_data(filepath: str, sheetname_flexibility1: str, sheetname_flexibility2: str):
    df_data = pd.read_excel(filepath, sheet_name=sheetname_flexibility1)
    df_spec = pd.read_excel(filepath, sheet_name=sheetname_flexibility2)

    df_spec["Power Plant Type"] = df_spec["Power Plant Type"].astype(str).str.strip()
    df_spec["Node"] = df_spec["Node"].astype(str).str.strip()
    if "Control Area" in df_spec.columns:
        df_spec["Control Area"] = df_spec["Control Area"].astype(str).str.strip()

    data_by_type = df_data.set_index("Power Plant Type")
    all_flex = df_spec.merge(data_by_type, on="Power Plant Type", how="left")

    dsr_spec = all_flex[all_flex["Power Plant Type"] == "DSR"].reset_index(drop=True)
    bat_spec = all_flex[all_flex["Power Plant Type"] == "Battery"].reset_index(drop=True)

    dsr_node_idx = {node: dsr_spec[dsr_spec["Node"] == node].index.tolist() for node in dsr_spec["Node"].unique()}
    bat_node_idx = {node: bat_spec[bat_spec["Node"] == node].index.tolist() for node in bat_spec["Node"].unique()}

    return dsr_spec, bat_spec, dsr_node_idx, bat_node_idx


# -----------------------------------------------------------------------------
# EXCHANGE + PTDF (NEW CSV IMPORT)
# -----------------------------------------------------------------------------

def load_ptdf_from_csv(
    ptdf_csv_path: str,
    *,
    connections: List[str] | None = None,
    nodes: List[str] | None = None,
    slack_node: str | None = None,
    verbose: bool = True,
) -> Dict:
    """
    Loads the NEW synchronized PTDF CSV and aligns it to the Exchange_Data ordering.

    Expected NEW CSV layout (from build_data_updated):
      - rows: line/branch ids (connections)
      - columns: node ids (INCLUDES slack column, slack column values should be ~0)

    Returns dict with:
      - PTDF: DataFrame index=connections, columns=non_slack_nodes
      - slack_node, nodes, connections, non_slack_nodes, source
    """
    if not os.path.exists(ptdf_csv_path):
        raise FileNotFoundError(f"[PTDF] CSV not found: {ptdf_csv_path}")

    ptdf = pd.read_csv(ptdf_csv_path, index_col=0)
    ptdf.index = ptdf.index.astype(str).str.strip()
    ptdf.columns = ptdf.columns.astype(str).str.strip()

    # ---- align to connections (rows) ----
    if connections is not None:
        connections = [str(x).strip() for x in connections]
        missing_lines = [c for c in connections if c not in ptdf.index]
        extra_lines = [i for i in ptdf.index if i not in connections]

        if missing_lines:
            raise ValueError(
                f"[PTDF] PTDF_Synchronized.csv is missing {len(missing_lines)} connection rows from Exchange_Data. "
                f"Examples: {missing_lines[:20]}"
            )

        ptdf = ptdf.loc[connections]
        if verbose and extra_lines:
            print(f"[PTDF] Note: CSV contains {len(extra_lines)} extra line-rows not in Exchange_Data (ignored).")

    # ---- align to nodes (columns) and drop slack ----
    if nodes is not None:
        nodes = [str(x).strip() for x in nodes]

        if slack_node is None:
            slack_node = nodes[0]
        slack_node = str(slack_node).strip()

        if slack_node not in nodes:
            raise ValueError(f"[PTDF] slack_node '{slack_node}' not in Exchange_Data nodes.")

        # For downstream you want non-slack nodes in THIS exact order
        expected_non_slack = [n for n in nodes if n != slack_node]

        # NEW CSV should include slack; if it doesn't, we still proceed if non-slack exists
        if slack_node in ptdf.columns:
            # optional sanity: slack column should be near zero
            slack_maxabs = float(np.abs(pd.to_numeric(ptdf[slack_node], errors="coerce")).max())
            if verbose:
                print(f"[PTDF] slack column present. max|PTDF[:,slack]| = {slack_maxabs:.3e} (should be ~0)")
            ptdf = ptdf.drop(columns=[slack_node])

        missing_cols = [n for n in expected_non_slack if n not in ptdf.columns]
        if missing_cols:
            raise ValueError(
                f"[PTDF] PTDF_Synchronized.csv is missing {len(missing_cols)} node-columns (non-slack) from Exchange_Data. "
                f"Examples: {missing_cols[:20]}"
            )

        ptdf = ptdf.loc[:, expected_non_slack]
        non_slack_nodes = expected_non_slack
    else:
        if slack_node is not None and slack_node in ptdf.columns:
            ptdf = ptdf.drop(columns=[slack_node])
        non_slack_nodes = list(ptdf.columns)

    # numeric coercion
    ptdf = ptdf.apply(pd.to_numeric, errors="coerce")
    if ptdf.isna().any().any():
        bad = ptdf.columns[ptdf.isna().any()].tolist()[:20]
        raise ValueError(f"[PTDF] Non-numeric/NaN values detected in PTDF CSV. Example bad columns: {bad}")

    return {
        "PTDF": ptdf,
        "slack_node": slack_node,
        "nodes": nodes,
        "connections": connections,
        "non_slack_nodes": non_slack_nodes,
        "source": ptdf_csv_path,
    }


def load_exchange_data(
    filepath: str,
    sheetname_exchange: str,
    *,
    ptdf_csv_path: str,
    slack_node: str | None = None,
    verbose: bool = True,
):
    """
    Loads Exchange_Data sheet (incidence, ntc, xborder, conductance) and loads PTDF from NEW CSV.

    Output:
      nodes, connections, incidence_matrix, ntc_pos, ntc_neg, xborder, conductance_series, ptdf_results
    """
    df = pd.read_excel(filepath, sheet_name=sheetname_exchange)

    connections = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    nodes = df.columns[5:].astype(str).str.strip().tolist()

    incidence_matrix = df.iloc[:, 5:].transpose()
    incidence_matrix.columns = connections
    incidence_matrix.index = nodes
    incidence_matrix = incidence_matrix.fillna(0)

    ntc_pos = pd.Series(df.iloc[:, 1].values, index=connections).to_dict()
    ntc_neg = pd.Series(df.iloc[:, 2].values, index=connections).to_dict()
    xborder = pd.Series(df.iloc[:, 4].values, index=connections).to_dict()
    conductance_series = pd.Series(df.iloc[:, 3].values, index=connections).astype(float)

    if slack_node is None:
        slack_node = nodes[0]
    slack_node = str(slack_node).strip()
    if slack_node not in nodes:
        raise ValueError(f"Slack node '{slack_node}' not found in nodes list.")

    ptdf_results = load_ptdf_from_csv(
        ptdf_csv_path,
        connections=connections,
        nodes=nodes,
        slack_node=slack_node,
        verbose=verbose,
    )

    if verbose:
        print(f"[PTDF] Loaded PTDF from CSV: {ptdf_results.get('source')}")
        print(f"Slack node: {slack_node}")
        print("n_nodes:", len(nodes), "| n_lines:", len(connections))
        print("PTDF shape (connections x non-slack nodes):", ptdf_results["PTDF"].shape)

    return (
        nodes,
        connections,
        incidence_matrix,
        ntc_pos,
        ntc_neg,
        xborder,
        conductance_series,
        ptdf_results,
    )


# =============================================================================
# EXAMPLE RUNS (OPTIONAL)
# =============================================================================

if __name__ == "__main__":

    weekly_storage_df, interpolated_storage_df = load_phs_storage_profile_data(
        FILEPATH_XLSX,
        sheetname_phs_storage_profiles="PHS_Storage_Profiles",
        mins=60,
    )

    print_storage_profile_quick_check(
        weekly_storage_df,
        interpolated_storage_df
    )
    
    inflow_weekly, inflow_hourly, totals = load_phs_inflow_data(
        FILEPATH_XLSX,
        sheetname_phs_inflow="PHS_Inflow_Profiles"
    )
    print_inflow_quick_check(
        inflow_weekly,
        inflow_hourly,
        totals,
        label="PHS INFLOW"
    )