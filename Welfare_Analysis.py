# welfare_analysis.py
from __future__ import annotations

import os
import re
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# =============================================================================
# IMPORT FROM MAIN (single source of truth)
# =============================================================================

from Main import CFG, INSTRUMENTS

from Parameters_Updated import (  # keep consistent with Main_New
    load_general_data,
    load_demand_data,
    load_thermal_power_plant_data,
    load_phs_power_plant_data,
    load_renewable_power_plant_data,
    load_flexibility_data,
    load_exchange_data,
)

# =============================================================================
# Paths (ALL PATHS SPECIFIED HERE)
# =============================================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_XLSX = os.path.join(BASE_DIR, "Data_Updated.xlsx")

# --- System results (Stage 1) ---
BASE_SYSTEM_RESULTS_PATH = os.path.join(BASE_DIR, "full_model_results_stage1_base_test.csv")
def stage1_instr_system_results_path(instrument: str) -> str:
    return os.path.join(
        BASE_DIR,
        f"full_model_results_stage1_instrument__{instrument_slug(instrument)}_test.csv",
    )

# --- Stage-1 dual price dumps (your test files) ---
STAGE1_BASE_PRICES_PATH = os.path.join(BASE_DIR, "full_stage1_base_dual_prices_test.csv")
def stage1_instr_prices_path(instrument: str) -> str:
    return os.path.join(
        BASE_DIR,
        f"full_stage1_dual_prices_instrument__{instrument_slug(instrument)}_test.csv",
    )

# --- Slug logic (copied from main convention) ---
def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# =============================================================================
# IO helpers
# =============================================================================

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def to_float(x) -> float:
    if isinstance(x, pd.DataFrame):
        return float(np.asarray(x.values, dtype="object").reshape(-1)[0])
    if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
        return float(np.asarray(x, dtype="object").reshape(-1)[0])
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


# =============================================================================
# Demand extraction (robust to tuple formats)
# =============================================================================

def extract_demand_profiles(demand_data_obj) -> pd.DataFrame:
    """
    Parameters_Updated.load_demand_data(...) may return a tuple with multiple items.
    We only need the demand profiles matrix:
      index: global hour t_global (int)
      columns: nodes (str)
    """
    if isinstance(demand_data_obj, pd.DataFrame):
        return demand_data_obj

    if isinstance(demand_data_obj, (tuple, list)):
        for item in demand_data_obj:
            if isinstance(item, pd.DataFrame):
                return item

    raise TypeError("Could not extract demand profiles DataFrame from demand_data.")


# =============================================================================
# System export parsing (dump_system_window format)
# =============================================================================

def reconstruct_time_from_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main_New.dump_system_window exports:
      window, component, value, index_0, index_1, ...

    We reconstruct:
      start_week = 1 + window*weeks_per_step
      t_local    = last index_* column (assumed timestep)
      t_global   = (start_week-1)*hours_per_week + t_local
    """
    out = df.copy()

    if "window" not in out.columns:
        out["window"] = 0
    out["window"] = pd.to_numeric(out["window"], errors="coerce").fillna(0).astype(int)

    weeks_per_step = int(CFG.horizon.weeks_per_step)
    hours_per_week = int(CFG.horizon.hours_per_week)

    out["start_week"] = 1 + out["window"] * weeks_per_step

    idx_cols = sorted(
        [c for c in out.columns if c.startswith("index_")],
        key=lambda s: int(s.split("_")[1]),
    )
    if not idx_cols:
        out["t_local"] = np.nan
        out["t_global"] = np.nan
        return out

    last_idx = idx_cols[-1]
    out["t_local"] = pd.to_numeric(out[last_idx], errors="coerce")
    out = out[out["t_local"].notna()].copy()
    out = out[out["t_local"] != 0].copy()

    out["t_global"] = (out["start_week"] - 1) * hours_per_week + out["t_local"]
    out["t_global"] = pd.to_numeric(out["t_global"], errors="coerce").astype(int)

    return out


# =============================================================================
# Prices (Stage-1) helpers
# =============================================================================

def load_prices(path: str) -> pd.DataFrame:
    """
    Expected columns: node, start_week, t, price
    where t is global hour. We'll provide (node, t_global) for merges.
    """
    df = read_csv(path).copy()

    missing = sorted({"node", "t", "price"} - set(df.columns))
    if missing:
        raise ValueError(f"Prices CSV missing columns {missing}: {path}")

    df["node"] = df["node"].astype(str)
    df["t_global"] = pd.to_numeric(df["t"], errors="coerce").astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # optional: if all negative, flip (keeps consistent with your stage0 processing)
    if df["price"].notna().any():
        s = df["price"].dropna()
        if not s.empty and (s <= 0).all():
            df["price"] = -df["price"]

    return df[["node", "t_global", "price"]]


# =============================================================================
# Welfare components (SYSTEM ONLY; no follower welfare)
# =============================================================================

def calculate_consumer_surplus(
    *,
    prices: pd.DataFrame,
    demand_profiles: pd.DataFrame,
    voll: float,
) -> float:
    """
    CS = sum_{n,t} (VOLL - price_{n,t}) * demand_{n,t}
    demand_profiles: index=t_global, columns=node
    prices: columns [node, t_global, price]
    """
    dem = demand_profiles.copy()
    dem.index = pd.to_numeric(dem.index, errors="coerce")
    dem = dem[dem.index.notna()].copy()
    dem.index = dem.index.astype(int)
    dem.columns = dem.columns.astype(str)

    dem_long = dem.stack().rename("demand").reset_index()
    dem_long.columns = ["t_global", "node", "demand"]
    dem_long["node"] = dem_long["node"].astype(str)
    dem_long["demand"] = pd.to_numeric(dem_long["demand"], errors="coerce").fillna(0.0)

    m = dem_long.merge(prices, on=["node", "t_global"], how="left")
    m["price"] = pd.to_numeric(m["price"], errors="coerce").fillna(0.0)

    return float(((voll - m["price"]) * m["demand"]).sum())


def _idxdict_to_df(indices_dict: Dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(
        [(str(plant), str(node)) for node, plants in indices_dict.items() for plant in plants],
        columns=["plant", "node"],
    )


def _cost_series(df_sys: pd.DataFrame, var_cost: str) -> pd.Series:
    """
    Reads plant cost entries from system dump.
    These cost components are typically time-invariant or repeated.
    We take last occurrence per plant (sorted by window if present).
    """
    sub = df_sys[df_sys["component"] == var_cost].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    sub["plant"] = sub["index_0"].astype(str)
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")

    if "window" in sub.columns:
        sub = sub.sort_values(["plant", "window"]).drop_duplicates(subset=["plant"], keep="last")
    else:
        sub = sub.drop_duplicates(subset=["plant"], keep="last")

    return pd.to_numeric(sub.set_index("plant")["value"], errors="coerce")


def _build_prod_block(
    df_sys: pd.DataFrame,
    *,
    var_prod: str,
    var_cost: str,
    indices_dict: Dict[str, list],
) -> pd.DataFrame:
    """
    Returns columns: node, t_global, qty, cost
    """
    map_df = _idxdict_to_df(indices_dict)

    prod = df_sys[df_sys["component"] == var_prod].copy()
    if prod.empty:
        return pd.DataFrame(columns=["node", "t_global", "qty", "cost"])

    prod = reconstruct_time_from_window(prod)

    prod["plant"] = prod["index_0"].astype(str)
    prod["qty"] = pd.to_numeric(prod["value"], errors="coerce").fillna(0.0)

    prod = prod.merge(map_df, on="plant", how="inner")
    costs = _cost_series(df_sys, var_cost)
    prod["cost"] = prod["plant"].map(costs).fillna(0.0)

    return prod[["node", "t_global", "qty", "cost"]]


def calculate_producer_surplus(
    *,
    df_sys: pd.DataFrame,
    prices: pd.DataFrame,
    thermal_indices: Dict[str, list],
    phs_indices: Dict[str, list],
    res_indices: Dict[str, list],
) -> float:
    """
    PS = sum (price - marginal_cost) * generation
       + battery arbitrage (node-aggregated) if available
       - pumping * price
    """
    # generation blocks
    th = _build_prod_block(df_sys, var_prod="thermal_generation", var_cost="thermal_generation_cost", indices_dict=thermal_indices)
    phs = _build_prod_block(df_sys, var_prod="phs_turbine_generation", var_cost="phs_generation_cost", indices_dict=phs_indices)
    res = _build_prod_block(df_sys, var_prod="renewable_generation", var_cost="renewable_generation_cost", indices_dict=res_indices)

    prod = pd.concat([th, phs, res], ignore_index=True)
    if prod.empty:
        ps_core = 0.0
    else:
        m = prod.merge(prices, on=["node", "t_global"], how="left")
        m["price"] = pd.to_numeric(m["price"], errors="coerce").fillna(0.0)
        ps_core = float(((m["price"] - m["cost"]) * m["qty"]).sum())

    # battery arbitrage (node-aggregated)
    batt_out = df_sys[df_sys["component"] == "total_battery_out_per_node"].copy()
    batt_in = df_sys[df_sys["component"] == "total_battery_in_per_node"].copy()
    if batt_out.empty and batt_in.empty:
        ps_batt = 0.0
    else:
        batt = pd.concat([batt_out, batt_in], ignore_index=True)
        batt = reconstruct_time_from_window(batt)
        batt["node"] = batt["index_0"].astype(str)
        batt["val"] = pd.to_numeric(batt["value"], errors="coerce").fillna(0.0)
        sign = {"total_battery_out_per_node": 1.0, "total_battery_in_per_node": -1.0}
        batt["signed"] = batt["val"] * batt["component"].map(sign).fillna(0.0)

        net = batt.groupby(["node", "t_global"], as_index=False)["signed"].sum().rename(columns={"signed": "net_battery"})
        net = net.merge(prices, on=["node", "t_global"], how="left")
        net["price"] = pd.to_numeric(net["price"], errors="coerce").fillna(0.0)
        ps_batt = float((net["net_battery"] * net["price"]).sum())

    # pumping cost valued at nodal price
    pump = df_sys[df_sys["component"] == "phs_pump_consumption"].copy()
    if pump.empty:
        ps_pump = 0.0
    else:
        pump = reconstruct_time_from_window(pump)
        pump_map = _idxdict_to_df(phs_indices)
        pump["plant"] = pump["index_0"].astype(str)
        pump["qty"] = pd.to_numeric(pump["value"], errors="coerce").fillna(0.0)
        pump = pump.merge(pump_map, on="plant", how="inner")
        pump = pump.merge(prices, on=["node", "t_global"], how="left")
        pump["price"] = pd.to_numeric(pump["price"], errors="coerce").fillna(0.0)
        ps_pump = float(-(pump["qty"] * pump["price"]).sum())

    return float(ps_core + ps_batt + ps_pump)


def calculate_congestion_rent(
    *,
    df_sys: pd.DataFrame,
    prices: pd.DataFrame,
    incidence_matrix: pd.DataFrame,
) -> float:
    """
    CR = sum_{l,t} |flow_{l,t}| * |price_{n1,t} - price_{n2,t}|
    where (n1,n2) from incidence matrix (only connections with exactly 2 incident nodes).
    """
    inc = incidence_matrix.copy()

    nz = inc.where(inc != 0).stack()
    nodes_by_conn = nz.groupby(level=1).apply(lambda s: list(s.index.get_level_values(0)))

    conn2nodes = {}
    for conn, nodes in nodes_by_conn.items():
        if len(nodes) != 2:
            continue
        conn2nodes[str(conn)] = (str(nodes[0]), str(nodes[1]))

    mapping = (
        pd.DataFrame.from_dict(conn2nodes, orient="index", columns=["node_1", "node_2"])
        .reset_index()
        .rename(columns={"index": "connection"})
    )

    flow = df_sys[df_sys["component"] == "flow"].copy()
    if flow.empty:
        return 0.0

    flow = reconstruct_time_from_window(flow)
    flow["connection"] = flow["index_0"].astype(str)
    flow["flow"] = pd.to_numeric(flow["value"], errors="coerce").fillna(0.0)

    f = flow.merge(mapping, on="connection", how="inner")

    p1 = prices.rename(columns={"node": "node_1", "price": "price_1"})
    p2 = prices.rename(columns={"node": "node_2", "price": "price_2"})

    f = f.merge(p1, on=["node_1", "t_global"], how="left")
    f = f.merge(p2, on=["node_2", "t_global"], how="left")

    f["price_1"] = pd.to_numeric(f["price_1"], errors="coerce").fillna(0.0)
    f["price_2"] = pd.to_numeric(f["price_2"], errors="coerce").fillna(0.0)

    return float((f["flow"].abs() * (f["price_1"] - f["price_2"]).abs()).sum())


def calculate_extra_cost(
    *,
    df_sys: pd.DataFrame,
    thermal_indices: Dict[str, list],
    phs_indices: Dict[str, list],
    dsr_indices: Dict[str, list],
    voll: float,
) -> float:
    """
    EC = start_thermal_generation*start_cost + phs_spill*spill_cost + dsr_down*dsr_down_cost + nse*VOLL
    (matches your old script logic)
    """
    def build_qty_cost_block(var_qty: str, var_cost: str, indices_dict: Dict[str, list]) -> pd.DataFrame:
        map_df = _idxdict_to_df(indices_dict)
        qty = df_sys[df_sys["component"] == var_qty].copy()
        if qty.empty:
            return pd.DataFrame(columns=["qty", "cost"])
        qty["plant"] = qty["index_0"].astype(str)
        qty["qty"] = pd.to_numeric(qty["value"], errors="coerce").fillna(0.0)
        qty = qty.merge(map_df, on="plant", how="inner")
        costs = _cost_series(df_sys, var_cost)
        qty["cost"] = qty["plant"].map(costs).fillna(0.0)
        return qty[["qty", "cost"]]

    th_start = build_qty_cost_block("start_thermal_generation", "start_thermal_generation_cost", thermal_indices)
    phs_spill = build_qty_cost_block("phs_spill", "phs_spill_cost", phs_indices)
    dsr_down = build_qty_cost_block("dsr_down", "dsr_down_cost", dsr_indices)

    cost_th_start = float((th_start["qty"] * th_start["cost"]).sum()) if not th_start.empty else 0.0
    cost_phs_spill = float((phs_spill["qty"] * phs_spill["cost"]).sum()) if not phs_spill.empty else 0.0
    cost_dsr_down = float((dsr_down["qty"] * dsr_down["cost"]).sum()) if not dsr_down.empty else 0.0

    nse = df_sys[df_sys["component"] == "nse"].copy()
    if nse.empty:
        cost_nse = 0.0
    else:
        nse["qty"] = pd.to_numeric(nse["value"], errors="coerce").fillna(0.0)
        cost_nse = float((nse["qty"] * float(voll)).sum())

    # curtailment penalty valued at VOLL (as in model objective)
    curt = df_sys[df_sys["component"] == "curtailment"].copy()
    if curt.empty:
        cost_curt = 0.0
    else:
        curt["qty"] = pd.to_numeric(curt["value"], errors="coerce").fillna(0.0)
        cost_curt = float((curt["qty"] * float(voll)).sum())

    return float(cost_th_start + cost_phs_spill + cost_dsr_down + cost_nse + cost_curt)



def calculate_total_welfare(
    *,
    df_sys: pd.DataFrame,
    prices: pd.DataFrame,
    demand_profiles: pd.DataFrame,
    voll: float,
    thermal_indices: Dict[str, list],
    phs_indices: Dict[str, list],
    res_indices: Dict[str, list],
    dsr_indices: Dict[str, list],
    incidence_matrix: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    cs = calculate_consumer_surplus(prices=prices, demand_profiles=demand_profiles, voll=voll)
    ps = calculate_producer_surplus(
        df_sys=df_sys,
        prices=prices,
        thermal_indices=thermal_indices,
        phs_indices=phs_indices,
        res_indices=res_indices,
    )
    cr = calculate_congestion_rent(df_sys=df_sys, prices=prices, incidence_matrix=incidence_matrix)
    ec = calculate_extra_cost(df_sys=df_sys, thermal_indices=thermal_indices, phs_indices=phs_indices, dsr_indices=dsr_indices, voll=voll)

    total = float(cs + ps + cr - ec)
    return total, {
        "consumer_surplus": float(cs),
        "producer_surplus": float(ps),
        "congestion_rent": float(cr),
        "extra_cost": float(ec),
        "total_welfare": float(total),
    }


# =============================================================================
# Main (no exporting of welfare results; prints only)
# =============================================================================

def main() -> None:
    print("\n==============================")
    print(" WELFARE ANALYSIS (SYSTEM ONLY)")
    print("==============================\n")

    # -------------------------
    # Load parameter data (for mappings + VOLL)
    # -------------------------
    co2_price, voll_raw, wind_pv_cost, phs_cost, dsr_cost, flow_cost = load_general_data(DATA_XLSX, "General_Data")
    voll = to_float(voll_raw)

    demand_data_obj = load_demand_data(DATA_XLSX, "Demand_Profiles")
    demand_profiles = extract_demand_profiles(demand_data_obj)

    thermal_specific, thermal_node_idx = load_thermal_power_plant_data(
        DATA_XLSX, "Thermal_Power_Data", "Thermal_Power_Specific_Data"
    )
    phs_specific, _, phs_node_idx = load_phs_power_plant_data(
        DATA_XLSX, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data"
    )
    renewable_specific, renewable_node_idx = load_renewable_power_plant_data(
        DATA_XLSX, "RES_Power_Data", "RES_Power_Specific_Data"
    )
    dsr_specific, battery_specific, dsr_node_idx, battery_node_idx = load_flexibility_data(
        DATA_XLSX, "Flexibility_Data", "Flexibility_Specific_Data"
    )

    base_dir = os.path.dirname(DATA_XLSX)
    ptdf_csv_path = os.path.join(base_dir, "PTDF_Synchronized.csv")

    _, connections, incidence_matrix, capacity_pos, capacity_neg, xborder, conductance_series, ptdf_results = load_exchange_data(
        DATA_XLSX,
        "Exchange_Data",
        ptdf_csv_path=ptdf_csv_path,
        slack_node=None,
        verbose=False,
    )

    # -------------------------
    # BASE scenario
    # -------------------------
    print("Loading BASE system results...")
    sys_base = read_csv(BASE_SYSTEM_RESULTS_PATH)
    prices_base = load_prices(STAGE1_BASE_PRICES_PATH)

    tw_base, parts_base = calculate_total_welfare(
        df_sys=sys_base,
        prices=prices_base,
        demand_profiles=demand_profiles,
        voll=voll,
        thermal_indices=thermal_node_idx,
        phs_indices=phs_node_idx,
        res_indices=renewable_node_idx,
        dsr_indices=dsr_node_idx,
        incidence_matrix=incidence_matrix,
    )

    print("\n===============================")
    print(" BASE WELFARE (SYSTEM ONLY) ")
    print("===============================")
    for k in ["consumer_surplus", "producer_surplus", "congestion_rent", "extra_cost", "total_welfare"]:
        print(f"{k:>18}: {parts_base.get(k, 0.0):,.2f} €")

    # -------------------------
    # Instruments + components
    # -------------------------
    print("\n===================================")
    print(" INSTRUMENT COMPARISON vs BASE (with components)")
    print("===================================\n")

    # store base parts for deltas
    base_parts = parts_base.copy()

    name_width = 22
    num_width = 18

    cols = [
        ("Total", "total_welfare"),
        ("CS", "consumer_surplus"),
        ("PS", "producer_surplus"),
        ("CR", "congestion_rent"),
        ("EC", "extra_cost"),
    ]

    header = f"{'Instrument':<{name_width}}"
    for label, _ in cols:
        header += f"{label + ' [€]':>{num_width}}"
    for label, _ in cols:
        header += f"{('Δ' + label + ' [€]'):>{num_width}}"

    line = "-" * len(header)
    print(header)
    print(line)

    # helper to print one row
    def print_row(name: str, parts: dict, base_parts: dict):
        row = f"{name:<{name_width}}"
        for _, key in cols:
            row += f"{parts.get(key, 0.0):>{num_width},.2f}"
        for _, key in cols:
            row += f"{(parts.get(key, 0.0) - base_parts.get(key, 0.0)):>{num_width},.2f}"
        print(row)

    # BASE row (deltas = 0)
    print_row("BASE", parts_base, parts_base)

    # instrument rows
    for instr in INSTRUMENTS:
        sys_instr = read_csv(stage1_instr_system_results_path(instr))
        prices_instr = load_prices(stage1_instr_prices_path(instr))

        tw_i, parts_i = calculate_total_welfare(
            df_sys=sys_instr,
            prices=prices_instr,
            demand_profiles=demand_profiles,
            voll=voll,
            thermal_indices=thermal_node_idx,
            phs_indices=phs_node_idx,
            res_indices=renewable_node_idx,
            dsr_indices=dsr_node_idx,
            incidence_matrix=incidence_matrix,
        )

        print_row(instr, parts_i, base_parts)

    print(line)
    print("\nDone.\n")

if __name__ == "__main__":
    main()
