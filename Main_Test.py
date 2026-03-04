# =============================================================================
# GENERAL OVERVIEW
# =============================================================================
# Rolling-horizon optimization pipeline (Stage 0 -> Stage 1 with follower coupling)
#
# - Stage 0 runs ONCE (system only): extract prices and PV
# - Stage 1 BASE runs ONCE (follower BASE + system BASE)
# - Stage 1 INSTRUMENT runs for EACH instrument (RTP / Peak-Shaving / Capacity-Tariff),
#   writing results into instrument-specific subfolders & files.
#
# CHANGE (requested):
# - Stage 1 BASE, Peak-Shaving, Capacity-Tariff use ONE FIXED PRICE PER CONTROL AREA,
#   identical for all nodes within that control area (based on node_to_control_area mapping).
# - If any control-area price cannot be computed (n/a), RAISE an error (no fallback).
# - RTP remains nodal time-varying (as before).
# =============================================================================

from __future__ import annotations

import os
import re
import inspect
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from pyomo.environ import Var, value

# =============================================================================
# Model + data imports
# =============================================================================

from Model_New import create_model as create_model_system
from Model_New import solve_model as solve_model_system
from Model_New import build_follower_model

from Parameters_Updated import (
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

# =============================================================================
# Configuration
# =============================================================================

FOLLOWER_FLOW_KEYS = [
    "pv_feed_to_system",
    "battery_to_system",
    "imports_to_demand",
    "imports_to_battery",
]
FOLLOWER_FLOW_COLS = ["node", "start_week", "t", *FOLLOWER_FLOW_KEYS]

# Which instruments to compute (Stage 1 instrument for each):
INSTRUMENTS = ["RTP", "Peak-Shaving", "Capacity-Tariff"]


@dataclass(frozen=True)
class Horizon:
    mins_per_hour: int = 60
    hours_per_week: int = 168
    num_weeks: int = 1  # 52
    weeks_per_step: int = 1  # 2

    @property
    def T(self) -> int:
        return int(self.hours_per_week * self.weeks_per_step)

    @property
    def num_windows(self) -> int:
        return int(np.ceil(self.num_weeks / self.weeks_per_step))

    def start_week_of_window(self, w: int) -> int:
        return int(1 + w * self.weeks_per_step)

    def global_hour_offset(self, start_week: int) -> int:
        return int((start_week - 1) * self.hours_per_week)


@dataclass(frozen=True)
class Shares:
    pv_share: float
    battery_share: float
    demand_fraction: float


@dataclass(frozen=True)
class Fees:
    fee_pv_feed: float = 0.0
    fee_battery_in: float = 100.0
    fee_battery_out: float = 0.0
    fee_imports_to_demand: float = 100.0

    def as_kwargs(self) -> dict:
        return {
            "fee_pv_feed": float(self.fee_pv_feed),
            "fee_battery_in": float(self.fee_battery_in),
            "fee_battery_out": float(self.fee_battery_out),
            "fee_imports_to_demand": float(self.fee_imports_to_demand),
        }


@dataclass(frozen=True)
class Outputs:
    weekly_stage0_dir: str = "weekly_results/stage0_system"
    weekly_stage1_base_dir: str = "weekly_results/stage1_base"
    weekly_follower_base_dir: str = "weekly_results/follower_base"

    # Instrument outputs will be stored under:
    # weekly_results/stage1_instrument/<instrument_slug>/
    weekly_stage1_instr_root: str = "weekly_results/stage1_instrument"
    weekly_follower_instr_root: str = "weekly_results/follower_instrument"

    full_stage0_results: str = "full_model_results_stage0.csv"
    full_stage0_prices: str = "full_stage0_prices.csv"
    full_stage0_pv: str = "full_stage0_pv_quantities.csv"

    full_follower_base: str = "full_follower_values_base.csv"
    full_stage1_base: str = "full_model_results_stage1_base.csv"


@dataclass(frozen=True)
class Config:
    base_dir: str = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
    data_file: str = "Data_Updated.xlsx"

    export_share: float = 0.70

    follower_control_area: Union[str, Tuple[str, ...]] = (
        "AT", "BE", "CH", "CZ", "DE", "FR", "HU", "IT", "LU", "NL", "PL", "SI", "SK"
    )

    horizon: Horizon = Horizon()

    # Stage 0 is system-only (no follower coupling)
    stage0: Shares = Shares(pv_share=0.0, battery_share=0.0, demand_fraction=0.0)

    # Stage 1 activates follower
    stage1: Shares = Shares(pv_share=0.9, battery_share=0.9, demand_fraction=0.05)

    fees: Fees = Fees()
    out: Outputs = Outputs()

    # Capacity tariff parameters (used only for Capacity-Tariff instrument)
    peak_rate_eur_per_mw: float = 2000.0  # EUR/MW per billing horizon
    billing_weeks: int = 4  # stylized month


CFG = Config()

# =============================================================================
# Paths + filesystem helpers
# =============================================================================

@dataclass(frozen=True)
class Paths:
    base_dir: str
    data_path: str

    weekly_stage0_dir: str
    weekly_stage1_base_dir: str
    weekly_follower_base_dir: str

    weekly_stage1_instr_root: str
    weekly_follower_instr_root: str

    full_stage0_results: str
    full_stage0_prices: str
    full_stage0_pv: str

    full_follower_base: str
    full_stage1_base: str


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def clean_csv_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        return
    for f in os.listdir(dir_path):
        if f.lower().endswith(".csv"):
            try:
                os.remove(os.path.join(dir_path, f))
            except Exception as e:
                print(f"⚠️ Could not remove {os.path.join(dir_path, f)}: {e}")


def concat_weekly_csvs(results_dir: str, out_path: str) -> pd.DataFrame:
    files = sorted([f for f in os.listdir(results_dir) if f.lower().endswith(".csv")]) if os.path.isdir(results_dir) else []
    if not files:
        print(f"⚠️ No weekly CSVs found in: {results_dir}")
        return pd.DataFrame()

    df_all = pd.concat(
        [pd.read_csv(os.path.join(results_dir, f), low_memory=False) for f in files],
        ignore_index=True,
    )
    df_all.to_csv(out_path, index=False)
    print(f"✅ Full results saved to: {out_path}")
    return df_all


def build_paths(cfg: Config) -> Paths:
    base_dir = cfg.base_dir
    data_path = os.path.join(base_dir, cfg.data_file)

    out = cfg.out

    weekly_stage0_dir = os.path.join(base_dir, out.weekly_stage0_dir)
    weekly_stage1_base_dir = os.path.join(base_dir, out.weekly_stage1_base_dir)
    weekly_follower_base_dir = os.path.join(base_dir, out.weekly_follower_base_dir)

    weekly_stage1_instr_root = os.path.join(base_dir, out.weekly_stage1_instr_root)
    weekly_follower_instr_root = os.path.join(base_dir, out.weekly_follower_instr_root)

    ensure_dirs(
        weekly_stage0_dir,
        weekly_stage1_base_dir,
        weekly_follower_base_dir,
        weekly_stage1_instr_root,
        weekly_follower_instr_root,
    )

    return Paths(
        base_dir=base_dir,
        data_path=data_path,
        weekly_stage0_dir=weekly_stage0_dir,
        weekly_stage1_base_dir=weekly_stage1_base_dir,
        weekly_follower_base_dir=weekly_follower_base_dir,
        weekly_stage1_instr_root=weekly_stage1_instr_root,
        weekly_follower_instr_root=weekly_follower_instr_root,
        full_stage0_results=os.path.join(base_dir, out.full_stage0_results),
        full_stage0_prices=os.path.join(base_dir, out.full_stage0_prices),
        full_stage0_pv=os.path.join(base_dir, out.full_stage0_pv),
        full_follower_base=os.path.join(base_dir, out.full_follower_base),
        full_stage1_base=os.path.join(base_dir, out.full_stage1_base),
    )


def prepare_output_dirs(paths: Paths) -> None:
    # Clean only the shared dirs (Stage0 and Stage1 base). Instrument dirs are per-instrument and will be cleaned per run.
    clean_csv_dir(paths.weekly_stage0_dir)
    clean_csv_dir(paths.weekly_stage1_base_dir)
    clean_csv_dir(paths.weekly_follower_base_dir)


def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def instrument_weekly_dirs(paths: Paths, instrument: str) -> Tuple[str, str]:
    slug = instrument_slug(instrument)
    sys_dir = os.path.join(paths.weekly_stage1_instr_root, slug)
    fol_dir = os.path.join(paths.weekly_follower_instr_root, slug)
    ensure_dirs(sys_dir, fol_dir)
    return sys_dir, fol_dir


def instrument_full_paths(paths: Paths, instrument: str) -> Tuple[str, str]:
    slug = instrument_slug(instrument)
    full_follower = os.path.join(paths.base_dir, f"full_follower_values_instrument__{slug}.csv")
    full_system = os.path.join(paths.base_dir, f"full_model_results_stage1_instrument__{slug}.csv")
    return full_follower, full_system

# =============================================================================
# Signature-safe call helper
# =============================================================================

def call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)

# =============================================================================
# Data loading
# =============================================================================

@dataclass(frozen=True)
class Inputs:
    general_data: tuple
    demand_data: tuple
    thermal_data: tuple
    phs_data: tuple
    phs_inflow: tuple
    hs_inflow: tuple
    phs_storage_profile_data: tuple
    renewable_data: tuple
    ror_profile_data: tuple
    windon_profile_data: tuple
    windoff_profile_data: tuple
    pv_profile_data: tuple
    flexibility_data: tuple
    exchange_data: tuple


def load_all_inputs(data_path: str, *, mins_per_hour: int) -> Inputs:
    # Parameters_Updated expects a synchronized PTDF CSV for exchange data
    # Default: look in the same folder as the Excel input file.
    base_dir = os.path.dirname(data_path)
    ptdf_csv_path = os.path.join(base_dir, "PTDF_Synchronized.csv")

    general_data = load_general_data(data_path, "General_Data")

    # UPDATED: demand loader takes only the demand profiles sheet
    demand_data = load_demand_data(data_path, "Demand_Profiles")

    # UPDATED: no RES_Shares argument anymore
    thermal_data = load_thermal_power_plant_data(
        data_path, "Thermal_Power_Data", "Thermal_Power_Specific_Data"
    )

    phs_data = load_phs_power_plant_data(
        data_path, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data"
    )
    phs_inflow = load_phs_inflow_data(data_path, "PHS_Inflow_Profiles")
    hs_inflow = load_phs_inflow_data(data_path, "HS_Inflow_Profiles")
    phs_storage_profile_data = load_phs_storage_profile_data(
        data_path, "PHS_Storage_Profiles", mins=mins_per_hour
    )

    # UPDATED: no RES_Shares argument anymore
    renewable_data = load_renewable_power_plant_data(
        data_path, "RES_Power_Data", "RES_Power_Specific_Data"
    )

    # UPDATED: plant_type is keyword-only
    ror_profile_data = load_res_profile_data(
        data_path, "RoR_Profile", plant_type="RoR"
    )
    windon_profile_data = load_res_profile_data(
        data_path, "WindOn_Profile", plant_type="WindOn"
    )
    windoff_profile_data = load_res_profile_data(
        data_path, "WindOff_Profile", plant_type="WindOff"
    )
    pv_profile_data = load_res_profile_data(
        data_path, "PV_Profile", plant_type="PV"
    )

    # UPDATED: no RES_Shares argument anymore
    flexibility_data = load_flexibility_data(
        data_path, "Flexibility_Data", "Flexibility_Specific_Data"
    )

    # UPDATED: requires ptdf_csv_path keyword-only
    exchange_data = load_exchange_data(
        data_path,
        "Exchange_Data",
        ptdf_csv_path=ptdf_csv_path,
        slack_node=None,
        verbose=False,
    )

    return Inputs(
        general_data=general_data,
        demand_data=demand_data,
        thermal_data=thermal_data,
        phs_data=phs_data,
        phs_inflow=phs_inflow,
        hs_inflow=hs_inflow,
        phs_storage_profile_data=phs_storage_profile_data,
        renewable_data=renewable_data,
        ror_profile_data=ror_profile_data,
        windon_profile_data=windon_profile_data,
        windoff_profile_data=windoff_profile_data,
        pv_profile_data=pv_profile_data,
        flexibility_data=flexibility_data,
        exchange_data=exchange_data,
    )

# =============================================================================
# Weekly dumping
# =============================================================================

def _is_local_t0(model, idx_tuple: Tuple) -> bool:
    if not idx_tuple:
        return False
    tset = getattr(model, "timesteps", None)
    if tset is None:
        return False
    try:
        last = idx_tuple[-1]
        return (last in tset) and (int(last) == 0)
    except Exception:
        return False


def dump_system_window(model, *, window_id: int, results_dir: str, label: str) -> None:
    records = []
    for _, var in model.component_map(Var, active=True).items():
        for index in var:
            idx = index if isinstance(index, tuple) else (index,)
            if _is_local_t0(model, idx):
                continue

            v = value(var[index], exception=False)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = None

            rec = {"window": int(window_id), "component": var.local_name, "value": v}
            for i, x in enumerate(idx):
                rec[f"index_{i}"] = str(x)
            records.append(rec)

    df = pd.DataFrame(records)
    out = os.path.join(results_dir, f"window_{window_id:03d}_system_{label}.csv")
    df.to_csv(out, index=False)
    print(f"✅ Saved system {label} results for window {window_id} to: {out} ({len(df)} records)")


def dump_follower_window(df: pd.DataFrame, *, window_id: int, results_dir: str, label: str) -> None:
    if df is None or df.empty:
        print(f"⚠️ Follower produced no data for window {window_id}.")
        return
    out = os.path.join(results_dir, f"window_{window_id:03d}_follower_{label}.csv")
    df.to_csv(out, index=False)
    print(f"✅ Saved follower {label} values for window {window_id} to: {out} ({len(df)} rows)")

# =============================================================================
# Stage 0 extraction: prices (duals) + total PV
# =============================================================================

def extract_prices_from_model(model, *, start_week: int, horizon: Horizon) -> pd.DataFrame:
    dual_suffix = getattr(model, "dual", None)
    comp = getattr(model, "demand_equilibrium_constraint", None)

    if dual_suffix is None or comp is None:
        print("⚠️ Could not extract prices (missing model.dual or demand_equilibrium_constraint).")
        return pd.DataFrame(columns=["node", "start_week", "t", "price"])

    t_off = horizon.global_hour_offset(start_week)

    rows = []
    for n in model.nodes:
        for t in model.timesteps:
            if int(t) == 0:
                continue

            con = comp[n, t]
            if (not con.active) or (con.body is None):
                continue

            pr = dual_suffix.get(con, None)
            if pr is None or (isinstance(pr, float) and np.isnan(pr)):
                continue

            rows.append(
                {"node": str(n), "start_week": int(start_week), "t": int(t_off + int(t)), "price": float(pr)}
            )

    return pd.DataFrame(rows)


def extract_total_pv_from_model(model, *, start_week: int, horizon: Horizon) -> pd.DataFrame:
    t_off = horizon.global_hour_offset(start_week)

    for name in ["pv_generation", "pv_total_generation", "total_pv_generation"]:
        comp = getattr(model, name, None)
        if comp is not None:
            try:
                rows = []
                for t in model.timesteps:
                    if int(t) == 0:
                        continue
                    rows.append(
                        {"start_week": int(start_week), "t": int(t_off + int(t)), "pv_generation": float(value(comp[t]))}
                    )
                return pd.DataFrame(rows)
            except Exception:
                pass

    ren_gen = getattr(model, "renewable_generation", None)
    ren_set = getattr(model, "renewable_power_plants", None)
    if ren_gen is not None and ren_set is not None:
        is_pv = getattr(model, "renewable_is_pv", None)
        rtype = getattr(model, "renewable_type", None)

        rows = []
        if is_pv is not None:
            for t in model.timesteps:
                if int(t) == 0:
                    continue
                total = 0.0
                for p in ren_set:
                    try:
                        if float(value(is_pv[p])) >= 0.5:
                            total += float(value(ren_gen[p, t]))
                    except Exception:
                        continue
                rows.append({"start_week": int(start_week), "t": int(t_off + int(t)), "pv_generation": total})
            return pd.DataFrame(rows)

        if rtype is not None:
            for t in model.timesteps:
                if int(t) == 0:
                    continue
                total = 0.0
                for p in ren_set:
                    try:
                        if str(value(rtype[p])) == "PV":
                            total += float(value(ren_gen[p, t]))
                    except Exception:
                        continue
                rows.append({"start_week": int(start_week), "t": int(t_off + int(t)), "pv_generation": total})
            return pd.DataFrame(rows)

        for t in model.timesteps:
            if int(t) == 0:
                continue
            total = 0.0
            for p in ren_set:
                try:
                    total += float(value(ren_gen[p, t]))
                except Exception:
                    continue
            rows.append({"start_week": int(start_week), "t": int(t_off + int(t)), "pv_generation": total})
        return pd.DataFrame(rows)

    print("⚠️ Could not extract PV generation. Returning empty.")
    return pd.DataFrame(columns=["start_week", "t", "pv_generation"])

# =============================================================================
# Stage 0 -> Stage 1 base price construction (PV-weighted)
# =============================================================================

def pv_weighted_price_over_horizon(prices_df: pd.DataFrame, pv_df: pd.DataFrame) -> float:
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df is empty.")
    if pv_df is None or pv_df.empty:
        raise ValueError("pv_df is empty.")
    if not {"t", "price"}.issubset(prices_df.columns):
        raise ValueError("prices_df must contain columns ['t','price'].")
    if not {"t", "pv_generation"}.issubset(pv_df.columns):
        raise ValueError("pv_df must contain columns ['t','pv_generation'].")

    df = prices_df[["t", "price"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if (df["price"] <= 0).all():
        df["price"] = -df["price"]
    elif (df["price"] >= 0).all():
        pass
    else:
        bad = df[df["price"] < 0].head()
        raise ValueError(f"Mixed positive/negative prices detected. Examples:\n{bad}")

    pv = pv_df[["t", "pv_generation"]].copy()
    pv["pv_generation"] = pd.to_numeric(pv["pv_generation"], errors="coerce").fillna(0.0).clip(lower=0.0)

    merged = df.merge(pv, on="t", how="inner")
    if merged.empty:
        raise ValueError("No overlap in 't' between prices_df and pv_df.")

    pv_sum = float(merged["pv_generation"].sum())
    if pv_sum <= 0.0:
        raise ValueError("Sum of PV generation is 0; cannot compute PV-weighted price.")

    weights = merged["pv_generation"] / pv_sum
    p_base = float((merged["price"] * weights).sum())

    if not np.isfinite(p_base):
        raise ValueError("PV-weighted price is not finite.")
    return p_base


def make_constant_stage1_price_df(
    nodes: Iterable[str], *, start_week: int, T: int, price: float
) -> pd.DataFrame:
    rows = []
    for n in nodes:
        for t in range(1, T + 1):
            rows.append({"node": str(n), "start_week": int(start_week), "t": int(t), "price": float(price)})
    return pd.DataFrame(rows)


def make_constant_stage1_price_df_by_ca(
    nodes: Iterable[str],
    *,
    start_week: int,
    T: int,
    node_to_control_area: Dict[str, str],
    ca_price_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Constant price per node, determined by the node's control area mapping.

    Raises errors if:
      - node is missing in node_to_control_area
      - node's control area has no entry in ca_price_dict
    """
    rows = []
    for n in nodes:
        n_str = str(n)
        if n_str not in node_to_control_area:
            raise ValueError(f"Missing node_to_control_area mapping for node '{n_str}'.")
        ca = str(node_to_control_area[n_str])
        if ca not in ca_price_dict:
            raise ValueError(f"No base price available for control area '{ca}' (node '{n_str}').")
        p = float(ca_price_dict[ca])
        for t in range(1, T + 1):
            rows.append({"node": n_str, "start_week": int(start_week), "t": int(t), "price": p})
    return pd.DataFrame(rows)


def choose_price_nodes(stage0_prices_df: pd.DataFrame, demand_data) -> List[str]:
    if stage0_prices_df is not None and not stage0_prices_df.empty and "node" in stage0_prices_df.columns:
        nodes = sorted(stage0_prices_df["node"].astype(str).unique().tolist())
        if nodes:
            return nodes
    _, _, _, nodes, _ = demand_data
    return [str(n) for n in nodes]


def make_rtp_price_df_from_stage0_prices(
    stage0_prices_df: pd.DataFrame,
    *,
    start_week: int,
    horizon: Horizon,
    nodes_for_follower: List[str],
) -> pd.DataFrame:
    if stage0_prices_df is None or stage0_prices_df.empty:
        raise ValueError("Stage-0 price DF is empty; cannot build RTP signal.")

    req = {"node", "start_week", "t", "price"}
    missing = req - set(stage0_prices_df.columns)
    if missing:
        raise ValueError(f"stage0_prices_df missing columns: {sorted(missing)}")

    T = horizon.T
    t_off = horizon.global_hour_offset(start_week)
    global_ts = np.arange(t_off + 1, t_off + T + 1)

    df = stage0_prices_df.copy()
    df["node"] = df["node"].astype(str)
    df = df[df["start_week"] == int(start_week)].copy()
    df = df[df["t"].isin(global_ts)].copy()
    df = df[df["node"].isin([str(n) for n in nodes_for_follower])].copy()

    if df.empty:
        raise ValueError(
            f"No Stage-0 prices found for start_week={start_week} on follower nodes "
            f"(sample: {nodes_for_follower[:5]}...)."
        )

    df["t_local"] = df["t"].astype(int) - int(t_off)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    out_parts = []
    full_index = pd.Index(range(1, T + 1), name="t_local")

    for n in nodes_for_follower:
        s = df[df["node"] == str(n)][["t_local", "price"]].drop_duplicates("t_local", keep="last").set_index("t_local")
        s = s.reindex(full_index)
        s["price"] = s["price"].interpolate().ffill().bfill()
        tmp = s.reset_index().rename(columns={"t_local": "t"})
        tmp.insert(0, "node", str(n))
        tmp.insert(1, "start_week", int(start_week))
        out_parts.append(tmp[["node", "start_week", "t", "price"]])

    return pd.concat(out_parts, ignore_index=True)

# =============================================================================
# Initial value helpers
# =============================================================================

def get_initial_values_for_next_week(model) -> dict:
    initial_values = {}
    T = max(model.timesteps)

    for p in getattr(model, "thermal_power_plants", []):
        initial_values[("thermal_generation", p)] = value(model.thermal_generation[p, T])
        initial_values[("on_1", p)] = value(model.on_1[p, T])
        initial_values[("on_2", p)] = value(model.on_2[p, T])
        initial_values[("start_thermal_generation", p)] = value(model.start_thermal_generation[p, T])

    for p in getattr(model, "phs_power_plants", []):
        initial_values[("phs_turbine_generation", p)] = value(model.phs_turbine_generation[p, T])
        initial_values[("phs_pump_consumption", p)] = value(model.phs_pump_consumption[p, T])
        initial_values[("phs_storage", p)] = value(model.phs_storage[p, T])
        initial_values[("phs_spill", p)] = value(model.phs_spill[p, T])

    for p in getattr(model, "renewable_power_plants", []):
        initial_values[("renewable_spill", p)] = value(model.renewable_spill[p, T])
        initial_values[("renewable_generation", p)] = value(model.renewable_generation[p, T])

    for dsr in getattr(model, "dsr_units", []):
        initial_values[("dsr", dsr)] = value(model.dsr_storage[dsr, T])
        initial_values[("dsr_down", dsr)] = value(model.dsr_down[dsr, T])
        initial_values[("dsr_up", dsr)] = value(model.dsr_up[dsr, T])

    for b in getattr(model, "battery_units", []):
        initial_values[("battery_storage", b)] = value(model.battery_storage[b, T])
        initial_values[("battery_out", b)] = value(model.battery_out[b, T])
        initial_values[("battery_in", b)] = value(model.battery_in[b, T])

    for c in getattr(model, "connections", []):
        initial_values[("flow", c)] = value(model.flow[c, T])
    for n in getattr(model, "nodes", []):
        initial_values[("nse", n)] = value(model.nse[n, T])

    return initial_values


def get_initial_values_for_next_week_price_follower(model) -> dict:
    initial_values = {}
    T = max(model.timesteps)
    for n in model.nodes:
        initial_values[("battery_storage", n)] = value(model.battery_storage[n, T])
    return initial_values

# =============================================================================
# Model builders
# =============================================================================

def build_system_model(
    inputs: Inputs,
    *,
    horizon: Horizon,
    shares: Shares,
    start_week: int,
    initial_values: dict,
    instrument: str,
    follower_control_area: str,
    follower_flows_df: Optional[pd.DataFrame] = None,
):
    return call_with_supported_kwargs(
        create_model_system,
        inputs.general_data,
        inputs.demand_data,
        inputs.thermal_data,
        inputs.phs_data,
        inputs.phs_inflow,
        inputs.hs_inflow,
        inputs.phs_storage_profile_data,
        inputs.renewable_data,
        inputs.ror_profile_data,
        inputs.windon_profile_data,
        inputs.windoff_profile_data,
        inputs.pv_profile_data,
        inputs.flexibility_data,
        inputs.exchange_data,
        initial_values,
        horizon.mins_per_hour,
        horizon.hours_per_week,
        int(start_week),
        horizon.weeks_per_step,
        pv_share=shares.pv_share,
        battery_share=shares.battery_share,
        demand_fraction=shares.demand_fraction,
        follower_control_area=follower_control_area,
        instrument=instrument,
        follower_flows_df=follower_flows_df,
    )


def follower_flows_for_system(df_follower: pd.DataFrame, *, start_week: int) -> pd.DataFrame:
    if df_follower is None or df_follower.empty:
        return pd.DataFrame(columns=FOLLOWER_FLOW_COLS)

    missing = sorted(set(FOLLOWER_FLOW_COLS) - set(df_follower.columns))
    if missing:
        raise ValueError(f"Follower results missing required flow columns: {missing}")

    flows = df_follower[FOLLOWER_FLOW_COLS].copy()
    flows["node"] = flows["node"].astype(str)
    flows["start_week"] = pd.to_numeric(flows["start_week"], errors="coerce").astype(int)
    flows["t"] = pd.to_numeric(flows["t"], errors="coerce").astype(int)

    flows = flows[flows["start_week"] == int(start_week)].copy()
    if flows.empty:
        raise ValueError(
            f"Follower results contain no rows for start_week={start_week}. "
            f"Available start_weeks: {sorted(df_follower['start_week'].unique().tolist())[:10]}"
        )

    flows = flows.sort_values(["node", "t"]).drop_duplicates(["node", "t"], keep="last")
    return flows

# =============================================================================
# Stage runners
# =============================================================================

def run_stage0(
    inputs: Inputs,
    *,
    horizon: Horizon,
    shares: Shares,
    follower_control_area: str,
    weekly_out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n==============================")
    print("=== STAGE 0: system model  ===")
    print("==============================\n")

    initial_values = {}
    all_prices: List[pd.DataFrame] = []
    all_pv: List[pd.DataFrame] = []

    for w in range(horizon.num_windows):
        start_week = horizon.start_week_of_window(w)
        print(f"\n🔄 [Stage 0] Window {w+1}/{horizon.num_windows}  (start_week={start_week})\n")

        m0 = build_system_model(
            inputs,
            horizon=horizon,
            shares=shares,
            start_week=start_week,
            initial_values=initial_values,
            instrument="BASE",
            follower_control_area=follower_control_area,
            follower_flows_df=None,
        )
        solve_model_system(m0)

        dump_system_window(m0, window_id=w, results_dir=weekly_out_dir, label="stage0")

        df_prices = extract_prices_from_model(m0, start_week=start_week, horizon=horizon)
        df_pv = extract_total_pv_from_model(m0, start_week=start_week, horizon=horizon)

        if not df_prices.empty:
            all_prices.append(df_prices)
        if not df_pv.empty:
            all_pv.append(df_pv)

        initial_values = get_initial_values_for_next_week(m0)

    prices_df = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame(columns=["node", "start_week", "t", "price"])
    pv_df = pd.concat(all_pv, ignore_index=True) if all_pv else pd.DataFrame(columns=["start_week", "t", "pv_generation"])
    return prices_df, pv_df


def run_stage1_base(
    inputs: Inputs,
    *,
    horizon: Horizon,
    shares: Shares,
    follower_control_area: str,
    fees: Fees,
    ca_price_dict: Dict[str, float],
    node_to_control_area: Dict[str, str],
    stage0_prices_df: pd.DataFrame,
    weekly_system_base_dir: str,
    weekly_follower_base_dir: str,
) -> List[pd.DataFrame]:
    """
    Runs Stage-1 BASE ONCE (follower + system) over all windows.

    The follower gets a constant price per node based on its control area mapping.
    """
    print("\n==============================")
    print("=== STAGE 1 BASE (once)     ===")
    print("==============================\n")

    initial_values_follower_base = {}
    initial_values_system_base = {}

    follower_base_dfs: List[pd.DataFrame] = []

    for w in range(horizon.num_windows):
        start_week = horizon.start_week_of_window(w)
        print(f"\n🔄 [Stage 1 BASE] Window {w+1}/{horizon.num_windows}  (start_week={start_week})\n")

        nodes_all = choose_price_nodes(stage0_prices_df, inputs.demand_data)

        stage1_base_prices_df_const = make_constant_stage1_price_df_by_ca(
            nodes_all,
            start_week=start_week,
            T=horizon.T,
            node_to_control_area=node_to_control_area,
            ca_price_dict=ca_price_dict,
        )

        m_f_base, df_f_base = build_follower_model(
            inputs.general_data,
            inputs.demand_data,
            inputs.renewable_data,
            inputs.flexibility_data,
            inputs.pv_profile_data,
            initial_values=initial_values_follower_base,
            mins_per_hour=horizon.mins_per_hour,
            hours_per_week=horizon.hours_per_week,
            start_week=int(start_week),
            weeks_per_step=horizon.weeks_per_step,
            stage1_base_prices_df=stage1_base_prices_df_const,
            follower_control_area=follower_control_area,
            pv_share=shares.pv_share,
            battery_share=shares.battery_share,
            instrument="BASE",
            base_case=True,
            export_share=CFG.export_share,
            demand_fraction=shares.demand_fraction,
            # ensure capacity tariff is off in BASE even if follower supports it
            peak_rate_eur_per_mw=0.0,
            peak_imports_old=None,
            peak_exports_old=None,
            reset_billing=True,
            **fees.as_kwargs(),
        )
        dump_follower_window(df_f_base, window_id=w, results_dir=weekly_follower_base_dir, label="base")
        follower_base_dfs.append(df_f_base)

        flows_base = follower_flows_for_system(df_f_base, start_week=start_week)

        m_sys_base = build_system_model(
            inputs,
            horizon=horizon,
            shares=shares,
            start_week=start_week,
            initial_values=initial_values_system_base,
            instrument="BASE",
            follower_control_area=follower_control_area,
            follower_flows_df=flows_base,
        )
        solve_model_system(m_sys_base)
        dump_system_window(m_sys_base, window_id=w, results_dir=weekly_system_base_dir, label="base")

        initial_values_follower_base = get_initial_values_for_next_week_price_follower(m_f_base)
        initial_values_system_base = get_initial_values_for_next_week(m_sys_base)

    return follower_base_dfs


def run_stage1_instrument(
    inputs: Inputs,
    *,
    horizon: Horizon,
    shares: Shares,
    instrument: str,
    follower_control_area: str,
    fees: Fees,
    ca_price_dict: Dict[str, float],
    node_to_control_area: Dict[str, str],
    stage0_prices_df: pd.DataFrame,
    weekly_system_instr_dir: str,
    weekly_follower_instr_dir: str,
    peak_rate_eur_per_mw: float,
    billing_weeks: int,
) -> List[pd.DataFrame]:
    """
    Runs Stage-1 INSTRUMENT for a single instrument over all windows.

    - RTP: nodal time series from Stage-0 (unchanged).
    - Peak-Shaving & Capacity-Tariff: constant per control area (mapped to nodes).
    """
    print("\n==============================================")
    print(f"=== STAGE 1 INSTRUMENT: {instrument} ===")
    print("==============================================\n")

    initial_values_follower_instr = {}
    initial_values_system_instr = {}

    follower_instr_dfs: List[pd.DataFrame] = []

    # Billing horizon in "steps" (rolling windows)
    steps_per_billing = int(np.ceil(float(billing_weeks) / float(horizon.weeks_per_step))) if billing_weeks > 0 else 1

    # Node-specific rolling states (MW)
    peak_imports_old_by_node: Dict[str, float] = {}
    peak_exports_old_by_node: Dict[str, float] = {}

    def _reset_peaks_for_nodes(nodes: Iterable[str]) -> None:
        for n in nodes:
            peak_imports_old_by_node[str(n)] = 0.0
            peak_exports_old_by_node[str(n)] = 0.0

    for w in range(horizon.num_windows):
        start_week = horizon.start_week_of_window(w)
        print(f"\n🔄 [Stage 1 {instrument}] Window {w+1}/{horizon.num_windows}  (start_week={start_week})\n")

        nodes_all = choose_price_nodes(stage0_prices_df, inputs.demand_data)
        nodes_all_str = [str(n) for n in nodes_all]

        # Ensure peak dicts have all nodes that may appear in follower results
        for n in nodes_all_str:
            peak_imports_old_by_node.setdefault(n, 0.0)
            peak_exports_old_by_node.setdefault(n, 0.0)

        # -------------------------
        # Price signal for follower
        # -------------------------
        if instrument == "RTP":
            stage1_instr_prices_df = make_rtp_price_df_from_stage0_prices(
                stage0_prices_df,
                start_week=start_week,
                horizon=horizon,
                nodes_for_follower=nodes_all_str,
            )
        else:
            stage1_instr_prices_df = make_constant_stage1_price_df_by_ca(
                nodes_all_str,
                start_week=start_week,
                T=horizon.T,
                node_to_control_area=node_to_control_area,
                ca_price_dict=ca_price_dict,
            )

        # -------------------------
        # Billing reset logic (Capacity-Tariff only)
        # -------------------------
        reset_billing = False
        if instrument == "Capacity-Tariff":
            reset_billing = (w % steps_per_billing == 0)
            if reset_billing:
                _reset_peaks_for_nodes(nodes_all_str)

        # -------------------------
        # Solve follower
        # -------------------------
        m_f_instr, df_f_instr = build_follower_model(
            inputs.general_data,
            inputs.demand_data,
            inputs.renewable_data,
            inputs.flexibility_data,
            inputs.pv_profile_data,
            initial_values=initial_values_follower_instr,
            mins_per_hour=horizon.mins_per_hour,
            hours_per_week=horizon.hours_per_week,
            start_week=int(start_week),
            weeks_per_step=horizon.weeks_per_step,
            stage1_base_prices_df=stage1_instr_prices_df,
            follower_control_area=follower_control_area,
            pv_share=shares.pv_share,
            battery_share=shares.battery_share,
            instrument=instrument,
            base_case=False,
            export_share=CFG.export_share,
            demand_fraction=shares.demand_fraction,
            # Capacity tariff args (ignored by follower if instrument != Capacity-Tariff)
            peak_rate_eur_per_mw=float(peak_rate_eur_per_mw) if instrument == "Capacity-Tariff" else 0.0,
            peak_imports_old=peak_imports_old_by_node if instrument == "Capacity-Tariff" else None,
            peak_exports_old=peak_exports_old_by_node if instrument == "Capacity-Tariff" else None,
            reset_billing=bool(reset_billing) if instrument == "Capacity-Tariff" else False,
            **fees.as_kwargs(),
        )
        dump_follower_window(
            df_f_instr,
            window_id=w,
            results_dir=weekly_follower_instr_dir,
            label=instrument_slug(instrument),
        )
        follower_instr_dfs.append(df_f_instr)

        # -------------------------
        # Update node-specific peaks after window (Capacity-Tariff only)
        # -------------------------
        if instrument == "Capacity-Tariff":
            if df_f_instr is None or df_f_instr.empty:
                raise ValueError(
                    f"Capacity-Tariff follower produced empty results in window {w} (start_week={start_week})."
                )

            # Expect node-specific MW telemetry columns in follower export
            needed = {"node", "imports_power", "exports_power"}
            missing = needed - set(df_f_instr.columns)
            if missing:
                raise ValueError(
                    f"Capacity-Tariff follower results missing required columns {sorted(missing)}. "
                    f"Available columns include: {list(df_f_instr.columns)[:40]}..."
                )

            tmp = df_f_instr.copy()
            tmp["node"] = tmp["node"].astype(str)
            tmp["imports_power"] = pd.to_numeric(tmp["imports_power"], errors="coerce")
            tmp["exports_power"] = pd.to_numeric(tmp["exports_power"], errors="coerce")

            # Peak within this window per node
            peak_imp_by_node = tmp.groupby("node", as_index=True)["imports_power"].max()
            peak_exp_by_node = tmp.groupby("node", as_index=True)["exports_power"].max()

            # Update realized billing-horizon peak per node (carry max forward)
            for n, v in peak_imp_by_node.items():
                v = float(v) if np.isfinite(v) else 0.0
                peak_imports_old_by_node[str(n)] = max(float(peak_imports_old_by_node.get(str(n), 0.0)), v)

            for n, v in peak_exp_by_node.items():
                v = float(v) if np.isfinite(v) else 0.0
                peak_exports_old_by_node[str(n)] = max(float(peak_exports_old_by_node.get(str(n), 0.0)), v)

        # -------------------------
        # Solve system with follower flows
        # -------------------------
        flows_instr = follower_flows_for_system(df_f_instr, start_week=start_week)

        m_sys_instr = build_system_model(
            inputs,
            horizon=horizon,
            shares=shares,
            start_week=start_week,
            initial_values=initial_values_system_instr,
            instrument=instrument,
            follower_control_area=follower_control_area,
            follower_flows_df=flows_instr,
        )
        solve_model_system(m_sys_instr)
        dump_system_window(
            m_sys_instr,
            window_id=w,
            results_dir=weekly_system_instr_dir,
            label=instrument_slug(instrument),
        )

        # Continuity
        initial_values_follower_instr = get_initial_values_for_next_week_price_follower(m_f_instr)
        initial_values_system_instr = get_initial_values_for_next_week(m_sys_instr)

    return follower_instr_dfs

# =============================================================================
# Main helpers
# =============================================================================

def _extract_node_to_control_area(demand_data: tuple, control_areas: Iterable[str]) -> Dict[str, str]:
    """
    demand_data contains node_to_control_area already (as you stated).
    We extract it robustly by scanning for a dict whose values look like CA codes.
    """
    ca_set = set(str(ca) for ca in control_areas)

    # Direct attribute (if demand_data is an object)
    if hasattr(demand_data, "node_to_control_area"):
        m = getattr(demand_data, "node_to_control_area")
        if isinstance(m, dict) and m:
            return {str(k): str(v) for k, v in m.items()}

    # Scan tuple parts
    for part in demand_data:
        if isinstance(part, dict) and part:
            vals = list(part.values())
            sample = vals[: min(200, len(vals))]
            if any(str(v) in ca_set for v in sample):
                return {str(k): str(v) for k, v in part.items()}

    raise ValueError(
        "Could not find node_to_control_area mapping in inputs.demand_data. "
        "Expected a dict-like object mapping node -> control area code."
    )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    paths = build_paths(CFG)
    prepare_output_dirs(paths)

    inputs = load_all_inputs(paths.data_path, mins_per_hour=CFG.horizon.mins_per_hour)

    # Node -> Control Area mapping (works even if AT nodes don't start with 'AT')
    node_to_control_area = _extract_node_to_control_area(inputs.demand_data, CFG.follower_control_area)

    # --- Stage 0 (ONCE)
    prices_df, pv_df = run_stage0(
        inputs,
        horizon=CFG.horizon,
        shares=CFG.stage0,
        follower_control_area=CFG.follower_control_area,
        weekly_out_dir=paths.weekly_stage0_dir,
    )

    prices_df.to_csv(paths.full_stage0_prices, index=False)
    pv_df.to_csv(paths.full_stage0_pv, index=False)
    print(f"\n✅ Stage 0 extracted prices saved to: {paths.full_stage0_prices}")
    print(f"✅ Stage 0 extracted PV saved to: {paths.full_stage0_pv}")

    concat_weekly_csvs(paths.weekly_stage0_dir, paths.full_stage0_results)

    # --- Aggregate PV by t (used for ALL CA base prices)
    pv_by_t = (
        pv_df.groupby("t", as_index=False)["pv_generation"].sum()
        if not pv_df.empty else
        pd.DataFrame(columns=["t", "pv_generation"])
    )

    # --- Stage 1 BASE price per control area (computed ONCE, printed ONCE)
    print("\n==============================")
    print("Stage-1 BASE price per control area")
    print("==============================")

    # Nodes per CA via mapping
    nodes_by_ca: Dict[str, List[str]] = {str(ca): [] for ca in CFG.follower_control_area}
    for n, ca in node_to_control_area.items():
        ca = str(ca)
        if ca in nodes_by_ca:
            nodes_by_ca[ca].append(str(n))
    for ca in nodes_by_ca:
        nodes_by_ca[ca] = sorted(nodes_by_ca[ca])

    ca_prices: List[Tuple[str, Optional[float]]] = []  # (CA, p_base_ca)

    for ca in CFG.follower_control_area:
        ca = str(ca)
        nodes_ca = nodes_by_ca.get(ca, [])

        if not nodes_ca:
            ca_prices.append((ca, None))
            continue

        prices_ca = prices_df[prices_df["node"].astype(str).isin(nodes_ca)].copy()
        if prices_ca.empty:
            ca_prices.append((ca, None))
            continue

        prices_ca_by_t = prices_ca.groupby("t", as_index=False)["price"].mean()

        try:
            p_base_ca = pv_weighted_price_over_horizon(prices_ca_by_t, pv_by_t)
            ca_prices.append((ca, float(p_base_ca)))
        except ValueError:
            ca_prices.append((ca, None))

    for ca, p in ca_prices:
        if p is None:
            print(f"{ca}: n/a")
        else:
            print(f"{ca}: {p:.6f} EUR/MWh")

    # --- Fail hard if any CA base price is n/a
    missing = [ca for ca, p in ca_prices if p is None]
    if missing:
        diag = []
        for ca in missing:
            nodes_ca = nodes_by_ca.get(str(ca), [])
            n_nodes = len(nodes_ca)
            n_price_rows = int(prices_df["node"].astype(str).isin(nodes_ca).sum()) if n_nodes else 0
            diag.append(f"{ca}: nodes={n_nodes}, price_rows={n_price_rows}")
        raise ValueError(
            "Stage-1 BASE price computation failed for control areas: "
            + ", ".join(missing)
            + "\nDiagnostics: " + " | ".join(diag)
            + "\nFix: ensure Stage-0 prices exist for nodes in those control areas and overlap with PV time index."
        )

    # CA -> base price dict (guaranteed floats now)
    ca_price_dict: Dict[str, float] = {ca: float(p) for ca, p in ca_prices}  # p is not None due to check above

    # --- Stage 1 BASE (ONCE)
    follower_base_dfs = run_stage1_base(
        inputs,
        horizon=CFG.horizon,
        shares=CFG.stage1,
        follower_control_area=CFG.follower_control_area,
        fees=CFG.fees,
        ca_price_dict=ca_price_dict,
        node_to_control_area=node_to_control_area,
        stage0_prices_df=prices_df,
        weekly_system_base_dir=paths.weekly_stage1_base_dir,
        weekly_follower_base_dir=paths.weekly_follower_base_dir,
    )

    if follower_base_dfs:
        pd.concat(follower_base_dfs, ignore_index=True).to_csv(paths.full_follower_base, index=False)
        print(f"\n✅ Full follower BASE saved to: {paths.full_follower_base}\n")

    concat_weekly_csvs(paths.weekly_stage1_base_dir, paths.full_stage1_base)

    # --- Stage 1 INSTRUMENT (for ALL instruments)
    for instrument in INSTRUMENTS:
        sys_dir, fol_dir = instrument_weekly_dirs(paths, instrument)

        # Clean instrument dirs each run so outputs don't mix across runs
        clean_csv_dir(sys_dir)
        clean_csv_dir(fol_dir)

        follower_instr_dfs = run_stage1_instrument(
            inputs,
            horizon=CFG.horizon,
            shares=CFG.stage1,
            instrument=instrument,
            follower_control_area=CFG.follower_control_area,
            fees=CFG.fees,
            ca_price_dict=ca_price_dict,
            node_to_control_area=node_to_control_area,
            stage0_prices_df=prices_df,
            weekly_system_instr_dir=sys_dir,
            weekly_follower_instr_dir=fol_dir,
            peak_rate_eur_per_mw=float(CFG.peak_rate_eur_per_mw),
            billing_weeks=int(CFG.billing_weeks),
        )

        full_follower_path, full_system_path = instrument_full_paths(paths, instrument)

        if follower_instr_dfs:
            pd.concat(follower_instr_dfs, ignore_index=True).to_csv(full_follower_path, index=False)
            print(f"\n✅ Full follower INSTRUMENT ({instrument}) saved to: {full_follower_path}\n")

        concat_weekly_csvs(sys_dir, full_system_path)

    print("\n✅ All stages finished.\n")


if __name__ == "__main__":
    main()
