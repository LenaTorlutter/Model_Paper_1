# horizon_window_debug.py
# -*- coding: utf-8 -*-

"""
Window-horizon debugger (SYSTEM model)

Goal
----
- Use WINDOW_INIT_CSV (e.g. window_013_...) to build carry-over initial_values
  from the LAST timestep in that window (typically t=336 for weeks_per_step=2).
- Solve WINDOW_REF (e.g. window_014_...) starting at START_WEEK_REF (the true window start).
- Choose any horizon length H (hours) within that window, e.g. H=1, 100, 336.
- No T_override in the debug script.
- Compare terminal values at local t=H against the reference window CSV for window 14
  at the SAME local t=H (if those rows exist).

Requires (model-side tiny patch)
--------------------------------
1) In create_model(...): accept keyword-only `horizon_hours: int | None = None`
   and set `T = horizon_hours if provided else hours_per_week*weeks_per_step`
   (keep time_offset based on start_week and hours_per_week).
2) In solve_model(...): accept keyword-only `write_lp: bool = False`
   and only write LP/IIS files if write_lp=True.
   (You can still compute IIS without writing LP each run.)

This script assumes:
- Pipeline dump CSV uses columns: window, component, value, index_0, index_1, ...
- Time index is the LAST index_* column (local timestep in 1..T of that window)
"""

import os
import sys
import ast
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
from pyomo.environ import Objective, Expression, Var, value

# =============================================================================
# USER SETTINGS
# =============================================================================

# (1) INIT window CSV: used ONLY to extract carry-over initial_values (end-of-window 13)
WINDOW_INIT_CSV = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\weekly_results\stage0_system\window_014_system_stage0_test.csv"

# (2) REF window CSV: used ONLY to compare (values inside window 14)
WINDOW_REF_CSV  = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\weekly_results\stage0_system\window_015_system_stage0_test.csv"

# Excel + modules
DATA_XLSX = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Data_Updated.xlsx"
PARAMETERS_PY = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Parameters_Updated.py"
MODEL_DEFS_PY = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Model_Test_New.py"

# Rolling window (the one you SOLVE / COMPARE)  -> this must match WINDOW_REF_CSV
START_WEEK_REF = 31         # start week of window 14 in your pipeline
WEEKS_PER_STEP = 2          # pipeline window length in weeks (e.g. 2)
HOURS_PER_WEEK = 168
MINS_PER_HOUR = 60

# Choose any H <= WEEKS_PER_STEP*HOURS_PER_WEEK
HORIZON_HOURS_TO_TEST = [336] #336 for comparison of model values

# Model knobs (system side)
PV_SHARE = 0.0
BATTERY_SHARE = 0.0
DEMAND_FRACTION = 0.0
FOLLOWER_CONTROL_AREA = ("AT", "BE", "CH", "CZ", "DE", "FR", "LU", "HU", "IT", "NL", "PL", "SI", "SK")
FOLLOWER_FLOWS_DF = None

# Comparison settings
ABS_TOL = 1e-4
REL_TOL = 1e-4
MAX_MISMATCH_REPORT = 50

# =============================================================================
# Dynamic import helpers
# =============================================================================

def import_module_from_path(module_path: str, module_name: str):
    module_path = os.path.abspath(module_path)
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def call_with_supported_kwargs(fn, *args, **kwargs):
    import inspect
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)

# =============================================================================
# Data loading (same as pipeline)
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


def load_all_inputs_like_main(data_path: str, *, mins_per_hour: int, parameters_mod) -> Inputs:
    base_dir = os.path.dirname(os.path.abspath(data_path))
    ptdf_csv_path = os.path.join(base_dir, "PTDF_Synchronized.csv")

    load_general_data = getattr(parameters_mod, "load_general_data")
    load_demand_data = getattr(parameters_mod, "load_demand_data")
    load_thermal_power_plant_data = getattr(parameters_mod, "load_thermal_power_plant_data")
    load_phs_power_plant_data = getattr(parameters_mod, "load_phs_power_plant_data")
    load_phs_inflow_data = getattr(parameters_mod, "load_phs_inflow_data")
    load_phs_storage_profile_data = getattr(parameters_mod, "load_phs_storage_profile_data")
    load_renewable_power_plant_data = getattr(parameters_mod, "load_renewable_power_plant_data")
    load_res_profile_data = getattr(parameters_mod, "load_res_profile_data")
    load_flexibility_data = getattr(parameters_mod, "load_flexibility_data")
    load_exchange_data = getattr(parameters_mod, "load_exchange_data")

    general_data = load_general_data(data_path, "General_Data")
    demand_data = load_demand_data(data_path, "Demand_Profiles")

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

    renewable_data = load_renewable_power_plant_data(
        data_path, "RES_Power_Data", "RES_Power_Specific_Data"
    )

    ror_profile_data = load_res_profile_data(data_path, "RoR_Profile", plant_type="RoR")
    windon_profile_data = load_res_profile_data(data_path, "WindOn_Profile", plant_type="WindOn")
    windoff_profile_data = load_res_profile_data(data_path, "WindOff_Profile", plant_type="WindOff")
    pv_profile_data = load_res_profile_data(data_path, "PV_Profile", plant_type="PV")

    flexibility_data = load_flexibility_data(
        data_path, "Flexibility_Data", "Flexibility_Specific_Data"
    )

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
# Index parsing + CSV snapshot extraction
# =============================================================================

def _parse_idx(v):
    if pd.isna(v):
        return None
    # numeric?
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        pass
    s = str(v).strip()
    # tuple/list/etc?
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _csv_idx_cols(df: pd.DataFrame, index_prefix: str = "index_") -> List[str]:
    return [c for c in df.columns if c.startswith(index_prefix)]


def _csv_time_col(df: pd.DataFrame, idx_cols: List[str]) -> str:
    """
    In your dump_system_window, the timestep is typically the LAST index_* column.
    We still sanity-check it is mostly numeric.
    """
    if not idx_cols:
        raise ValueError("No index_* columns found.")
    cand = idx_cols[-1]
    x = pd.to_numeric(df[cand], errors="coerce")
    if float(x.notna().mean()) < 0.80:
        # fallback search from end
        for c in reversed(idx_cols):
            x = pd.to_numeric(df[c], errors="coerce")
            if float(x.notna().mean()) >= 0.80:
                return c
        raise ValueError("Could not identify timestep column (no mostly-numeric index_* column).")
    return cand


def reference_values_from_window_csv_at_t(
    csv_path: str,
    *,
    t_local: int,
    value_col: str = "value",
    component_col: str = "component",
    index_prefix: str = "index_",
) -> Dict[Tuple[Any, ...], float]:
    """
    Build reference dict at exactly local timestep t_local from the pipeline dump CSV.

    Returns keys: (component, idx0, idx1, ...) WITHOUT the time index.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    if component_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Reference CSV missing required columns '{component_col}'/'{value_col}'")

    idx_cols = _csv_idx_cols(df, index_prefix=index_prefix)
    if not idx_cols:
        return {}

    df[component_col] = df[component_col].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).copy()

    tcol = _csv_time_col(df, idx_cols)
    df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.dropna(subset=["_t"]).copy()
    df["_t"] = df["_t"].astype(int)

    df = df[df["_t"] == int(t_local)].copy()
    if df.empty:
        return {}

    other_cols = [c for c in idx_cols if c != tcol]

    ref: Dict[Tuple[Any, ...], float] = {}
    if not other_cols:
        for _, r in df.iterrows():
            ref[(str(r[component_col]),)] = float(r[value_col])
        return ref

    for _, r in df.iterrows():
        comp = str(r[component_col])
        parts = tuple(_parse_idx(r[c]) for c in other_cols)
        ref[(comp, *parts)] = float(r[value_col])

    return ref


def initial_values_from_window_csv_last_t(
    csv_path: str,
    *,
    value_col: str = "value",
    component_col: str = "component",
    index_prefix: str = "index_",
) -> Dict[Tuple[Any, ...], float]:
    """
    Carry-over initial_values for the *next* window:
    take the LAST timestep in this window CSV and return keys WITHOUT time.

    Example:
      ('phs_storage', plant, t=336)  -> ('phs_storage', plant) = value_at_t336
    """
    df = pd.read_csv(csv_path, low_memory=False)

    if component_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Init CSV missing required columns '{component_col}'/'{value_col}'")

    idx_cols = _csv_idx_cols(df, index_prefix=index_prefix)
    if not idx_cols:
        # no indices: last row per component
        df[component_col] = df[component_col].astype(str)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col]).copy()
        last = df.groupby(component_col, as_index=False).tail(1)
        return {(str(r[component_col]),): float(r[value_col]) for _, r in last.iterrows()}

    df[component_col] = df[component_col].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).copy()

    tcol = _csv_time_col(df, idx_cols)
    df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.dropna(subset=["_t"]).copy()
    df["_t"] = df["_t"].astype(int)

    T_last = int(df["_t"].max())
    df = df[df["_t"] == T_last].copy()
    if df.empty:
        return {}

    other_cols = [c for c in idx_cols if c != tcol]

    out: Dict[Tuple[Any, ...], float] = {}
    if not other_cols:
        for _, r in df.iterrows():
            out[(str(r[component_col]),)] = float(r[value_col])
        return out

    for _, r in df.iterrows():
        comp = str(r[component_col])
        parts = tuple(_parse_idx(r[c]) for c in other_cols)
        out[(comp, *parts)] = float(r[value_col])

    return out

# =============================================================================
# Terminal snapshot from solved model (t = H)
# =============================================================================

def model_terminal_snapshot(model, *, t_local: int) -> Dict[Tuple[Any, ...], float]:
    """
    Extract values of ALL active Vars at exactly t=t_local (local time in model.timesteps),
    keyed as (var.local_name, *indices_without_t).
    Only includes indices whose *last* index equals t_local.
    """
    out: Dict[Tuple[Any, ...], float] = {}
    t_local = int(t_local)

    for _, var in model.component_map(Var, active=True).items():
        vname = var.local_name
        for index in var:
            idx = index if isinstance(index, tuple) else (index,)
            if not idx:
                continue
            last = idx[-1]
            try:
                if int(last) != t_local:
                    continue
            except Exception:
                continue

            key = (vname, *idx[:-1])
            val = value(var[index], exception=False)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            out[key] = float(val)

    return out


# =============================================================================
# Comparison helper
# =============================================================================

def maybe_load_reference_snapshot(ref_csv_path: str | None, *, t_local: int) -> Dict[Tuple[Any, ...], float]:
    """
    Returns an empty dict if:
      - ref_csv_path is None/empty
      - file does not exist
      - file exists but has no rows for t_local
    Otherwise returns the reference snapshot dict.
    """
    if ref_csv_path is None:
        return {}
    ref_csv_path = str(ref_csv_path).strip()
    if not ref_csv_path:
        return {}
    if not os.path.isfile(ref_csv_path):
        print(f"ℹ️ Reference CSV not found -> skip compare: {ref_csv_path}")
        return {}

    ref = reference_values_from_window_csv_at_t(ref_csv_path, t_local=int(t_local))
    if not ref:
        print(f"ℹ️ Reference CSV exists but has no rows for t={t_local} -> skip compare.")
    return ref

def compare_snapshots(
    solved: Dict[Tuple[Any, ...], float],
    reference: Dict[Tuple[Any, ...], float],
    *,
    abs_tol: float,
    rel_tol: float,
    max_report: int,
) -> None:
    solved_keys = set(solved.keys())
    ref_keys = set(reference.keys())

    common = solved_keys & ref_keys
    only_solved = solved_keys - ref_keys
    only_ref = ref_keys - solved_keys

    mismatches = []
    for k in common:
        a = solved[k]
        b = reference[k]
        diff = abs(a - b)
        tol = abs_tol + rel_tol * max(1.0, abs(b))
        if diff > tol:
            mismatches.append((k, a, b, diff, tol))

    mismatches.sort(key=lambda x: x[3], reverse=True)

    print("\n==============================")
    print("TERMINAL SNAPSHOT COMPARISON")
    print("==============================")
    print(f"Common keys:        {len(common)}")
    print(f"Only in solved:     {len(only_solved)}")
    print(f"Only in reference:  {len(only_ref)}")
    print(f"Mismatches:         {len(mismatches)} (tol=abs {abs_tol}, rel {rel_tol})")

    if only_ref:
        sample = list(only_ref)[:min(10, len(only_ref))]
        print("\nSample keys ONLY in reference (missing in solved snapshot):")
        for k in sample:
            print("  ", k)

    if only_solved:
        sample = list(only_solved)[:min(10, len(only_solved))]
        print("\nSample keys ONLY in solved (not in reference snapshot):")
        for k in sample:
            print("  ", k)

    if mismatches:
        print(f"\nTop {min(max_report, len(mismatches))} mismatches:")
        for k, a, b, diff, tol in mismatches[:max_report]:
            print(f"  {k}: solved={a:.6g}  ref={b:.6g}  |diff|={diff:.3g}  tol={tol:.3g}")
            
# =============================================================================
# DEBUGGING
# =============================================================================

def _active_objective(model):
    objs = list(model.component_objects(Objective, active=True))
    return objs[0] if objs else None

def print_penalty_terms(model, *, top_n=50, name_hints=("penalty", "slack", "viol", "nse", "curtail")):
    """
    Prints:
      1) total objective value
      2) values of any active Expression components whose name matches name_hints
      3) fallback: nonzero Var entries whose varname matches name_hints (top_n by abs value)

    Tip: If you define explicit penalty Expressions in the model (recommended),
    they'll show up in section (2) with exact costs.
    """
    print("\n==============================")
    print("PENALTY TERMS / OBJECTIVE BREAKDOWN")
    print("==============================")

    obj = _active_objective(model)
    if obj is not None:
        try:
            print(f"Total objective ({obj.name}): {value(obj.expr):.6g}")
        except Exception:
            print(f"Total objective ({obj.name}): <could not evaluate>")
    else:
        print("No active Objective found.")

    # (2) Preferred: explicit penalty Expressions in the model
    hits_expr = []
    for e in model.component_objects(Expression, active=True):
        ename = e.local_name.lower()
        if any(h in ename for h in name_hints):
            try:
                hits_expr.append((e.name, float(value(e, exception=False) or 0.0)))
            except Exception:
                hits_expr.append((e.name, float("nan")))

    if hits_expr:
        print("\n-- Matching Expression components (best signal) --")
        for n, v in sorted(hits_expr, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {n}: {v:.6g}")
    else:
        print("\n-- Matching Expression components --")
        print("  (none found)")

    # (3) Fallback: list nonzero slack-like vars (not costs, but shows where penalties may apply)
    hits_var = []
    for var in model.component_map(Var, active=True).values():
        vname = var.local_name.lower()
        if not any(h in vname for h in name_hints):
            continue
        for idx in var:
            val = value(var[idx], exception=False)
            if val is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if abs(fv) > 1e-9:
                hits_var.append((var.name, idx, fv))

    if hits_var:
        hits_var.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n-- Matching Var entries with nonzero value (top {top_n}) --")
        for name, idx, fv in hits_var[:top_n]:
            print(f"  {name}{idx}: {fv:.6g}")
        if len(hits_var) > top_n:
            print(f"  ... ({len(hits_var)-top_n} more)")
    else:
        print("\n-- Matching Var entries with nonzero value --")
        print("  (none found)")            


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading model defs...")
    model_mod = import_module_from_path(MODEL_DEFS_PY, "model_defs_mod")
    create_model_func = getattr(model_mod, "create_model")
    solve_model_func = getattr(model_mod, "solve_model")

    print("Loading Parameters_Updated...")
    parameters_mod = import_module_from_path(PARAMETERS_PY, "parameters_updated_mod")

    print("Loading inputs (like pipeline)...")
    inputs = load_all_inputs_like_main(DATA_XLSX, mins_per_hour=MINS_PER_HOUR, parameters_mod=parameters_mod)

    window_len_hours = int(WEEKS_PER_STEP * HOURS_PER_WEEK)
    time_offset = int((START_WEEK_REF - 1) * HOURS_PER_WEEK)

    print("\n==============================")
    print(f"WINDOW DEBUG | start_week={START_WEEK_REF} | weeks_per_step={WEEKS_PER_STEP}")
    print(f"Window length: {window_len_hours} h | time_offset={time_offset}")
    print(f"INIT CSV: {WINDOW_INIT_CSV}")
    print(f"REF  CSV: {WINDOW_REF_CSV}")
    print("==============================\n")

    # Carry-over from previous window (end-of-window 13 -> initial_values for window 14)
    print("Extracting carry-over initial_values from INIT CSV (LAST timestep)...")
    initial_values = initial_values_from_window_csv_last_t(WINDOW_INIT_CSV)
    print(f"Initial values extracted: {len(initial_values)} keys\n")

    for H in HORIZON_HOURS_TO_TEST:
        H = int(H)
        if H < 1 or H > window_len_hours:
            raise ValueError(f"H={H} must be in [1, {window_len_hours}]")

        tag = f"w{START_WEEK_REF}_H{H}"
        print("\n--------------------------------")
        print(f"Solving window start_week={START_WEEK_REF} for H={H} hours (within window)")
        print("--------------------------------")

        model = call_with_supported_kwargs(
            create_model_func,
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
            MINS_PER_HOUR,
            HOURS_PER_WEEK,
            int(START_WEEK_REF),
            int(WEEKS_PER_STEP),
            pv_share=float(PV_SHARE),
            battery_share=float(BATTERY_SHARE),
            demand_fraction=float(DEMAND_FRACTION),
            follower_control_area=FOLLOWER_CONTROL_AREA,
            follower_flows_df=FOLLOWER_FLOWS_DF,
            # NEW (model patch required): limit model horizon to H local hours
            horizon_hours=int(H),
        )

        # Solve (model patch required): avoid LP writing
        call_with_supported_kwargs(
            solve_model_func,
            model,
            tee=False,
            tag=tag,
            write_lp=False,
        )
        
        print_penalty_terms(model, top_n=50)

        # Extract terminal snapshot at t=H
        solved_snap = model_terminal_snapshot(model, t_local=H)
        print(f"Terminal snapshot keys extracted (t={H}): {len(solved_snap)}")
        
        # Optional reference compare (only if file exists)
        ref_snap = maybe_load_reference_snapshot(WINDOW_REF_CSV, t_local=H)
        
        if ref_snap:
            print(f"Reference snapshot keys found in REF CSV (t={H}): {len(ref_snap)}")
            compare_snapshots(
                solved_snap,
                ref_snap,
                abs_tol=ABS_TOL,
                rel_tol=REL_TOL,
                max_report=MAX_MISMATCH_REPORT,
            )
        else:
            print("✅ Compare skipped (no reference available).")

if __name__ == "__main__":
    main()