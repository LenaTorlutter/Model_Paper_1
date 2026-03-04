from __future__ import annotations

import os
import re
import warnings
import inspect
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
DATA_FILE = "Data.xlsx"
DATA_XLSX = os.path.join(BASE_DIR, DATA_FILE)

DEFAULT_BASE_DATE = "2030-01-01"
HOURS_PER_WEEK = 168

CONTROL_AREA = "AT"
NODE: str | None = None  # set e.g. "VBG" and set CONTROL_AREA=None for node-scope

INSTRUMENTS = ["RTP", "Peak-Shaving", "Capacity-Tariff"]

START = "2030-01-01 00:00"
END = "2030-01-02 00:00"

SHOW_PF_FLOWS = True

# Set VERBOSE=True if you want warnings and small info messages.
VERBOSE = False

PRINT_STACK_DEBUG = True      # set False to silence the table output
STACK_DEBUG_N = 10            # number of timesteps to print
STACK_DEBUG_WHICH = "first"   # "first" or "last"
STACK_DEBUG_DECIMALS = 2

# ============================================================
# Helpers: pipeline naming
# ============================================================

def instrument_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def stage1_instr_path(base_dir: str, instrument: str) -> str:
    return os.path.join(base_dir, f"full_model_results_stage1_instrument__{instrument_slug(instrument)}.csv")

def follower_instr_path(base_dir: str, instrument: str) -> str:
    return os.path.join(base_dir, f"full_follower_values_instrument__{instrument_slug(instrument)}.csv")


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
# Time helpers
# ============================================================

DATE_FMT = "%Y-%m-%d %H:%M"

def coerce_base_date(base_date: str) -> str:
    return str(base_date).strip()

def parse_user_datetime(dt: str) -> pd.Timestamp:
    return pd.to_datetime(dt, format=DATE_FMT)

def parse_user_range(start: str, end: str) -> tuple[str, str]:
    ts0 = parse_user_datetime(start)
    ts1 = parse_user_datetime(end)
    if ts1 < ts0:
        raise ValueError("End datetime must be >= start datetime.")
    return ts0.strftime(DATE_FMT), ts1.strftime(DATE_FMT)

def date_to_abs_hour(ts: str, base_date: str = DEFAULT_BASE_DATE) -> int:
    # 1-based abs_hour
    t0 = pd.to_datetime(coerce_base_date(base_date) + " 00:00", format=DATE_FMT)
    t = pd.to_datetime(ts, format=DATE_FMT)
    return int((t - t0).total_seconds() // 3600) + 1

def ensure_abs_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "abs_hour" in df.columns:
        return df

    df = df.copy()

    if "start_week" not in df.columns:
        if "window" in df.columns:
            df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1
        else:
            return df

    if "timestep" in df.columns:
        time_col = "timestep"
    elif "t" in df.columns:
        time_col = "t"
    elif "index_1" in df.columns:
        df["timestep"] = pd.to_numeric(df["index_1"], errors="coerce").fillna(0).astype(int)
        time_col = "timestep"
    else:
        return df

    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").fillna(0).astype(int)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(int)
    df["abs_hour"] = df[time_col] + HOURS_PER_WEEK * (df["start_week"] - 1)
    return df


# ============================================================
# IO
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
        df["start_week"] = pd.to_numeric(df["window"], errors="coerce").fillna(0).astype(int) + 1

    return ensure_abs_hour(df)


# ============================================================
# Excel demand (CA-wide)
# ============================================================

def demand_series_from_excel_by_ca(
    data_xlsx_path: str,
    *,
    sheet_name: str,
    control_area: str,
    hours_per_week: int = HOURS_PER_WEEK,
) -> pd.DataFrame:
    ca = str(control_area).strip()
    raw = pd.read_excel(data_xlsx_path, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]

    if "Hour" not in raw.columns:
        raise ValueError(f"Sheet {sheet_name!r} must contain column 'Hour'. Found: {list(raw.columns)[:20]}")

    if ca not in raw.columns:
        raise ValueError(f"Control area '{ca}' not found in sheet {sheet_name!r}. Columns: {list(raw.columns)}")

    df = raw[["Hour", ca]].copy().rename(columns={"Hour": "timestep", ca: "demand"})
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce").fillna(0).astype(int)

    # decimal comma -> float
    df["demand"] = df["demand"].astype(str).str.replace(",", ".", regex=False)
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)

    df = df[(df["timestep"] >= 1) & (df["timestep"] <= int(hours_per_week))].copy()
    df["start_week"] = 1
    df["abs_hour"] = df["timestep"] + hours_per_week * (df["start_week"] - 1)
    return df[["abs_hour", "demand"]]


# ============================================================
# Scope resolution (robust to Data vs Data_Updated mismatch)
# ============================================================

def resolve_nodes_for_scope(
    *,
    stage_df: pd.DataFrame,
    pf_nodewise: Optional[pd.DataFrame],
    node_to_control_area: Optional[dict],
    control_area: Optional[str],
    node: Optional[str],
    verbose: bool = False,
) -> Tuple[Set[str], str, str]:
    """
    Returns: (nodes_set, scope_label, ca_use_for_excel)
    """
    # --- node scope ---
    if node is not None:
        node_str = str(node)
        ca_use = str(node_to_control_area.get(node_str, "UNKNOWN")) if node_to_control_area else "UNKNOWN"
        return {node_str}, f"Node_{node_str}", ca_use

    # --- CA scope ---
    if control_area is None:
        raise ValueError("Specify exactly one of `control_area` or `node`.")
    ca_use = str(control_area)
    scope_label = f"Area_{ca_use}"

    # Nodes present in stage results (prefer node-indexed vars)
    stage_nodes = set(
        stage_df.loc[stage_df["variable"].isin(["exchange", "nse", "curtailment"]), "index_0"]
        .astype(str)
        .unique()
    )
    if not stage_nodes and "index_0" in stage_df.columns:
        stage_nodes = set(stage_df["index_0"].astype(str).unique())

    # Nodes present in PF file (if available)
    pf_nodes: Set[str] = set()
    if pf_nodewise is not None and not pf_nodewise.empty and "node" in pf_nodewise.columns:
        pf_nodes = set(pf_nodewise["node"].astype(str).unique())

    mapped_nodes: Set[str] = set()
    if node_to_control_area:
        mapped_nodes = {str(n) for n, ca in node_to_control_area.items() if str(ca) == ca_use}

    # Use mapping only if it meaningfully overlaps with stage nodes; else fall back to stage nodes.
    if mapped_nodes and stage_nodes and len(mapped_nodes.intersection(stage_nodes)) >= 2:
        nodes_set = mapped_nodes.intersection(stage_nodes)
    else:
        nodes_set = stage_nodes.copy()
        if verbose:
            warnings.warn(
                f"node_to_control_area mapping does not match result nodes for control_area={ca_use}. "
                f"Using nodes from results file instead (n={len(nodes_set)})."
            )

    # Keep PF + stage consistent if both exist
    if pf_nodes:
        inter = nodes_set.intersection(pf_nodes)
        if inter:
            nodes_set = inter

    if not nodes_set:
        raise ValueError(
            f"Could not determine nodes for scope control_area={ca_use}. "
            f"Stage nodes={len(stage_nodes)}, mapped nodes={len(mapped_nodes)}, PF nodes={len(pf_nodes)}."
        )

    return nodes_set, scope_label, ca_use


# ============================================================
# Plot 1: System stacks (instrument-only)
# ============================================================

def print_stackplot_variables(
    *,
    time_index: pd.DatetimeIndex,
    gen_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    demand_s: pd.Series,
    exchange_s: pd.Series,
    n: int = 10,
    which: str = "first",   # "first" or "last"
    decimals: int = 2,
    title: str | None = None,
) -> None:
    """
    Print all variables that appear in the stack plot (gen_df + cons_df + demand line),
    plus signed exchange, for first/last n timesteps of the plotted window.
    """
    demand_s = demand_s.reindex(time_index).fillna(0.0)
    exchange_s = exchange_s.reindex(time_index).fillna(0.0)

    out = pd.concat(
        [
            gen_df.reindex(time_index).fillna(0.0),
            cons_df.reindex(time_index).fillna(0.0).add_prefix("-"),
            pd.DataFrame(
                {
                    "Exchange (signed)": exchange_s.values,
                    "Demand (from Excel)": demand_s.values,
                },
                index=time_index,
            ),
        ],
        axis=1,
    )

    block = out.head(n) if which.lower() == "first" else out.tail(n)
    block = block.round(decimals)

    if title:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_rows", None,
    ):
        print(block)


def plot_total_values_instrument_only(
    *,
    stage_df: pd.DataFrame,
    node_to_control_area: dict,
    case_name: str,
    control_area: str | None,
    node: str | None,
    pf_nodewise: pd.DataFrame | None,
    thermal_node_idx: Dict,
    phs_node_idx: Dict,
    renewable_node_idx: Dict,
    dsr_node_idx: Dict,
    battery_node_idx: Dict,
    base_date: str,
    start: str,
    end: str,
    demand_fraction: float = 0.0,
    apply_demand_fraction_to_excel: bool = True,
    verbose: bool = False,
) -> None:
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    nodes_set, scope_label, ca_use = resolve_nodes_for_scope(
        stage_df=stage_df,
        pf_nodewise=pf_nodewise,
        node_to_control_area=node_to_control_area,
        control_area=control_area,
        node=node,
        verbose=verbose,
    )

    def plant_map_df(plant_to_nodes: Dict) -> pd.DataFrame:
        """
        Accept both orientations:
          A) node -> [plants]
          B) plant -> node
        """
        if not plant_to_nodes:
            return pd.DataFrame(columns=["plant", "node"])
        items = list(plant_to_nodes.items())
        val0 = items[0][1]
        if isinstance(val0, (list, tuple, set, np.ndarray, pd.Series)):
            m = [(str(p), str(n)) for n, plants in plant_to_nodes.items() for p in plants]
        else:
            m = [(str(plant), str(node_)) for plant, node_ in plant_to_nodes.items()]
        return pd.DataFrame(m, columns=["plant", "node"])

    def agg_by_nodes(
        varname: str,
        plant_to_nodes: Optional[Dict],
        *,
        is_node_indexed: bool = False,
    ) -> pd.DataFrame:
        part = stage_df.loc[stage_df["variable"] == varname].copy()
        if part.empty:
            return pd.DataFrame(columns=["start_week", "timestep", "value"])

        part["start_week"] = pd.to_numeric(part.get("start_week", 1), errors="coerce").fillna(1).astype(int)

        if is_node_indexed:
            part["node"] = part["index_0"].astype(str)
        else:
            if plant_to_nodes is None:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])
            map_df = plant_map_df(plant_to_nodes)
            if map_df.empty:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])
            part["plant"] = part["index_0"].astype(str)
            part = part.merge(map_df, on="plant", how="inner")
            if part.empty:
                return pd.DataFrame(columns=["start_week", "timestep", "value"])

        # timestep
        if "index_1" in part.columns:
            part["timestep"] = pd.to_numeric(part["index_1"], errors="coerce").fillna(0).astype(int)
        else:
            part["timestep"] = pd.to_numeric(part.get("timestep", 0), errors="coerce").fillna(0).astype(int)

        part["node"] = part["node"].astype(str)
        part = part.loc[part["node"].isin(nodes_set)]
        if part.empty:
            return pd.DataFrame(columns=["start_week", "timestep", "value"])

        part["value"] = pd.to_numeric(part["value"], errors="coerce").fillna(0.0)
        return part.groupby(["start_week", "timestep"], as_index=False)["value"].sum()

    # --- system supply
    thermal = agg_by_nodes("thermal_generation", thermal_node_idx).rename(columns={"value": "thermal"})
    phs_gen = agg_by_nodes("phs_turbine_generation", phs_node_idx).rename(columns={"value": "phs_gen"})
    res_gen = agg_by_nodes("renewable_generation", renewable_node_idx).rename(columns={"value": "res"})
    dsr_dn  = agg_by_nodes("dsr_down", dsr_node_idx).rename(columns={"value": "dsr_out"})
    bat_out = agg_by_nodes("battery_out", battery_node_idx).rename(columns={"value": "battery_out"})

    # --- system consumption
    phs_pu  = agg_by_nodes("phs_pump_consumption", phs_node_idx).rename(columns={"value": "phs_cons"})
    dsr_up  = agg_by_nodes("dsr_up", dsr_node_idx).rename(columns={"value": "dsr_in"})
    bat_in  = agg_by_nodes("battery_in", battery_node_idx).rename(columns={"value": "battery_in"})

    # --- balance terms
    nse     = agg_by_nodes("nse", None, is_node_indexed=True).rename(columns={"value": "nse"})
    curt    = agg_by_nodes("curtailment", None, is_node_indexed=True).rename(columns={"value": "curtailment"})

    exch = agg_by_nodes("exchange", None, is_node_indexed=True)
    if exch.empty:
        exch = agg_by_nodes("expr:exchange", None, is_node_indexed=True)
    exch = exch.rename(columns={"value": "exchange"})

    # --- merge helper (outer join on keys)
    def outer_merge(base: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
        if base.empty:
            return comp.copy()
        if comp.empty:
            return base.copy()
        return base.merge(comp, on=["start_week", "timestep"], how="outer")

    base = pd.DataFrame(columns=["start_week", "timestep"])
    for df_ in [thermal, phs_gen, res_gen, dsr_dn, bat_out, phs_pu, dsr_up, bat_in, nse, curt, exch]:
        base = outer_merge(base, df_)

    # fill missing numeric columns with 0
    for col in ["thermal", "phs_gen", "res", "dsr_out", "battery_out",
                "phs_cons", "dsr_in", "battery_in",
                "nse", "curtailment", "exchange"]:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    base["start_week"] = pd.to_numeric(base["start_week"], errors="coerce").fillna(1).astype(int)
    base["timestep"] = pd.to_numeric(base["timestep"], errors="coerce").fillna(0).astype(int)
    base["abs_hour"] = base["timestep"] + HOURS_PER_WEEK * (base["start_week"] - 1)

    # --- demand from Excel (CA-wide)
    demand_excel = demand_series_from_excel_by_ca(
        DATA_XLSX, sheet_name="Demand_Profiles", control_area=ca_use, hours_per_week=HOURS_PER_WEEK
    )
    if apply_demand_fraction_to_excel and demand_fraction > 0.0 and pf_nodewise is not None and not pf_nodewise.empty:
        pf_tmp = ensure_abs_hour(pf_nodewise.copy())
        pf_tmp["node"] = pf_tmp["node"].astype(str)
        pf_tmp = pf_tmp.loc[pf_tmp["node"].isin(nodes_set)]
        follower_nodes_present = set(pf_tmp["node"].unique())

        if node is not None and str(node) in follower_nodes_present:
            demand_excel["demand"] *= (1.0 - float(demand_fraction))
        elif node is None and nodes_set:
            frac = len(follower_nodes_present) / float(len(nodes_set))
            demand_excel["demand"] *= (1.0 - float(demand_fraction) * float(frac))

    base = base.merge(demand_excel, on="abs_hour", how="left")
    base["demand"] = pd.to_numeric(base["demand"], errors="coerce").fillna(0.0)

    # --- PF bridge terms
    if pf_nodewise is not None and not pf_nodewise.empty and "node" in pf_nodewise.columns:
        pf_scope = ensure_abs_hour(pf_nodewise.copy())
        pf_scope["node"] = pf_scope["node"].astype(str)
        pf_scope = pf_scope.loc[pf_scope["node"].isin(nodes_set)].copy()

        pf_scope["pf_exports_injection"] = (
            pd.to_numeric(pf_scope.get("pv_feed_to_system", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(pf_scope.get("battery_to_system", 0.0), errors="coerce").fillna(0.0)
        )
        pf_scope["pf_imports_to_demand"] = pd.to_numeric(pf_scope.get("imports_to_demand", 0.0), errors="coerce").fillna(0.0)
        pf_scope["pf_imports_to_battery"] = pd.to_numeric(pf_scope.get("imports_to_battery", 0.0), errors="coerce").fillna(0.0)

        pf_agg = pf_scope.groupby("abs_hour", as_index=False).agg(
            pf_exports_injection=("pf_exports_injection", "sum"),
            pf_imports_to_demand=("pf_imports_to_demand", "sum"),
            pf_imports_to_battery=("pf_imports_to_battery", "sum"),
        )
        base = base.merge(pf_agg, on="abs_hour", how="left")
    else:
        base["pf_exports_injection"] = 0.0
        base["pf_imports_to_demand"] = 0.0
        base["pf_imports_to_battery"] = 0.0

    for c in ["pf_exports_injection", "pf_imports_to_demand", "pf_imports_to_battery"]:
        base[c] = pd.to_numeric(base.get(c, 0.0), errors="coerce").fillna(0.0)

    # --- window slice
    use_start, use_end = parse_user_range(start, end)
    ah0 = date_to_abs_hour(use_start, base_date)
    ah1 = date_to_abs_hour(use_end, base_date)

    w = base.loc[(base["abs_hour"] >= ah0) & (base["abs_hour"] <= ah1)].copy()
    if w.empty:
        raise ValueError(
            f"Window [{use_start} → {use_end}] produced an empty slice. "
            f"Available abs_hour min/max: {base['abs_hour'].min()} / {base['abs_hour'].max()}."
        )

    idx = np.unique(w["abs_hour"].astype(int).values)
    base_ts = pd.to_datetime(coerce_base_date(base_date) + " 00:00", format=DATE_FMT)
    time_index = pd.to_datetime([base_ts + pd.Timedelta(hours=int(h - 1)) for h in idx])

    def s(col: str) -> pd.Series:
        return w.set_index("abs_hour")[col].reindex(idx).fillna(0.0)

    exch_s = s("exchange")
    imports = exch_s.clip(lower=0.0)
    exports = (-exch_s).clip(lower=0.0)

    gen_df = pd.DataFrame(
        {
            "Thermal": s("thermal").values,
            "PHS (gen)": s("phs_gen").values,
            "RES": s("res").values,
            "DSR (down)": s("dsr_out").values,
            "Battery (out)": s("battery_out").values,
            "Imports": imports.values,
            "Follower exports": s("pf_exports_injection").values,
            "NSE": s("nse").values,
        },
        index=time_index,
    ).clip(lower=0.0)

    cons_df = pd.DataFrame(
        {
            "PHS (pump)": s("phs_cons").values,
            "DSR (up)": s("dsr_in").values,
            "Battery (in)": s("battery_in").values,
            "Exports": exports.values,
            "Follower imports (to demand)": s("pf_imports_to_demand").values,
            "Follower imports (to battery)": s("pf_imports_to_battery").values,
            "Curtailment": s("curtailment").values,
        },
        index=time_index,
    ).clip(lower=0.0)

    dem_excel_s = s("demand")
    
    if PRINT_STACK_DEBUG:
        print_stackplot_variables(
            time_index=time_index,
            gen_df=gen_df,
            cons_df=cons_df,
            demand_s=dem_excel_s,
            exchange_s=exch_s,
            n=STACK_DEBUG_N,
            which=STACK_DEBUG_WHICH,
            decimals=STACK_DEBUG_DECIMALS,
            title=f"{scope_label} ({case_name}) — stackplot variables ({STACK_DEBUG_WHICH} {STACK_DEBUG_N})",
        )

    # --- colors
    COLORS_SUPPLY = {
        "Thermal": "#D62728",
        "PHS (gen)": "#1F77B4",
        "RES": "#2CA02C",
        "DSR (down)": "#9467BD",
        "Battery (out)": "#FF7F0E",
        "Imports": "#17BECF",
        "Follower exports": "#2ECC71",
        "NSE": "#8E44AD",
    }
    COLORS_CONS = {
        "PHS (pump)": "#AEC7E8",
        "DSR (up)": "#7F7F7F",
        "Battery (in)": "#F1C40F",
        "Exports": "#4D4D4D",
        "Follower imports (to demand)": "#E74C3C",
        "Follower imports (to battery)": "#C0392B",
        "Curtailment": "#34495E",
    }

    fig, ax = plt.subplots(figsize=(18, 7))
    gen_cols = list(gen_df.columns)
    cons_cols = list(cons_df.columns)

    ax.stackplot(
        gen_df.index,
        [gen_df[c].values for c in gen_cols],
        labels=gen_cols,
        colors=[COLORS_SUPPLY[c] for c in gen_cols],
        alpha=0.9,
    )
    ax.stackplot(
        cons_df.index,
        [-cons_df[c].values for c in cons_cols],
        labels=[f"-{c}" for c in cons_cols],
        colors=[COLORS_CONS[c] for c in cons_cols],
        alpha=0.9,
    )

    ax.plot(time_index, dem_excel_s.values, color="black", linewidth=2.0, label="Demand (from Excel)", zorder=6)
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_title(f"{scope_label} ({case_name})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power [MW]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 2: PF flows & price (instrument-only)
# ============================================================

def plot_pf_import_export_and_prices(
    *,
    pf_nodewise: pd.DataFrame,
    node_to_control_area: dict,
    case_name: str,
    control_area: str | None,
    node: str | int | None,
    base_date: str,
    start: str,
    end: str,
    show_flows: bool = True,
    verbose: bool = False,
) -> None:
    if (control_area is None) == (node is None):
        raise ValueError("Specify exactly one of `control_area` or `node`.")

    pf = ensure_abs_hour(pf_nodewise.copy())
    pf["node"] = pf["node"].astype(str)

    if node is not None:
        node_str = str(node)
        ca_of_node = str(node_to_control_area.get(node_str, "UNKNOWN"))
        scope_label = (
            f"{ca_of_node} — Node {node_str} (PF flows & price, {case_name})"
            if show_flows else f"{ca_of_node} — Node {node_str} (PF price, {case_name})"
        )
        pf_scope = pf.loc[pf["node"] == node_str].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for node {node_str}.")
        price_agg = pf_scope[["abs_hour", "price"]].copy()
    else:
        ca = str(control_area)
        scope_label = f"{ca} — PF flows & price ({case_name})" if show_flows else f"{ca} — PF price ({case_name})"

        pf_nodes = set(pf["node"].unique())
        mapped_nodes = {str(n) for n, ca_map in node_to_control_area.items() if str(ca_map) == ca}

        if mapped_nodes and len(mapped_nodes.intersection(pf_nodes)) >= 2:
            node_set = mapped_nodes.intersection(pf_nodes)
        else:
            node_set = pf_nodes
            if verbose:
                warnings.warn(
                    f"node_to_control_area mapping does not match PF nodes for control_area={ca}. "
                    f"Using PF nodes from PF file instead (n={len(pf_nodes)})."
                )

        pf_scope = pf.loc[pf["node"].isin(node_set)].copy()
        if pf_scope.empty:
            raise ValueError(f"No PF data for control area {ca}.")
        price_agg = pf_scope.groupby("abs_hour", as_index=False)["price"].mean()

    # Construct aggregated exports/imports
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
    agg["price"] = pd.to_numeric(agg.get("price", 0.0), errors="coerce").fillna(0.0)

    # Window
    use_start, use_end = parse_user_range(start, end)
    s_abs = date_to_abs_hour(use_start, base_date)
    e_abs = date_to_abs_hour(use_end, base_date)

    agg = agg.loc[(agg["abs_hour"] >= s_abs) & (agg["abs_hour"] <= e_abs)].copy()
    if agg.empty:
        raise ValueError(f"Chosen window [{use_start} → {use_end}] produced an empty slice.")

    base_ts = pd.Timestamp(coerce_base_date(base_date))
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

def _pick_first_dict(x) -> Optional[dict]:
    if isinstance(x, dict):
        return x
    if isinstance(x, tuple):
        for obj in x:
            if isinstance(obj, dict):
                return obj
    return None

def main() -> None:
    if VERBOSE:
        print(f"Using parameter module: {_PARAMS_FLAVOR}")
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"DATA_XLSX: {DATA_XLSX}")

    # --- Load indices/mappings from Excel via Parameters(_Updated) ---
    demand_out = call_with_supported_kwargs(load_demand_data, DATA_XLSX, "Demand_Profiles")

    # Keep behavior simple: get *a* dict; we only use it when it overlaps anyway.
    node_to_control_area = None
    if isinstance(demand_out, tuple):
        dicts = [obj for obj in demand_out if isinstance(obj, dict)]
        node_to_control_area = dicts[-1] if dicts else None
    elif isinstance(demand_out, dict):
        node_to_control_area = demand_out

    if node_to_control_area is None:
        raise ValueError("Could not obtain node_to_control_area from load_demand_data output.")

    thermal_out = call_with_supported_kwargs(load_thermal_power_plant_data, DATA_XLSX, "Thermal_Power_Data", "Thermal_Power_Specific_Data")
    phs_out = call_with_supported_kwargs(load_phs_power_plant_data, DATA_XLSX, "(P)HS_Power_Data", "(P)HS_Power_Specific_Data")
    res_out = call_with_supported_kwargs(load_renewable_power_plant_data, DATA_XLSX, "RES_Power_Data", "RES_Power_Specific_Data")
    flex_out = call_with_supported_kwargs(load_flexibility_data, DATA_XLSX, "Flexibility_Data", "Flexibility_Specific_Data")

    thermal_node_idx = next((obj for obj in thermal_out if isinstance(obj, dict)), None) if isinstance(thermal_out, tuple) else None
    phs_node_idx = next((obj for obj in phs_out if isinstance(obj, dict)), None) if isinstance(phs_out, tuple) else None
    renewable_node_idx = next((obj for obj in res_out if isinstance(obj, dict)), None) if isinstance(res_out, tuple) else None

    dsr_node_idx = None
    battery_node_idx = None
    if isinstance(flex_out, tuple):
        dicts = [obj for obj in flex_out if isinstance(obj, dict)]
        if len(dicts) >= 2:
            dsr_node_idx = dicts[-2]
            battery_node_idx = dicts[-1]

    if thermal_node_idx is None:
        raise ValueError("Could not obtain thermal_node_idx from load_thermal_power_plant_data.")
    if phs_node_idx is None:
        raise ValueError("Could not obtain phs_node_idx from load_phs_power_plant_data.")
    if renewable_node_idx is None:
        raise ValueError("Could not obtain renewable_node_idx from load_renewable_power_plant_data.")
    if dsr_node_idx is None or battery_node_idx is None:
        raise ValueError("Could not obtain dsr_node_idx / battery_node_idx from load_flexibility_data.")

    # exchange_data is no longer used for plotting, but you can keep it if your code requires it elsewhere.
    # We still call it here only if you want to ensure Parameters_Updated is “healthy”.
    _ = call_with_supported_kwargs(
        load_exchange_data,
        DATA_XLSX,
        "Exchange_Data",
        ptdf_csv_path=os.path.join(BASE_DIR, "PTDF_Synchronized.csv"),
        slack_node=None,
        verbose=False,
    )

    for inst in INSTRUMENTS:
        print("\n" + "=" * 70)
        print(f"INSTRUMENT CASE: {inst}")
        print("=" * 70)

        stage_df = read_results(stage1_instr_path(BASE_DIR, inst))
        pf_df = read_results(follower_instr_path(BASE_DIR, inst))

        plot_total_values_instrument_only(
            stage_df=stage_df,
            node_to_control_area=node_to_control_area,
            case_name=inst,
            control_area=CONTROL_AREA if NODE is None else None,
            node=NODE,
            pf_nodewise=pf_df,
            thermal_node_idx=thermal_node_idx,
            phs_node_idx=phs_node_idx,
            renewable_node_idx=renewable_node_idx,
            dsr_node_idx=dsr_node_idx,
            battery_node_idx=battery_node_idx,
            base_date=DEFAULT_BASE_DATE,
            start=START,
            end=END,
            demand_fraction=0.1,
            apply_demand_fraction_to_excel=True,
            verbose=VERBOSE,
        )

        plot_pf_import_export_and_prices(
            pf_nodewise=pf_df,
            node_to_control_area=node_to_control_area,
            case_name=inst,
            control_area=CONTROL_AREA if NODE is None else None,
            node=NODE,
            base_date=DEFAULT_BASE_DATE,
            start=START,
            end=END,
            show_flows=SHOW_PF_FLOWS,
            verbose=VERBOSE,
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
