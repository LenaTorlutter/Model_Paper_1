# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd

from pyomo.core import value
from pyomo.environ import *

# =============================================================================
# Utilities
# =============================================================================

def as_float(x) -> float:
    """Robustly cast scalars/1-element containers to float."""
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


def init_or_zero(initial_values: dict, start_week: int, varname: str, *idx, default: float = 0.0) -> float:
    """
    Fetch initial condition from `initial_values` if start_week>1 else 0.0.
    Keys expected like: (varname, *idx) or ('phs_storage', plant).
    """
    if start_week == 1:
        return 0.0
    return value(initial_values.get((varname, *idx), default))

# =============================================================================
# Follower PV + Battery + Demand (Stage 1)
# =============================================================================

def build_follower_model(
    general_data,
    demand_data,
    renewable_data,
    flexibility_data,
    pv_profile_data,
    initial_values,
    mins_per_hour,
    hours_per_week,
    start_week,
    weeks_per_step,
    *,
    # Imported from main:
    stage1_base_prices_df: pd.DataFrame,
    follower_control_area,
    pv_share: float,
    battery_share: float,
    instrument: str,
    base_case: bool,
    # Instrument knob (from main; for now default 70%)
    export_share: float,
    # Optional knobs:
    demand_fraction: float,
    fee_pv_feed: float,
    fee_battery_in: float,
    fee_battery_out: float,
    fee_imports_to_demand: float,
    # Capacity-based (power) tariff knobs (applies to BOTH imports and exports)
    peak_rate_eur_per_mw: float = 0.0,     # EUR per MW per billing horizon
    peak_imports_old: dict | None = None,  # dict[node -> MW] already realized peak IMPORTS in current billing horizon
    peak_exports_old: dict | None = None,  # dict[node -> MW] already realized peak EXPORTS in current billing horizon
    reset_billing: bool = False,           # if True: treat old peaks as zero inside this solve
    solver_name: str = "gurobi",
):
    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    allowed = {"BASE", "RTP", "Peak-Shaving", "Capacity-Tariff"}
    if instrument not in allowed:
        raise ValueError(f"instrument must be one of: {sorted(allowed)}")

    if stage1_base_prices_df is None or not isinstance(stage1_base_prices_df, pd.DataFrame) or stage1_base_prices_df.empty:
        raise ValueError("stage1_base_prices_df must be a non-empty pandas DataFrame.")
    if not {"node", "t", "price"}.issubset(stage1_base_prices_df.columns):
        raise ValueError("stage1_base_prices_df must contain columns: ['node', 't', 'price'] (optional: 'start_week').")
    if not (0.0 <= demand_fraction <= 1.0):
        raise ValueError("demand_fraction must be in [0, 1].")
    if not (0.0 < export_share <= 1.0):
        raise ValueError("export_share must be in (0, 1].")

    if peak_rate_eur_per_mw < 0.0:
        raise ValueError("peak_rate_eur_per_mw must be >= 0.0.")

    # Node-specific old peaks (dicts); reset if new billing horizon
    peak_imports_old = {} if peak_imports_old is None else {str(k): float(v) for k, v in peak_imports_old.items()}
    peak_exports_old = {} if peak_exports_old is None else {str(k): float(v) for k, v in peak_exports_old.items()}
    if reset_billing:
        peak_imports_old = {}
        peak_exports_old = {}

    # Use capacity tariff only in the correct instrument AND when rate > 0
    use_capacity_tariff = (instrument == "Capacity-Tariff") and (float(peak_rate_eur_per_mw) > 0.0)

    # -------------------------------------------------------------------------
    # Unpack + time
    # -------------------------------------------------------------------------
    _, _, wind_pv_cost, _, _, _ = general_data
    _, node_demand_data, _, nodes, node_to_control_area = demand_data
    ren_df, ren_nodes_idx = renewable_data
    _, battery_df, _, battery_nodes_idx = flexibility_data

    dt = mins_per_hour / 60.0
    if dt <= 0:
        raise ValueError("mins_per_hour must imply dt > 0.")
    T = int(hours_per_week * weeks_per_step)
    time_offset = int((start_week - 1) * hours_per_week)
    t0 = 0

    # -------------------------------------------------------------------------
    # Allowed control areas for follower scope
    # -------------------------------------------------------------------------
    fca = follower_control_area
    if fca in {None, "ALL", "*", ""}:
        allowed_cas = None  # means: all control areas
    elif isinstance(fca, (set, list, tuple)):
        allowed_cas = {str(x) for x in fca}
    else:
        allowed_cas = {str(fca)}

    # -------------------------------------------------------------------------
    # Map assets to nodes (inside follower_control_area)
    # -------------------------------------------------------------------------
    nodes_with_pv = {}
    for n, plant_ids in ren_nodes_idx.items():
        pv_list = []
        for p in plant_ids:
            if str(ren_df.loc[p, "Power Plant Type"]) != "PV":
                continue
            ca_p = str(ren_df.loc[p, "Control Area"])
            if (allowed_cas is None) or (ca_p in allowed_cas):
                pv_list.append(p)
        if pv_list:
            nodes_with_pv[n] = pv_list[0]

    nodes_with_battery = {}
    for n, b_ids in battery_nodes_idx.items():
        b_list = []
        for b in b_ids:
            ca_b = str(battery_df.loc[b].get("Control Area", node_to_control_area.get(str(n), "")))
            if (allowed_cas is None) or (ca_b in allowed_cas):
                b_list.append(b)
        if b_list:
            nodes_with_battery[n] = b_list[0]

    nodes_in_ca = sorted(set(nodes_with_pv) | set(nodes_with_battery))
    if not nodes_in_ca:
        raise ValueError(
            f"No PV or Battery units found in control area '{follower_control_area}'. "
            "Check your indices and dataframes."
        )

    # -------------------------------------------------------------------------
    # PV availability (MWh per timestep) + PV export cap (MWh per timestep)
    # -------------------------------------------------------------------------
    pv_availability = {(n, t): 0.0 for n in nodes_in_ca for t in range(1, T + 1)}
    pv_export_cap_dt = {n: 0.0 for n in nodes_in_ca}  # MWh per timestep (dt-scaled), node-specific

    for n, p in nodes_with_pv.items():
        ca = str(ren_df.loc[p, "Control Area"])
        cap_gw = float(ren_df.loc[p, "Installed Capacity [GW]"]) * float(pv_share)

        # installed PV export cap per timestep (MWh) = export_share * MW * dt
        pv_export_cap_dt[n] = float(export_share) * (cap_gw * 1000.0 * dt)

        for t in range(1, T + 1):
            prof = float(pv_profile_data.loc[t + time_offset, ca])
            pv_availability[(n, t)] = cap_gw * 1000.0 * prof * dt

    # -------------------------------------------------------------------------
    # Battery params per node + battery export cap (MWh per timestep)
    # -------------------------------------------------------------------------
    bat_eta_in = {n: 0.0 for n in nodes_in_ca}
    bat_eta_out = {n: 1.0 for n in nodes_in_ca}
    bat_out_cap_dt = {n: 0.0 for n in nodes_in_ca}
    bat_in_cap_dt = {n: 0.0 for n in nodes_in_ca}
    bat_e_max = {n: 0.0 for n in nodes_in_ca}
    bat_export_cap_dt = {n: 0.0 for n in nodes_in_ca}
    bat_import_cap_dt = {n: 0.0 for n in nodes_in_ca}

    for n, b in nodes_with_battery.items():
        bat_eta_in[n] = float(battery_df.loc[b, "Efficiency Pump"])
        bat_eta_out[n] = float(battery_df.loc[b, "Efficiency Turbine"])
        bat_out_cap_dt[n] = float(battery_df.loc[b, "Turbine Capacity [GW]"]) * float(battery_share) * 1000.0 * dt
        bat_in_cap_dt[n] = float(battery_df.loc[b, "Pump Capacity [GW]"]) * float(battery_share) * 1000.0 * dt
        bat_e_max[n] = float(battery_df.loc[b, "Energy (max) [GWh]"]) * float(battery_share) * 1000.0

        bat_export_cap_dt[n] = float(export_share) * float(bat_out_cap_dt[n])
        bat_import_cap_dt[n] = float(export_share) * float(bat_in_cap_dt[n])

    # -------------------------------------------------------------------------
    # Prices (node-specific) from Stage-1 DF
    # -------------------------------------------------------------------------
    dfp_all = stage1_base_prices_df.copy()
    dfp_all["price"] = pd.to_numeric(dfp_all["price"], errors="coerce")
    if dfp_all["price"].isna().all():
        raise ValueError("stage1_base_prices_df: price column is all NaN.")

    '''
    if (dfp_all["price"] <= 0).all():
        dfp_all["price"] = -dfp_all["price"]
    elif (dfp_all["price"] >= 0).all():
        pass
    else:
        bad = dfp_all[dfp_all["price"] < 0].head()
        raise ValueError(
            "Mixed positive and negative prices detected in stage1_base_prices_df. "
            f"Examples:\n{bad}"
        )
    '''

    if "start_week" in dfp_all.columns:
        dfp = dfp_all.query("node in @nodes_in_ca and start_week == @start_week")[["t", "price", "node"]]
    else:
        dfp = dfp_all.query("node in @nodes_in_ca")[["t", "price", "node"]]

    if dfp.empty:
        raise ValueError(
            f"No node-level prices found for control area '{follower_control_area}' "
            f"(start_week={start_week} if provided). Expected nodes like: {nodes_in_ca[:5]}..."
        )

    price = {(str(r.node), int(r.t)): float(r.price) for _, r in dfp.iterrows()}

    for t in range(1, T + 1):
        for n in nodes_in_ca:
            if (str(n), t) not in price:
                raise ValueError(f"Missing price for node={n}, hour t={t} (start_week={start_week}).")

    # =========================================================================
    # Pyomo model
    # =========================================================================
    m = ConcreteModel()

    # -------------------------------------------------------------------------
    # Sets
    # -------------------------------------------------------------------------
    m.nodes = Set(initialize=nodes_in_ca)
    m.timesteps = Set(initialize=range(0, T + 1))

    # -------------------------------------------------------------------------
    # Params
    # -------------------------------------------------------------------------
    m.dt = Param(initialize=float(dt), mutable=False)

    def pv_availability_init(_m, n, t):
        return 0.0 if t == t0 else float(pv_availability[(n, t)])

    m.pv_availability = Param(m.nodes, m.timesteps, initialize=pv_availability_init, mutable=False)
    m.pv_export_cap_dt = Param(m.nodes, initialize=lambda _m, n: float(pv_export_cap_dt.get(n, 0.0)), mutable=False)

    m.bat_eta_in = Param(m.nodes, initialize=lambda _m, n: float(bat_eta_in[n]), mutable=False)
    m.bat_eta_out = Param(m.nodes, initialize=lambda _m, n: float(bat_eta_out[n]), mutable=False)
    m.bat_out_cap_dt = Param(m.nodes, initialize=lambda _m, n: float(bat_out_cap_dt[n]), mutable=False)
    m.bat_in_cap_dt = Param(m.nodes, initialize=lambda _m, n: float(bat_in_cap_dt[n]), mutable=False)
    m.bat_e_max = Param(m.nodes, initialize=lambda _m, n: float(bat_e_max[n]), mutable=False)
    m.bat_export_cap_dt = Param(m.nodes, initialize=lambda _m, n: float(bat_export_cap_dt.get(n, 0.0)), mutable=False)
    m.bat_import_cap_dt = Param(m.nodes, initialize=lambda _m, n: float(bat_import_cap_dt.get(n, 0.0)), mutable=False)

    def local_demand_init(_m, n, t):
        if t == t0:
            return 0.0
        base = float(node_demand_data.loc[t + time_offset, n])
        return float(demand_fraction) * base

    m.local_demand = Param(m.nodes, m.timesteps, initialize=local_demand_init, mutable=False)
    m.pv_availability_cost = Param(initialize=float(wind_pv_cost), mutable=False)

    # Capacity-based tariff params (node-specific old peaks)
    m.peak_rate_eur_per_mw = Param(initialize=float(peak_rate_eur_per_mw), mutable=False)
    m.peak_imports_old = Param(
        m.nodes,
        initialize=lambda _m, n: float(peak_imports_old.get(str(n), 0.0)),
        mutable=False,
    )
    m.peak_exports_old = Param(
        m.nodes,
        initialize=lambda _m, n: float(peak_exports_old.get(str(n), 0.0)),
        mutable=False,
    )

    # -------------------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------------------
    m.pv_to_demand = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.pv_to_battery = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.pv_feed_to_system = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.pv_curtailment = Var(m.nodes, m.timesteps, within=NonNegativeReals)

    m.imports_to_demand = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.imports_to_battery = Var(m.nodes, m.timesteps, within=NonNegativeReals)

    m.battery_to_demand = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.battery_to_system = Var(m.nodes, m.timesteps, within=NonNegativeReals)
    m.battery_storage = Var(m.nodes, m.timesteps, within=NonNegativeReals)

    # Capacity-based tariff peak variables (MW), node-specific
    m.peak_imports = Var(m.nodes, within=NonNegativeReals)
    m.peak_imports_new = Var(m.nodes, within=NonNegativeReals)

    m.peak_exports = Var(m.nodes, within=NonNegativeReals)
    m.peak_exports_new = Var(m.nodes, within=NonNegativeReals)

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------
    def battery_in_expr(_m, n, t):
        return 0.0 if t == t0 else (_m.pv_to_battery[n, t] + _m.imports_to_battery[n, t])

    def battery_out_expr(_m, n, t):
        return 0.0 if t == t0 else (_m.battery_to_demand[n, t] + _m.battery_to_system[n, t])

    m.battery_in = Expression(m.nodes, m.timesteps, rule=battery_in_expr)
    m.battery_out = Expression(m.nodes, m.timesteps, rule=battery_out_expr)

    # Imports/exports power (MW) derived from energy variables (MWh per timestep)
    def imports_power_expr(_m, n, t):
        if t == t0:
            return 0.0
        imports_energy = _m.imports_to_demand[n, t] + _m.imports_to_battery[n, t]
        return imports_energy / _m.dt

    def exports_power_expr(_m, n, t):
        if t == t0:
            return 0.0
        exports_energy = _m.pv_feed_to_system[n, t] + _m.battery_to_system[n, t]
        return exports_energy / _m.dt

    m.imports_power = Expression(m.nodes, m.timesteps, rule=imports_power_expr)
    m.exports_power = Expression(m.nodes, m.timesteps, rule=exports_power_expr)

    # -------------------------------------------------------------------------
    # Initial storage (t=0)
    # -------------------------------------------------------------------------
    def battery_initial_storage_rule(_m, n):
        if start_week == 1:
            return _m.battery_storage[n, t0] == 0.0
        return _m.battery_storage[n, t0] == float(value(initial_values.get(("battery_storage", n), 0.0)))

    m.battery_initial_storage_constraint = Constraint(m.nodes, rule=battery_initial_storage_rule)

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    def pv_split_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.pv_availability[n, t] == (
            _m.pv_to_demand[n, t]
            + _m.pv_to_battery[n, t]
            + _m.pv_feed_to_system[n, t]
            + _m.pv_curtailment[n, t]
        )

    def pv_export_limit_rule(_m, n, t):
        if t == t0 or instrument != "Peak-Shaving":
            return Constraint.Skip
        return _m.pv_feed_to_system[n, t] <= _m.pv_export_cap_dt[n]

    def battery_out_cap_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.battery_out[n, t] <= _m.bat_out_cap_dt[n]

    def battery_export_limit_rule(_m, n, t):
        if t == t0 or instrument != "Peak-Shaving":
            return Constraint.Skip
        return _m.battery_to_system[n, t] <= _m.bat_export_cap_dt[n]

    def battery_in_cap_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.battery_in[n, t] <= _m.bat_in_cap_dt[n]

    def battery_import_limit_rule(_m, n, t):
        if t == t0 or instrument != "Peak-Shaving":
            return Constraint.Skip
        return _m.imports_to_battery[n, t] <= _m.bat_import_cap_dt[n]

    def battery_e_cap_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.battery_storage[n, t] <= _m.bat_e_max[n]

    def battery_dyn_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.battery_storage[n, t] == (
            _m.battery_storage[n, t - 1]
            + _m.bat_eta_in[n] * _m.battery_in[n, t]
            - (1.0 / _m.bat_eta_out[n]) * _m.battery_out[n, t]
        )

    def local_demand_balance_rule(_m, n, t):
        if t == t0:
            return Constraint.Skip
        return _m.local_demand[n, t] == (
            _m.pv_to_demand[n, t]
            + _m.battery_to_demand[n, t]
            + _m.imports_to_demand[n, t]
        )

    m.pv_split_constraint = Constraint(m.nodes, m.timesteps, rule=pv_split_rule)
    m.pv_export_limit_constraint = Constraint(m.nodes, m.timesteps, rule=pv_export_limit_rule)

    m.battery_out_cap_constraint = Constraint(m.nodes, m.timesteps, rule=battery_out_cap_rule)
    m.battery_export_limit_constraint = Constraint(m.nodes, m.timesteps, rule=battery_export_limit_rule)

    m.battery_in_cap_constraint = Constraint(m.nodes, m.timesteps, rule=battery_in_cap_rule)
    m.battery_import_limit_constraint = Constraint(m.nodes, m.timesteps, rule=battery_import_limit_rule)

    m.battery_e_cap_constraint = Constraint(m.nodes, m.timesteps, rule=battery_e_cap_rule)
    m.battery_dyn_constraint = Constraint(m.nodes, m.timesteps, rule=battery_dyn_rule)
    m.local_demand_balance_constraint = Constraint(m.nodes, m.timesteps, rule=local_demand_balance_rule)

    # -------------------------------------------------------------------------
    # Capacity-based tariff constraints (only if use_capacity_tariff)
    # -------------------------------------------------------------------------
    def peak_imports_lower_old_rule(_m, n):
        if not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_imports[n] >= _m.peak_imports_old[n]

    def peak_imports_cover_rule(_m, n, t):
        if t == t0 or not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_imports[n] >= _m.imports_power[n, t]

    def peak_imports_new_rule(_m, n):
        if not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_imports_new[n] >= _m.peak_imports[n] - _m.peak_imports_old[n]

    def peak_exports_lower_old_rule(_m, n):
        if not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_exports[n] >= _m.peak_exports_old[n]

    def peak_exports_cover_rule(_m, n, t):
        if t == t0 or not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_exports[n] >= _m.exports_power[n, t]

    def peak_exports_new_rule(_m, n):
        if not use_capacity_tariff:
            return Constraint.Skip
        return _m.peak_exports_new[n] >= _m.peak_exports[n] - _m.peak_exports_old[n]

    m.peak_imports_old_constraint = Constraint(m.nodes, rule=peak_imports_lower_old_rule)
    m.peak_imports_cover_constraint = Constraint(m.nodes, m.timesteps, rule=peak_imports_cover_rule)
    m.peak_imports_new_constraint = Constraint(m.nodes, rule=peak_imports_new_rule)

    m.peak_exports_old_constraint = Constraint(m.nodes, rule=peak_exports_lower_old_rule)
    m.peak_exports_cover_constraint = Constraint(m.nodes, m.timesteps, rule=peak_exports_cover_rule)
    m.peak_exports_new_constraint = Constraint(m.nodes, rule=peak_exports_new_rule)

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------
    def objective_rule(_m):
        revenue = sum(
            price[(str(n), int(t))] * (_m.pv_feed_to_system[n, t] + _m.battery_to_system[n, t])
            for n in _m.nodes
            for t in _m.timesteps
            if t != t0
        )

        pv_used_cost = sum(
            _m.pv_availability_cost * (_m.pv_availability[n, t] - _m.pv_curtailment[n, t])
            for n in _m.nodes
            for t in _m.timesteps
            if t != t0
        )

        import_energy_costs = sum(
            price[(str(n), int(t))] * (_m.imports_to_demand[n, t] + _m.imports_to_battery[n, t])
            for n in _m.nodes
            for t in _m.timesteps
            if t != t0
        )

        fees = (
            float(fee_pv_feed) * sum(_m.pv_feed_to_system[n, t] for n in _m.nodes for t in _m.timesteps if t != t0)
            + float(fee_battery_in) * sum(_m.imports_to_battery[n, t] for n in _m.nodes for t in _m.timesteps if t != t0)
            + float(fee_battery_out) * sum(_m.battery_to_system[n, t] for n in _m.nodes for t in _m.timesteps if t != t0)
            + float(fee_imports_to_demand) * sum(_m.imports_to_demand[n, t] for n in _m.nodes for t in _m.timesteps if t != t0)
        )

        power_charge = 0.0
        if use_capacity_tariff:
            rate = float(value(_m.peak_rate_eur_per_mw))
            power_charge = rate * (
                sum(_m.peak_imports_new[n] for n in _m.nodes) + sum(_m.peak_exports_new[n] for n in _m.nodes)
            )

        return revenue - pv_used_cost - import_energy_costs - fees - power_charge

    m.obj = Objective(rule=objective_rule, sense=maximize)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    SolverFactory(solver_name).solve(m)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    rows = []
    pv_cost_scalar = float(value(m.pv_availability_cost))
    peak_rate_val = float(value(m.peak_rate_eur_per_mw))

    for t in range(1, T + 1):
        for n in nodes_in_ca:
            price_t = float(price[(str(n), int(t))])

            pv_av = float(value(m.pv_availability[n, t]))
            pv_to_demand = float(value(m.pv_to_demand[n, t]))
            pv_to_battery = float(value(m.pv_to_battery[n, t]))
            pv_feed = float(value(m.pv_feed_to_system[n, t]))
            pv_curt = float(value(m.pv_curtailment[n, t]))

            imp_dem = float(value(m.imports_to_demand[n, t]))
            imp_bat = float(value(m.imports_to_battery[n, t]))

            bat_in = float(value(m.battery_in[n, t]))
            bat_to_dem = float(value(m.battery_to_demand[n, t]))
            bat_to_sys = float(value(m.battery_to_system[n, t]))
            bat_out = float(value(m.battery_out[n, t]))
            bat_store = float(value(m.battery_storage[n, t]))

            local_dem = float(value(m.local_demand[n, t]))

            revenue = price_t * (pv_feed + bat_to_sys)
            pv_used_cost = pv_cost_scalar * pv_av
            import_energy_costs = price_t * (imp_dem + imp_bat)

            fees_val = (
                float(fee_pv_feed) * pv_feed
                + float(fee_battery_in) * imp_bat
                + float(fee_battery_out) * bat_to_sys
                + float(fee_imports_to_demand) * imp_dem
            )

            imports_power_nt = float(value(m.imports_power[n, t]))
            exports_power_nt = float(value(m.exports_power[n, t]))

            if use_capacity_tariff:
                peak_imports_old_n = float(value(m.peak_imports_old[n]))
                peak_imports_n = float(value(m.peak_imports[n]))
                peak_imports_new_n = float(value(m.peak_imports_new[n]))

                peak_exports_old_n = float(value(m.peak_exports_old[n]))
                peak_exports_n = float(value(m.peak_exports[n]))
                peak_exports_new_n = float(value(m.peak_exports_new[n]))

                power_charge_n = peak_rate_val * (peak_imports_new_n + peak_exports_new_n)
            else:
                peak_imports_old_n = 0.0
                peak_imports_n = 0.0
                peak_imports_new_n = 0.0
                peak_exports_old_n = 0.0
                peak_exports_n = 0.0
                peak_exports_new_n = 0.0
                power_charge_n = 0.0

            # Node-level accounting (power charge is node-specific here by construction)
            objective = revenue - pv_used_cost - import_energy_costs - fees_val - power_charge_n

            rows.append({
                "node": str(n),
                "start_week": int(start_week),
                "t": int(t),
                "instrument": str(instrument),
                "base_case": bool(base_case),

                "export_share": float(export_share),
                "pv_export_cap_dt": float(value(m.pv_export_cap_dt[n])),
                "bat_export_cap_dt": float(value(m.bat_export_cap_dt[n])),
                "bat_import_cap_dt": float(value(m.bat_import_cap_dt[n])),

                "price": price_t,
                "local_demand": local_dem,

                "pv_availability": pv_av,
                "pv_to_demand": pv_to_demand,
                "pv_to_battery": pv_to_battery,
                "pv_feed_to_system": pv_feed,
                "pv_curtailment": pv_curt,

                "imports_to_demand": imp_dem,
                "imports_to_battery": imp_bat,

                "battery_in": bat_in,
                "battery_to_demand": bat_to_dem,
                "battery_to_system": bat_to_sys,
                "battery_out": bat_out,
                "battery_storage": bat_store,

                # Power telemetry (MW)
                "imports_power": imports_power_nt,
                "exports_power": exports_power_nt,

                # Capacity tariff telemetry (node-specific)
                "peak_rate_eur_per_mw": peak_rate_val if use_capacity_tariff else 0.0,

                "peak_imports_old": peak_imports_old_n,
                "peak_imports": peak_imports_n,
                "peak_imports_new": peak_imports_new_n,

                "peak_exports_old": peak_exports_old_n,
                "peak_exports": peak_exports_n,
                "peak_exports_new": peak_exports_new_n,

                "power_charge": power_charge_n,

                "revenue": revenue,
                "pv_used_cost": pv_used_cost,
                "import_energy_costs": import_energy_costs,
                "fees": fees_val,
                "objective": objective,
            })

    results_df = pd.DataFrame(rows)
    return m, results_df



# =============================================================================
# Model
# =============================================================================

def create_model(
    general_data,
    demand_data,
    thermal_data,
    phs_data,
    phs_inflow,
    hs_inflow,
    phs_storage_profile_data,
    renewable_data,
    ror_profile_data,
    windon_profile_data,
    windoff_profile_data,
    pv_profile_data,
    flexibility_data,
    exchange_data,
    initial_values,
    mins_per_hour,
    hours_per_week,
    start_week,
    weeks_per_step,
    *,
    pv_share: float,
    battery_share: float,
    demand_fraction: float,
    follower_control_area="AT",  # can be str OR set/list/tuple of strings OR "ALL"/"*"/None
    follower_flows_df: pd.DataFrame | None = None,
):

    # =========================================================================
    # Unpack Inputs
    # =========================================================================

    co2_price, voll, wind_pv_cost, phs_cost, dsr_cost, flow_cost = general_data

    _, node_demand_data, control_areas, nodes, _ = demand_data

    thermal_df, th_nodes_idx = thermal_data

    phs_df, phs_ca_idx, phs_nodes_idx = phs_data
    _, phs_inflow_hourly, _ = phs_inflow
    _, hs_inflow_hourly, _ = hs_inflow
    _, phs_storage_profile = phs_storage_profile_data

    ren_df, ren_nodes_idx = renewable_data

    dsr_df, bat_df, dsr_nodes_idx, bat_nodes_idx = flexibility_data

    _, connections, incidence_matrix, capacity_pos, capacity_neg, xborder, conductance_series, ptdf_results = exchange_data
    
    ptdf_df = ptdf_results["PTDF"]
    slack_node = ptdf_results["slack_node"]  # kept for completeness
    non_slack_nodes = list(ptdf_df.columns)

    # =========================================================================
    # Time Setup
    # =========================================================================

    dt = mins_per_hour / 60.0
    T = int(hours_per_week * weeks_per_step)
    time_offset = int((start_week - 1) * hours_per_week)
    t0 = 0

    # =========================================================================
    # Normalize follower_control_area to a set for membership checks
    #   - fca_set is None => "ALL control areas"
    # =========================================================================
    fca = follower_control_area
    if fca in (None, "ALL", "*", ""):
        fca_set = None
    elif isinstance(fca, (set, list, tuple)):
        fca_set = {str(x) for x in fca}
    else:
        fca_set = {str(fca)}


    # =========================================================================
    # Model Definition
    # =========================================================================

    model = ConcreteModel()

    # =========================================================================
    # Sets
    # =========================================================================

    # --- Time ---
    model.timesteps = Set(initialize=range(0, T + 1))

    # --- Geography ---
    model.control_areas = Set(initialize=control_areas)
    model.nodes = Set(initialize=nodes)

    # --- Thermal ---
    model.thermal_power_plants = Set(initialize=thermal_df.index.tolist())
    model.thermal_power_plant_nodes = Set(initialize=th_nodes_idx.keys())
    model.thermal_power_plants_by_node = Set(
        model.thermal_power_plant_nodes,
        initialize=lambda m, node: th_nodes_idx[node],
    )

    # --- PHS / HS ---
    model.phs_power_plants = Set(initialize=phs_df.index.tolist())
    model.phs_power_plant_control_areas = Set(initialize=phs_ca_idx.keys())
    model.phs_power_plants_by_control_area = Set(
        model.phs_power_plant_control_areas,
        initialize=lambda m, ca: phs_ca_idx[ca],
    )
    model.phs_power_plant_nodes = Set(initialize=phs_nodes_idx.keys())
    model.phs_power_plants_by_node = Set(
        model.phs_power_plant_nodes,
        initialize=lambda m, node: phs_nodes_idx[node],
    )

    # --- Renewables ---
    model.renewable_power_plants = Set(initialize=ren_df.index.tolist())
    model.renewable_power_plant_nodes = Set(initialize=ren_nodes_idx.keys())
    model.renewable_power_plants_by_node = Set(
        model.renewable_power_plant_nodes,
        initialize=lambda m, node: ren_nodes_idx[node],
    )

    # --- Flexibility ---
    model.dsr_units = Set(initialize=dsr_df.index.tolist())
    model.dsr_nodes = Set(initialize=dsr_nodes_idx.keys())
    model.dsr_by_node = Set(model.dsr_nodes, initialize=lambda m, node: dsr_nodes_idx[node])

    model.battery_units = Set(initialize=bat_df.index.tolist())
    model.battery_nodes = Set(initialize=bat_nodes_idx.keys())
    model.battery_by_node = Set(model.battery_nodes, initialize=lambda m, node: bat_nodes_idx[node])

    # --- Network ---
    model.connections = Set(initialize=connections)
    model.non_slack_nodes = Set(initialize=non_slack_nodes)

    # =========================================================================
    # Parameters
    # =========================================================================

    # --- Scalars (keep as Python floats; but could be Params if you prefer) ---
    # co2_price, voll, wind_pv_cost, phs_cost, dsr_cost, flow_cost already unpacked

    # --- Network ---
    model.capacity_pos = Param(model.connections, initialize=capacity_pos, mutable=False)
    model.capacity_neg = Param(model.connections, initialize=capacity_neg, mutable=False)
    model.xborder = Param(model.connections, initialize=xborder, within=Binary, mutable=False)
    
    model.PTDF = Param(
        model.connections,
        model.non_slack_nodes,
        initialize=lambda m, c, n: float(ptdf_df.loc[c, n]),
        mutable=False,
    )
    
    # --- NSE ---
    model.voll = Param(model.nodes, initialize={n: float(voll) for n in nodes}, mutable=False)

    # =========================================================================
    # Variables + Derived Expressions
    # =========================================================================

    # -------------------------------------------------------------------------
    # Thermal
    # -------------------------------------------------------------------------

    model.thermal_generation = Var(model.thermal_power_plants, model.timesteps, within=NonNegativeReals)
    model.on_1 = Var(model.thermal_power_plants, model.timesteps, bounds=(0, 1))
    model.on_2 = Var(model.thermal_power_plants, model.timesteps, bounds=(0, 1))
    model.start_thermal_generation = Var(model.thermal_power_plants, model.timesteps, within=NonNegativeReals)

    def total_thermal_expr(m, n, t):
        if t == t0 or n not in m.thermal_power_plant_nodes:
            return 0.0
        return sum(m.thermal_generation[p, t] for p in m.thermal_power_plants_by_node[n])

    model.total_thermal_generation_per_node = Expression(model.nodes, model.timesteps, rule=total_thermal_expr)

    def thermal_mc_rule(m, p):
        eff = as_float(thermal_df.loc[p, "Efficiency"])
        pe = as_float(thermal_df.loc[p, "Primary Energy Price [€/MWh]"])
        em = as_float(thermal_df.loc[p, "Emissions [t/MWh]"])
        nfo = as_float(thermal_df.loc[p, "Non-Fuel O&M Cost [€/MWh]"])
        return (pe / eff) + (em * float(co2_price) / eff) + nfo

    model.thermal_generation_cost = Expression(model.thermal_power_plants, rule=thermal_mc_rule)
    model.start_thermal_generation_cost = Expression(
        model.thermal_power_plants,
        rule=lambda m, p: as_float(thermal_df.loc[p, "Start-Up Cost [€]"]),
    )

    # -------------------------------------------------------------------------
    # PHS / HS
    # -------------------------------------------------------------------------

    model.phs_turbine_generation = Var(model.phs_power_plants, model.timesteps, within=NonNegativeReals)
    model.phs_pump_consumption = Var(model.phs_power_plants, model.timesteps, within=NonNegativeReals)
    model.phs_storage = Var(model.phs_power_plants, model.timesteps, within=NonNegativeReals)
    model.phs_spill = Var(model.phs_power_plants, model.timesteps, within=NonNegativeReals)

    def total_phs_gen_expr(m, n, t):
        if t == t0 or n not in m.phs_power_plant_nodes:
            return 0.0
        return sum(m.phs_turbine_generation[p, t] for p in m.phs_power_plants_by_node[n])

    def total_phs_cons_expr(m, n, t):
        if t == t0 or n not in m.phs_power_plant_nodes:
            return 0.0
        return sum(m.phs_pump_consumption[p, t] for p in m.phs_power_plants_by_node[n])

    model.total_phs_generation_per_node = Expression(model.nodes, model.timesteps, rule=total_phs_gen_expr)
    model.total_phs_consumption_per_node = Expression(model.nodes, model.timesteps, rule=total_phs_cons_expr)

    model.phs_generation_cost = Expression(model.phs_power_plants, rule=lambda m, p: float(phs_cost))
    model.phs_spill_cost = Expression(model.phs_power_plants, rule=lambda m, p: float(voll))

    # -------------------------------------------------------------------------
    # Renewables
    # -------------------------------------------------------------------------

    model.renewable_spill = Var(model.renewable_power_plants, model.timesteps, within=NonNegativeReals)
    model.renewable_generation = Var(model.renewable_power_plants, model.timesteps, within=NonNegativeReals)

    def total_res_expr(m, n, t):
        if t == t0 or n not in m.renewable_power_plant_nodes:
            return 0.0
        return sum(m.renewable_generation[p, t] for p in m.renewable_power_plants_by_node[n])

    model.total_renewable_generation_per_node = Expression(model.nodes, model.timesteps, rule=total_res_expr)

    def ren_cost_rule(m, p):
        typ = str(ren_df.loc[p, "Power Plant Type"])
        if typ == "RoR":
            return float(phs_cost) / 100.0
        if typ in {"WindOn", "WindOff", "PV"}:
            return float(wind_pv_cost)
        raise ValueError(f"Unknown renewable type '{typ}' for plant '{p}'")

    model.renewable_generation_cost = Expression(model.renewable_power_plants, rule=ren_cost_rule)

    # -------------------------------------------------------------------------
    # Flexibility: DSR + Batteries
    # -------------------------------------------------------------------------

    model.dsr_storage = Var(model.dsr_units, model.timesteps, within=Reals)
    model.dsr_down = Var(model.dsr_units, model.timesteps, within=NonNegativeReals)
    model.dsr_up = Var(model.dsr_units, model.timesteps, within=NonNegativeReals)
    model.dsr_down_cost = Expression(model.dsr_units, rule=lambda m, d: float(dsr_cost))

    model.battery_storage = Var(model.battery_units, model.timesteps, within=NonNegativeReals)
    model.battery_out = Var(model.battery_units, model.timesteps, within=NonNegativeReals)
    model.battery_in = Var(model.battery_units, model.timesteps, within=NonNegativeReals)

    def total_dsr_down_expr(m, n, t):
        if t == t0 or n not in m.dsr_nodes:
            return 0.0
        return sum(m.dsr_down[d, t] for d in m.dsr_by_node[n])

    def total_dsr_up_expr(m, n, t):
        if t == t0 or n not in m.dsr_nodes:
            return 0.0
        return sum(m.dsr_up[d, t] for d in m.dsr_by_node[n])

    def total_bat_out_expr(m, n, t):
        if t == t0 or n not in m.battery_nodes:
            return 0.0
        return sum(m.battery_out[b, t] for b in m.battery_by_node[n])

    def total_bat_in_expr(m, n, t):
        if t == t0 or n not in m.battery_nodes:
            return 0.0
        return sum(m.battery_in[b, t] for b in m.battery_by_node[n])

    model.total_dsr_down_per_node = Expression(model.nodes, model.timesteps, rule=total_dsr_down_expr)
    model.total_dsr_up_per_node = Expression(model.nodes, model.timesteps, rule=total_dsr_up_expr)
    model.total_battery_out_per_node = Expression(model.nodes, model.timesteps, rule=total_bat_out_expr)
    model.total_battery_in_per_node = Expression(model.nodes, model.timesteps, rule=total_bat_in_expr)

    # -------------------------------------------------------------------------
    # Network + NSE + Demand (as Expression because it uses follower flows)
    # -------------------------------------------------------------------------

    model.flow = Var(model.connections, model.timesteps, within=Reals)
    model.flow_pos = Var(model.connections, model.timesteps, domain=NonNegativeReals)
    model.flow_neg = Var(model.connections, model.timesteps, domain=NonNegativeReals)
    model.exchange = Var(model.nodes, model.timesteps, within=Reals)
    model.curtailment = Var(model.nodes, model.timesteps, within=NonNegativeReals)

    model.flow_cost = Expression(model.connections, rule=lambda m, c: float(flow_cost))
    
    model.nse = Var(model.nodes, model.timesteps, within=NonNegativeReals)

    # =========================================================================
    # Initial Conditions (t = 0)
    # =========================================================================

    # --- Thermal at t0 ---
    model.init_thermal_generation = Constraint(
        model.thermal_power_plants,
        rule=lambda m, p: m.thermal_generation[p, t0] == init_or_zero(initial_values, start_week, "thermal_generation", p),
    )
    model.init_on_1 = Constraint(
        model.thermal_power_plants,
        rule=lambda m, p: m.on_1[p, t0] == init_or_zero(initial_values, start_week, "on_1", p),
    )
    model.init_on_2 = Constraint(
        model.thermal_power_plants,
        rule=lambda m, p: m.on_2[p, t0] == init_or_zero(initial_values, start_week, "on_2", p),
    )
    model.init_start_thermal = Constraint(
        model.thermal_power_plants,
        rule=lambda m, p: m.start_thermal_generation[p, t0] == init_or_zero(initial_values, start_week, "start_thermal_generation", p),
    )

    # --- PHS / HS at t0 ---
    def phs_initial_storage_rule(m, p):
        if start_week == 1:
            ca = phs_df.loc[p, "Control Area"]
            initial_share = float(phs_storage_profile.loc[1, ca])
            return m.phs_storage[p, t0] == float(phs_df.loc[p, "Energy (max) [GWh]"]) * 1000.0 * initial_share
        return m.phs_storage[p, t0] == value(initial_values.get(("phs_storage", p), 0.0))

    model.phs_initial_storage_constraint = Constraint(model.phs_power_plants, rule=phs_initial_storage_rule)
    model.init_phs_prod = Constraint(
        model.phs_power_plants,
        rule=lambda m, p: m.phs_turbine_generation[p, t0] == init_or_zero(initial_values, start_week, "phs_turbine_generation", p),
    )
    model.init_phs_pump = Constraint(
        model.phs_power_plants,
        rule=lambda m, p: m.phs_pump_consumption[p, t0] == init_or_zero(initial_values, start_week, "phs_pump_consumption", p),
    )
    model.init_phs_spill = Constraint(
        model.phs_power_plants,
        rule=lambda m, p: m.phs_spill[p, t0] == init_or_zero(initial_values, start_week, "phs_spill", p),
    )

    # --- Renewables at t0 ---
    model.init_renewable_spill = Constraint(
        model.renewable_power_plants,
        rule=lambda m, p: m.renewable_spill[p, t0] == init_or_zero(initial_values, start_week, "renewable_spill", p),
    )
    model.init_renewable_prod = Constraint(
        model.renewable_power_plants,
        rule=lambda m, p: m.renewable_generation[p, t0] == init_or_zero(initial_values, start_week, "renewable_generation", p),
    )

    # --- Battery at t0 (adapted for multi-CA follower_control_area) ---
    def battery_initial_storage_rule(m, b):
        if start_week == 1:
            initial_share = 0.5
            emax_gwh = float(bat_df.loc[b, "Energy (max) [GWh]"])

            ca_b = str(bat_df.loc[b, "Control Area"]) if "Control Area" in bat_df.columns else ""
            in_pf_ca = True if (fca_set is None) else (ca_b in fca_set)

            scale = (1.0 - battery_share) if in_pf_ca else 1.0
            return m.battery_storage[b, 0] == emax_gwh * scale * 1000.0 * initial_share

        return m.battery_storage[b, 0] == value(initial_values.get(("battery_storage", b), 0.0))

    model.battery_initial_storage_constraint = Constraint(model.battery_units, rule=battery_initial_storage_rule)
    model.init_battery_out = Constraint(
        model.battery_units,
        rule=lambda m, b: m.battery_out[b, t0] == init_or_zero(initial_values, start_week, "battery_out", b),
    )
    model.init_battery_in = Constraint(
        model.battery_units,
        rule=lambda m, b: m.battery_in[b, t0] == init_or_zero(initial_values, start_week, "battery_in", b),
    )

    # --- Network & NSE at t0 ---
    model.init_flow = Constraint(
        model.connections,
        rule=lambda m, c: m.flow[c, t0] == init_or_zero(initial_values, start_week, "flow", c),
    )
    model.init_nse = Constraint(
        model.nodes,
        rule=lambda m, n: m.nse[n, t0] == init_or_zero(initial_values, start_week, "nse", n),
    )

    # =========================================================================
    # Constraints
    # =========================================================================

    # -------------------------------------------------------------------------
    # Thermal Constraints
    # -------------------------------------------------------------------------

    def thermal_generation_capacity_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        cap_gw = as_float(thermal_df.loc[p, "Installed Capacity [GW]"])
        minload = as_float(thermal_df.loc[p, "Minimal Possible Load [-]"])
        avail = as_float(thermal_df.loc[p, "Availability (Forced Outage)"])
        return (
            m.thermal_generation[p, t]
            - (cap_gw * minload * m.on_1[p, t] + cap_gw * (1 - minload) * m.on_2[p, t]) * 1000.0 * dt * avail
            == 0
        )

    model.thermal_generation_capacity_constraint = Constraint(
        model.thermal_power_plants, model.timesteps, rule=thermal_generation_capacity_rule
    )

    def auxiliary_variables_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        return m.on_1[p, t] - m.on_2[p, t] >= 0

    model.auxiliary_variables_constraint = Constraint(
        model.thermal_power_plants, model.timesteps, rule=auxiliary_variables_rule
    )

    def start_thermal_generation_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        return m.start_thermal_generation[p, t] - m.on_1[p, t] + m.on_1[p, t - 1] >= 0

    model.start_thermal_generation_constraint = Constraint(
        model.thermal_power_plants, model.timesteps, rule=start_thermal_generation_rule
    )

    def ramp_up_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        cap_gw = as_float(thermal_df.loc[p, "Installed Capacity [GW]"])
        grad = as_float(thermal_df.loc[p, "Load Gradient [-/min]"])
        return m.thermal_generation[p, t] - m.thermal_generation[p, t - 1] <= cap_gw * grad * 1000.0 * mins_per_hour

    def ramp_down_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        cap_gw = as_float(thermal_df.loc[p, "Installed Capacity [GW]"])
        grad = as_float(thermal_df.loc[p, "Load Gradient [-/min]"])
        return m.thermal_generation[p, t - 1] - m.thermal_generation[p, t] <= cap_gw * grad * 1000.0 * mins_per_hour

    model.load_gradient_thermal_generation_constraint_1 = Constraint(
        model.thermal_power_plants, model.timesteps, rule=ramp_up_rule
    )
    model.load_gradient_thermal_generation_constraint_2 = Constraint(
        model.thermal_power_plants, model.timesteps, rule=ramp_down_rule
    )

    # -------------------------------------------------------------------------
    # PHS / HS Constraints
    # -------------------------------------------------------------------------

    def phs_turbine_cap_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        cap = as_float(phs_df.loc[p, "Turbine Capacity [GW]"]) * 1000.0 * dt
        return m.phs_turbine_generation[p, t] <= cap

    def phs_pump_cap_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        cap = as_float(phs_df.loc[p, "Pump Capacity [GW]"]) * 1000.0 * dt
        return m.phs_pump_consumption[p, t] <= cap

    model.phs_turbine_capacity_limit_constraint = Constraint(
        model.phs_power_plants, model.timesteps, rule=phs_turbine_cap_rule
    )
    model.phs_pump_capacity_limit_constraint = Constraint(
        model.phs_power_plants, model.timesteps, rule=phs_pump_cap_rule
    )

    def phs_storage_dyn_rule(m, p, t):
        if t == t0:
            return Constraint.Skip

        ca = phs_df.loc[p, "Control Area"]
        typ = str(phs_df.loc[p, "Power Plant Type"])

        inflow_df = phs_inflow_hourly if typ == "PHS" else hs_inflow_hourly
        inflow = as_float(inflow_df.loc[t + time_offset, ca])

        ep = as_float(phs_df.loc[p, "Efficiency Pump"])
        et = as_float(phs_df.loc[p, "Efficiency Turbine"])
        inflow_abs = as_float(phs_df.loc[p, "Inflow [GWh/a]"]) * 1000.0 * dt * inflow

        return (
            m.phs_storage[p, t]
            - m.phs_storage[p, t - 1]
            - m.phs_pump_consumption[p, t] * ep
            + m.phs_turbine_generation[p, t] / et
            - inflow_abs
            + m.phs_spill[p, t]
            == 0
        )

    model.phs_storage_level_constraint = Constraint(
        model.phs_power_plants, model.timesteps, rule=phs_storage_dyn_rule
    )

    def phs_storage_cap_rule(m, p, t):
        if t == t0:
            return Constraint.Skip
        emax = as_float(phs_df.loc[p, "Energy (max) [GWh]"]) * 1000.0
        return m.phs_storage[p, t] <= emax

    model.phs_storage_capacity_limit_constraint = Constraint(
        model.phs_power_plants, model.timesteps, rule=phs_storage_cap_rule
    )

    def phs_profile_lb(m, ca, t):
        if t == t0 or ca not in m.phs_power_plant_control_areas:
            return Constraint.Skip
        total_cap = sum(
            as_float(phs_df.loc[p, "Energy (max) [GWh]"]) * 1000.0
            for p in m.phs_power_plants_by_control_area[ca]
        )
        target = float(phs_storage_profile.loc[t + time_offset, ca])
        return sum(m.phs_storage[p, t] for p in m.phs_power_plants_by_control_area[ca]) >= total_cap * target * 0.9

    def phs_profile_ub(m, ca, t):
        if t == t0 or ca not in m.phs_power_plant_control_areas:
            return Constraint.Skip
        total_cap = sum(
            as_float(phs_df.loc[p, "Energy (max) [GWh]"]) * 1000.0
            for p in m.phs_power_plants_by_control_area[ca]
        )
        target = float(phs_storage_profile.loc[t + time_offset, ca])
        return sum(m.phs_storage[p, t] for p in m.phs_power_plants_by_control_area[ca]) <= total_cap * target * 1.1

    model.phs_weekly_storage_profile_constraint_1 = Constraint(model.control_areas, model.timesteps, rule=phs_profile_lb)
    model.phs_weekly_storage_profile_constraint_2 = Constraint(model.control_areas, model.timesteps, rule=phs_profile_ub)

    # -------------------------------------------------------------------------
    # Renewables Constraints
    # -------------------------------------------------------------------------

    def ren_prod_rule(m, p, t):
        if t == t0:
            return Constraint.Skip

        typ = str(ren_df.loc[p, "Power Plant Type"])
        ca = ren_df.loc[p, "Control Area"]

        profile_map = {
            "RoR": ror_profile_data,
            "WindOn": windon_profile_data,
            "WindOff": windoff_profile_data,
            "PV": pv_profile_data,
        }
        if typ not in profile_map:
            raise ValueError(f"Unknown plant type '{typ}' for plant '{p}'")

        prof = float(profile_map[typ].loc[t + time_offset, ca])

        if typ == "RoR":
            cap_term = float(ren_df.loc[p, "Inflow [GWh/a]"]) * 1000.0 * prof
        else:
            cap_gw = float(ren_df.loc[p, "Installed Capacity [GW]"])

            # scale-out PV in final run for ALL follower control areas (supports multi-CA)
            if typ == "PV":
                ca_str = str(ca)
                in_pf_ca = True if (fca_set is None) else (ca_str in fca_set)
                if in_pf_ca:
                    cap_gw = cap_gw * (1.0 - pv_share)

            cap_term = cap_gw * 1000.0 * prof

        return cap_term * dt - m.renewable_generation[p, t] - m.renewable_spill[p, t] == 0

    model.renewable_generation_constraint = Constraint(
        model.renewable_power_plants, model.timesteps, rule=ren_prod_rule
    )

    # -------------------------------------------------------------------------
    # Flexibility Constraints
    # -------------------------------------------------------------------------

    def dsr_storage_balance_rule(m, d, t):
        if t == t0:
            return Constraint.Skip
        et = float(dsr_df.loc[d, "Efficiency Turbine"])
        ep = float(dsr_df.loc[d, "Efficiency Pump"])
        return m.dsr_storage[d, t] - m.dsr_storage[d, t - 1] + m.dsr_down[d, t] / et - m.dsr_up[d, t] * ep == 0

    def dsr_down_cap_rule(m, d, t):
        if t == t0:
            return Constraint.Skip
        cap = float(dsr_df.loc[d, "Turbine Capacity [GW]"]) * 1000.0 * dt
        return m.dsr_down[d, t] <= cap

    def dsr_up_cap_rule(m, d, t):
        if t == t0:
            return Constraint.Skip
        cap = float(dsr_df.loc[d, "Pump Capacity [GW]"]) * 1000.0 * dt
        return m.dsr_up[d, t] <= cap

    def dsr_store_upper(m, d, t):
        if t == t0:
            return Constraint.Skip
        cap = float(dsr_df.loc[d, "Energy (max) [GWh]"]) * 1000.0
        return m.dsr_storage[d, t] <= cap

    def dsr_store_lower(m, d, t):
        if t == t0:
            return Constraint.Skip
        cap = float(dsr_df.loc[d, "Energy (max) [GWh]"]) * 1000.0
        return m.dsr_storage[d, t] >= -cap

    model.dsr_storage_balance_constraint = Constraint(model.dsr_units, model.timesteps, rule=dsr_storage_balance_rule)
    model.dsr_down_capacity_limit_constraint = Constraint(model.dsr_units, model.timesteps, rule=dsr_down_cap_rule)
    model.dsr_up_capacity_limit_constraint = Constraint(model.dsr_units, model.timesteps, rule=dsr_up_cap_rule)
    model.dsr_storage_capacity_upper_constraint = Constraint(model.dsr_units, model.timesteps, rule=dsr_store_upper)
    model.dsr_storage_capacity_lower_constraint = Constraint(model.dsr_units, model.timesteps, rule=dsr_store_lower)

    def bat_storage_balance_rule(m, b, t):
        if t == t0:
            return Constraint.Skip
        et = float(bat_df.loc[b, "Efficiency Turbine"])
        ep = float(bat_df.loc[b, "Efficiency Pump"])
        return m.battery_storage[b, t] - m.battery_storage[b, t - 1] + m.battery_out[b, t] / et - m.battery_in[b, t] * ep == 0

    def bat_out_cap_rule(m, b, t):
        if t == t0:
            return Constraint.Skip
        cap_gw = float(bat_df.loc[b, "Turbine Capacity [GW]"])

        ca_b = str(bat_df.loc[b, "Control Area"]) if "Control Area" in bat_df.columns else ""
        in_pf_ca = True if (fca_set is None) else (ca_b in fca_set)
        if in_pf_ca:
            cap_gw = cap_gw * (1.0 - battery_share)

        return m.battery_out[b, t] <= cap_gw * 1000.0 * dt

    def bat_in_cap_rule(m, b, t):
        if t == t0:
            return Constraint.Skip
        cap_gw = float(bat_df.loc[b, "Pump Capacity [GW]"])

        ca_b = str(bat_df.loc[b, "Control Area"]) if "Control Area" in bat_df.columns else ""
        in_pf_ca = True if (fca_set is None) else (ca_b in fca_set)
        if in_pf_ca:
            cap_gw = cap_gw * (1.0 - battery_share)

        return m.battery_in[b, t] <= cap_gw * 1000.0 * dt

    def bat_storage_cap_rule(m, b, t):
        if t == t0:
            return Constraint.Skip
        emax_gwh = float(bat_df.loc[b, "Energy (max) [GWh]"])

        ca_b = str(bat_df.loc[b, "Control Area"]) if "Control Area" in bat_df.columns else ""
        in_pf_ca = True if (fca_set is None) else (ca_b in fca_set)
        if in_pf_ca:
            emax_gwh = emax_gwh * (1.0 - battery_share)

        return m.battery_storage[b, t] <= emax_gwh * 1000.0

    model.battery_storage_balance_constraint = Constraint(model.battery_units, model.timesteps, rule=bat_storage_balance_rule)
    model.battery_out_capacity_limit_constraint = Constraint(model.battery_units, model.timesteps, rule=bat_out_cap_rule)
    model.battery_in_capacity_limit_constraint = Constraint(model.battery_units, model.timesteps, rule=bat_in_cap_rule)
    model.battery_storage_capacity_limit_constraint = Constraint(model.battery_units, model.timesteps, rule=bat_storage_cap_rule)

    # -------------------------------------------------------------------------
    # Network Constraints
    # -------------------------------------------------------------------------

    def flow_balance_rule(m, c, t):
        return m.flow[c, t] == m.flow_pos[c, t] - m.flow_neg[c, t]

    model.flow_balance = Constraint(model.connections, model.timesteps, rule=flow_balance_rule)
    
    def ptdf_flow_rule(m, c, t):
        if t == t0:
            return Constraint.Skip
        return m.flow[c, t] == sum(m.PTDF[c, n] * m.exchange[n, t] for n in m.non_slack_nodes)

    model.ptdf_flow_constraint = Constraint(model.connections, model.timesteps, rule=ptdf_flow_rule)

    def flow_upper_bound_rule(m, c, t):
        if t == t0:
        #if t == t0 or m.xborder[c] != 1:
            return Constraint.Skip
        return m.flow[c, t] <= m.capacity_pos[c] * dt

    def flow_lower_bound_rule(m, c, t):
        if t == t0:
        #if t == t0 or m.xborder[c] != 1:
            return Constraint.Skip
        return m.flow[c, t] >= -m.capacity_neg[c] * dt

    model.flow_upper_bound = Constraint(model.connections, model.timesteps, rule=flow_upper_bound_rule)
    model.flow_lower_bound = Constraint(model.connections, model.timesteps, rule=flow_lower_bound_rule)
    
    def global_exchange_balance_rule(m, t):
        if t == t0:
            return Constraint.Skip
        return sum(m.exchange[n, t] for n in m.nodes) == 0

    model.global_exchange_balance = Constraint(model.timesteps, rule=global_exchange_balance_rule)

    # =========================================================================
    # Demand Balance
    # =========================================================================

    # --- Build per-node,time follower flow lookup for this window (local t=1..T)
    flow_map: Dict[Tuple[str, int], Dict[str, float]] = {}

    if follower_flows_df is not None and not follower_flows_df.empty:
        required = {"node", "start_week", "t",
                    "pv_feed_to_system", "battery_to_system",
                    "imports_to_demand", "imports_to_battery"}
        missing = required - set(follower_flows_df.columns)
        if missing:
            raise ValueError(f"follower_flows_df missing columns: {sorted(missing)}")

        tmp = follower_flows_df.copy()
        tmp["node"] = tmp["node"].astype(str)
        tmp["start_week"] = pd.to_numeric(tmp["start_week"], errors="coerce").astype(int)
        tmp["t"] = pd.to_numeric(tmp["t"], errors="coerce").astype(int)

        tmp = tmp[tmp["start_week"] == int(start_week)].copy()

        # Keep last if duplicates exist
        tmp = tmp.sort_values(["node", "t"]).drop_duplicates(["node", "t"], keep="last")

        flow_map = {
            (str(r.node), int(r.t)): {
                "pv_feed_to_system": float(r.pv_feed_to_system),
                "battery_to_system": float(r.battery_to_system),
                "imports_to_demand": float(r.imports_to_demand),
                "imports_to_battery": float(r.imports_to_battery),
            }
            for _, r in tmp.iterrows()
        }

    nodes_with_follower = {n for (n, tt) in flow_map.keys()}

    # --- Base demand reduced only at follower nodes
    base_demand_map = {}
    for n in model.nodes:
        in_follower_block = (str(n) in nodes_with_follower)
        for tt in range(1, T + 1):
            base = float(node_demand_data.loc[tt + time_offset, n])
            if in_follower_block:
                base = base * (1.0 - float(demand_fraction))
            base_demand_map[(n, tt)] = base

    model.base_demand = Param(
        model.nodes, model.timesteps,
        initialize=lambda m, n, t: 0.0 if t == t0 else base_demand_map.get((n, t), 0.0),
        mutable=False,
    )

    def follower_flow_get(n, t, key):
        if t == t0:
            return 0.0
        rec = flow_map.get((str(n), int(t)))
        return 0.0 if rec is None else float(rec.get(key, 0.0))

    model.follower_pv_feed_to_system = Expression(
        model.nodes, model.timesteps,
        rule=lambda m, n, t: follower_flow_get(n, t, "pv_feed_to_system"),
    )
    model.follower_battery_to_system = Expression(
        model.nodes, model.timesteps,
        rule=lambda m, n, t: follower_flow_get(n, t, "battery_to_system"),
    )
    model.follower_imports_to_demand = Expression(
        model.nodes, model.timesteps,
        rule=lambda m, n, t: follower_flow_get(n, t, "imports_to_demand"),
    )
    model.follower_imports_to_battery = Expression(
        model.nodes, model.timesteps,
        rule=lambda m, n, t: follower_flow_get(n, t, "imports_to_battery"),
    )

    def curtailment_cap_rule(m, n, t):
        if t == t0:
            return Constraint.Skip
        return m.curtailment[n, t] <= m.follower_pv_feed_to_system[n, t] + m.follower_battery_to_system[n, t]

    model.curtailment_cap_constraint = Constraint(model.nodes, model.timesteps, rule=curtailment_cap_rule)

    def effective_demand_rule(m, n, t):
        if t == t0:
            return 0.0
        return (
            m.base_demand[n, t]
            + m.follower_imports_to_demand[n, t]
            + m.follower_imports_to_battery[n, t]
            - m.follower_pv_feed_to_system[n, t]
            - m.follower_battery_to_system[n, t]
            + m.curtailment[n, t]
        )

    model.demand = Expression(model.nodes, model.timesteps, rule=effective_demand_rule)

    def demand_equilibrium_rule(m, n, t):
        if t == t0:
            return Constraint.Skip

        supply = (
            m.total_thermal_generation_per_node[n, t]
            + m.total_phs_generation_per_node[n, t]
            + m.total_renewable_generation_per_node[n, t]
            + m.total_dsr_down_per_node[n, t]
            + m.total_battery_out_per_node[n, t]
        )

        consumption = (
            m.total_phs_consumption_per_node[n, t]
            + m.total_dsr_up_per_node[n, t]
            + m.total_battery_in_per_node[n, t]
            + m.demand[n, t]
        )

        return supply + m.exchange[n, t] + m.nse[n, t] == consumption

    model.demand_equilibrium_constraint = Constraint(model.nodes, model.timesteps, rule=demand_equilibrium_rule)

    # =========================================================================
    # Objective
    # =========================================================================

    model.obj = Objective(
        expr=(
            sum(
                sum(model.thermal_generation[p, t] * model.thermal_generation_cost[p] for p in model.thermal_power_plants)
                for t in model.timesteps
            )
            + sum(
                sum(model.start_thermal_generation[p, t] * model.start_thermal_generation_cost[p] for p in model.thermal_power_plants)
                for t in model.timesteps
            )
            + sum(
                sum(model.phs_turbine_generation[p, t] * model.phs_generation_cost[p] for p in model.phs_power_plants)
                for t in model.timesteps
            )
            + sum(
                sum(model.phs_spill[p, t] * model.phs_spill_cost[p] for p in model.phs_power_plants)
                for t in model.timesteps
            )
            + sum(
                sum(model.renewable_generation[p, t] * model.renewable_generation_cost[p] for p in model.renewable_power_plants)
                for t in model.timesteps
            )
            + sum(
                sum(model.dsr_down[d, t] * model.dsr_down_cost[d] for d in model.dsr_units)
                for t in model.timesteps
            )
            + sum(
                (model.flow_pos[c, t] + model.flow_neg[c, t]) * model.flow_cost[c]
                for c in model.connections
                for t in model.timesteps
            )
            + sum(
                sum(model.nse[n, t] * model.voll[n] for n in model.nodes)
                for t in model.timesteps
            )
            + sum(
                sum(model.curtailment[n, t] * model.voll[n] for n in model.nodes)
                for t in model.timesteps
            )
        ),
        sense=minimize,
    )

    # =========================================================================
    # Duals
    # =========================================================================

    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    return model



# =============================================================================
# Solver Wrapper
# =============================================================================

def solve_model(model, solver_name: str = "gurobi", tee: bool = False):
    """
    Solve the model and return the solver results.
    """
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=tee)
    return results
