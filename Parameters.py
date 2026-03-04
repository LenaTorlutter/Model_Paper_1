import pandas as pd

filepath = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Data.xlsx"

##########################
### GENERAL PARAMETERS ###
##########################

def load_general_data(filepath, sheetname_general):

    general_data = pd.read_excel(filepath, sheet_name = sheetname_general)

    co2_price = general_data['CO2 Price [€/t]'].to_numpy()
    voll = general_data['VoLL [€/MWh]'].to_numpy()
    wind_pv_cost = general_data['Wind PV Cost [€/MWh]'].to_numpy()
    phs_cost = general_data['Hydro Cost [€/MWh]'].to_numpy()
    dsr_cost = general_data['DSR Cost [€/MWh]'].to_numpy()
    flow_cost = general_data['Flow Cost [€/MWh]'].to_numpy()

    return co2_price, voll, wind_pv_cost, phs_cost, dsr_cost, flow_cost


#########################
### PARAMETERS DEMAND ###
#########################

def get_node_to_control_area_mapping(demand_shares_df):
    node_to_ca = {}
    for node in demand_shares_df.columns:
        control_areas_with_share = demand_shares_df[node][demand_shares_df[node] > 0]
        if len(control_areas_with_share) > 1:
            print(f"⚠️ Node {node} has shares in multiple control areas: {list(control_areas_with_share.index)}")
        elif len(control_areas_with_share) == 1:
            node_to_ca[node] = control_areas_with_share.idxmax()
        else:
            node_to_ca[node] = None  # No mapping
    return node_to_ca

def load_demand_data(filepath, sheetname_demand1, sheetname_demand2, sheetname_demand3):
    # Load input data from Excel
    total_demand_df = pd.read_excel(filepath, sheet_name=sheetname_demand1, index_col=0)     # Total annual demand [TWh] per node
    demand_shares_df = pd.read_excel(filepath, sheet_name=sheetname_demand2, index_col=0)    # Share of each node's demand assigned to control areas
    demand_profiles_df = pd.read_excel(filepath, sheet_name=sheetname_demand3, index_col=0)  # Hourly demand profiles per control area

    # Extract node and control area labels
    nodes = total_demand_df.columns.tolist()
    control_areas = demand_shares_df.index.tolist()

    # Scale demand shares per node by total node demand (still in TWh)
    scaled_shares_df = demand_shares_df.copy()
    common_nodes = scaled_shares_df.columns.intersection(total_demand_df.columns)
    for node in common_nodes:
        scaled_shares_df[node] *= total_demand_df.loc[total_demand_df.index[0], node]

    # Normalize each control area's profile over time (columns sum to 1)
    normalized_profiles_df = demand_profiles_df.div(demand_profiles_df.sum(axis=0), axis=1)

    # Multiply: (hours × control areas) @ (control areas × nodes) → (hours × nodes)
    node_demand_df = normalized_profiles_df.values @ scaled_shares_df.values
    node_demand_df = pd.DataFrame(node_demand_df, index=normalized_profiles_df.index, columns=scaled_shares_df.columns)

    # Convert from TWh to MWh
    node_demand_df *= 1e6

    # Build node-to-control-area mapping (based on largest share per node)
    node_to_control_area = get_node_to_control_area_mapping(demand_shares_df)

    return normalized_profiles_df, node_demand_df, control_areas, nodes, node_to_control_area


#######################################
### PARAMETERS THERMAL POWER PLANTS ###
#######################################

def load_thermal_power_plant_data(filepath, sheetname_thermal1, sheetname_thermal2, sheetname_res_shares):
    # Load thermal data
    df_thermal_power_plant_data = pd.read_excel(filepath, sheet_name=sheetname_thermal1)
    df_thermal_power_plant_specific_data = pd.read_excel(filepath, sheet_name=sheetname_thermal2)

    # Clean up whitespace in key columns
    df_thermal_power_plant_specific_data['Power Plant Type'] = df_thermal_power_plant_specific_data['Power Plant Type'].astype(str).str.strip()
    df_thermal_power_plant_specific_data['Node'] = df_thermal_power_plant_specific_data['Node'].astype(str).str.strip()
    if 'Control Area' in df_thermal_power_plant_specific_data.columns:
        df_thermal_power_plant_specific_data['Control Area'] = df_thermal_power_plant_specific_data['Control Area'].astype(str).str.strip()

    # Load RES parts data (first column contains plant types)
    res_shares_raw = pd.read_excel(filepath, sheet_name=sheetname_res_shares)
    res_shares_data = res_shares_raw.set_index(res_shares_raw.columns[0])
    res_shares_data.columns = res_shares_data.columns.astype(str).str.strip()

    # Merge thermal general data into specific data
    thermal_power_plant_data = df_thermal_power_plant_data.set_index('Power Plant Type')
    df_thermal_power_plant_specific_data = df_thermal_power_plant_specific_data.merge(
        thermal_power_plant_data, on='Power Plant Type', how='left'
    )

    # Define AT nodes
    at_nodes = [
        'VBG', 'TIR_w', 'TIR_e', 'OTIR', 'SBG_s', 'SBG_n', 'OOE_e', 'OOE_w', 
        'NOE', 'NOE_n', 'BGLD', 'W', 'STMK_w', 'STMK', 'STMK_s', 'KTN_e', 'KTN_w'
    ]

    # Select rows to expand
    rows_to_expand = df_thermal_power_plant_specific_data[
        (df_thermal_power_plant_specific_data['Power Plant Type'].isin(['OtherRES', 'OtherNonRES'])) &
        (df_thermal_power_plant_specific_data['Node'] == 'AT')
    ]

    # All other rows will be kept
    rows_to_keep = df_thermal_power_plant_specific_data.drop(rows_to_expand.index)

    # Expand rows
    expanded_rows = []
    for _, row in rows_to_expand.iterrows():
        plant_type = row['Power Plant Type']
        original_capacity = row['Installed Capacity [GW]']
        for node in at_nodes:
            share = res_shares_data.loc[plant_type, node]
            new_row = row.copy()
            new_row['Node'] = node
            new_row['Installed Capacity [GW]'] = original_capacity * share
            new_row['Control Area'] = 'AT'
            expanded_rows.append(new_row)

    # Final assignment
    thermal_power_plant_specific_data = pd.concat([rows_to_keep, pd.DataFrame(expanded_rows)], ignore_index=True)

    # Node index mapping
    thermal_power_plant_node_indices = {
        node: thermal_power_plant_specific_data[thermal_power_plant_specific_data['Node'] == node].index.tolist()
        for node in thermal_power_plant_specific_data['Node'].unique()
    }

    return (thermal_power_plant_specific_data, 
            thermal_power_plant_node_indices)

thermal_power_plant_specific_data, thermal_power_plant_node_indices = load_thermal_power_plant_data(
    filepath,
    sheetname_thermal1="Thermal_Power_Data",      # 🔁 adapt to your Excel
    sheetname_thermal2="Thermal_Power_Specific_Data",
    sheetname_res_shares="RES_Shares"
)


####################################################
### PARAMETERS PUMPED HYDRO STORAGE POWER PLANTS ###
####################################################

def load_phs_power_plant_data(filepath, sheetname_phs1, sheetname_phs2):

    df_phs_power_plant_data = pd.read_excel(filepath, sheet_name=sheetname_phs1)
    df_phs_power_plant_specific_data = pd.read_excel(filepath, sheet_name=sheetname_phs2)

    phs_power_plant_data = df_phs_power_plant_data.set_index('Power Plant Type')

    phs_power_plant_specific_data = df_phs_power_plant_specific_data.merge(phs_power_plant_data, on='Power Plant Type', how='left')

    phs_power_plant_control_area_indices = {}
    for control_area in phs_power_plant_specific_data['Control Area']:
        phs_power_plant_control_area_indices[control_area] = phs_power_plant_specific_data[phs_power_plant_specific_data['Control Area'] == control_area].index.tolist()

    phs_power_plant_node_indices = {}
    for node in phs_power_plant_specific_data['Node']:
        phs_power_plant_node_indices[node] = phs_power_plant_specific_data[phs_power_plant_specific_data['Node'] == node].index.tolist()

    return phs_power_plant_specific_data, phs_power_plant_control_area_indices, phs_power_plant_node_indices

import numpy as np
from scipy.interpolate import interp1d

def load_phs_inflow_data(filepath, sheetname_phs_inflow):
    
    # Load original inflow data
    phs_inflow_data = pd.read_excel(filepath, sheet_name=sheetname_phs_inflow)

    total_hours = 8760

    # Rename the first two columns for time steps
    phs_inflow_data.rename(columns={phs_inflow_data.columns[0]: 'Start',
                                    phs_inflow_data.columns[1]: 'End'}, inplace=True)

    time_steps = phs_inflow_data['End'].values
    hourly_time_steps = np.arange(1, total_hours + 1)  # 1-based indexing

    interpolated_data = {}
    totals = {}

    for control_area in phs_inflow_data.columns[2:]:
        inflow_values = phs_inflow_data[control_area].values

        # Interpolate normalized data to hourly values
        interp_func = interp1d(time_steps, inflow_values, kind='linear', fill_value='extrapolate')
        hourly_values = interp_func(hourly_time_steps)

        total = np.sum(hourly_values)
        totals[control_area] = total

        interpolated_data[control_area] = hourly_values / total

    # Construct DataFrame with index starting at 1
    hourly_phs_inflow_data = pd.DataFrame(interpolated_data, index=hourly_time_steps)
    hourly_phs_inflow_data.index.name = 'Hour'

    return phs_inflow_data, hourly_phs_inflow_data, totals

import matplotlib.pyplot as plt

def plot_hourly_inflow(phs_inflow_data, hourly_phs_inflow_data, totals):
    
    plt.figure(figsize=(12, 6))

    # Normalize the original AT inflow by its own sum (same as in loading function)
    normalized_original_at = phs_inflow_data['AT'] / totals['AT']

    # Plot normalized interpolated data
    plt.plot(hourly_phs_inflow_data.index, hourly_phs_inflow_data['AT'],
             label='AT (Interpolated)', alpha=0.8)

    # Plot normalized original data
    plt.scatter(phs_inflow_data['End'], normalized_original_at,
                marker='o', s=10, alpha=0.8, label='AT (Original Normalized)')
    
    plt.xlabel('Hour')
    plt.ylabel('Normalized Inflow')
    plt.title('Normalized Hourly Inflow Profile for AT')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def load_phs_storage_profile_data(filepath, sheetname_phs_storage_profiles, mins):
    
    # Load weekly data
    weekly_phs_storage_profile_data = pd.read_excel(filepath, sheet_name=sheetname_phs_storage_profiles)
    weekly_phs_storage_profile_data = weekly_phs_storage_profile_data.set_index('Week')

    points_per_week = int(168 * 60 / mins)
    total_weeks = len(weekly_phs_storage_profile_data)

    # Shift original points and interpolated index by 1
    original_points = np.arange(total_weeks) * points_per_week + 1
    total_points = original_points[-1]
    interpolated_index = np.arange(1, total_points + 1)

    interpolated = {}

    for control_area in weekly_phs_storage_profile_data.columns:
        weekly_values = weekly_phs_storage_profile_data[control_area].values
        interp_func = interp1d(original_points, weekly_values, kind='linear', fill_value='extrapolate')
        interpolated[control_area] = interp_func(interpolated_index)

    interpolated_phs_storage_profile_data = pd.DataFrame(interpolated, index=interpolated_index)
    interpolated_phs_storage_profile_data.index.name = 'Timestep'

    return weekly_phs_storage_profile_data, interpolated_phs_storage_profile_data


###########################################
### Parameters RENEWABLE ENERGY SOURCES ###
###########################################´

# Define dataset for renewable power plants
def load_renewable_power_plant_data(filepath, sheetname_renewable1, sheetname_renewable2, sheetname_res_shares):
    # Load general and specific renewable data
    df_renewable_power_plant_data = pd.read_excel(filepath, sheet_name=sheetname_renewable1)
    df_renewable_power_plant_specific_data = pd.read_excel(filepath, sheet_name=sheetname_renewable2)

    # Load RES share matrix
    res_shares_raw = pd.read_excel(filepath, sheet_name=sheetname_res_shares)
    res_shares_data = res_shares_raw.set_index(res_shares_raw.columns[0])
    res_shares_data.columns = res_shares_data.columns.astype(str).str.strip()

    # Merge general data
    renewable_power_plant_data = df_renewable_power_plant_data.set_index('Power Plant Type')
    df_renewable_power_plant_specific_data = df_renewable_power_plant_specific_data.merge(
        renewable_power_plant_data, on='Power Plant Type', how='left'
    )

    # Define AT nodes
    at_nodes = [
        'VBG', 'TIR_w', 'TIR_e', 'OTIR', 'SBG_s', 'SBG_n', 'OOE_e', 'OOE_w', 
        'NOE', 'NOE_n', 'BGLD', 'W', 'STMK_w', 'STMK', 'STMK_s', 'KTN_e', 'KTN_w'
    ]

    # Types to expand
    expand_types = ['RoR', 'WindOn', 'PV']

    # Select rows to expand
    rows_to_expand = df_renewable_power_plant_specific_data[
        (df_renewable_power_plant_specific_data['Power Plant Type'].isin(expand_types)) &
        (df_renewable_power_plant_specific_data['Node'] == 'AT')
    ]

    rows_to_keep = df_renewable_power_plant_specific_data.drop(rows_to_expand.index)

    # Expand rows
    expanded_rows = []
    for _, row in rows_to_expand.iterrows():
        plant_type = row['Power Plant Type']
        original_capacity = row['Installed Capacity [GW]']
        original_inflow = row['Inflow [GWh/a]']
        for node in at_nodes:
            share = res_shares_data.loc[plant_type, node]
            new_row = row.copy()
            new_row['Node'] = node
            new_row['Installed Capacity [GW]'] = original_capacity * share
            new_row['Inflow [GWh/a]'] = original_inflow * share
            new_row['Control Area'] = 'AT'
            expanded_rows.append(new_row)

    # Combine final dataset
    renewable_power_plant_specific_data = pd.concat([rows_to_keep, pd.DataFrame(expanded_rows)], ignore_index=True)

    # Index mappings
    renewable_power_plant_node_indices = {
        node: renewable_power_plant_specific_data[renewable_power_plant_specific_data['Node'] == node].index.tolist()
        for node in renewable_power_plant_specific_data['Node'].unique()
    }

    return (renewable_power_plant_specific_data,
            renewable_power_plant_node_indices)

# Define load profiles for control areas
def load_res_profile_data(filepath, sheetname_res_profile, plant_type):
    # Load Excel sheet
    res_profile_data = pd.read_excel(filepath, sheet_name=sheetname_res_profile)

    # Set first column as index (assumes it contains time or ID)
    res_profile_data.set_index(res_profile_data.columns[0], inplace=True)

    totals = {}

    if plant_type == 'RoR':
        for control_area in res_profile_data.columns:
            total = res_profile_data[control_area].sum()
            totals[control_area] = total
            if total != 0:
                res_profile_data[control_area] = res_profile_data[control_area] / total
            else:
                res_profile_data[control_area] = 0  # Avoid division by zero

    return res_profile_data

##############################
### Parameters FLEXIBILITY ###
##############################

def load_flexibility_data(filepath, sheetname_flexibility1, sheetname_flexibility2, sheetname_res_shares):
    # Load flexibility data
    df_flexibility_data = pd.read_excel(filepath, sheet_name=sheetname_flexibility1)
    df_flexibility_specific_data = pd.read_excel(filepath, sheet_name=sheetname_flexibility2)

    # Clean up whitespace in key columns
    df_flexibility_specific_data['Power Plant Type'] = df_flexibility_specific_data['Power Plant Type'].astype(str).str.strip()
    df_flexibility_specific_data['Node'] = df_flexibility_specific_data['Node'].astype(str).str.strip()
    if 'Control Area' in df_flexibility_specific_data.columns:
        df_flexibility_specific_data['Control Area'] = df_flexibility_specific_data['Control Area'].astype(str).str.strip()

    # Load RES parts data
    res_shares_raw = pd.read_excel(filepath, sheet_name=sheetname_res_shares)
    res_shares_data = res_shares_raw.set_index(res_shares_raw.columns[0])
    res_shares_data.columns = res_shares_data.columns.astype(str).str.strip()

    # Merge general into specific
    flexibility_data = df_flexibility_data.set_index('Power Plant Type')
    df_flexibility_specific_data = df_flexibility_specific_data.merge(
        flexibility_data, on='Power Plant Type', how='left'
    )

    # Define AT nodes
    at_nodes = [
        'VBG', 'TIR_w', 'TIR_e', 'OTIR', 'SBG_s', 'SBG_n', 'OOE_e', 'OOE_w', 
        'NOE', 'NOE_n', 'BGLD', 'W', 'STMK_w', 'STMK', 'STMK_s', 'KTN_e', 'KTN_w'
    ]

    # Types to expand
    types_to_expand = ['DSR', 'Battery']
    rows_to_expand = df_flexibility_specific_data[
        (df_flexibility_specific_data['Power Plant Type'].isin(types_to_expand)) &
        (df_flexibility_specific_data['Node'] == 'AT')
    ]
    rows_to_keep = df_flexibility_specific_data.drop(rows_to_expand.index)

    # Expand rows
    expanded_rows = []
    for _, row in rows_to_expand.iterrows():
        plant_type = row['Power Plant Type']
        original_capacity_in = row['Turbine Capacity [GW]']
        original_capacity_out = row['Pump Capacity [GW]']
        original_energy = row['Energy (max) [GWh]']
        for node in at_nodes:
            share = res_shares_data.loc[plant_type, node]
            new_row = row.copy()
            new_row['Node'] = node
            new_row['Turbine Capacity [GW]'] = original_capacity_in * share
            new_row['Pump Capacity [GW]'] = original_capacity_out * share
            new_row['Energy (max) [GWh]'] = original_energy * share
            expanded_rows.append(new_row)

    # Final full data
    all_flex_data = pd.concat([rows_to_keep, pd.DataFrame(expanded_rows)], ignore_index=True)

    # Split into separate DataFrames
    dsr_specific_data = all_flex_data[all_flex_data['Power Plant Type'] == 'DSR'].reset_index(drop=True)
    battery_specific_data = all_flex_data[all_flex_data['Power Plant Type'] == 'Battery'].reset_index(drop=True)

    # Index mapping
    dsr_node_indices = {
        node: dsr_specific_data[dsr_specific_data['Node'] == node].index.tolist()
        for node in dsr_specific_data['Node'].unique()
    }
    battery_node_indices = {
        node: battery_specific_data[battery_specific_data['Node'] == node].index.tolist()
        for node in battery_specific_data['Node'].unique()
    }

    return dsr_specific_data, battery_specific_data, dsr_node_indices, battery_node_indices

###########################
### Parameters EXCHANGE ###
###########################


def load_exchange_data(filepath, sheetname_exchange, slack_node=None, verbose=True):
    """
    Load exchange data from Excel, build incidence matrix and capacities,
    and compute PTDFs (including B_l, B_bus, B_red, B_red_inv).

    Returns
    -------
    nodes : list[str]
    connections : list[str]
    incidence_matrix : pd.DataFrame (nodes x lines)
    ntc_pos : dict (line -> NTC A->B)
    ntc_neg : dict (line -> NTC B->A)
    xborder : dict (line -> XBorder flag)
    conductance_series : pd.Series (line -> 'Komplexe Leitwerte')
    ptdf_results : dict with keys
        - "B_l"        : DataFrame (lines x lines)
        - "B_bus"      : DataFrame (nodes x nodes)
        - "B_red"      : DataFrame (non-slack x non-slack)
        - "B_red_inv"  : DataFrame (non-slack x non-slack)
        - "PTDF"       : DataFrame (lines x non-slack nodes)
        - "slack_node" : str
    """

    # --- read Excel ----------------------------------------------------
    df = pd.read_excel(filepath, sheet_name=sheetname_exchange)

    # 1) Line identifiers (connection names)
    connections = df.iloc[:, 0].dropna().tolist()

    # 2) Node / bidding-zone names (columns 5 onward)
    nodes = df.columns[5:].tolist()

    # 3) Incidence matrix A (nodes × lines)
    incidence_matrix = df.iloc[:, 5:].transpose()
    incidence_matrix.columns = connections
    incidence_matrix.index = nodes
    incidence_matrix = incidence_matrix.fillna(0)

    # 4) NTC capacities
    ntc_pos = pd.Series(df.iloc[:, 1].values, index=connections).to_dict()
    ntc_neg = pd.Series(df.iloc[:, 2].values, index=connections).to_dict()

    # 5) XBorder info
    xborder = pd.Series(df.iloc[:, 4].values, index=connections).to_dict()

    # 6) Line conductances ("Komplexe Leitwerte")
    conductance_series = pd.Series(df.iloc[:, 3].values, index=connections)

    # ==================================================================
    # PTDF PART
    # ==================================================================

    # Ensure numeric incidence
    A_df = incidence_matrix.loc[nodes, connections].astype(float)
    A = A_df.values                          # shape (N_nodes, N_lines)

    # --- 1) B_l: diagonal of line conductances ------------------------
    b_vec = conductance_series.loc[connections].astype(float).values
    B_l = np.diag(b_vec)                     # (L_lines, L_lines)

    # --- 2) B_bus = A B_l A^T -----------------------------------------
    B_bus = A @ B_l @ A.T                    # (N_nodes, N_nodes)

    # --- 3) Choose slack and build reduced matrices -------------------
    if slack_node is None:
        slack_node = nodes[0]
    slack_idx = nodes.index(slack_node)

    keep_mask = np.ones(len(nodes), dtype=bool)
    keep_mask[slack_idx] = False
    non_slack_nodes = [n for i, n in enumerate(nodes) if keep_mask[i]]

    B_red = B_bus[np.ix_(keep_mask, keep_mask)]   # (N-1, N-1)
    A_red = A[keep_mask, :]                       # (N-1, L)

    # --- 4) Invert B_red ----------------------------------------------
    # B_red is the nodal susceptance matrix with the slack removed.
    # For a connected grid it is symmetric positive definite:
    # - mathematically invertible
    # - small enough (zone level) to invert directly.
    B_red_inv = np.linalg.inv(B_red)

    # --- 5) PTDF = B_l * A_red^T * B_red_inv --------------------------
    # f = PTDF @ P_red  (P_red: injections at non-slack nodes)
    PTDF = B_l @ A_red.T @ B_red_inv         # (L, N-1)

    # --- 6) Wrap in DataFrames for readability ------------------------
    B_l_df = pd.DataFrame(B_l, index=connections, columns=connections)
    B_bus_df = pd.DataFrame(B_bus, index=nodes, columns=nodes)
    B_red_df = pd.DataFrame(B_red, index=non_slack_nodes, columns=non_slack_nodes)
    B_red_inv_df = pd.DataFrame(B_red_inv, index=non_slack_nodes, columns=non_slack_nodes)
    PTDF_df = pd.DataFrame(PTDF, index=connections, columns=non_slack_nodes)

    if verbose:
        print(f"Slack node: {slack_node}")

        print("\n=== B_l (diag of 'Komplexe Leitwerte') ===")
        print(B_l_df.round(6))

        print("\n=== B_bus (nodal susceptance matrix) ===")
        print(B_bus_df.round(6))

        print("\n=== B_red (B_bus without slack row/column) ===")
        print(B_red_df.round(6))

        print("\n=== B_red_inv (inverse of B_red) ===")
        print(B_red_inv_df.round(6))

        print("\n=== PTDF (lines x non-slack nodes) ===")
        print(PTDF_df.round(6))

    ptdf_results = {
        "B_l": B_l_df,
        "B_bus": B_bus_df,
        "B_red": B_red_df,
        "B_red_inv": B_red_inv_df,
        "PTDF": PTDF_df,
        "slack_node": slack_node,
    }

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

def generate_ptdf_synchronized_csv(
    filepath: str,
    sheetname_exchange: str,
    output_csv_path: str,
    *,
    slack_node: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute PTDF from the Exchange_Data sheet and export a *synchronized* CSV.

    The exported CSV matches the format expected by Parameters_Updated:
      - rows   : connections (line IDs) in the exact Exchange_Data order
      - cols   : nodes (INCLUDING the slack node column)
      - slack column values are set to 0.0
      - values : PTDF coefficients (connections x nodes)
    """
    (
        nodes,
        connections,
        _incidence_matrix,
        _ntc_pos,
        _ntc_neg,
        _xborder,
        _conductance_series,
        ptdf_results,
    ) = load_exchange_data(filepath, sheetname_exchange, slack_node=slack_node, verbose=False)

    slack = str(ptdf_results.get("slack_node")).strip()
    ptdf_non_slack = ptdf_results["PTDF"].copy()
    ptdf_non_slack.index = ptdf_non_slack.index.astype(str).str.strip()

    full_ptdf = pd.DataFrame(0.0, index=connections, columns=nodes)
    for col in ptdf_non_slack.columns:
        if col not in full_ptdf.columns:
            raise ValueError(
                f"[PTDF export] Non-slack PTDF column '{col}' not found in Exchange_Data nodes. "
                f"Check node naming / whitespace."
            )
        full_ptdf[col] = pd.to_numeric(ptdf_non_slack[col], errors="raise").values

    if slack not in full_ptdf.columns:
        raise ValueError(f"[PTDF export] slack node '{slack}' not found in Exchange_Data nodes list.")
    full_ptdf[slack] = 0.0

    full_ptdf = full_ptdf.loc[connections, nodes]
    full_ptdf.to_csv(output_csv_path)

    if verbose:
        print(f"[PTDF export] Wrote synchronized PTDF CSV to: {output_csv_path}")
        print(f"[PTDF export] shape = {full_ptdf.shape} (connections x nodes)")
        print(f"[PTDF export] slack node = '{slack}' (column forced to 0.0)")

    return full_ptdf

generate_ptdf_synchronized_csv(
    filepath=r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\Data.xlsx",
    sheetname_exchange="Exchange_Data",
    output_csv_path=r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1\PTDF_Synchronized.csv",
    slack_node=None,   # or e.g. "AT" / "VBG" etc.
    verbose=True
)