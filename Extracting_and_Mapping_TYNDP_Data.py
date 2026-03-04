# =============================================================================
# TYNDP → NODE-LEVEL INPUT BUILDER (GENERATION + STORAGE + DEMAND) WITHOUT PLANT MATCHING
# =============================================================================
# This script constructs model-ready, node-level CSV inputs from ENTSO-E/TYNDP
# source workbooks and a combined grid-elements export. It is designed to create
# consistent (country, node/bus, technology) tables *without* any powerplant-level
# matching or allocation based on individual assets.
#
# What it produces
# ----------------
#   - Generation_Data.csv:
#       Node-level installed generation capacities (MW) by technology.
#       Also includes storage-related fields where applicable (e.g., e_nom_MWh,
#       p_nom_pump_MW, p_nom_storage_MW), aggregated to the same node/tech keys.
#
#   - Demand_Data.csv:
#       Hourly demand time series (MW) at node level, derived from TYNDP demand
#       sheets and averaged over a configurable set of weather years.
#
# Node concept and Austria handling
# --------------------------------
# Nodes are derived from a grid-elements CSV containing bus_u / bus_v:
#   - Non-AT: one node per country code (e.g. "DE", "CZ", ...).
#   - AT: if substation-like bus labels exist, Austria is represented nodally
#         (substations become nodes). Otherwise Austria remains aggregated as "AT".
#
# Allocation philosophy (high-level)
# ----------------------------------
#   - Generation & storage start as country totals extracted from PEMMDB workbooks.
#   - Demand is parsed from control-area sheets and can be aggregated to country level
#     before node allocation (configurable).
#   - Country totals are then distributed to nodes:
#       * uniform distribution for countries with a single node (typical non-AT case)
#       * optional share-based distribution for Austria if AT substations exist and
#         Substations_AT.py provides the necessary metadata and shares
#
# Outputs and checks
# ------------------
#   - Writes CSVs to the configured TYNDP_* directories.
#   - Prints sanity summaries of capacities by (country, technology) and basic
#     demand conservation diagnostics (per snapshot, depending on settings).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class Config:
    BASE_DIR: Path = Path(r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1")

    # Austrian substation metadata (state mapping + GSK)
    SUBSTATIONS_AT_FILE: Path = BASE_DIR / "Substations_AT.py"

    # Node source (combined grid export with bus_u/bus_v)
    GRID_ELEMENTS_FILE: Path = (
        BASE_DIR / "Generation_Grid_Data" / "Grid_Data" / "Lines_AT_and_Tielines.csv"
    )

    # TYNDP generation (PEMMDB)
    TYNDP_GEN_DIR: Path = BASE_DIR / "Generation_Grid_Data" / "TYNDP_Generation_Data"
    PEMMDB_GLOB: str = "PEMMDB_*.xlsx"
    GEN_EXPORT_CSV: Path = TYNDP_GEN_DIR / "Generation_Data.csv"

    # TYNDP demand workbook
    TYNDP_DEMAND_DIR: Path = BASE_DIR / "Generation_Grid_Data" / "TYNDP_Demand_Data"
    DEMAND_XLSX: Path = TYNDP_DEMAND_DIR / "2030_National Trends.xlsx"
    DEMAND_XLSX_FALLBACK: Path = Path(r"/mnt/data/2030_National Trends.xlsx")
    DEMAND_EXPORT_CSV: Path = TYNDP_DEMAND_DIR / "Demand_Data.csv"

    # Demand configuration
    LAST_N_WEATHER_YEARS: int = 10

    CONTROL_AREAS: Optional[List[str]] = field(default_factory=lambda: [
        "AT00",
        "BE00",
        "CH00",
        "CZ00",
        "DE00",
        "FR00",
        "HU00",
        "ITCA",
        "ITCN",
        "ITCS",
        "ITN1",
        "ITS1",
        "ITSA",
        "ITSI",
        "NL00",
        "PL00",
        "SI00",
        "SK00",
    ])

    AGGREGATE_DEMAND_TO_COUNTRY: bool = True


CFG = Config()


# =============================================================================
# Small utilities
# =============================================================================

def _norm_str(x: Any) -> str:
    return str(x).strip()

def _to_float(x: Any) -> Optional[float]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _control_area_to_country(control_area: str) -> str:
    s = _norm_str(control_area)
    return s[:2].upper() if len(s) >= 2 else s.upper()

def _ensure_workbook_path(cfg: Config) -> Path:
    if cfg.DEMAND_XLSX.exists():
        return cfg.DEMAND_XLSX
    if cfg.DEMAND_XLSX_FALLBACK.exists():
        print(f"[INFO] Using fallback workbook path: {cfg.DEMAND_XLSX_FALLBACK}")
        return cfg.DEMAND_XLSX_FALLBACK
    raise FileNotFoundError(
        "Demand workbook not found at either:\n"
        f"  - {cfg.DEMAND_XLSX}\n"
        f"  - {cfg.DEMAND_XLSX_FALLBACK}\n"
    )

def _filter_nonzero_caps(
    df: pd.DataFrame,
    cap_cols: List[str],
    *,
    eps: float = 0.0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    for c in cap_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    mask = np.zeros(len(out), dtype=bool)
    for c in cap_cols:
        mask |= out[c].abs().gt(eps)

    return out.loc[mask].copy()

def _read_csv_flexible(path: Path) -> pd.DataFrame:
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8")

def _is_country_code(x: str) -> bool:
    s = _norm_str(x).upper()
    return len(s) == 2 and s.isalpha()


# =============================================================================
# TECH -> GSK GROUP mapping
# =============================================================================

TECH_TO_GSK_GROUP: Dict[str, str] = {
    "onshore_wind": "wind",
    "offshore_wind": "wind",

    "solar_pv": "pv",
    "solar_rooftop": "pv",
    "solar_thermal": "pv",
    "solar_thermal_storage": "pv",

    "run_of_river": "ror",
    "pondage": "hydro_turbine",
    "reservoir": "hydro_turbine",
    "phs_open": "hydro_turbine",
    "phs_pure": "hydro_turbine",

    "battery": "batteries",
    "dsr": "batteries",

    "other_non_res": "other_non_res",

    "small_biomass": "other_res",
    "geothermal": "other_res",
    "marine": "other_res",
    "waste": "other_res",
    "other_res_not_defined": "other_res",
}

THERMAL_FUEL_KEYWORDS_TO_GROUP: List[Tuple[str, str]] = [
    ("NATURAL GAS", "gas"),
    ("GAS", "gas"),
    ("HARD COAL", "coal"),
    ("COAL", "coal"),
    ("LIGNITE", "coal"),
]

def tech_to_gsk_group(tech: str) -> Optional[str]:
    if tech is None:
        return None
    t = str(tech).strip()
    if not t:
        return None

    key = t.lower()
    if key in TECH_TO_GSK_GROUP:
        return TECH_TO_GSK_GROUP[key]

    for k, g in TECH_TO_GSK_GROUP.items():
        if k.lower() == key:
            return g

    up = t.upper()
    for kw, grp in THERMAL_FUEL_KEYWORDS_TO_GROUP:
        if kw in up:
            return grp

    return None


# =============================================================================
# STEP 1) Load nodes from combined grid export
# =============================================================================

def load_substations_at_objects(cfg: Config) -> Tuple[dict, dict]:
    """
    Robust import of Substations_AT.py via absolute file path (no sys.path dependency).

    Returns:
      SUBSTATION_LOOKUP, GSK_STATE_TECH_SHARES
    """
    import importlib.util

    if not cfg.SUBSTATIONS_AT_FILE.exists():
        return {}, {}

    try:
        spec = importlib.util.spec_from_file_location("Substations_AT", str(cfg.SUBSTATIONS_AT_FILE))
        if spec is None or spec.loader is None:
            return {}, {}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        sub_lookup = getattr(mod, "SUBSTATION_LOOKUP", {})
        gsk = getattr(mod, "GSK_STATE_TECH_SHARES", {})

        if not isinstance(sub_lookup, dict):
            sub_lookup = {}
        if not isinstance(gsk, dict):
            gsk = {}

        return sub_lookup, gsk
    except Exception:
        return {}, {}

def load_nodes_from_grid(cfg: Config) -> Tuple[pd.DataFrame, bool]:
    """
    Build node list from GRID_ELEMENTS_FILE containing:
      - bus_u, bus_v

    Node policy:
      - If bus is a 2-letter code => country node
      - Else => treat as AT substation node

    >>> NEW/CHANGED:
    Returns (nodes_df, has_at_substations).
    If NO AT substations exist, Austria remains only "AT" (country node).
    """
    if not cfg.GRID_ELEMENTS_FILE.exists():
        raise FileNotFoundError(f"Grid elements file not found: {cfg.GRID_ELEMENTS_FILE}")

    df = _read_csv_flexible(cfg.GRID_ELEMENTS_FILE)
    df.columns = [_norm_str(c) for c in df.columns]

    required = {"bus_u", "bus_v"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(
            f"GRID_ELEMENTS_FILE missing required columns {sorted(missing)}. "
            f"Columns found: {list(df.columns)}"
        )

    buses = pd.concat(
        [df["bus_u"].astype(str).str.strip(), df["bus_v"].astype(str).str.strip()],
        ignore_index=True,
    ).dropna()

    buses = [b for b in buses.tolist() if b and str(b).upper() != "NAN"]
    buses = [str(b).strip() for b in buses]

    # "AT substations" are any non-2-letter bus labels
    at_substations = sorted({b.upper() for b in buses if not _is_country_code(b)})
    other_countries = sorted({b.upper() for b in buses if _is_country_code(b)})

    has_at_substations = len(at_substations) > 0

    rows = []

    # ensure country nodes exist for all detected country buses
    for cc in other_countries:
        rows.append({"country": cc, "bus_id": cc, "node_type": "country"})

    # >>> NEW/CHANGED: only add AT substations if they exist; else ensure "AT" exists as country node
    if has_at_substations:
        for ss in at_substations:
            rows.append({"country": "AT", "bus_id": ss, "node_type": "substation_at"})
        # also add "AT" as country node only if it is explicitly present as a bus
        # (usually not needed; you want substations)
        if "AT" in other_countries:
            # already added above
            pass
    else:
        # No substations in the grid file => Austria is country-level node "AT"
        rows.append({"country": "AT", "bus_id": "AT", "node_type": "country"})

    out = pd.DataFrame(rows).drop_duplicates(subset=["country", "bus_id"]).reset_index(drop=True)
    return out, has_at_substations

def country_to_nodes(country_nodes: pd.DataFrame) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for cc, sub in country_nodes.groupby("country"):
        m[str(cc)] = sub["bus_id"].astype(str).tolist()
    return m


# =============================================================================
# STEP 2) TYNDP Generation extraction (country totals) from PEMMDB files
# =============================================================================

def infer_country_code_from_filename(p: Path) -> str:
    parts = p.stem.split("_")
    if len(parts) >= 2 and parts[0].upper() == "PEMMDB":
        token = parts[1]
        cc = token[:2].upper()
        if cc.isalpha():
            return cc
    return p.stem.upper()

def _safe_read_sheet(xls: pd.ExcelFile, sheet_name: str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(xls, sheet_name, **kwargs)
    except ValueError:
        return None

def extract_wind_caps(xls: pd.ExcelFile) -> Dict[str, float]:
    df = _safe_read_sheet(xls, "Wind")
    if df is None or df.empty:
        return {}

    caps: Dict[str, float] = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0])
        val = _to_float(row.iloc[1])
        if val is None:
            continue

        if "Onshore wind" in label:
            caps["onshore_wind"] = val * 1e3
        elif "Offshore wind" in label:
            caps["offshore_wind"] = val * 1e3

    return caps

def extract_solar_caps(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    df = _safe_read_sheet(xls, "Solar")
    if df is None or df.empty:
        return {}, {}

    gen_caps: Dict[str, float] = {}
    stor_caps: Dict[str, Dict[str, float]] = {}

    e_gwh: Optional[float] = None

    for _, row in df.iterrows():
        label = str(row.iloc[0])
        val = _to_float(row.iloc[1])
        if val is None:
            continue

        if "Thermal Solar (GW)" in label and "Storage" not in label:
            gen_caps["solar_thermal"] = gen_caps.get("solar_thermal", 0.0) + val * 1e3
        elif "Photovoltaic (GW)" in label:
            gen_caps["solar_pv"] = gen_caps.get("solar_pv", 0.0) + val * 1e3
        elif "Rooftop (GW)" in label:
            gen_caps["solar_rooftop"] = gen_caps.get("solar_rooftop", 0.0) + val * 1e3
        elif "Solar Thermal with Storage (GW)" in label:
            gen_caps["solar_thermal_storage"] = gen_caps.get("solar_thermal_storage", 0.0) + val * 1e3
        elif "Storage capacities Solar Thermal with Storage (GWh)" in label:
            e_gwh = val

    if e_gwh is not None and e_gwh > 0:
        stor_caps["solar_thermal_storage"] = {
            "p_nom_MW": gen_caps.get("solar_thermal_storage", 0.0),
            "p_nom_pump_MW": 0.0,
            "e_nom_MWh": e_gwh * 1e3,
        }

    return gen_caps, stor_caps

def extract_hydro_caps(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    df = _safe_read_sheet(xls, "Hydro")
    if df is None or df.empty:
        return {}, {}

    gen_caps: Dict[str, float] = {}
    storage_caps: Dict[str, Dict[str, float]] = {}

    tmp = {
        "pondage": {"e_GWh": None, "p_turb": None, "p_pump": 0.0},
        "reservoir": {"e_GWh": None, "p_turb": None, "p_pump": 0.0},
        "phs_open": {"e_GWh": None, "p_turb": None, "p_pump": None},
        "phs_pure": {"e_GWh": None, "p_turb": None, "p_pump": None},
    }

    for _, row in df.iterrows():
        label = str(row.iloc[0])
        val = _to_float(row.iloc[1])
        if val is None:
            continue

        if "Run of River" in label and "turbining capacity" in label:
            gen_caps["run_of_river"] = float(val)

        elif "Pondage - Total reservoir capacity (GWh)" in label:
            tmp["pondage"]["e_GWh"] = val
        elif "Pondage - Total turbining capacity (MW)" in label:
            tmp["pondage"]["p_turb"] = val

        elif "Reservoir - Total reservoir capacity (GWh)" in label:
            tmp["reservoir"]["e_GWh"] = val
        elif "Reservoir - Total turbining capacity (MW)" in label:
            tmp["reservoir"]["p_turb"] = val

        elif "Pump Storage (open loop" in label and "reservoir capacity" in label:
            tmp["phs_open"]["e_GWh"] = val
        elif "Pump Storage (open loop" in label and "turbining capacity" in label:
            tmp["phs_open"]["p_turb"] = val
        elif "Pump Storage (open loop" in label and "pumping capacity" in label:
            tmp["phs_open"]["p_pump"] = abs(val)

        elif "Pure Pump Storage" in label and "reservoir capacity" in label:
            tmp["phs_pure"]["e_GWh"] = val
        elif "Pure Pump Storage" in label and "turbining capacity" in label:
            tmp["phs_pure"]["p_turb"] = val
        elif "Pure Pump Storage" in label and "pumping capacity" in label:
            tmp["phs_pure"]["p_pump"] = abs(val)

    for tech, vals in tmp.items():
        if vals["p_turb"] is not None and vals["p_turb"] > 0:
            gen_caps[tech] = float(vals["p_turb"])

        if vals["e_GWh"] is None or vals["p_turb"] is None:
            continue

        storage_caps[tech] = {
            "p_nom_MW": float(vals["p_turb"]),
            "p_nom_pump_MW": float(vals["p_pump"] or 0.0),
            "e_nom_MWh": float(vals["e_GWh"]) * 1e3,
        }

    return gen_caps, storage_caps

def extract_battery_caps(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    sheet = next((s for s in xls.sheet_names if s.lower().startswith("batter")), None)
    if sheet is None:
        return {}, {}

    df = pd.read_excel(xls, sheet)
    if df.empty:
        return {}, {}

    mask = df.iloc[:, 0].astype(str).str.strip().eq("Battery")
    if not mask.any():
        return {}, {}

    row = df.loc[mask].iloc[0]
    p_gen = _to_float(row.iloc[2])
    p_dem = _to_float(row.iloc[3])
    e_nom = _to_float(row.iloc[4])

    gen_caps: Dict[str, float] = {}
    stor_caps: Dict[str, Dict[str, float]] = {}

    if p_gen is not None and p_gen > 0:
        gen_caps["battery"] = float(p_gen)

        if e_nom is not None and e_nom > 0:
            stor_caps["battery"] = {
                "p_nom_MW": float(p_gen),
                "p_nom_pump_MW": float(p_dem) if (p_dem is not None and p_dem > 0) else float(p_gen),
                "e_nom_MWh": float(e_nom),
            }

    return gen_caps, stor_caps

def extract_dsr_caps(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    sheet_name = next((sn for sn in xls.sheet_names if sn.strip().lower() == "dsr"), None)
    if sheet_name is None:
        return {}, {}

    df = pd.read_excel(xls, sheet_name, header=None)
    if df.empty:
        return {}, {}

    def _first_non_empty_idx(row: pd.Series) -> Optional[int]:
        for j, cell in enumerate(row):
            if pd.isna(cell):
                continue
            s = str(cell).strip()
            if s:
                return j
        return None

    def _next_numeric_to_right(row: pd.Series, start_j: int) -> Optional[float]:
        for j in range(start_j + 1, len(row)):
            v = _to_float(row.iloc[j])
            if v is not None:
                return v
        return None

    capacity_val: Optional[float] = None

    for _, row in df.iterrows():
        j0 = _first_non_empty_idx(row)
        if j0 is None:
            continue
        if str(row.iloc[j0]).strip().lower() == "capacity":
            capacity_val = _next_numeric_to_right(row, j0)
            if capacity_val is not None:
                break

    if capacity_val is None or capacity_val <= 0:
        return {}, {}

    cap = float(capacity_val)
    gen_caps = {"dsr": cap}
    storage_caps = {
        "dsr": {
            "p_nom_MW": cap / 2.0,
            "p_nom_pump_MW": cap / 2.0,
            "e_nom_MWh": cap,
        }
    }
    return gen_caps, storage_caps

def extract_thermal_caps(xls: pd.ExcelFile) -> Dict[str, float]:
    df = _safe_read_sheet(xls, "Thermal")
    if df is None or df.empty:
        return {}

    caps: Dict[str, float] = {}
    current_fuel: Optional[str] = None

    for _, row in df.iterrows():
        fuel = row.iloc[0]
        subtype = row.iloc[1]
        cap = _to_float(row.iloc[2])

        if isinstance(fuel, str) and fuel.strip():
            current_fuel = fuel.strip()

        if not current_fuel:
            continue

        if isinstance(subtype, str) and "installed capacity" in subtype.lower():
            continue

        if cap is None or cap <= 0:
            continue

        if isinstance(subtype, str) and subtype.strip():
            tech = f"{current_fuel} {subtype.strip()}"
        else:
            tech = current_fuel

        caps[tech] = caps.get(tech, 0.0) + float(cap)

    return caps

def extract_other_res_caps(xls: pd.ExcelFile) -> Dict[str, float]:
    sheet_name = next((sn for sn in xls.sheet_names if "other res" in sn.lower()), None)
    if sheet_name is None:
        return {}

    df = pd.read_excel(xls, sheet_name, header=None)
    if df.empty:
        return {}

    def _norm(x: Any) -> str:
        if pd.isna(x):
            return ""
        return " ".join(str(x).replace("\n", " ").replace("\r", " ").split()).strip()

    def _norm_lower(x: Any) -> str:
        return _norm(x).lower()

    header_row_idx: Optional[int] = None
    for i in range(len(df)):
        if any(_norm_lower(c) == "small biomass" for c in df.iloc[i]):
            header_row_idx = i
            break
    if header_row_idx is None:
        return {}

    cap_row_idx: Optional[int] = None
    for i in range(header_row_idx + 1, len(df)):
        if any(("installed capacity" in _norm_lower(c) and "(mw)" in _norm_lower(c)) for c in df.iloc[i]):
            cap_row_idx = i
            break
    if cap_row_idx is None:
        return {}

    header_row = df.iloc[header_row_idx]
    cap_row = df.iloc[cap_row_idx]

    keep = {
        "small biomass": "small_biomass",
        "geothermal": "geothermal",
        "marine": "marine",
        "waste": "waste",
        "not defined / splitting not known": "other_res_not_defined",
        "not defined": "other_res_not_defined",
        "splitting not known": "other_res_not_defined",
    }

    caps: Dict[str, float] = {}
    for col_idx, tech_cell in enumerate(header_row):
        tech_key = keep.get(_norm_lower(tech_cell))
        if tech_key is None:
            continue

        v = _to_float(cap_row.iloc[col_idx])
        if v is None or v <= 0:
            continue

        caps[tech_key] = caps.get(tech_key, 0.0) + float(v)

    return caps

def extract_other_nonres_caps(xls: pd.ExcelFile) -> Dict[str, float]:
    sheet_name = next(
        (sn for sn in xls.sheet_names if ("other" in sn.lower() and "non" in sn.lower() and "res" in sn.lower())),
        None,
    )
    if sheet_name is None:
        return {}

    df = pd.read_excel(xls, sheet_name, header=None)
    if df.empty:
        return {}

    def _norm_label(x: Any) -> str:
        if pd.isna(x):
            return ""
        return " ".join(str(x).replace("\n", " ").replace("\r", " ").split()).lower()

    def _next_numeric_to_right(row: pd.Series, start_j: int) -> Optional[float]:
        for j in range(start_j + 1, len(row)):
            v = _to_float(row.iloc[j])
            if v is not None:
                return v
        return None

    installed_cap: Optional[float] = None

    for _, row in df.iterrows():
        for j, cell in enumerate(row):
            lab = _norm_label(cell)
            if ("installed capacity" in lab) and ("(mw)" in lab):
                installed_cap = _next_numeric_to_right(row, j)
                if installed_cap is not None:
                    break
        if installed_cap is not None:
            break

    if installed_cap is None or installed_cap <= 0:
        return {}

    return {"other_non_res": float(installed_cap)}

def build_country_tables(pemmdb_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cc = infer_country_code_from_filename(pemmdb_path)

    gen_parts: List[dict] = []
    stor_parts: List[dict] = []

    with pd.ExcelFile(pemmdb_path) as xls:
        solar_gen, solar_stor = extract_solar_caps(xls)

        for d in (
            extract_wind_caps(xls),
            solar_gen,
            extract_thermal_caps(xls),
            extract_other_res_caps(xls),
            extract_other_nonres_caps(xls),
        ):
            for tech, cap in d.items():
                gen_parts.append({"country": cc, "tech": tech, "p_nom_MW": cap})

        for tech, vals in solar_stor.items():
            stor_parts.append({"country": cc, "tech": tech, **vals})

        hydro_gen, hydro_stor = extract_hydro_caps(xls)
        for tech, cap in hydro_gen.items():
            gen_parts.append({"country": cc, "tech": tech, "p_nom_MW": cap})
        for tech, vals in hydro_stor.items():
            stor_parts.append({"country": cc, "tech": tech, **vals})

        batt_gen, batt_stor = extract_battery_caps(xls)
        for tech, cap in batt_gen.items():
            gen_parts.append({"country": cc, "tech": tech, "p_nom_MW": cap})
        for tech, vals in batt_stor.items():
            stor_parts.append({"country": cc, "tech": tech, **vals})

        dsr_gen, dsr_stor = extract_dsr_caps(xls)
        for tech, cap in dsr_gen.items():
            gen_parts.append({"country": cc, "tech": tech, "p_nom_MW": cap})
        for tech, vals in dsr_stor.items():
            stor_parts.append({"country": cc, "tech": tech, **vals})

    return pd.DataFrame(gen_parts), pd.DataFrame(stor_parts)

def build_all_countries_generation(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gen_all: List[pd.DataFrame] = []
    stor_all: List[pd.DataFrame] = []

    for f in sorted(cfg.TYNDP_GEN_DIR.glob(cfg.PEMMDB_GLOB)):
        try:
            g, s = build_country_tables(f)
            if not g.empty:
                gen_all.append(g)
            if not s.empty:
                stor_all.append(s)
        except Exception as e:
            print(f"[WARN] Failed {f.name}: {e}")

    gen_df = pd.concat(gen_all, ignore_index=True) if gen_all else pd.DataFrame(columns=["country", "tech", "p_nom_MW"])
    stor_df = (
        pd.concat(stor_all, ignore_index=True)
        if stor_all
        else pd.DataFrame(columns=["country", "tech", "p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"])
    )

    for df in [gen_df, stor_df]:
        if df.empty:
            continue
        df["country"] = df["country"].astype(str).str.upper().str.strip()
        df["tech"] = df["tech"].astype(str).str.strip()

    if not gen_df.empty:
        gen_df["p_nom_MW"] = pd.to_numeric(gen_df["p_nom_MW"], errors="coerce").fillna(0.0)

    if not stor_df.empty:
        for c in ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]:
            if c not in stor_df.columns:
                stor_df[c] = 0.0
            stor_df[c] = pd.to_numeric(stor_df[c], errors="coerce").fillna(0.0)

    return gen_df, stor_df


# =============================================================================
# STEP 3) Allocation helpers
# =============================================================================

def allocate_country_totals_to_nodes_uniform(
    totals: pd.DataFrame,
    country_nodes_map: Dict[str, List[str]],
    value_cols: List[str],
    *,
    country_col: str = "country",
    tech_col: str = "tech",
) -> pd.DataFrame:
    if totals.empty:
        return pd.DataFrame(columns=[country_col, "bus_id", tech_col] + value_cols)

    needed = {country_col, tech_col}.union(value_cols)
    missing = needed.difference(totals.columns)
    if missing:
        raise ValueError(f"allocate_country_totals_to_nodes_uniform: missing columns {sorted(missing)}")

    nat = totals.copy()
    nat[country_col] = nat[country_col].astype(str).str.upper().str.strip()
    nat[tech_col] = nat[tech_col].astype(str).str.strip()
    for c in value_cols:
        nat[c] = pd.to_numeric(nat[c], errors="coerce").fillna(0.0)

    nat = nat.groupby([country_col, tech_col], as_index=False)[value_cols].sum()

    rows = []
    for r in nat.itertuples(index=False):
        cc = getattr(r, country_col)
        tech = getattr(r, tech_col)
        nodes = country_nodes_map.get(cc, [])
        if not nodes:
            nodes = [cc]

        n = float(len(nodes))
        for node in nodes:
            out = {country_col: cc, "bus_id": str(node), tech_col: tech}
            for c in value_cols:
                out[c] = float(getattr(r, c)) / n if n > 0 else 0.0
            rows.append(out)

    return pd.DataFrame(rows)

def _at_substation_state_map(at_nodes: List[str], substation_lookup: dict) -> Dict[str, Optional[str]]:
    m: Dict[str, Optional[str]] = {}
    for s in at_nodes:
        info = substation_lookup.get(s, {})
        st = info.get("state") if isinstance(info, dict) else None
        m[s] = st if (isinstance(st, str) and st.strip()) else None
    return m

def allocate_at_by_gsk_group(
    at_totals: pd.DataFrame,
    at_nodes: List[str],
    *,
    substation_lookup: dict,
    gsk_state_group_shares: dict,
    value_cols: List[str],
) -> pd.DataFrame:
    if at_totals is None or at_totals.empty:
        return pd.DataFrame(columns=["country", "bus_id", "tech"] + value_cols)

    if not at_nodes:
        at_nodes = ["AT"]

    at = at_totals.copy()
    at["country"] = "AT"
    at["tech"] = at["tech"].astype(str).str.strip()
    for c in value_cols:
        at[c] = pd.to_numeric(at[c], errors="coerce").fillna(0.0)

    at = at.groupby(["tech"], as_index=False)[value_cols].sum()

    sub_to_state = _at_substation_state_map(at_nodes, substation_lookup)
    state_to_subs: Dict[str, List[str]] = {}
    for s, st in sub_to_state.items():
        if st:
            state_to_subs.setdefault(st, []).append(s)

    no_state_subs = [s for s in at_nodes if not sub_to_state.get(s)]

    rows: List[dict] = []

    for r in at.itertuples(index=False):
        tech = getattr(r, "tech")
        group = tech_to_gsk_group(tech)

        if group is None or not state_to_subs or not isinstance(gsk_state_group_shares, dict):
            for node in at_nodes:
                out = {"country": "AT", "bus_id": node, "tech": tech}
                for c in value_cols:
                    total = float(getattr(r, c))
                    out[c] = total / float(len(at_nodes)) if len(at_nodes) else 0.0
                rows.append(out)
            continue

        state_shares: Dict[str, Optional[float]] = {}
        for st in state_to_subs.keys():
            share = None
            if st in gsk_state_group_shares and isinstance(gsk_state_group_shares[st], dict):
                share = gsk_state_group_shares[st].get(group)
            state_shares[st] = float(share) if share is not None else None

        if any(v is None for v in state_shares.values()):
            for node in at_nodes:
                out = {"country": "AT", "bus_id": node, "tech": tech}
                for c in value_cols:
                    total = float(getattr(r, c))
                    out[c] = total / float(len(at_nodes)) if len(at_nodes) else 0.0
                rows.append(out)
            continue

        for st, subs in state_to_subs.items():
            st_share = float(state_shares[st])
            for node in subs:
                out = {"country": "AT", "bus_id": node, "tech": tech}
                for c in value_cols:
                    total = float(getattr(r, c))
                    st_total = total * st_share
                    out[c] = st_total / float(len(subs)) if len(subs) else 0.0
                rows.append(out)

        if no_state_subs:
            share_sum = float(sum(state_shares.values()))
            rem_factor = max(0.0, 1.0 - share_sum)
            for node in no_state_subs:
                out = {"country": "AT", "bus_id": node, "tech": tech}
                for c in value_cols:
                    total = float(getattr(r, c))
                    rem = total * rem_factor
                    out[c] = rem / float(len(no_state_subs)) if len(no_state_subs) else 0.0
                rows.append(out)

    return pd.DataFrame(rows)


# =============================================================================
# STEP 4) Demand extraction + allocation
# =============================================================================

def _to_int_year(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None
        y = int(float(s))
        if 1800 <= y <= 2200:
            return y
        return None
    except Exception:
        return None

def _find_hourly_header_row(raw: pd.DataFrame) -> int:
    scan_lim = min(len(raw), 300)
    for i in range(scan_lim):
        first = raw.iat[i, 0]
        if isinstance(first, str) and first.strip().lower() == "date":
            row_vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if "hour" in row_vals:
                return i
    raise ValueError("Could not locate hourly header row (expected a 'Date' row containing 'Hour').")

def _coerce_decimal_comma_to_float(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def parse_control_area_sheet(xlsx_path: Path, sheet_name: str, last_n_weather_years: int) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    header_row = _find_hourly_header_row(raw)
    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header

    for col in ["Date", "Hour"]:
        if col not in data.columns:
            raise ValueError(f"[{sheet_name}] Missing required column '{col}' after header detection.")

    data = data.dropna(subset=["Date", "Hour"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Hour"] = pd.to_numeric(data["Hour"], errors="coerce")
    data = data.dropna(subset=["Date", "Hour"]).copy()

    hour = pd.to_numeric(data["Hour"], errors="coerce")

    if hour.notna().any() and hour.min() >= 1 and hour.max() <= 24 and float(hour.max()) == 24.0:
        hour = hour - 1

    data["snapshot"] = data["Date"] + pd.to_timedelta(hour, unit="h")

    year_cols = []
    year_map = {}
    for c in data.columns:
        y = _to_int_year(c)
        if y is not None:
            year_cols.append(c)
            year_map[c] = y

    if not year_cols:
        raise ValueError(f"[{sheet_name}] No weather-year columns detected.")

    year_cols_sorted = sorted(year_cols, key=lambda c: year_map[c])
    used = year_cols_sorted[-last_n_weather_years:] if last_n_weather_years > 0 else year_cols_sorted

    vals = _coerce_decimal_comma_to_float(data[used])
    mean_mw = vals.mean(axis=1, skipna=True).astype(float)

    out = pd.DataFrame(
        {
            "control_area": sheet_name,
            "snapshot": data["snapshot"].values,
            "demand_MW_mean": mean_mw.values,
            "used_weather_years": ",".join(map(str, [year_map[c] for c in used])),
        }
    )
    return out

def load_all_control_areas_demand(cfg: Config) -> pd.DataFrame:
    xlsx_path = _ensure_workbook_path(cfg)
    xl = pd.ExcelFile(xlsx_path)

    sheet_names = xl.sheet_names
    if cfg.CONTROL_AREAS is not None:
        wanted = set(cfg.CONTROL_AREAS)
        missing = sorted(wanted.difference(sheet_names))
        if missing:
            raise ValueError(f"Requested control areas not found in workbook: {missing}")
        sheet_names = [s for s in sheet_names if s in wanted]

    parts = []
    for s in sheet_names:
        parts.append(parse_control_area_sheet(xlsx_path, s, cfg.LAST_N_WEATHER_YEARS))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["control_area", "snapshot", "demand_MW_mean", "used_weather_years"]
    )

def allocate_demand_to_country_nodes(
    ca_demand: pd.DataFrame,
    country_nodes_map: Dict[str, List[str]],
    *,
    aggregate_to_country: bool,
) -> pd.DataFrame:
    if ca_demand.empty:
        return pd.DataFrame(columns=["control_area", "country", "bus_id", "snapshot", "p_set_MW"])

    tmp = ca_demand.copy()
    tmp["control_area"] = tmp["control_area"].astype(str).str.strip()
    tmp["country"] = tmp["control_area"].map(_control_area_to_country)
    tmp["snapshot"] = pd.to_datetime(tmp["snapshot"], errors="coerce")
    tmp["demand_MW_mean"] = pd.to_numeric(tmp["demand_MW_mean"], errors="coerce").fillna(0.0)

    if aggregate_to_country:
        tmp = (
            tmp.groupby(["country", "snapshot"], as_index=False)["demand_MW_mean"]
            .sum()
            .assign(control_area=lambda df: df["country"])
        )

    rows = []
    for (ca, cc, snap), sub in tmp.groupby(["control_area", "country", "snapshot"], as_index=False):
        demand = float(sub["demand_MW_mean"].iloc[0])
        nodes = country_nodes_map.get(cc, [])
        if not nodes:
            nodes = [cc]

        per_node = demand / float(len(nodes)) if nodes else 0.0
        for node in nodes:
            rows.append(
                {
                    "control_area": ca,
                    "country": cc,
                    "bus_id": str(node),
                    "snapshot": snap,
                    "p_set_MW": per_node,
                }
            )

    out = pd.DataFrame(rows)
    out["p_set_MW"] = pd.to_numeric(out["p_set_MW"], errors="coerce").fillna(0.0)
    return out


# =============================================================================
# SANITY CHECKING (unchanged)
# =============================================================================

def print_generation_capacity_by_country_tech(gen_totals: pd.DataFrame) -> None:
    print("\n=== GENERATION CAPACITY BY (COUNTRY -> TECHNOLOGY) (MW) ===\n")

    if gen_totals is None or gen_totals.empty:
        print("(no generation data)")
        return

    g = gen_totals.copy()
    g["country"] = g["country"].astype(str).str.upper().str.strip()
    g["tech"] = g["tech"].astype(str).str.strip()
    g["p_nom_MW"] = pd.to_numeric(g["p_nom_MW"], errors="coerce").fillna(0.0)

    g = _filter_nonzero_caps(g, ["p_nom_MW"])
    if g.empty:
        print("(no non-zero generation capacities)")
        return

    g = g.groupby(["country", "tech"], as_index=False)["p_nom_MW"].sum()
    g = _filter_nonzero_caps(g, ["p_nom_MW"])

    for cc in sorted(g["country"].unique()):
        sub = g.loc[g["country"] == cc].copy()
        ser = sub.set_index("tech")["p_nom_MW"].sort_values(ascending=False)
        print(f"--- {cc} ---")
        print(ser.to_string())
        print()

def print_storage_capacity_by_country_tech(stor_totals: pd.DataFrame) -> None:
    print("\n=== STORAGE CAPACITY BY (COUNTRY -> TECHNOLOGY) ===\n")

    if stor_totals is None or stor_totals.empty:
        print("(no storage data)")
        return

    s = stor_totals.copy()
    s["country"] = s["country"].astype(str).str.upper().str.strip()
    s["tech"] = s["tech"].astype(str).str.strip()

    for c in ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]:
        if c not in s.columns:
            s[c] = 0.0
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)

    s = _filter_nonzero_caps(s, ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"])
    if s.empty:
        print("(no non-zero storage capacities)")
        return

    s = s.groupby(["country", "tech"], as_index=False)[["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]].sum()
    s = _filter_nonzero_caps(s, ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"])

    for cc in sorted(s["country"].unique()):
        sub = s.loc[s["country"] == cc].copy()
        sub = sub.set_index("tech")[["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]].sort_values(
            by="p_nom_MW", ascending=False
        )
        print(f"--- {cc} ---")
        print(sub.to_string())
        print()


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: Config = CFG) -> None:
    # 1) Nodes from combined grid file
    nodes, has_at_substations = load_nodes_from_grid(cfg)  # >>> NEW/CHANGED
    c2n = country_to_nodes(nodes)

    print("\n[INFO] Node detection summary")
    print("  Total nodes:", len(nodes))
    print("  Countries:", sorted(nodes["country"].unique().tolist()))
    print("  has_at_substations:", has_at_substations)
    if "AT" in c2n:
        print("  AT nodes:", c2n["AT"])

    # Optional: load AT substation state + GSK shares (ONLY needed if we actually split AT)
    if has_at_substations:
        sub_lookup, gsk = load_substations_at_objects(cfg)  # >>> NEW/CHANGED: robust path import
        print("\n[INFO] Loaded Substations_AT objects")
        print("  SUB_LOOKUP size:", len(sub_lookup))
        print("  GSK size:", len(gsk))
    else:
        sub_lookup, gsk = {}, {}
        print("\n[INFO] No AT substations detected -> Austria handled at country level (no GSK).")

    # 2) Generation totals
    gen_totals, stor_totals = build_all_countries_generation(cfg)

    print_generation_capacity_by_country_tech(gen_totals)
    print_storage_capacity_by_country_tech(stor_totals)

    # ---------------------------
    # Generation allocation
    # ---------------------------
    gen_totals = gen_totals.copy()
    gen_totals["country"] = gen_totals["country"].astype(str).str.upper().str.strip()

    # >>> NEW/CHANGED: if no AT substations, treat AT like any other country (uniform onto single node "AT")
    if not has_at_substations:
        gen_alloc = allocate_country_totals_to_nodes_uniform(
            totals=gen_totals,
            country_nodes_map=c2n,
            value_cols=["p_nom_MW"],
        )
    else:
        gen_at = gen_totals.loc[gen_totals["country"] == "AT"].copy()
        gen_non = gen_totals.loc[gen_totals["country"] != "AT"].copy()

        gen_non_alloc = allocate_country_totals_to_nodes_uniform(
            totals=gen_non,
            country_nodes_map=c2n,
            value_cols=["p_nom_MW"],
        )

        at_nodes = c2n.get("AT", ["AT"])
        gen_at_alloc = allocate_at_by_gsk_group(
            at_totals=gen_at,
            at_nodes=at_nodes,
            substation_lookup=sub_lookup,
            gsk_state_group_shares=gsk,
            value_cols=["p_nom_MW"],
        )

        gen_alloc = pd.concat([gen_non_alloc, gen_at_alloc], ignore_index=True)

    # ---------------------------
    # Storage allocation
    # ---------------------------
    stor_totals = stor_totals.copy()
    if not stor_totals.empty:
        stor_totals["country"] = stor_totals["country"].astype(str).str.upper().str.strip()

    if stor_totals.empty:
        stor_alloc = pd.DataFrame(columns=["country", "bus_id", "tech", "p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"])
    else:
        # >>> NEW/CHANGED: if no AT substations, treat AT like any other country (uniform)
        if not has_at_substations:
            stor_alloc = allocate_country_totals_to_nodes_uniform(
                totals=stor_totals,
                country_nodes_map=c2n,
                value_cols=["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"],
            )
        else:
            stor_at = stor_totals.loc[stor_totals["country"] == "AT"].copy()
            stor_non = stor_totals.loc[stor_totals["country"] != "AT"].copy()

            stor_non_alloc = allocate_country_totals_to_nodes_uniform(
                totals=stor_non,
                country_nodes_map=c2n,
                value_cols=["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"],
            )

            at_nodes = c2n.get("AT", ["AT"])
            if stor_at.empty:
                stor_at_alloc = pd.DataFrame(columns=["country", "bus_id", "tech", "p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"])
            else:
                stor_at2 = stor_at.copy()
                stor_at2["tech"] = stor_at2["tech"].astype(str).str.strip()
                for c in ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]:
                    if c not in stor_at2.columns:
                        stor_at2[c] = 0.0
                    stor_at2[c] = pd.to_numeric(stor_at2[c], errors="coerce").fillna(0.0)

                stor_at2 = stor_at2.groupby(["tech"], as_index=False)[["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]].sum()

                stor_at_alloc = allocate_at_by_gsk_group(
                    at_totals=stor_at2.assign(country="AT"),
                    at_nodes=at_nodes,
                    substation_lookup=sub_lookup,
                    gsk_state_group_shares=gsk,
                    value_cols=["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"],
                )

                stor_at_alloc["tech"] = stor_at_alloc["tech"].astype(str).str.strip()
                for c in ["p_nom_MW", "p_nom_pump_MW", "e_nom_MWh"]:
                    if c not in stor_at_alloc.columns:
                        stor_at_alloc[c] = 0.0
                    stor_at_alloc[c] = pd.to_numeric(stor_at_alloc[c], errors="coerce").fillna(0.0)

            stor_alloc = pd.concat([stor_non_alloc, stor_at_alloc], ignore_index=True)

    # ---------------------------
    # Merge storage onto generation
    # ---------------------------
    gen_out = gen_alloc.merge(
        stor_alloc.rename(columns={"p_nom_MW": "p_nom_storage_MW"}),
        on=["country", "bus_id", "tech"],
        how="outer",
    )

    gen_out["p_nom_MW"] = pd.to_numeric(gen_out.get("p_nom_MW"), errors="coerce").fillna(0.0)
    gen_out["p_nom_storage_MW"] = pd.to_numeric(gen_out.get("p_nom_storage_MW"), errors="coerce").fillna(0.0)

    if "p_nom_pump_MW" not in gen_out.columns:
        gen_out["p_nom_pump_MW"] = 0.0
    if "e_nom_MWh" not in gen_out.columns:
        gen_out["e_nom_MWh"] = 0.0

    gen_out["p_nom_pump_MW"] = pd.to_numeric(gen_out["p_nom_pump_MW"], errors="coerce").fillna(0.0)
    gen_out["e_nom_MWh"] = pd.to_numeric(gen_out["e_nom_MWh"], errors="coerce").fillna(0.0)

    gen_out = _filter_nonzero_caps(gen_out, ["p_nom_MW", "p_nom_storage_MW", "p_nom_pump_MW", "e_nom_MWh"])
    gen_out = gen_out.sort_values(["country", "bus_id", "tech"]).reset_index(drop=True)

    cfg.TYNDP_GEN_DIR.mkdir(parents=True, exist_ok=True)
    gen_out.to_csv(cfg.GEN_EXPORT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] Generation exported: {cfg.GEN_EXPORT_CSV}")
    print("Generation rows:", len(gen_out))

    # ---------------------------
    # Demand
    # ---------------------------
    ca_demand = load_all_control_areas_demand(cfg)
    demand_out = allocate_demand_to_country_nodes(
        ca_demand=ca_demand,
        country_nodes_map=c2n,
        aggregate_to_country=cfg.AGGREGATE_DEMAND_TO_COUNTRY,
    )

    cfg.TYNDP_DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    demand_out.to_csv(cfg.DEMAND_EXPORT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Demand exported: {cfg.DEMAND_EXPORT_CSV}")
    print("Demand rows:", len(demand_out))

    # Quick sanity: total demand conservation (per snapshot)
    if not ca_demand.empty and not demand_out.empty:
        if cfg.AGGREGATE_DEMAND_TO_COUNTRY:
            lhs = ca_demand.assign(country=lambda df: df["control_area"].map(_control_area_to_country))
            lhs = lhs.groupby(["country", "snapshot"], as_index=False)["demand_MW_mean"].sum()
            rhs = demand_out.groupby(["country", "snapshot"], as_index=False)["p_set_MW"].sum()
            chk = lhs.merge(rhs, on=["country", "snapshot"], how="outer").fillna(0.0)
            chk["diff"] = chk["p_set_MW"] - chk["demand_MW_mean"]
            max_abs = float(chk["diff"].abs().max()) if len(chk) else 0.0
            print(f"[SANITY] Demand conservation max|diff| = {max_abs:.6e} MW")

    print("\nDone.")


if __name__ == "__main__":
    main()