# =============================================================================
# GRID-ELEMENTS → COMBINED NODE BACKBONE EXPORTER (TIELINES + OPTIONAL AT LINES)
# =============================================================================
# This script exports a *single, combined* grid-elements table (CSV) from the
# Static_Grid_Model workbook that provides a consistent node representation for:
#   (a) cross-border interconnectors ("tielines"), and optionally
#   (b) Austrian internal transmission lines ("line_at", APG-owned).
#
# The resulting CSV is a core backbone for downstream country→node disaggregation
# (e.g., allocating TYNDP demand/generation to nodes), because it defines the
# model’s node set via the columns `bus_u` / `bus_v`.
#
# What it produces
# ----------------
#   - Lines_AT_and_Tielines.csv (aka OUT_CSV):
#       A combined elements table with (at minimum):
#         element_type, element_id,
#         country_u, country_v,
#         bus_u, bus_v,
#         electrical parameters (voltage_kv, r_ohm, x_ohm, b_series_S, b_shunt_S, ...),
#         capacity proxies (imax_A_used, smax_MVA, fmax_MW),
#         provenance (source_sheet, row_index) and selected raw fields (eic_code, ne_name_old).
#
# Inputs
# ------
#   - Static_Grid_Model.xlsx
#       * Sheet `Tielines_New`: cross-border tielines (with free-text country hints)
#       * Sheet `Lines_New`: internal lines (only used if INCLUDE_AT_INTERNAL_LINES=True)
#   - Substations_AT.py (optional but recommended when keeping AT substations):
#       Provides canonical AT substation keys (SUBSTATION_LOOKUP) used to map AT-side
#       tieline endpoints to stable node names.
#
# Node concept and Austria handling
# --------------------------------
# Nodes are derived from exported endpoints (`bus_u`, `bus_v`), with a deliberate
# toggle controlling Austria’s granularity:
#
#   - Non-AT countries:
#       Always represented as a single country node (e.g., "DE", "CZ", ...).
#
#   - Austria (AT):
#       Controlled by INCLUDE_AT_INTERNAL_LINES:
#         * INCLUDE_AT_INTERNAL_LINES = False
#             - Austria behaves like other countries: single node "AT"
#             - AT-side tieline endpoints are aggregated (bus_u/bus_v = "AT")
#             - No internal AT lines are exported
#         * INCLUDE_AT_INTERNAL_LINES = True
#             - Austrian substations are kept as explicit nodes (e.g., "BISAMBERG", ...)
#             - AT-side tieline endpoints are normalized and mapped to canonical keys
#               from Substations_AT.py when possible (fallback to "AT" otherwise)
#             - Austrian internal APG-owned lines are exported with element_type="line_at"
#
# Key mechanisms
# --------------
#   - Robust Excel parsing:
#       Reads multi-header sheets (header=[0,1]), safely handles duplicate bottom labels,
#       and cleans cells consistently for downstream parsing.
#
#   - Country pair inference for tielines:
#       Uses the tieline `comment` text plus the owning `tso` to infer the country pair
#       (e.g., "AT-DE"), including alias replacement and regex-based patterns.
#
#   - Austrian substation canonicalization:
#       Normalizes substation strings (removes "UW"/"SUBSTATION", folds umlauts, strips
#       punctuation) and matches against canonical keys from Substations_AT.py.
#
#   - Electrical parameter & limit extraction:
#       Pulls voltage and line parameters from the sheet, chooses an effective `imax`
#       (fixed > max(period1..6) > dlrmax > dlrmin), and derives smax_MVA and fmax_MW
#       from voltage_kv and imax.
#
# Outputs and checks
# ------------------
#   - Prints a cross-border AT partner summary (sum of smax_MVA by neighbor country).
#   - Prints sanity tables for all tielines, and (if enabled) AT internal lines.
#   - Writes the combined CSV to OUT_CSV with a stable, model-friendly column order.
#
# Design intent
# -------------
# This exporter is intentionally conservative and modular:
#   - With the toggle off, the model sees Austria as a single-node control area.
#   - With the toggle on, Austria gains nodal detail and internal topology suitable
#     for later within-AT allocation steps.
#   - You can evolve Substations_AT.py (lookup keys, shares, metadata) without changing
#     this exporter, as long as canonical substation keys remain stable.
# =============================================================================

import re
import math
import warnings
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Lena\Documents\PSS 2030+")
MODEL_DIR = PROJECT_ROOT / "Power_System_Models" / "Model_Paper_1"
GRID_DIR = MODEL_DIR / "Generation_Grid_Data" / "Grid_Data"
GRID_FILE = GRID_DIR / "Static_Grid_Model.xlsx"

SHEET_TIELINES = "Tielines_New"
SHEET_LINES = "Lines_New"

FILTER_TSO_AT = "APG"  # internal Austrian lines

# Toggle whether to include AT internal lines AND keep AT substations as nodes
INCLUDE_AT_INTERNAL_LINES: bool = True

OUT_CSV = GRID_DIR / "Lines_AT_and_Tielines.csv"

COUNTRY_RE = r"(AT|DE|CZ|SK|HU|SI|IT|CH|PL|FR|LU|BE|NL|HR|BA|RS|ME|RO|UA|BG|MD|LT|SE|ES|UK)"

SUBSTATIONS_AT_FILE = MODEL_DIR / "Substations_AT.py"

# ============================================================
# COUNTRY / TSO MAPPINGS
# ============================================================

COUNTRY_ALIASES = {
    "AUSTRIA": "AT",
    "BELGIUM": "BE",
    "SWITZERLAND": "CH",
    "CZECHIA": "CZ",
    "FRANCE": "FR",
    "GERMANY": "DE",
    "HUNGARY": "HU",
    "ITALY": "IT",
    "LUXEMBOURG": "LU",
    "NETHERLAND": "NL",
    "POLAND": "PL",
    "SLOVENIA": "SI",
    "SLOVAKIA": "SK",
}

TSO_TO_COUNTRY = {
    "50HERTZ": "DE",
    "AMPRION": "DE",
    "AMPRION GMBH": "DE",
    "APG": "AT",
    "CEPS": "CZ",
    "CREOS": "LU",
    "ELIA": "BE",
    "ELES": "SI",
    "MAVIR": "HU",
    "PSE": "PL",
    "RTE": "FR",
    "SEPS": "SK",
    "SWISSGRID": "CH",
    "TENNETGMBH": "DE",
    "TENNETNL": "NL",
    "TENNET NL": "NL",
    "TRANSNETBW": "DE",
}

def norm_tso(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or x is pd.NA:
        return None
    s = str(x).upper().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_tso_to_country(tso):
    tso = norm_tso(tso)
    if tso is None:
        return None
    return TSO_TO_COUNTRY.get(tso)

# ============================================================
# OPTIONAL: AUSTRIAN SUBSTATION LOOKUP (used to identify AT-side tieline node)
# ============================================================

def load_substation_lookup_keys(substations_file: Path) -> set[str]:
    """
    Loads canonical AT substation keys from Substations_AT.py via absolute file path.
    """
    import importlib.util

    if not substations_file.exists():
        raise FileNotFoundError(f"Substations_AT.py not found: {substations_file}")

    spec = importlib.util.spec_from_file_location("Substations_AT", str(substations_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import from: {substations_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lookup = getattr(mod, "SUBSTATION_LOOKUP", {})
    if not isinstance(lookup, dict):
        return set()

    return {str(k).strip().upper() for k in lookup.keys()}

# ============================================================
# ROBUST EXCEL READER (multiheader + duplicates-safe cleaning)
# ============================================================

def _clean_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip().replace("\u00a0", " ")
    if s.lower() in {"nan", "none", ""}:
        return pd.NA
    return s

def read_sheet_with_headers(path: Path, sheet_name: str):
    """
    Returns:
      df: columns renamed to bottom header (lowercased; duplicates allowed)
      cols_top: original top header labels (strings)
      cols_bottom: original bottom header labels (strings)
    """
    df0 = pd.read_excel(path, sheet_name=sheet_name, header=[0, 1])

    cols_top, cols_bottom = [], []
    for top, bottom in df0.columns:
        cols_top.append("" if pd.isna(top) else str(top).strip())
        cols_bottom.append("" if pd.isna(bottom) else str(bottom).strip())

    df0.columns = [b.lower() for b in cols_bottom]

    df0 = df0.astype("object")
    for j in range(df0.shape[1]):
        df0.iloc[:, j] = df0.iloc[:, j].map(_clean_cell)

    return df0, cols_top, cols_bottom

def detect_endpoint_columns(cols_top, cols_bottom):
    """
    Finds Substation_1 Full_name and Substation_2 Full_name column indices.
    Prefers multiheader names; falls back to first two 'full_name' columns.
    """
    top_norm = [str(t).strip().lower() for t in cols_top]
    bottom_norm = [str(b).strip().lower() for b in cols_bottom]

    idx1 = idx2 = None
    for i, (t, b) in enumerate(zip(top_norm, bottom_norm)):
        if b == "full_name":
            if "substation_1" in t and idx1 is None:
                idx1 = i
            elif "substation_2" in t and idx2 is None:
                idx2 = i

    if idx1 is not None and idx2 is not None:
        return idx1, idx2

    fulls = [i for i, b in enumerate(bottom_norm) if b == "full_name"]
    if len(fulls) >= 2:
        return fulls[0], fulls[1]

    raise ValueError("Could not detect two 'Full_name' endpoint columns.")

# ============================================================
# SUBSTATION NAME NORMALIZATION
# ============================================================

_UW_WORDS = ["UMSPANNWERK", "UW", "SUBSTATION", "SS", "STATION", "SCHALTANLAGE", "SCHALTWERK"]

def normalize_substation_name(x) -> str | None:
    if x is None or x is pd.NA or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None

    s = s.upper()
    s = s.replace("_", " ").replace("-", " ").replace("–", " ").replace("/", " ")
    s = re.sub(r"[(),.;:]", " ", s)

    for w in _UW_WORDS:
        s = re.sub(rf"\b{re.escape(w)}\b", " ", s)

    s = (s.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ẞ", "SS"))
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

# ============================================================
# PARSE COUNTRY PAIR FROM COMMENT + TSO
# ============================================================

def parse_country_pair(comment, tso_owner=None):
    if comment is None or comment is pd.NA or (isinstance(comment, float) and pd.isna(comment)):
        return None

    s = str(comment).upper()
    for name, code in COUNTRY_ALIASES.items():
        s = s.replace(name, code)

    m = re.search(rf"\bBETWEEN\s+{COUNTRY_RE}\s*[-–]\s*{COUNTRY_RE}\b", s)
    if m:
        return (m.group(1), m.group(2))

    m = re.search(rf"\b{COUNTRY_RE}\s*[-–]\s*{COUNTRY_RE}\b", s)
    if m:
        return (m.group(1), m.group(2))

    origin = map_tso_to_country(tso_owner)

    m = (
        re.search(rf"\bTIE[- ]?LINE\s+TO\s+{COUNTRY_RE}\b", s) or
        re.search(rf"\bTIELINE\s+TO\s+{COUNTRY_RE}\b", s) or
        re.search(rf"\bTIE\s+LINE\s+TO\s+{COUNTRY_RE}\b", s)
    )
    if m:
        dest = m.group(1)
        if origin is None or origin == dest:
            return None
        return (origin, dest)

    m = (
        re.search(r"\bTIE[- ]?LINE\s+WITH\s+([A-Z0-9 _-]+)\b", s) or
        re.search(r"\bTIELINE\s+WITH\s+([A-Z0-9 _-]+)\b", s) or
        re.search(r"\bTIE\s+LINE\s+WITH\s+([A-Z0-9 _-]+)\b", s)
    )
    if m:
        partner_txt = m.group(1).strip()
        partner_country = map_tso_to_country(partner_txt)

        if partner_country == "AT":
            if origin and origin != "AT":
                return ("AT", origin)
            return None

        if origin == "AT" and partner_country and partner_country != "AT":
            return ("AT", partner_country)

        if origin and partner_country and origin != partner_country:
            return (origin, partner_country)

    return None

# ============================================================
# PARAMETERS
# ============================================================

def to_float(x):
    if x is None or x is pd.NA or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace("\u00a0", " ")
    if s.lower() in {"nan", ""}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def choose_imax_A(row: pd.Series) -> float | None:
    fixed = to_float(row.get("fixed"))
    if fixed is not None:
        return fixed

    vals = []
    for p in range(1, 7):
        v = to_float(row.get(f"period {p}"))
        if v is not None:
            vals.append(v)
    if vals:
        return max(vals)

    dlrmax = to_float(row.get("dlrmax(a)"))
    if dlrmax is not None:
        return dlrmax

    dlrmin = to_float(row.get("dlrmin(a)"))
    if dlrmin is not None:
        return dlrmin

    return None

def safe_get(row: pd.Series, colname: str):
    if colname not in row.index:
        return pd.NA
    val = row[colname]
    if isinstance(val, pd.Series):
        for x in val.tolist():
            if x is not pd.NA and not (isinstance(x, float) and pd.isna(x)):
                return x
        return pd.NA
    return val

# ============================================================
# BUILD: TIELINES (AT endpoint depends on INCLUDE_AT_INTERNAL_LINES)
# ============================================================

def build_tielines_with_at_bus(
    df: pd.DataFrame,
    idx_u: int,
    idx_v: int,
    at_keys: set[str],
    *,
    keep_at_substations: bool,
):
    export_rows = []
    pair_counter = defaultdict(int)

    for i in range(len(df)):
        r = df.iloc[i, :]

        tso_owner = safe_get(r, "tso")
        origin = map_tso_to_country(tso_owner)
        if origin is None:
            continue

        pair = parse_country_pair(safe_get(r, "comment"), tso_owner)
        if not pair:
            continue

        a, b = pair[0], pair[1]
        if origin == a:
            dest = b
        elif origin == b:
            dest = a
        else:
            continue

        if origin == dest:
            continue

        s1_can = normalize_substation_name(df.iloc[i, idx_u])
        s2_can = normalize_substation_name(df.iloc[i, idx_v])

        key_sorted = tuple(sorted([origin, dest]))
        pair_counter[key_sorted] += 1
        k = pair_counter[key_sorted]
        id_left, id_right = key_sorted
        tieline_id = f"{id_left}{k}-{id_right}{k}"

        vk = to_float(safe_get(r, "voltage_level(kv)"))
        R = to_float(safe_get(r, "resistance_r(ω)"))
        X = to_float(safe_get(r, "reactance_x(ω)"))
        B_uS = to_float(safe_get(r, "susceptance_b(μs)"))
        L = to_float(safe_get(r, "length_(km)"))
        imax = choose_imax_A(r)

        b_series = (1.0 / X) if (X is not None and abs(X) > 1e-12) else None
        b_shunt_S = (B_uS * 1e-6) if B_uS is not None else None

        smax_MVA = None
        if vk is not None and imax is not None:
            smax_MVA = math.sqrt(3.0) * vk * (imax / 1000.0)

        # Default directional buses = country nodes
        bus_u = origin
        bus_v = dest

        # If we are NOT keeping AT substations, AT must be aggregated to "AT"
        if not keep_at_substations:
            if origin == "AT":
                bus_u = "AT"
            if dest == "AT":
                bus_v = "AT"
        else:
            # Keep AT as substation endpoint if possible
            if dest == "AT":
                bus_v = s2_can if (s2_can and s2_can in at_keys) else "AT"
            elif origin == "AT":
                bus_u = s1_can if (s1_can and s1_can in at_keys) else "AT"

        export_rows.append({
            "element_type": "tieline",
            "element_id": tieline_id,

            "country_u": origin,
            "country_v": dest,
            "bus_u": bus_u,
            "bus_v": bus_v,

            "voltage_kv": vk,
            "r_ohm": R,
            "x_ohm": X,
            "b_series_S": b_series,
            "b_shunt_S": b_shunt_S,
            "length_km": L,
            "imax_A_used": imax,
            "smax_MVA": smax_MVA,
            "fmax_MW": smax_MVA,

            "eic_code": safe_get(r, "eic_code"),
            "ne_name_old": safe_get(r, "ne_name"),
            "source_sheet": SHEET_TIELINES,
            "row_index": i,
        })

    return pd.DataFrame(export_rows)

# ============================================================
# BUILD: AUSTRIAN INTERNAL LINES (APG) (optional)
# ============================================================

def build_at_internal_lines(df: pd.DataFrame, idx_u: int, idx_v: int, *, enabled: bool = True):
    if not enabled:
        return pd.DataFrame(columns=[
            "element_type", "element_id",
            "country_u", "country_v", "bus_u", "bus_v",
            "voltage_kv", "r_ohm", "x_ohm", "b_series_S", "b_shunt_S",
            "length_km", "imax_A_used", "smax_MVA", "fmax_MW",
            "eic_code", "ne_name_old", "source_sheet", "row_index",
        ])

    export_rows = []
    pair_counter = defaultdict(int)

    for i in range(len(df)):
        r = df.iloc[i, :]

        tso = safe_get(r, "tso")
        if FILTER_TSO_AT:
            if tso is pd.NA or str(tso).strip().upper() != FILTER_TSO_AT.upper():
                continue

        s1_can = normalize_substation_name(df.iloc[i, idx_u])
        s2_can = normalize_substation_name(df.iloc[i, idx_v])
        if not s1_can or not s2_can or s1_can == s2_can:
            continue

        a, b = sorted([s1_can, s2_can])
        pair_counter[(a, b)] += 1
        k = pair_counter[(a, b)]
        line_id = f"{a}__{b}__{k}"

        vk = to_float(safe_get(r, "voltage_level(kv)"))
        R = to_float(safe_get(r, "resistance_r(ω)"))
        X = to_float(safe_get(r, "reactance_x(ω)"))
        B_uS = to_float(safe_get(r, "susceptance_b(μs)"))
        L = to_float(safe_get(r, "length_(km)"))
        imax = choose_imax_A(r)

        b_series = (1.0 / X) if (X is not None and abs(X) > 1e-12) else None
        b_shunt_S = (B_uS * 1e-6) if B_uS is not None else None

        smax_MVA = None
        if vk is not None and imax is not None:
            smax_MVA = math.sqrt(3.0) * vk * (imax / 1000.0)

        export_rows.append({
            "element_type": "line_at",
            "element_id": line_id,

            "country_u": "AT",
            "country_v": "AT",
            "bus_u": s1_can,
            "bus_v": s2_can,

            "voltage_kv": vk,
            "r_ohm": R,
            "x_ohm": X,
            "b_series_S": b_series,
            "b_shunt_S": b_shunt_S,
            "length_km": L,
            "imax_A_used": imax,
            "smax_MVA": smax_MVA,
            "fmax_MW": smax_MVA,

            "eic_code": safe_get(r, "eic_code"),
            "ne_name_old": safe_get(r, "ne_name"),
            "source_sheet": SHEET_LINES,
            "row_index": i,
        })

    return pd.DataFrame(export_rows)

# ============================================================
# SANITY CHECK
# ============================================================

def print_tielines_sanity(df: pd.DataFrame):
    df_tie = df[df["element_type"] == "tieline"].copy()

    print("\n================ ALL TIELINES (SANITY VIEW) ================\n")
    if df_tie.empty:
        print("No tielines found.\n")
        return

    cols = [
        "element_id",
        "country_u", "country_v",
        "bus_u", "bus_v",
        "voltage_kv",
        "smax_MVA",
        "length_km",
    ]
    cols = [c for c in cols if c in df_tie.columns]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    print(df_tie[cols].to_string(index=False))
    print("\nTotal tielines:", len(df_tie))

def print_lines_at_sanity(df: pd.DataFrame):
    df_at = df[df["element_type"] == "line_at"].copy()

    print("\n================ AUSTRIAN INTERNAL LINES (SANITY VIEW) ================\n")
    if df_at.empty:
        print("No Austrian lines found.\n")
        return

    cols = [
        "element_id",
        "bus_u", "bus_v",
        "voltage_kv",
        "smax_MVA",
        "length_km",
    ]
    cols = [c for c in cols if c in df_at.columns]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    print(df_at[cols].to_string(index=False))
    print("\nTotal Austrian internal lines:", len(df_at))
    
# ============================================================
# AT CROSS-BORDER SUMMARY (sum nominal apparent power)
# ============================================================

AT_COUNTRY_ORDER = ["CH", "CZ", "DE", "HU", "IT", "SI", "SK"]

def print_at_crossborder_snom(df: pd.DataFrame):
    """
    Identify all tielines with Austria on one side, group by partner country,
    sum up smax_MVA, and print in the requested order.
    """
    df_tie = df[df["element_type"] == "tieline"].copy()
    if df_tie.empty:
        print("\n[INFO] No tielines found -> no AT cross-border summary.\n")
        return

    # Keep only AT cross-border tielines
    mask = (df_tie["country_u"] == "AT") | (df_tie["country_v"] == "AT")
    df_at = df_tie.loc[mask].copy()
    if df_at.empty:
        print("\n[INFO] No AT cross-border tielines found.\n")
        return

    # Identify partner country (the non-AT side)
    df_at["partner_country"] = df_at["country_v"].where(
        df_at["country_u"] == "AT", df_at["country_u"]
    )

    # Ensure numeric
    df_at["smax_MVA"] = pd.to_numeric(df_at["smax_MVA"], errors="coerce").fillna(0.0)

    sums = df_at.groupby("partner_country")["smax_MVA"].sum()

    print("\n================ AT CROSS-BORDER CAPACITY SUMMARY ================\n")
    for country in AT_COUNTRY_ORDER:
        val = float(sums.get(country, 0.0))
        print(f"AT-{country}: {val:.6g}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # If we include AT internal lines, we also keep AT substations as nodes for tielines.
    keep_at_substations = bool(INCLUDE_AT_INTERNAL_LINES)

    at_keys = load_substation_lookup_keys(SUBSTATIONS_AT_FILE)
    print(f"[INFO] Loaded {len(at_keys)} AT substation keys for tieline assignment")
    print(f"[INFO] INCLUDE_AT_INTERNAL_LINES = {INCLUDE_AT_INTERNAL_LINES}")
    print(f"[INFO] keep_at_substations (tielines) = {keep_at_substations}")

    # --- Tielines
    df_tie, top_tie, bot_tie = read_sheet_with_headers(GRID_FILE, SHEET_TIELINES)
    tie_u, tie_v = detect_endpoint_columns(top_tie, bot_tie)
    df_tie_out = build_tielines_with_at_bus(
        df_tie, tie_u, tie_v, at_keys,
        keep_at_substations=keep_at_substations
    )

    # --- AT Lines (optional)
    if INCLUDE_AT_INTERNAL_LINES:
        df_lines, top_lines, bot_lines = read_sheet_with_headers(GRID_FILE, SHEET_LINES)
        line_u, line_v = detect_endpoint_columns(top_lines, bot_lines)
        df_lines_out = build_at_internal_lines(df_lines, line_u, line_v, enabled=True)
    else:
        df_lines_out = build_at_internal_lines(pd.DataFrame(), 0, 0, enabled=False)

    # --- Combined export
    out = pd.concat([df_tie_out, df_lines_out], ignore_index=True)

    print_at_crossborder_snom(out)

    # Optional: stable column order
    col_order = [
        "element_type", "element_id",
        "country_u", "country_v", "bus_u", "bus_v",
        "voltage_kv", "r_ohm", "x_ohm", "b_series_S", "b_shunt_S",
        "length_km", "imax_A_used", "smax_MVA", "fmax_MW",
        "eic_code", "ne_name_old", "source_sheet", "row_index",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    print_tielines_sanity(out)
    if INCLUDE_AT_INTERNAL_LINES:
        print_lines_at_sanity(out)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] Exported: {OUT_CSV}")
    print("Rows:", len(out))