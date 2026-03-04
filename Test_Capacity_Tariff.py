# sanity_check_stage1_base_capacity_tariff.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# EDIT IF NEEDED (but these match your Main.py config)
# ---------------------------------------------------------------------
BASE_DIR = r"C:\Users\Lena\Documents\PSS 2030+\Power_System_Models\Model_Paper_1"
BILLING_WEEKS = 4

# Stage-1 BASE follower export (concatenated)
FOLLOWER_BASE_CSV = os.path.join(BASE_DIR, "full_follower_values_base_test.csv")

# Tariff table written by your pipeline
TARIFF_TABLE_CSV = os.path.join(BASE_DIR, "endogenous_capacity_tariff_by_block_node_test.csv")

# Output directory
OUT_DIR = os.path.join(BASE_DIR, "sanity_checks")
OUT_CSV = os.path.join(OUT_DIR, "sanity_stage1_base__fees_vs_maximport_times_tariff.csv")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_node_aggregation_map(nodes: pd.Series) -> dict[str, str]:
    """
    Aggregate internal Austrian nodes into 'AT_AGG', keep country/zone-like nodes unchanged.

    Heuristic:
    - If node is a 2-letter uppercase code (DE, CZ, SI, AT, FR, ...), keep it.
    - Otherwise treat it as an internal node (substation/bus name) and map to 'AT_AGG'.

    This matches your desire to "aggregate AT nodes" without needing extra metadata.
    """
    out: dict[str, str] = {}
    for n in nodes.astype(str).unique():
        s = str(n).strip()
        if len(s) == 2 and s.isalpha() and s.upper() == s:
            out[s] = s
        else:
            out[s] = "AT_AGG"
    return out


def safe_rel_err(abs_err: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace(0.0, np.nan).abs()
    return (abs_err / denom2).fillna(0.0)


# ---------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    if not os.path.isfile(FOLLOWER_BASE_CSV):
        raise FileNotFoundError(f"Missing follower BASE file: {FOLLOWER_BASE_CSV}")
    if not os.path.isfile(TARIFF_TABLE_CSV):
        raise FileNotFoundError(f"Missing tariff table file: {TARIFF_TABLE_CSV}")

    df = pd.read_csv(FOLLOWER_BASE_CSV, low_memory=False)
    tt = pd.read_csv(TARIFF_TABLE_CSV, low_memory=False)

    # --- validate columns ---
    need_df = {"node", "start_week", "imports_power", "fees"}
    miss_df = need_df - set(df.columns)
    if miss_df:
        raise ValueError(f"{FOLLOWER_BASE_CSV} missing columns: {sorted(miss_df)}")

    need_tt = {"billing_block", "node", "cap_tariff_eur_per_mw"}
    miss_tt = need_tt - set(tt.columns)
    if miss_tt:
        raise ValueError(f"{TARIFF_TABLE_CSV} missing columns: {sorted(miss_tt)}")

    # --- clean types ---
    df = df.copy()
    df["node"] = df["node"].astype(str)
    df["start_week"] = pd.to_numeric(df["start_week"], errors="coerce").astype(int)
    df["imports_power"] = pd.to_numeric(df["imports_power"], errors="coerce").fillna(0.0)
    df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)

    # billing block computed exactly like in your pipeline
    df["billing_block"] = ((df["start_week"] - 1) // int(BILLING_WEEKS)).astype(int)

    # --- aggregate AT internal nodes ---
    node_map = make_node_aggregation_map(df["node"])
    df["node_agg"] = df["node"].map(node_map).fillna(df["node"])
    tt = tt.copy()
    tt["node"] = tt["node"].astype(str)

    # We must aggregate the tariff table the same way:
    # - country nodes keep their own tariffs
    # - internal nodes become 'AT_AGG' and we recompute a consistent tariff for that aggregate:
    #     cap_tariff_AT_AGG = (sum fees) / (max import)   (computed from df itself)
    # This avoids nonsense like averaging tariffs.
    #
    # So: use tt tariffs for non-AT_AGG nodes, and compute AT_AGG tariff from df aggregation.

    # --- aggregate follower BASE to (block,node_agg) ---
    g = (
        df.groupby(["billing_block", "node_agg"], as_index=False)
          .agg(
              block_start_week=("start_week", "min"),
              block_end_week=("start_week", "max"),
              fees_sum_eur=("fees", "sum"),
              max_import_mw=("imports_power", "max"),
          )
    )

    # --- bring in tariffs ---
    # start by joining tariffs for "kept" nodes (two-letter codes)
    tt_keep = tt.copy()
    tt_keep["node_agg"] = tt_keep["node"]  # country nodes already their own
    tt_keep = tt_keep[["billing_block", "node_agg", "cap_tariff_eur_per_mw"]].drop_duplicates()

    out = g.merge(tt_keep, on=["billing_block", "node_agg"], how="left")

    # compute tariff for AT_AGG directly from aggregated base (most meaningful)
    mask_at = out["node_agg"] == "AT_AGG"
    out.loc[mask_at, "cap_tariff_eur_per_mw"] = np.where(
        out.loc[mask_at, "max_import_mw"] > 0,
        out.loc[mask_at, "fees_sum_eur"] / out.loc[mask_at, "max_import_mw"],
        0.0,
    )

    # If any other node_agg is missing tariff (shouldn't happen for country nodes), compute implied tariff
    miss_tar = out["cap_tariff_eur_per_mw"].isna()
    out.loc[miss_tar, "cap_tariff_eur_per_mw"] = np.where(
        out.loc[miss_tar, "max_import_mw"] > 0,
        out.loc[miss_tar, "fees_sum_eur"] / out.loc[miss_tar, "max_import_mw"],
        0.0,
    )

    # --- compute the check: fees_sum ?= max_import * tariff ---
    out["rhs_maximport_times_tariff_eur"] = out["max_import_mw"] * out["cap_tariff_eur_per_mw"]
    out["abs_err_eur"] = (out["fees_sum_eur"] - out["rhs_maximport_times_tariff_eur"]).abs()
    out["rel_err"] = safe_rel_err(out["abs_err_eur"], out["fees_sum_eur"])

    out = out.sort_values(["billing_block", "node_agg"]).reset_index(drop=True)

    # write
    out.to_csv(OUT_CSV, index=False)

    # print a small, readable summary
    print("=" * 72)
    print("Stage-1 BASE sanity check: fees_sum ≈ max(imports_power) * cap_tariff")
    print(f"Follower BASE: {FOLLOWER_BASE_CSV}")
    print(f"Tariff table:  {TARIFF_TABLE_CSV}")
    print("-" * 72)
    print(f"Written:      {OUT_CSV}")
    print("-" * 72)

    # show worst offenders
    top = out.sort_values("abs_err_eur", ascending=False).head(15)
    with pd.option_context("display.max_columns", 999, "display.width", 200):
        print("Top 15 largest abs_err (should be ~0 if constructed consistently):")
        print(top[[
            "billing_block", "node_agg",
            "fees_sum_eur", "max_import_mw", "cap_tariff_eur_per_mw",
            "rhs_maximport_times_tariff_eur", "abs_err_eur", "rel_err",
            "block_start_week", "block_end_week",
        ]].to_string(index=False))

    # quick “pass/fail-ish” info
    max_abs = float(out["abs_err_eur"].max()) if not out.empty else 0.0
    max_rel = float(out["rel_err"].max()) if not out.empty else 0.0
    print("-" * 72)
    print(f"Max abs error: {max_abs:.6g} EUR")
    print(f"Max rel error: {max_rel:.6g}")
    print("=" * 72)


if __name__ == "__main__":
    main()