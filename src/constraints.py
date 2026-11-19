"""
constraints.py
──────────────
Applies hard engineering, hydrological, and environmental constraints
to prune unfeasible stream candidates before AHP multi-criteria scoring
and ML model training.

Why Hard Constraints?
----------------------
Machine learning models are good at subtle patterns, but obvious physical
impossibilities (like building inside a lake, in a protected national forest,
or across a 300-metre-wide chasm) should be eliminated upfront by strict rules.
Furthermore, negative samples for ML training must ONLY be drawn from stream
locations that passed physical feasibility; otherwise, the ML model wastes capacity
distinguishing dry riverbeds from deep lakes.

All thresholds are read from config.yaml and marked:
  ASSUMPTION - verify against IMSD / Karnataka check dam siting guidelines.

Outputs:
  - data/tables/candidates_v2.csv
  - outputs/stage1b_constraint_funnel.csv

SYNTHETIC - pipeline test, not a real finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_hard_constraints(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter candidate sites sequentially and construct a constraint funnel table.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stream candidate rows (synthetic v2 or real geospatial table).
    config : dict, optional
        Loaded configuration dictionary.

    Returns
    -------
    passed_df : pd.DataFrame
        Sites that met all feasibility constraints.
    funnel_df : pd.DataFrame
        Step-by-step funnel audit tracking candidate and positive counts.
    """
    if config is None:
        config = load_config()

    c_cfg = config.get("constraints", {})

    # Extract thresholds from config (no magic numbers in code)
    # Every threshold below is ASSUMPTION - verify against Karnataka guidelines
    so_min = c_cfg.get("stream_order_min", 1)  # ASSUMPTION - verify
    so_max = c_cfg.get("stream_order_max", 3)  # ASSUMPTION - verify
    slope_min = c_cfg.get("slope_deg_min", 1.0)  # ASSUMPTION - verify
    slope_max = c_cfg.get("slope_deg_max", 15.0)  # ASSUMPTION - verify
    min_catchment = c_cfg.get("min_catchment_area_km2", 1.0)  # ASSUMPTION - verify
    max_valley_w = c_cfg.get("max_valley_width_m", 120.0)  # ASSUMPTION - verify
    excluded_lulc = set(c_cfg.get("excluded_lulc_classes", ["built_up", "water", "forest"]))  # ASSUMPTION - verify
    min_dist_settlement = c_cfg.get("min_dist_to_settlement_m", 150.0)  # ASSUMPTION - verify
    exclude_protected = c_cfg.get("exclude_protected_areas", True)  # ASSUMPTION - verify

    # Prepare funnel tracking list
    funnel_rows = []
    current_df = df.copy()

    def record_funnel_step(step_name: str, before_df: pd.DataFrame, after_df: pd.DataFrame):
        """Helper to track how many total sites and positive labels were dropped."""
        n_before = len(before_df)
        n_after = len(after_df)
        n_removed = n_before - n_after

        pos_before = int(before_df["label"].sum()) if "label" in before_df else 0
        pos_after = int(after_df["label"].sum()) if "label" in after_df else 0
        pos_removed = pos_before - pos_after

        funnel_rows.append({
            "step": step_name,
            "sites_remaining": n_after,
            "sites_removed": n_removed,
            "positives_remaining": pos_after,
            "positives_removed": pos_removed,
            "pct_remaining": round((n_after / len(df)) * 100.0, 2),
        })

    # Initial baseline state
    record_funnel_step("0_initial_raw_candidates", current_df, current_df)

    # ── Rule 1: Stream Order (check dams are built on 1st to 3rd order streams) ──
    # ASSUMPTION - verify: 4th and 5th order streams have destructive monsoon torrents
    mask_so = (current_df["stream_order"] >= so_min) & (current_df["stream_order"] <= so_max)
    next_df = current_df[mask_so]
    record_funnel_step(f"1_stream_order_within_{so_min}_to_{so_max}", current_df, next_df)
    current_df = next_df

    # ── Rule 2: Slope (gentle to moderate slope required) ────────────────────────
    # ASSUMPTION - verify: <1 deg causes stagnant backwater; >15 deg risks scouring & dam breach
    mask_slope = (current_df["slope_deg"] >= slope_min) & (current_df["slope_deg"] <= slope_max)
    next_df = current_df[mask_slope]
    record_funnel_step(f"2_slope_deg_within_{slope_min}_to_{slope_max}", current_df, next_df)
    current_df = next_df

    # ── Rule 3: Catchment Area (minimum upstream drainage) ───────────────────────
    # ASSUMPTION - verify: too small catchment cannot capture sufficient runoff
    mask_catchment = current_df["catchment_area_km2"] >= min_catchment
    next_df = current_df[mask_catchment]
    record_funnel_step(f"3_min_catchment_area_{min_catchment}_km2", current_df, next_df)
    current_df = next_df

    # ── Rule 4: Valley Width (narrow cross-section preferred) ───────────────────
    # ASSUMPTION - verify: wider valleys require excessive construction material/cost
    mask_valley = current_df["valley_width_m"] <= max_valley_w
    next_df = current_df[mask_valley]
    record_funnel_step(f"4_max_valley_width_{max_valley_w}_m", current_df, next_df)
    current_df = next_df

    # ── Rule 5: LULC Inclusions & Exclusions ────────────────────────────────────
    # ASSUMPTION - verify: cannot construct on built-up settlements, open water bodies, or reserved forests
    mask_lulc = ~current_df["lulc_class"].isin(excluded_lulc)
    next_df = current_df[mask_lulc]
    record_funnel_step("5_exclude_built_up_water_forest", current_df, next_df)
    current_df = next_df

    # ── Rule 6: Protected Areas ─────────────────────────────────────────────────
    # ASSUMPTION - verify: statutory ecological / wildlife sanctuary exclusion
    if exclude_protected and "protected_area_flag" in current_df.columns:
        mask_prot = current_df["protected_area_flag"] == 0
        next_df = current_df[mask_prot]
        record_funnel_step("6_exclude_protected_areas", current_df, next_df)
        current_df = next_df

    # ── Rule 7: Minimum Distance from Settlements ───────────────────────────────
    # ASSUMPTION - verify: avoid inundating houses or posing direct flash-flood hazard
    mask_settlement = current_df["dist_to_settlement_m"] >= min_dist_settlement
    next_df = current_df[mask_settlement]
    record_funnel_step(f"7_min_dist_settlement_{min_dist_settlement}_m", current_df, next_df)
    current_df = next_df

    funnel_df = pd.DataFrame(funnel_rows)
    passed_df = current_df.reset_index(drop=True)
    return passed_df, funnel_df


def main():
    parser = argparse.ArgumentParser(description="Filter candidates by hard engineering constraints.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)

    in_path = project_root / config["paths"]["synthetic_v2_csv"]
    out_candidates = project_root / config["paths"]["candidates_filtered_csv"]
    out_funnel = project_root / config["paths"]["constraint_funnel_csv"]

    print("=" * 65)
    print("STAGE 1b - STEP 1: HARD CONSTRAINT FILTERING")
    print("=" * 65)
    print(f"Loading raw candidates: {in_path.relative_to(project_root)}")
    df_raw = pd.read_csv(in_path)

    passed_df, funnel_df = apply_hard_constraints(df_raw, config=config)

    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    out_funnel.parent.mkdir(parents=True, exist_ok=True)

    passed_df.to_csv(out_candidates, index=False)
    funnel_df.to_csv(out_funnel, index=False)

    print("\nConstraint Funnel Summary:")
    print(funnel_df.to_string(index=False))

    total_raw = len(df_raw)
    total_passed = len(passed_df)
    pos_raw = int(df_raw["label"].sum())
    pos_passed = int(passed_df["label"].sum())
    pos_removed = pos_raw - pos_passed

    print("\n" + "-" * 65)
    print(f"Total sites:      {total_raw} → {total_passed} passed ({total_passed/total_raw:.1%})")
    print(f"Positive labels:  {pos_raw} → {pos_passed} preserved ({pos_removed} removed, {pos_removed/pos_raw:.1%} drop rate)")
    print(f"Candidates saved: {out_candidates.relative_to(project_root)}")
    print(f"Funnel saved:     {out_funnel.relative_to(project_root)}")
    print("SYNTHETIC - pipeline test, not a real finding.")
    print("-" * 65)


if __name__ == "__main__":
    main()
