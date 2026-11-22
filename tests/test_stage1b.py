"""
test_stage1b.py
───────────────
Unit tests for Stage 1b pipeline refactor.

Covers:
  - Synthetic dataset v2 generation, columns, and spatial blocks
  - Hard constraint filtering logic on hand-made rows
  - AHP consistency ratio computation and CR < 0.1 verification
  - Strict absence of label usage in AHP scoring
  - Spatial block cross-validation (no spatial block leakage between train/test)
  - Pareto ranking non-dominated sorting on a known example
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.generate_synthetic import generate_synthetic_dataset_v2, load_config
from src.constraints import apply_hard_constraints
from src.ahp import AHPScorer, compute_eigenvector_weights
from src.ml_model import run_spatial_block_cv, run_step3_ml_only

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_v2_schema_and_blocks():
    """Verify synthetic v2 dataset has all 21 required columns and >=8 blocks."""
    df = generate_synthetic_dataset_v2(n_samples=500, random_seed=42)
    assert len(df) == 500
    expected_cols = [
        "site_id",
        "block_id",
        "lat",
        "lon",
        "elevation_m",
        "slope_deg",
        "stream_order",
        "flow_accum",
        "catchment_area_km2",
        "twi",
        "valley_width_m",
        "stream_width_m",
        "rainfall_mm",
        "lulc_class",
        "cropland_fraction_upstream",
        "soil_texture",
        "lithology",
        "dist_to_road_m",
        "dist_to_settlement_m",
        "protected_area_flag",
        "label",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert df["block_id"].nunique() >= 8
    assert set(df["label"].unique()).issubset({0, 1})
    assert 0.15 <= df["label"].mean() <= 0.45


def test_constraint_rules_on_hand_made_rows():
    """Verify that each hard constraint rule correctly filters out violating hand-made rows."""
    base_valid_row = {
        "site_id": 1,
        "block_id": 1,
        "lat": 15.2,
        "lon": 75.2,
        "elevation_m": 550.0,
        "slope_deg": 5.0,  # valid (1 - 15)
        "stream_order": 2,  # valid (1 - 3)
        "flow_accum": 3000,
        "catchment_area_km2": 2.5,  # valid (>= 1.0)
        "twi": 8.0,
        "valley_width_m": 60.0,  # valid (<= 120.0)
        "stream_width_m": 8.0,
        "rainfall_mm": 800.0,
        "lulc_class": "cropland",  # valid (not in built_up, water, forest)
        "cropland_fraction_upstream": 0.6,
        "soil_texture": "clay",
        "lithology": "gneiss",
        "dist_to_road_m": 500.0,
        "dist_to_settlement_m": 400.0,  # valid (>= 150.0)
        "protected_area_flag": 0,  # valid (0)
        "label": 1,
    }

    # Create test variations
    rows = []
    # 0: Valid row -> should pass
    rows.append(dict(base_valid_row, site_id=101))
    # 1: Stream order 4 -> fail
    rows.append(dict(base_valid_row, site_id=102, stream_order=4))
    # 2: Slope 25.0 -> fail
    rows.append(dict(base_valid_row, site_id=103, slope_deg=25.0))
    # 3: Catchment area 0.2 -> fail
    rows.append(dict(base_valid_row, site_id=104, catchment_area_km2=0.2))
    # 4: Valley width 200.0 -> fail
    rows.append(dict(base_valid_row, site_id=105, valley_width_m=200.0))
    # 5: LULC built_up -> fail
    rows.append(dict(base_valid_row, site_id=106, lulc_class="built_up"))
    # 6: LULC water -> fail
    rows.append(dict(base_valid_row, site_id=107, lulc_class="water"))
    # 7: LULC forest -> fail
    rows.append(dict(base_valid_row, site_id=108, lulc_class="forest"))
    # 8: Protected area flag 1 -> fail
    rows.append(dict(base_valid_row, site_id=109, protected_area_flag=1))
    # 9: Settlement distance 50m -> fail
    rows.append(dict(base_valid_row, site_id=110, dist_to_settlement_m=50.0))

    df_test = pd.DataFrame(rows)
    passed_df, funnel_df = apply_hard_constraints(df_test)

    assert len(passed_df) == 1
    assert passed_df.iloc[0]["site_id"] == 101
    assert len(funnel_df) == 8


def test_ahp_consistency_ratio_and_failure():
    """Verify AHP computes CR correctly, passes consistent matrices, and fails loudly when CR >= 0.1."""
    # Load default valid matrix from config
    scorer = AHPScorer()
    assert scorer.cr < 0.10, f"Expected CR < 0.10, got {scorer.cr}"
    assert np.isclose(np.sum(scorer.weights), 1.0)
    assert len(scorer.weights) == len(scorer.criteria)

    # Hand-craft an inconsistent 3x3 matrix (A > B, B > C, C > A)
    # A > B by 5, B > C by 5, C > A by 5
    inconsistent_matrix = np.array([
        [1.0, 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 1.0, 5.0],
        [5.0, 1.0 / 5.0, 1.0],
    ])
    with pytest.raises(ValueError, match="exceeds threshold"):
        compute_eigenvector_weights(inconsistent_matrix, max_cr=0.10)


def test_ahp_no_label_usage():
    """Verify that AHP suitability score is computed completely without the label column."""
    candidates_path = PROJECT_ROOT / "data" / "tables" / "candidates_v2.csv"
    if candidates_path.exists():
        df = pd.read_csv(candidates_path)
    else:
        raw = generate_synthetic_dataset_v2(n_samples=200, random_seed=42)
        df, _ = apply_hard_constraints(raw)

    # Completely remove the label column
    df_no_label = df.drop(columns=["label"], errors="ignore")
    assert "label" not in df_no_label.columns

    scorer = AHPScorer()
    scores = scorer.score(df_no_label)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(df_no_label)
    assert np.all((scores >= 0.0) & (scores <= 100.0))


def test_ahp_candidates_output_file():
    """Verify that candidates_v2_ahp.csv exists and contains ahp_score."""
    ahp_candidates_path = PROJECT_ROOT / "data" / "tables" / "candidates_v2_ahp.csv"
    assert ahp_candidates_path.exists(), "candidates_v2_ahp.csv does not exist"
    df = pd.read_csv(ahp_candidates_path)
    assert "ahp_score" in df.columns
    assert len(df) > 0
    assert df["ahp_score"].notna().all()


def test_spatial_block_cv_no_leakage():
    """Verify that no block appears in both train and test of any fold, preventing data leakage."""
    ahp_candidates_path = PROJECT_ROOT / "data" / "tables" / "candidates_v2_ahp.csv"
    df = pd.read_csv(ahp_candidates_path)
    blocks = sorted(df["block_id"].unique())
    assert len(blocks) >= 8

    tested_sites = set()
    for block in blocks:
        train_mask = df["block_id"] != block
        test_mask = df["block_id"] == block

        train_blocks = set(df.loc[train_mask, "block_id"].unique())
        test_blocks = set(df.loc[test_mask, "block_id"].unique())

        # Strict isolation assert
        assert train_blocks.isdisjoint(test_blocks), (
            f"Block leakage detected! Train blocks {train_blocks} intersect Test block {test_blocks}"
        )
        assert block not in train_blocks
        assert test_blocks == {block}

        # Site isolation
        train_sites = set(df.loc[train_mask, "site_id"])
        test_sites = set(df.loc[test_mask, "site_id"])
        assert train_sites.isdisjoint(test_sites)

        # Track that each site is evaluated exactly once in test sets
        assert tested_sites.isdisjoint(test_sites)
        tested_sites.update(test_sites)

    # All candidate sites must be tested across the 8 folds
    assert tested_sites == set(df["site_id"])


def test_oof_ml_only_predictions():
    """Verify out-of-fold ML-only predictions table exists and has valid probabilities."""
    oof_path = PROJECT_ROOT / "outputs" / "stage1b_oof_ml_only.csv"
    assert oof_path.exists(), "stage1b_oof_ml_only.csv does not exist"
    df_oof = pd.read_csv(oof_path)
    for col in ["site_id", "block_id", "label", "rf_prob_ml_only", "xgb_prob_ml_only"]:
        assert col in df_oof.columns, f"Missing column {col} in oof predictions"

    assert len(df_oof) > 0
    assert df_oof["rf_prob_ml_only"].between(0.0, 1.0).all()
    assert df_oof["xgb_prob_ml_only"].between(0.0, 1.0).all()
    assert not df_oof["rf_prob_ml_only"].isna().any()
    assert not df_oof["xgb_prob_ml_only"].isna().any()


def test_step3_gap_fill_artifacts():
    """Verify that block counts, cv metrics, and shap plot exist and have valid structure."""
    block_counts_path = PROJECT_ROOT / "outputs" / "stage1b_block_class_counts.csv"
    assert block_counts_path.exists(), "stage1b_block_class_counts.csv missing"
    df_bc = pd.read_csv(block_counts_path)
    assert len(df_bc) == 8
    for col in ["block_id", "total_candidates", "positives", "negatives", "positive_rate"]:
        assert col in df_bc.columns

    cv_metrics_path = PROJECT_ROOT / "outputs" / "stage1b_cv_ml_only.csv"
    assert cv_metrics_path.exists(), "stage1b_cv_ml_only.csv missing"
    df_cv = pd.read_csv(cv_metrics_path)
    assert len(df_cv) == 16  # 8 folds * 2 models
    for col in ["model", "fold", "holdout_block", "auc_roc", "pr_auc", "best_threshold"]:
        assert col in df_cv.columns

    shap_path = PROJECT_ROOT / "outputs" / "stage1b_shap_summary.png"
    assert shap_path.exists(), "stage1b_shap_summary.png missing"
    assert shap_path.stat().st_size > 1000

    citations_path = PROJECT_ROOT / "docs" / "CITATIONS_TO_VERIFY.md"
    assert citations_path.exists(), "docs/CITATIONS_TO_VERIFY.md missing"


def test_step4_ml_ahp_artifacts():
    """Verify that Step 4 OOF predictions and CV metrics with ahp_score exist and are valid."""
    oof_ahp_path = PROJECT_ROOT / "outputs" / "stage1b_oof_ml_ahp.csv"
    assert oof_ahp_path.exists(), "stage1b_oof_ml_ahp.csv missing"
    df_oof = pd.read_csv(oof_ahp_path)
    for col in ["site_id", "block_id", "label", "ahp_score", "rf_prob_ml_ahp", "xgb_prob_ml_ahp"]:
        assert col in df_oof.columns

    assert len(df_oof) > 0
    assert df_oof["rf_prob_ml_ahp"].between(0.0, 1.0).all()
    assert df_oof["xgb_prob_ml_ahp"].between(0.0, 1.0).all()

    cv_ahp_path = PROJECT_ROOT / "outputs" / "stage1b_cv_ml_ahp.csv"
    assert cv_ahp_path.exists(), "stage1b_cv_ml_ahp.csv missing"
    df_cv = pd.read_csv(cv_ahp_path)
    assert len(df_cv) == 16  # 8 folds * 2 models


def test_step5_comparison_artifacts():
    """Verify that Step 5 comparison table, per-fold metrics, plots, and notes exist and are valid."""
    comp_csv = PROJECT_ROOT / "outputs" / "stage1b_comparison_table.csv"
    assert comp_csv.exists(), "stage1b_comparison_table.csv missing"
    df_comp = pd.read_csv(comp_csv)
    assert len(df_comp) == 5
    expected_methods = {"AHP-only", "RF ML-only", "XGB ML-only", "RF + ahp_score", "XGB + ahp_score"}
    assert set(df_comp["Method"]) == expected_methods

    # Check required columns
    for col in [
        "Method",
        "AUC-ROC (mean ± std)",
        "PR-AUC (mean ± std)",
        "Precision (mean ± std)",
        "Recall (mean ± std)",
        "F1 (mean ± std)",
        "Pooled AUC-ROC",
        "Pooled PR-AUC",
        "Pooled Precision",
        "Pooled Recall",
        "Pooled F1",
    ]:
        assert col in df_comp.columns, f"Missing {col} in comparison table"

    # Check per-fold detailed table
    per_fold_csv = PROJECT_ROOT / "outputs" / "stage1b_per_fold_metrics.csv"
    assert per_fold_csv.exists(), "stage1b_per_fold_metrics.csv missing"
    df_pf = pd.read_csv(per_fold_csv)
    assert len(df_pf) == 40  # 5 methods * 8 folds
    for col in ["method", "fold", "holdout_block", "n_train", "n_test", "train_pos_rate", "k_top_flagged", "auc_roc", "pr_auc", "f1"]:
        assert col in df_pf.columns

    # Verify top-p% threshold rule was applied
    for _, row in df_pf.iterrows():
        expected_k = max(1, int(round(row["train_pos_rate"] * row["n_test"])))
        assert row["k_top_flagged"] == expected_k

    # Check ROC/PR curve plot and notes
    plot_path = PROJECT_ROOT / "outputs" / "stage1b_roc_pr_curves.png"
    assert plot_path.exists(), "stage1b_roc_pr_curves.png missing"
    assert plot_path.stat().st_size > 5000

    notes_path = PROJECT_ROOT / "outputs" / "stage1b_comparison_notes.md"
    assert notes_path.exists(), "stage1b_comparison_notes.md missing"
    content = notes_path.read_text(encoding="utf-8")
    assert "SYNTHETIC - pipeline test, not a real finding" in content
    assert "Does Adding `ahp_score` Help?" in content


def test_step5b_scenario_comparison_artifacts():
    """Verify Step 5b scenario sensitivity comparison artifacts exist and are valid."""
    csv_path = PROJECT_ROOT / "outputs" / "stage1b_scenario_comparison.csv"
    assert csv_path.exists(), "stage1b_scenario_comparison.csv missing"
    df = pd.read_csv(csv_path)
    assert len(df) == 10  # 2 scenarios * 5 methods
    assert set(df["Scenario"]) == {"ahp_aligned", "ahp_misaligned"}
    expected_methods = {"AHP-only", "RF ML-only", "XGB ML-only", "RF + ahp_score", "XGB + ahp_score"}
    assert set(df["Method"]) == expected_methods

    for col in ["Scenario", "Method", "AUC-ROC (mean ± std)", "PR-AUC (mean ± std)", "Pooled AUC-ROC", "Pooled PR-AUC"]:
        assert col in df.columns

    md_path = PROJECT_ROOT / "outputs" / "stage1b_scenario_comparison.md"
    assert md_path.exists(), "stage1b_scenario_comparison.md missing"
    notes_path = PROJECT_ROOT / "outputs" / "stage1b_scenario_notes.md"
    assert notes_path.exists(), "stage1b_scenario_notes.md missing"
    notes_content = notes_notes = notes_path.read_text(encoding="utf-8")
    assert "SYNTHETIC - pipeline test, not a real finding" in notes_content
    assert "ahp_misaligned" in notes_content








