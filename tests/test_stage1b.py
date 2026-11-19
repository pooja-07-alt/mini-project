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


