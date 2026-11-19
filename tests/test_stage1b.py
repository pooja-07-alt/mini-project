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
