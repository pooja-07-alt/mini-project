"""
generate_synthetic.py
─────────────────────
Generates a synthetic feature table for Stage 1 ML warm-up.

The distributions are approximate — calibrated to resemble real Karnataka
sub-watershed hydrology, but NOT derived from real measurements.
Do NOT use this data for any real-world conclusion.

Output: data/tables/synthetic_checkdam_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 1000
# Positive class: ~25% of sites have a check dam (realistic imbalance)
POSITIVE_FRACTION = 0.25

def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    positive_fraction: float = POSITIVE_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of candidate check dam sites.

    Each row is one candidate stream pixel. Features are drawn from
    distributions designed so that certain feature combinations strongly
    predict the label (making the ML problem learnable but not trivial).

    Returns
    -------
    pd.DataFrame with columns matching ARCHITECTURE.md feature schema.
    """
    rng = np.random.default_rng(random_seed)
    n_pos = int(n_samples * positive_fraction)
    n_neg = n_samples - n_pos

    # ── Positive class (label = 1): existing check dam sites ──────────────────
    # Check dams tend to be on moderate slopes, moderate flow accumulation,
    # moderate-to-high TWI, lower stream orders, higher rainfall areas,
    # loamy/clay soils, moderate NDVI, closer to roads.

    pos = {
        "slope_deg":           rng.normal(loc=8.0,  scale=3.0,  size=n_pos).clip(1, 35),
        "flow_accum":          rng.lognormal(mean=7.5, sigma=0.8, size=n_pos),      # ~1800 pixels median
        "twi":                 rng.normal(loc=8.5,  scale=1.5,  size=n_pos).clip(3, 15),
        "stream_order":        rng.integers(1, 4,   size=n_pos),                   # 1, 2, or 3
        "annual_rainfall_mm":  rng.normal(loc=900,  scale=150,  size=n_pos).clip(400, 1600),
        "soil_texture_class":  rng.choice([2, 3], p=[0.55, 0.45], size=n_pos),     # loamy or clay
        "ndvi":                rng.normal(loc=0.40, scale=0.12, size=n_pos).clip(-0.1, 0.9),
        "dist_to_road_m":      rng.lognormal(mean=6.5, sigma=0.7, size=n_pos),     # ~665 m median
        "lulc_class":          rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15], size=n_pos),  # ag/shrub/forest
        "geology_permeability":rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2], size=n_pos),
        "label":               np.ones(n_pos, dtype=int),
    }

    # ── Negative class (label = 0): non-dam stream pixels ─────────────────────
    # Non-dam sites spread more widely across feature space
    neg = {
        "slope_deg":           rng.normal(loc=15.0, scale=7.0,  size=n_neg).clip(0.5, 60),
        "flow_accum":          rng.lognormal(mean=6.0, sigma=1.2, size=n_neg),
        "twi":                 rng.normal(loc=6.0,  scale=2.5,  size=n_neg).clip(2, 15),
        "stream_order":        rng.integers(1, 6,   size=n_neg),                   # 1–5
        "annual_rainfall_mm":  rng.normal(loc=750,  scale=200,  size=n_neg).clip(300, 1600),
        "soil_texture_class":  rng.choice([1, 2, 3], p=[0.35, 0.40, 0.25], size=n_neg),
        "ndvi":                rng.normal(loc=0.30, scale=0.18, size=n_neg).clip(-0.1, 0.9),
        "dist_to_road_m":      rng.lognormal(mean=7.8, sigma=1.0, size=n_neg),     # farther from roads
        "lulc_class":          rng.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15], size=n_neg),
        "geology_permeability":rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25], size=n_neg),
        "label":               np.zeros(n_neg, dtype=int),
    }

    df_pos = pd.DataFrame(pos)
    df_neg = pd.DataFrame(neg)
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # Add site_id and shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df.insert(0, "site_id", df.index + 1)

    # Add synthetic spatial block ID for Stage 1 spatial cross-validation
    # (5 geographic blocks — in real data these come from actual coordinates)
    df["spatial_block"] = rng.integers(1, 6, size=len(df))

    # Round to realistic precision
    df["slope_deg"] = df["slope_deg"].round(2)
    df["flow_accum"] = df["flow_accum"].round(0).astype(int)
    df["twi"] = df["twi"].round(3)
    df["annual_rainfall_mm"] = df["annual_rainfall_mm"].round(1)
    df["ndvi"] = df["ndvi"].round(4)
    df["dist_to_road_m"] = df["dist_to_road_m"].round(1)

    return df


def main():
    output_path = Path(__file__).parent.parent / "data" / "tables" / "synthetic_checkdam_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset()
    df.to_csv(output_path, index=False)

    print(f"✅  Synthetic dataset saved to: {output_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Class balance (label=1): {df['label'].mean():.1%}")
    print(f"\nColumn summary:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
