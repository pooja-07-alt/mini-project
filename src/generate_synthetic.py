"""
generate_synthetic.py
─────────────────────
Generates synthetic feature tables for check dam suitability testing.

Supports:
  - Version 1: Original 1,000-sample warmup dataset (preserved for backward compatibility)
  - Version 2: Stage 1b refactored dataset (~3,000 stream candidates, 8 spatial blocks,
               expanded features, and a latent generating process with unobserved confounding
               and realistic noise targeting AUC ~0.75-0.90).

Outputs:
  - v1: data/tables/synthetic_checkdam_features.csv (left unchanged)
  - v2: data/tables/synthetic_checkdam_candidates_v2.csv

SYNTHETIC - pipeline test, not a real finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yaml

# ── Load Default Configuration ───────────────────────────────────────────────
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Version 1 Dataset Generator (Legacy)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset_v1(
    n_samples: int = 1000,
    positive_fraction: float = 0.25,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate legacy Stage 1 synthetic dataset."""
    rng = np.random.default_rng(random_seed)
    n_pos = int(n_samples * positive_fraction)
    n_neg = n_samples - n_pos

    pos = {
        "slope_deg": rng.normal(loc=8.0, scale=3.0, size=n_pos).clip(1, 35),
        "flow_accum": rng.lognormal(mean=7.5, sigma=0.8, size=n_pos),
        "twi": rng.normal(loc=8.5, scale=1.5, size=n_pos).clip(3, 15),
        "stream_order": rng.integers(1, 4, size=n_pos),
        "annual_rainfall_mm": rng.normal(loc=900, scale=150, size=n_pos).clip(400, 1600),
        "soil_texture_class": rng.choice([2, 3], p=[0.55, 0.45], size=n_pos),
        "ndvi": rng.normal(loc=0.40, scale=0.12, size=n_pos).clip(-0.1, 0.9),
        "dist_to_road_m": rng.lognormal(mean=6.5, sigma=0.7, size=n_pos),
        "lulc_class": rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15], size=n_pos),
        "geology_permeability": rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2], size=n_pos),
        "label": np.ones(n_pos, dtype=int),
    }

    neg = {
        "slope_deg": rng.normal(loc=15.0, scale=7.0, size=n_neg).clip(0.5, 60),
        "flow_accum": rng.lognormal(mean=6.0, sigma=1.2, size=n_neg),
        "twi": rng.normal(loc=6.0, scale=2.5, size=n_neg).clip(2, 15),
        "stream_order": rng.integers(1, 6, size=n_neg),
        "annual_rainfall_mm": rng.normal(loc=750, scale=200, size=n_neg).clip(300, 1600),
        "soil_texture_class": rng.choice([1, 2, 3], p=[0.35, 0.40, 0.25], size=n_neg),
        "ndvi": rng.normal(loc=0.30, scale=0.18, size=n_neg).clip(-0.1, 0.9),
        "dist_to_road_m": rng.lognormal(mean=7.8, sigma=1.0, size=n_neg),
        "lulc_class": rng.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15], size=n_neg),
        "geology_permeability": rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25], size=n_neg),
        "label": np.zeros(n_neg, dtype=int),
    }

    df_pos = pd.DataFrame(pos)
    df_neg = pd.DataFrame(neg)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df.insert(0, "site_id", df.index + 1)
    df["spatial_block"] = rng.integers(1, 6, size=len(df))

    df["slope_deg"] = df["slope_deg"].round(2)
    df["flow_accum"] = df["flow_accum"].round(0).astype(int)
    df["twi"] = df["twi"].round(3)
    df["annual_rainfall_mm"] = df["annual_rainfall_mm"].round(1)
    df["ndvi"] = df["ndvi"].round(4)
    df["dist_to_road_m"] = df["dist_to_road_m"].round(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Version 2 Dataset Generator (Stage 1b)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset_v2(
    config: Optional[dict] = None,
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic v2 candidate stream sites.

    Key improvements over v1:
      1. Features are generated for stream candidates first, covering realistic terrain,
         hydrology, soil, geology, land use, and infrastructure.
      2. Spatial blocks (>=8) are determined by geographic latitude/longitude grid cells.
      3. Labels are produced via a latent suitability process that includes observed
         environmental drivers, an unobserved geotechnical/subsurface factor, and noise.
         This prevents artificial 0.99 AUC and bounds model performance to realistic 0.75-0.90.
    """
    if config is None:
        config = load_config()

    syn_cfg = config.get("synthetic_v2", {})
    n = n_samples or syn_cfg.get("n_samples", 3000)
    seed = random_seed if random_seed is not None else syn_cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    lat_min, lat_max = syn_cfg.get("lat_range", [15.10, 15.65])
    lon_min, lon_max = syn_cfg.get("lon_range", [75.05, 75.60])

    # ── 1. Coordinates and Spatial Blocks (8 contiguous geographic blocks) ───
    # 2 latitude slices x 4 longitude slices = 8 spatial blocks
    lat = rng.uniform(lat_min, lat_max, size=n)
    lon = rng.uniform(lon_min, lon_max, size=n)

    lat_mid = (lat_min + lat_max) / 2.0
    lon_cuts = np.linspace(lon_min, lon_max, 5)

    lat_bin = np.where(lat >= lat_mid, 1, 0)
    lon_bin = np.clip(np.digitize(lon, lon_cuts[1:-1]), 0, 3)
    block_id = (lat_bin * 4 + lon_bin + 1).astype(int)

    # ── 2. Terrain & Elevation ───────────────────────────────────────────────
    # Elevation trends with latitude and longitude (sloping topography)
    rel_elev = (lat - lat_min) / (lat_max - lat_min) * 180 + (lon - lon_min) / (lon_max - lon_min) * 120
    elevation_m = 500.0 + rel_elev + rng.normal(0, 25, size=n)
    elevation_m = np.clip(elevation_m, 450.0, 950.0).round(1)

    # Slope in degrees (exponential distribution resembling typical Deccan terrain)
    slope_deg = rng.exponential(scale=5.5, size=n) + 0.5
    slope_deg = np.clip(slope_deg, 0.5, 32.0).round(2)

    # Stream order (Strahler order: 1=smallest, up to 5)
    stream_order = rng.choice([1, 2, 3, 4, 5], p=[0.42, 0.32, 0.16, 0.07, 0.03], size=n)

    # Flow accumulation (pixels) correlated with stream order
    base_fa = np.exp(stream_order * 1.5 + 4.0)
    flow_accum = (base_fa * rng.lognormal(0.0, 0.5, size=n)).round(0).astype(int)

    # Catchment area in km2 (proportional to flow accumulation: 30m x 30m pixel = 900 m2)
    catchment_area_km2 = (flow_accum * 0.0009 * rng.uniform(0.85, 1.15, size=n)).round(2)
    catchment_area_km2 = np.clip(catchment_area_km2, 0.1, 150.0)

    # Topographic Wetness Index (TWI) = ln(a / tan(beta))
    slope_rad = np.radians(np.clip(slope_deg, 0.5, 45.0))
    twi = (np.log(np.maximum(flow_accum, 1.0) / np.tan(slope_rad)) * 0.8 + rng.normal(0, 0.5, size=n)).round(3)
    twi = np.clip(twi, 2.0, 18.0)

    # Valley width (m) and stream width (m)
    valley_width_m = (rng.normal(60.0 + stream_order * 15.0, 25.0, size=n)).clip(15.0, 250.0).round(1)
    stream_width_m = (rng.normal(4.0 + stream_order * 3.5, 3.0, size=n)).clip(1.5, 45.0).round(1)

    # ── 3. Climate & Land Use ────────────────────────────────────────────────
    # Mean annual rainfall (mm) across North Karnataka / Dharwad region
    rainfall_mm = (720.0 + (lat - lat_min) * 200.0 + rng.normal(0, 40, size=n)).clip(500.0, 1300.0).round(1)

    # Land use / Land cover
    lulc_classes = ["cropland", "scrub", "grassland", "forest", "built_up", "water"]
    lulc_probs = [0.52, 0.22, 0.12, 0.07, 0.04, 0.03]
    lulc_class = rng.choice(lulc_classes, p=lulc_probs, size=n)

    # Cropland fraction in upstream catchment
    cropland_fraction_upstream = rng.beta(a=3.5, b=2.0, size=n).round(3)

    # Soil texture
    soil_textures = ["clay", "loamy", "sandy", "gravelly"]
    soil_texture = rng.choice(soil_textures, p=[0.40, 0.40, 0.12, 0.08], size=n)

    # Lithology / Rock type
    lithologies = ["gneiss", "granite", "basalt", "sandstone", "alluvium"]
    lithology = rng.choice(lithologies, p=[0.35, 0.30, 0.18, 0.10, 0.07], size=n)

    # Infrastructure distances
    dist_to_road_m = rng.lognormal(mean=6.2, sigma=0.8, size=n).clip(20.0, 6000.0).round(1)
    dist_to_settlement_m = rng.lognormal(mean=6.9, sigma=0.7, size=n).clip(50.0, 10000.0).round(1)

    # Protected area flag (1 = environmentally protected forest/reserve, 0 = regular)
    protected_area_flag = np.where(
        (lulc_class == "forest") & (rng.uniform(0, 1, size=n) < 0.45), 1,
        np.where(rng.uniform(0, 1, size=n) < 0.03, 1, 0)
    ).astype(int)

    # ── 4. Latent Data-Generating Process for Label ───────────────────────────
    # A check dam was built if both observed physical conditions were favorable
    # AND an unobserved subsurface geotechnical / community factor was present.
    # Unobserved factor: foundation bedrock depth, aquifer storage capacity, and local sanction
    unobserved_factor = rng.standard_normal(size=n)  # NOT in feature table!

    # Normalized environmental components in [-1, +1] range
    # Slope suitability: optimal around 3-8 deg
    slope_suit = 1.0 - np.abs(slope_deg - 5.5) / 10.0

    # Stream order suitability: 1-3 is good, 4-5 is difficult
    so_suit = np.where(stream_order <= 3, 0.8, -0.8)

    # Catchment suitability: adequate drainage
    fa_suit = (np.log1p(flow_accum) - 8.0) / 2.5

    # Rainfall suitability
    rf_suit = (rainfall_mm - 780.0) / 150.0

    # Soil / Lithology permeability suitability (impermeable bed is better for storage)
    soil_map = {"clay": 0.8, "loamy": 0.5, "sandy": -0.6, "gravelly": -0.8}
    lith_map = {"gneiss": 0.7, "granite": 0.6, "basalt": 0.3, "sandstone": -0.4, "alluvium": -0.5}
    soil_suit = np.array([soil_map[s] for s in soil_texture])
    lith_suit = np.array([lith_map[l] for l in lithology])

    # Road accessibility
    road_suit = - (np.log1p(dist_to_road_m) - 6.5) / 1.2

    # Valley geometry: moderate valley width preferred
    vw_suit = 1.0 - (valley_width_m - 60.0) / 80.0

    # Disqualification penalties
    penalty = np.zeros(n)
    penalty[lulc_class == "built_up"] -= 3.0
    penalty[lulc_class == "water"] -= 3.0
    penalty[lulc_class == "forest"] -= 1.0
    penalty[protected_area_flag == 1] -= 4.0
    penalty[stream_order > 3] -= 1.5

    # Latent logit
    latent_z = (
        0.35 * slope_suit
        + 0.30 * so_suit
        + 0.25 * fa_suit
        + 0.20 * rf_suit
        + 0.25 * soil_suit
        + 0.20 * lith_suit
        + 0.15 * road_suit
        + 0.15 * vw_suit
        + 0.55 * unobserved_factor  # Crucial unobserved factor
        + penalty
        - 1.4  # Intercept controlling overall positive rate
    )

    # Logistic noise
    noise = rng.logistic(loc=0.0, scale=syn_cfg.get("logistic_noise_scale", 0.65), size=n)
    prob_true = 1.0 / (1.0 + np.exp(-(latent_z + noise)))
    label = (rng.uniform(0, 1, size=n) < prob_true).astype(int)

    # ── 5. Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "site_id": np.arange(1, n + 1),
        "block_id": block_id,
        "lat": lat.round(5),
        "lon": lon.round(5),
        "elevation_m": elevation_m,
        "slope_deg": slope_deg,
        "stream_order": stream_order,
        "flow_accum": flow_accum,
        "catchment_area_km2": catchment_area_km2,
        "twi": twi,
        "valley_width_m": valley_width_m,
        "stream_width_m": stream_width_m,
        "rainfall_mm": rainfall_mm,
        "lulc_class": lulc_class,
        "cropland_fraction_upstream": cropland_fraction_upstream,
        "soil_texture": soil_texture,
        "lithology": lithology,
        "dist_to_road_m": dist_to_road_m,
        "dist_to_settlement_m": dist_to_settlement_m,
        "protected_area_flag": protected_area_flag,
        "label": label,
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic check dam datasets.")
    parser.add_argument("--version", type=int, default=2, choices=[1, 2], help="Dataset version (1 or 2)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)

    if args.version == 1:
        out_path = project_root / config["paths"]["synthetic_v1_csv"]
        df = generate_synthetic_dataset_v1()
    else:
        out_path = project_root / config["paths"]["synthetic_v2_csv"]
        df = generate_synthetic_dataset_v2(config=config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Synthetic v{args.version} dataset saved to: {out_path.relative_to(project_root)}")
    print(f"   Shape: {df.shape}")
    print(f"   Spatial blocks: {df['block_id'].nunique() if 'block_id' in df else df['spatial_block'].nunique()}")
    print(f"   Positive class balance (label=1): {df['label'].mean():.2%}")
    print("\n   [SYNTHETIC - pipeline test, not a real finding]")


if __name__ == "__main__":
    main()
