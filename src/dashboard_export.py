"""
dashboard_export.py
───────────────────
Exports model outputs to GeoJSON for the Leaflet dashboard.

In Stage 1 (synthetic data): exports a GeoJSON with placeholder coordinates.
In Stage 7 (real data): exports real lat/long coordinates with all feature
values and SHAP explanations as GeoJSON properties.

The Leaflet dashboard (docs/index.html) reads this file directly.
No backend server is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
GEOJSON_PATH = OUTPUT_DIR / "recommended_sites.geojson"
DOCS_GEOJSON_PATH = Path(__file__).parent.parent / "docs" / "data" / "recommended_sites.geojson"


def df_to_geojson(
    df: pd.DataFrame,
    lat_col: Optional[str] = "lat",
    lon_col: Optional[str] = "lon",
    output_path: Optional[Path] = None,
    title: str = "Recommended Check Dam Sites",
) -> dict:
    """
    Convert a DataFrame of recommended sites to GeoJSON FeatureCollection.

    Parameters
    ----------
    df          : DataFrame with site attributes. Must have lat/lon columns
                  (or placeholder coordinates for Stage 1).
    lat_col     : Column name for latitude
    lon_col     : Column name for longitude
    output_path : If given, saves the GeoJSON to this path
    title       : Human-readable title stored in GeoJSON metadata

    Returns
    -------
    GeoJSON dict (FeatureCollection)
    """
    features = []
    for _, row in df.iterrows():
        # Build properties dict from all non-geometry columns
        props = {}
        for col in df.columns:
            if col in (lat_col, lon_col):
                continue
            val = row[col]
            # Convert numpy types to native Python for JSON serialisation
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = round(float(val), 4)
            elif isinstance(val, float) and np.isnan(val):
                val = None
            props[col] = val

        # Geometry
        lat = float(row.get(lat_col, 0.0))
        lon = float(row.get(lon_col, 0.0))

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # GeoJSON is [lon, lat]
            },
            "properties": props,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "title":        title,
            "n_sites":      len(features),
            "generated_by": "checkdam-siting pipeline",
        },
        "features": features,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        print(f"✅  GeoJSON saved to: {output_path}")

    return geojson


def export_for_dashboard(
    pareto_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> None:
    """
    Export Pareto-optimal sites to both outputs/ and docs/data/ for the dashboard.

    Saves two copies:
      - outputs/recommended_sites.geojson (project record)
      - docs/data/recommended_sites.geojson (served by GitHub Pages)
    """
    for path in [GEOJSON_PATH, DOCS_GEOJSON_PATH]:
        df_to_geojson(pareto_df, lat_col=lat_col, lon_col=lon_col, output_path=path)

    print(f"\n📍 {len(pareto_df)} sites exported for dashboard.")
    print(f"   Open docs/index.html in a browser to view.")


def create_placeholder_geojson() -> None:
    """
    Create a placeholder GeoJSON for Stage 0/1 so the dashboard renders something.
    Uses approximate Karnataka coordinates (Dharwad district sub-watershed area).
    """
    import random
    random.seed(42)

    # Approximate centre of a typical Karnataka sub-watershed (placeholder)
    # These are NOT real recommended sites — they are placeholder examples only
    base_lat, base_lon = 15.35, 75.15

    placeholder_sites = []
    for i in range(5):
        placeholder_sites.append({
            "site_id":            i + 1,
            "lat":                round(base_lat + random.uniform(-0.05, 0.05), 5),
            "lon":                round(base_lon + random.uniform(-0.05, 0.05), 5),
            "suitability_score":  round(random.uniform(0.65, 0.92), 3),
            "storage_m3":         round(random.uniform(5000, 25000)),
            "cost_proxy":         round(random.uniform(0.5, 3.5), 2),
            "submergence_ha":     round(random.uniform(0.2, 2.5), 2),
            "slope_deg":          round(random.uniform(4, 14), 1),
            "annual_rainfall_mm": round(random.uniform(750, 1050)),
            "note":               "⚠️ PLACEHOLDER — not a real recommended site",
        })

    df = pd.DataFrame(placeholder_sites)
    export_for_dashboard(df)
    print("\n⚠️  Placeholder GeoJSON created. Replace after Stage 7 with real results.")
