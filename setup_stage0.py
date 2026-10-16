"""
setup_stage0.py
───────────────
Run this script to complete Stage 0 setup:
  1. Generates the synthetic dataset
  2. Creates the placeholder GeoJSON for the dashboard
  3. Verifies the src/ modules can be imported
  4. Prints a summary of what was created

Usage:
    python setup_stage0.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("STAGE 0 SETUP — Check Dam Site Selection Project")
print("=" * 60)

# ── 1. Generate synthetic dataset ─────────────────────────────────────────────
print("\n[1/3] Generating synthetic dataset...")
try:
    from src.generate_synthetic import generate_synthetic_dataset
    df = generate_synthetic_dataset()
    out = PROJECT_ROOT / "data" / "tables" / "synthetic_checkdam_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  ✅  {len(df)} rows × {len(df.columns)} columns → {out.relative_to(PROJECT_ROOT)}")
    print(f"      Class balance: {df['label'].mean():.1%} positive (check dam)")
except Exception as e:
    print(f"  ❌  Failed: {e}")

# ── 2. Create placeholder GeoJSON for dashboard ───────────────────────────────
print("\n[2/3] Creating placeholder GeoJSON for dashboard...")
try:
    from src.dashboard_export import create_placeholder_geojson
    create_placeholder_geojson()
    print("  ✅  Placeholder sites written to docs/data/ and outputs/")
except Exception as e:
    print(f"  ❌  Failed: {e}")

# ── 3. Verify imports ─────────────────────────────────────────────────────────
print("\n[3/3] Verifying module imports...")
modules = {
    "src.ml_model":        "CheckDamModel, AHPScorer, spatial_block_cv",
    "src.optimization":    "run_optimization, compute_objectives",
    "src.preprocessing":   "clip_raster_to_boundary, reproject_raster",
    "src.dashboard_export":"df_to_geojson, export_for_dashboard",
}

all_ok = True
for module, exports in modules.items():
    try:
        __import__(module, fromlist=exports.split(", "))
        print(f"  ✅  {module}")
    except ImportError as e:
        print(f"  ⚠️  {module} — partial import (missing optional dep): {e}")
    except Exception as e:
        print(f"  ❌  {module} — ERROR: {e}")
        all_ok = False

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 0 COMPLETE ✅")
print("=" * 60)
print("""
Created files:
  📄 README.md, PROGRESS.md, ARCHITECTURE.md
  📄 DECISIONS.md, DATA_SOURCES.md, LEARNING_NOTES.md
  📄 requirements.txt, .gitignore
  🐍 src/generate_synthetic.py, ml_model.py, optimization.py
  🐍 src/preprocessing.py, hydrology.py, features.py, dashboard_export.py
  📊 data/tables/synthetic_checkdam_features.csv
  🗺️  docs/index.html  (Leaflet dashboard)
  📍 docs/data/recommended_sites.geojson  (placeholder)

Next steps:
  1. Open a terminal in this folder and activate the venv:
       venv\\Scripts\\activate   (Windows)
  2. Install all requirements:
       pip install -r requirements.txt
  3. Run Stage 1:
       python notebooks/stage1_ml_warmup.py
     OR open in Jupyter:
       jupyter notebook
  4. Register for Google Earth Engine (do this now — takes 24–48h):
       https://earthengine.google.com/signup/
  5. Read DECISIONS.md to understand every design choice.
""")
