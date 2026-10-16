# Stage 1 ML Warm-up Notebook
# Run this as a Python script or open in Jupyter: jupyter notebook notebooks/stage1_ml_warmup.ipynb

"""
STAGE 1: ML WARM-UP ON SYNTHETIC DATA
======================================
Goal: Build, validate, and explain a suitability model end-to-end
      using the synthetic dataset before touching real geospatial data.

You will learn:
  - How to load and explore a feature table
  - How to check class balance
  - How to train a Random Forest and XGBoost model
  - How to run spatial block cross-validation (NOT random split)
  - How to interpret AUC-ROC, precision, recall
  - How to compute and read SHAP values
  - How AHP compares to ML

Definition of done:
  ✅ Both RF and XGBoost trained and validated
  ✅ Spatial CV results table saved to outputs/stage1_cv_results.csv
  ✅ Feature importance plot saved to outputs/stage1_feature_importance.png
  ✅ SHAP summary plot saved to outputs/stage1_shap_summary.png
  ✅ AHP vs ML comparison table printed
  ✅ LEARNING_NOTES.md updated with your explanations
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path so we can import src/ modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_model import (
    CheckDamModel,
    AHPScorer,
    spatial_block_cv,
    compute_shap_values,
    evaluate_ahp_vs_ml,
    FEATURE_COLS,
    LABEL_COL,
    BLOCK_COL,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = PROJECT_ROOT / "data" / "tables" / "synthetic_checkdam_features.csv"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("STAGE 1: ML WARM-UP ON SYNTHETIC DATA")
print("=" * 60)


# ── 1. LOAD AND EXPLORE ───────────────────────────────────────────────────────
print("\n[1/7] Loading synthetic dataset...")
df = pd.read_csv(DATA_PATH)

print(f"\nShape:          {df.shape}")
print(f"Columns:        {list(df.columns)}")
print(f"\nClass balance:")
vc = df[LABEL_COL].value_counts()
print(f"  label=1 (check dam):       {vc.get(1, 0)} ({vc.get(1,0)/len(df):.1%})")
print(f"  label=0 (non-dam stream):  {vc.get(0, 0)} ({vc.get(0,0)/len(df):.1%})")

print("\nFeature statistics:")
print(df[FEATURE_COLS].describe().round(2).to_string())

# ── Plot: class distributions ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle("Feature Distributions by Label (1=Check Dam, 0=Non-Dam)", fontsize=13)
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    for label_val, colour, linestyle in [(0, '#ef5350', '--'), (1, '#16c784', '-')]:
        subset = df[df[LABEL_COL] == label_val][col]
        axes[i].hist(subset, bins=20, alpha=0.5, color=colour,
                     label=f"label={label_val}", density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(labelsize=7)
    if i == 0:
        axes[i].legend(fontsize=7)

plt.tight_layout()
save_path = OUTPUT_DIR / "stage1_feature_distributions.png"
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n✅  Feature distributions saved: {save_path.name}")


# ── 2. SPATIAL CROSS-VALIDATION ───────────────────────────────────────────────
print("\n[2/7] Running spatial block cross-validation...")
print("\n── Random Forest ──")
rf_model = CheckDamModel(model_type="random_forest")
rf_cv = spatial_block_cv(df, rf_model)

try:
    print("\n── XGBoost ──")
    xgb_model = CheckDamModel(model_type="xgboost")
    xgb_cv = spatial_block_cv(df, xgb_model)
    rf_cv["model"]  = "Random Forest"
    xgb_cv["model"] = "XGBoost"
    cv_results = pd.concat([rf_cv, xgb_cv], ignore_index=True)
except Exception as e:
    print(f"  XGBoost skipped: {e}")
    rf_cv["model"] = "Random Forest"
    cv_results = rf_cv

cv_path = OUTPUT_DIR / "stage1_cv_results.csv"
cv_results.to_csv(cv_path, index=False)
print(f"\n✅  Cross-validation results saved: {cv_path.name}")

print("\nMean CV scores:")
print(cv_results.groupby("model")[["auc","precision","recall","f1"]].mean().round(3).to_string())


# ── 3. TRAIN FINAL MODEL ──────────────────────────────────────────────────────
print("\n[3/7] Training final model on all data...")
X_all = df[FEATURE_COLS]
y_all = df[LABEL_COL]

rf_final = CheckDamModel(model_type="random_forest")
rf_final.fit(X_all, y_all)
print("  ✅  Random Forest trained.")

try:
    xgb_final = CheckDamModel(model_type="xgboost")
    xgb_final.fit(X_all, y_all)
    print("  ✅  XGBoost trained.")
    best_model = xgb_final   # use XGBoost for SHAP if available
except Exception:
    best_model = rf_final
    print("  ⚠️  XGBoost not available, using RF for SHAP.")


# ── 4. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
print("\n[4/7] Plotting feature importances...")
fi = rf_final.feature_importance()
print(fi.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#2d6ef0' if i < 3 else '#8b949e' for i in range(len(fi))]
bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=10)
ax.set_title("Random Forest — Feature Importances", fontsize=12)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=9)

save_path = OUTPUT_DIR / "stage1_feature_importance.png"
plt.tight_layout()
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"✅  Feature importance plot saved: {save_path.name}")


# ── 5. SHAP VALUES ────────────────────────────────────────────────────────────
print("\n[5/7] Computing SHAP values...")
try:
    import shap
    shap_values = compute_shap_values(best_model, X_all)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_all[FEATURE_COLS], show=False)
    save_path = OUTPUT_DIR / "stage1_shap_summary.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅  SHAP summary plot saved: {save_path.name}")

    # Show SHAP for one high-scoring site
    top_site_idx = df[LABEL_COL].idxmax()
    print(f"\nSHAP explanation for site {df.loc[top_site_idx, 'site_id']} (label=1):")
    sv = shap_values[top_site_idx].values if hasattr(shap_values[top_site_idx], 'values') else []
    if len(sv):
        for feat, val in sorted(zip(FEATURE_COLS, sv), key=lambda x: abs(x[1]), reverse=True):
            direction = "+" if val > 0 else ""
            print(f"  {feat:<25} {direction}{val:.4f}")
except ImportError:
    print("  ⚠️  shap not installed. Run: pip install shap")


# ── 6. AHP vs ML COMPARISON ───────────────────────────────────────────────────
print("\n[6/7] AHP baseline vs ML comparison...")
comparison = evaluate_ahp_vs_ml(df, rf_final)
print(comparison.to_string(index=False))

comp_path = OUTPUT_DIR / "stage1_ahp_vs_ml.csv"
comparison.to_csv(comp_path, index=False)
print(f"✅  Comparison saved: {comp_path.name}")


# ── 7. SCORE ALL CANDIDATES ───────────────────────────────────────────────────
print("\n[7/7] Scoring all candidates with final RF model...")
df["suitability_score"] = rf_final.predict_proba(df)
df["ahp_score"]         = AHPScorer().score(df)

scored_path = OUTPUT_DIR / "stage1_all_candidates_scored.csv"
df.to_csv(scored_path, index=False)

top10 = df.nlargest(10, "suitability_score")[["site_id","suitability_score","ahp_score"] + FEATURE_COLS[:5]]
print("\nTop 10 sites by ML suitability score:")
print(top10.to_string(index=False))
print(f"\n✅  All scores saved: {scored_path.name}")


print("\n" + "=" * 60)
print("STAGE 1 COMPLETE")
print("=" * 60)
print("""
Next steps:
  1. Open LEARNING_NOTES.md and answer the Stage 1 prompts.
  2. Look at outputs/ — can you explain what you see?
  3. Update PROGRESS.md with your metrics.
  4. Move to Stage 2: pick your study area and collect real data.
""")
