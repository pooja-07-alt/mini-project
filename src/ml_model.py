"""
ml_model.py
───────────
Machine learning suitability models with Spatial Block Cross-Validation.

Contains:
  - CheckDamMLPipeline: Preprocessing (OneHotEncoder + Passthrough) and ML models
  - find_optimal_threshold: Chooses classification threshold on training folds by maximizing F1
  - run_spatial_block_cv: Leave-one-spatial-block-out CV strictly grouped by block_id
  - Out-of-fold predicted probability generation for Random Forest and XGBoost

Key Concepts:
  1. Spatial Autocorrelation: Adjacent stream pixels are not independent. Random train/test
     splits cause data leakage. Spatial block CV groups points geographically so an entire
     block is held out as unseen test data.
  2. Class Imbalance: Check dam sites are rarer than ordinary stream reaches. We handle
     imbalance using balanced class weights (RF) and scale_pos_weight (XGBoost).
  3. No Leakage: Categorical one-hot encoders are fitted on training folds only.

SYNTHETIC - pipeline test, not a real finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor(
    base_features: List[str],
    categorical_features: List[str],
    include_ahp: bool = False,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build a leak-free ColumnTransformer for continuous and categorical features.

    Parameters
    ----------
    base_features : list of str
        Continuous numeric features (e.g. slope, twi, rainfall, flow_accum).
    categorical_features : list of str
        Categorical features to one-hot encode (e.g. lulc_class, soil_texture, lithology).
    include_ahp : bool
        If True, appends 'ahp_score' as an extra numeric feature (for Step 4).

    Returns
    -------
    preprocessor : ColumnTransformer
    feature_list : list of str
    """
    numeric_cols = list(base_features)
    if include_ahp:
        numeric_cols.append("ahp_score")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    all_cols = numeric_cols + list(categorical_features)
    return preprocessor, all_cols


def build_classifier(
    model_type: str,
    config: dict,
    scale_pos_weight: Optional[float] = None,
):
    """
    Build an un-fitted classifier with parameters from config.yaml.

    Parameters
    ----------
    model_type : 'random_forest' or 'xgboost'
    config : dict
        Loaded project configuration.
    scale_pos_weight : float, optional
        Ratio of negative to positive training samples for XGBoost.
    """
    ml_cfg = config.get("ml", {})
    if model_type == "random_forest":
        rf_params = ml_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 250),
            max_depth=rf_params.get("max_depth", 8),
            min_samples_leaf=rf_params.get("min_samples_leaf", 6),
            class_weight=rf_params.get("class_weight", "balanced"),
            random_state=rf_params.get("random_state", 42),
            n_jobs=rf_params.get("n_jobs", -1),
        )
    elif model_type == "xgboost":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed. Install via pip install xgboost.")
        xgb_params = ml_cfg.get("xgboost", {})
        spw = scale_pos_weight if scale_pos_weight is not None else xgb_params.get("scale_pos_weight", 3.0)
        return xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 250),
            max_depth=xgb_params.get("max_depth", 4),
            learning_rate=xgb_params.get("learning_rate", 0.04),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            scale_pos_weight=spw,
            random_state=xgb_params.get("random_state", 42),
            eval_metric=xgb_params.get("eval_metric", "logloss"),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'random_forest' or 'xgboost'.")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Find probability decision threshold that maximizes F1 score on training folds.

    Parameters
    ----------
    y_true : 1D array of binary true labels
    y_prob : 1D array of predicted probabilities

    Returns
    -------
    best_threshold : float
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)

    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh


def run_spatial_block_cv(
    df: pd.DataFrame,
    model_type: str,
    config: Optional[dict] = None,
    include_ahp: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run leave-one-spatial-block-out cross validation grouped by block_id.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates table (must have block_id, label, and feature columns).
    model_type : 'random_forest' or 'xgboost'
    config : dict, optional
    include_ahp : bool
        If True, adds ahp_score as a feature (Method A, Step 4).
        If False, ML-only (Step 3).

    Returns
    -------
    fold_metrics_df : pd.DataFrame
        Per-fold validation metrics (AUC-ROC, PR-AUC, precision, recall, F1, optimal_threshold).
    oof_predictions : pd.Series
        Out-of-fold predicted probabilities for every candidate site (aligned with df.index).
    """
    if config is None:
        config = load_config()

    ml_cfg = config.get("ml", {})
    block_col = ml_cfg.get("block_column", "block_id")
    label_col = ml_cfg.get("label_column", "label")
    base_feats = ml_cfg.get("base_features", [])
    cat_feats = ml_cfg.get("categorical_features", [])

    blocks = sorted(df[block_col].unique())
    oof_probs = pd.Series(index=df.index, dtype=float)
    fold_records = []

    for fold_num, holdout_block in enumerate(blocks, start=1):
        # ── 1. Create Train / Test Masks (Strict Spatial Isolation) ───────────
        train_mask = df[block_col] != holdout_block
        test_mask = df[block_col] == holdout_block

        # Assert no spatial block leakage between train and test
        train_blocks = set(df.loc[train_mask, block_col].unique())
        test_blocks = set(df.loc[test_mask, block_col].unique())
        assert train_blocks.isdisjoint(test_blocks), (
            f"Spatial leakage detected! Fold {fold_num}: Train blocks {train_blocks} intersect Test blocks {test_blocks}"
        )

        # ── 2. Split Features & Labels ────────────────────────────────────────
        preprocessor, feature_cols = build_preprocessor(base_feats, cat_feats, include_ahp=include_ahp)

        X_train_raw = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, label_col].values
        X_test_raw = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, label_col].values

        # Fit preprocessor on training data only
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        # Compute dynamic scale_pos_weight for XGBoost on training fold
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        # ── 3. Train Classifier ───────────────────────────────────────────────
        model = build_classifier(model_type, config, scale_pos_weight=scale_pos_weight)
        model.fit(X_train_proc, y_train)

        # ── 4. Optimize Decision Threshold on Train Fold ──────────────────────
        y_train_probs = model.predict_proba(X_train_proc)[:, 1]
        best_threshold = find_optimal_threshold(y_train, y_train_probs)

        # ── 5. Predict on Held-Out Test Block ─────────────────────────────────
        y_test_probs = model.predict_proba(X_test_proc)[:, 1]
        oof_probs.loc[test_mask] = y_test_probs

        # Apply training-derived threshold to held-out test predictions
        y_test_preds = (y_test_probs >= best_threshold).astype(int)

        auc = roc_auc_score(y_test, y_test_probs)
        pr_auc = average_precision_score(y_test, y_test_probs)
        prec = precision_score(y_test, y_test_preds, zero_division=0)
        rec = recall_score(y_test, y_test_preds, zero_division=0)
        f1 = f1_score(y_test, y_test_preds, zero_division=0)

        fold_records.append({
            "model": f"{model_type}_{'with_ahp' if include_ahp else 'ml_only'}",
            "fold": fold_num,
            "holdout_block": holdout_block,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_positives": int(y_test.sum()),
            "best_threshold": round(best_threshold, 3),
            "auc_roc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

    fold_metrics_df = pd.DataFrame(fold_records)
    return fold_metrics_df, oof_probs


def run_step3_ml_only(config: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 3: Train Random Forest and XGBoost ML-only models with Spatial CV.

    Saves:
      - outputs/stage1b_oof_ml_only.csv
    """
    if config is None:
        config = load_config()

    project_root = Path(__file__).resolve().parent.parent
    candidates_path = project_root / config["paths"]["candidates_ahp_csv"]
    if not candidates_path.exists():
        candidates_path = project_root / config["paths"]["candidates_filtered_csv"]

    print("=" * 65)
    print("STAGE 1b - STEP 3: ML-ONLY MODELS WITH SPATIAL CROSS-VALIDATION")
    print("=" * 65)
    print(f"Loading candidates: {candidates_path.relative_to(project_root)}")
    df = pd.read_csv(candidates_path)

    # 1. Random Forest ML-Only
    print("\nRunning Spatial CV: Random Forest (ML-Only)...")
    rf_folds, rf_oof = run_spatial_block_cv(df, "random_forest", config=config, include_ahp=False)

    # 2. XGBoost ML-Only
    print("Running Spatial CV: XGBoost (ML-Only)...")
    xgb_folds, xgb_oof = run_spatial_block_cv(df, "xgboost", config=config, include_ahp=False)

    all_folds = pd.concat([rf_folds, xgb_folds], ignore_index=True)

    # 3. Assemble Out-of-Fold Predictions Table
    oof_df = pd.DataFrame({
        "site_id": df["site_id"],
        "block_id": df["block_id"],
        "lat": df["lat"],
        "lon": df["lon"],
        "label": df["label"],
        "rf_prob_ml_only": rf_oof.round(4),
        "xgb_prob_ml_only": xgb_oof.round(4),
    })
    if "ahp_score" in df.columns:
        oof_df["ahp_score"] = df["ahp_score"]

    oof_out = project_root / config["paths"]["oof_ml_only_csv"]
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_out, index=False)

    print("\nOut-of-Fold Spatial CV Summary (Mean +/- Std):")
    summary = all_folds.groupby("model")[["auc_roc", "pr_auc", "precision", "recall", "f1"]].agg(
        ["mean", "std"]
    ).round(3)
    print(summary.to_string())

    print(f"\n✅ Out-of-fold predictions saved to: {oof_out.relative_to(project_root)}")
    print("\n[SYNTHETIC - pipeline test, not a real finding]")
    return all_folds, oof_df


def main():
    parser = argparse.ArgumentParser(description="Run ML suitability models with spatial CV.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)
    run_step3_ml_only(config=config)


if __name__ == "__main__":
    main()
