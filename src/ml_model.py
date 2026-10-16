"""
ml_model.py
───────────
Suitability scoring models for check dam site selection.

Contains:
  - AHPScorer       : Rule-based baseline (no training required)
  - CheckDamModel   : Wrapper around RandomForest / XGBoost
  - spatial_block_cv: Spatial cross-validation utility
  - compute_shap    : SHAP explanation utility

Design note: the `structure_type` parameter is reserved for future extension
to farm ponds and percolation tanks without rewriting this module.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed. XGBoostModel will not be available.")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    warnings.warn("shap not installed. SHAP explanations will not be available.")


# ── Feature columns used by ML models ────────────────────────────────────────
FEATURE_COLS = [
    "slope_deg",
    "flow_accum",
    "twi",
    "stream_order",
    "annual_rainfall_mm",
    "soil_texture_class",
    "ndvi",
    "dist_to_road_m",
    "lulc_class",
    "geology_permeability",
]

LABEL_COL = "label"
BLOCK_COL = "spatial_block"  # column used for spatial cross-validation


# ─────────────────────────────────────────────────────────────────────────────
# AHP Baseline Scorer
# ─────────────────────────────────────────────────────────────────────────────

class AHPScorer:
    """
    Analytic Hierarchy Process (AHP) rule-based suitability scorer.

    Weights are assigned based on expert knowledge and literature for
    check dam siting in semi-arid India. This is the baseline that the
    ML models must beat to be considered useful.

    Weights sum to 1.0. Higher weight = more important criterion.

    References:
        Mukherjee et al. (2012), Nitheshnirmal et al. (2019)
        (Do NOT cite these without verifying they contain this method.)
    """

    # Feature → (weight, direction, optimal_range_or_class)
    # direction: +1 means higher raw value → higher score, -1 means inverse
    WEIGHTS = {
        "slope_deg":           0.20,   # moderate slope preferred (~5–15°)
        "flow_accum":          0.20,   # higher flow → more water
        "twi":                 0.15,   # higher TWI → better wetness
        "annual_rainfall_mm":  0.15,   # higher rainfall → more storage
        "soil_texture_class":  0.12,   # clay/loamy preferred (higher class)
        "geology_permeability":0.08,   # lower permeability → less seepage loss
        "dist_to_road_m":      0.05,   # closer to road → lower cost
        "ndvi":                0.05,   # moderate NDVI preferred
    }

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute AHP suitability score (0–1) for each row.

        Returns
        -------
        np.ndarray of shape (n_samples,) with scores in [0, 1].
        """
        scores = np.zeros(len(df))

        # Slope: optimal 5–15°, penalise <3° and >25°
        slope_norm = np.where(
            (df["slope_deg"] >= 5) & (df["slope_deg"] <= 15), 1.0,
            np.where(df["slope_deg"] < 3, 0.2,
            np.where(df["slope_deg"] > 25, 0.1, 0.5))
        )
        scores += self.WEIGHTS["slope_deg"] * slope_norm

        # Flow accumulation: log-normalise
        fa_log = np.log1p(df["flow_accum"])
        scores += self.WEIGHTS["flow_accum"] * (fa_log / fa_log.max())

        # TWI: normalise
        twi_norm = (df["twi"] - df["twi"].min()) / (df["twi"].max() - df["twi"].min() + 1e-9)
        scores += self.WEIGHTS["twi"] * twi_norm

        # Rainfall: normalise
        rf_norm = (df["annual_rainfall_mm"] - df["annual_rainfall_mm"].min()) / (
            df["annual_rainfall_mm"].max() - df["annual_rainfall_mm"].min() + 1e-9
        )
        scores += self.WEIGHTS["annual_rainfall_mm"] * rf_norm

        # Soil texture: clay=3 best, sandy=1 worst
        soil_norm = (df["soil_texture_class"] - 1) / 2.0
        scores += self.WEIGHTS["soil_texture_class"] * soil_norm

        # Geology permeability: LOW permeability (class 1) is BEST (less seepage)
        geo_norm = (3 - df["geology_permeability"]) / 2.0
        scores += self.WEIGHTS["geology_permeability"] * geo_norm

        # Distance to road: closer = better (inverse)
        dist_norm = 1.0 - (df["dist_to_road_m"] - df["dist_to_road_m"].min()) / (
            df["dist_to_road_m"].max() - df["dist_to_road_m"].min() + 1e-9
        )
        scores += self.WEIGHTS["dist_to_road_m"] * dist_norm

        # NDVI: moderate preferred (peak at 0.35–0.55)
        ndvi_norm = 1.0 - np.abs(df["ndvi"] - 0.45) / 0.55
        ndvi_norm = ndvi_norm.clip(0, 1)
        scores += self.WEIGHTS["ndvi"] * ndvi_norm

        return scores.clip(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ML Model Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CheckDamModel:
    """
    Wrapper around RandomForest or XGBoost for check dam suitability scoring.

    Parameters
    ----------
    model_type      : 'random_forest' or 'xgboost'
    structure_type  : 'check_dam' (reserved for future: 'farm_pond', 'percolation_tank')
    random_seed     : For reproducibility
    """

    SUPPORTED_MODELS = ("random_forest", "xgboost")

    def __init__(
        self,
        model_type: Literal["random_forest", "xgboost"] = "random_forest",
        structure_type: str = "check_dam",
        random_seed: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        if model_type == "xgboost" and not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.model_type = model_type
        self.structure_type = structure_type
        self.random_seed = random_seed
        self.model = self._build_model()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",  # handles class imbalance
                random_state=self.random_seed,
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,  # compensates for ~25% positive class
                random_state=self.random_seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CheckDamModel":
        """Train the model on the full feature matrix and labels."""
        self.model.fit(X[self.feature_cols], y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return suitability probability (class 1) for each row."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance().")
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Block Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def spatial_block_cv(
    df: pd.DataFrame,
    model: CheckDamModel,
    n_blocks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Leave-one-block-out spatial cross-validation.

    Splits data by `spatial_block` column (not randomly). Each fold uses
    one block as the test set and all other blocks as the training set.
    This avoids spatial autocorrelation inflating performance metrics.

    Parameters
    ----------
    df      : DataFrame with FEATURE_COLS, LABEL_COL, BLOCK_COL
    model   : CheckDamModel instance (will be re-initialised for each fold)
    n_blocks: If None, uses all unique blocks.

    Returns
    -------
    DataFrame with one row per fold: block_id, auc, precision, recall, f1, n_test
    """
    blocks = sorted(df[BLOCK_COL].unique())
    if n_blocks is not None:
        blocks = blocks[:n_blocks]

    results = []
    for block_id in blocks:
        train_mask = df[BLOCK_COL] != block_id
        test_mask  = df[BLOCK_COL] == block_id

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, LABEL_COL]
        X_test  = df.loc[test_mask,  FEATURE_COLS]
        y_test  = df.loc[test_mask,  LABEL_COL]

        if y_test.nunique() < 2:
            print(f"  Block {block_id}: skipped (only one class in test set)")
            continue

        # Re-build and train a fresh model for each fold
        fold_model = CheckDamModel(
            model_type=model.model_type,
            structure_type=model.structure_type,
            random_seed=model.random_seed,
        )
        fold_model.fit(X_train, y_train)
        y_prob = fold_model.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        results.append({
            "block_id":  block_id,
            "auc":       roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "n_test":    len(y_test),
        })
        print(f"  Block {block_id}: AUC={results[-1]['auc']:.3f}  "
              f"P={results[-1]['precision']:.3f}  R={results[-1]['recall']:.3f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explanation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model: CheckDamModel,
    X: pd.DataFrame,
    max_display: int = 10,
) -> Optional[object]:
    """
    Compute SHAP values for a fitted model.

    Parameters
    ----------
    model       : Fitted CheckDamModel
    X           : Feature DataFrame
    max_display : Max features to show in summary plot

    Returns
    -------
    shap.Explanation object, or None if shap is not installed.
    """
    if not _SHAP_AVAILABLE:
        print("⚠️  shap not installed. Run: pip install shap")
        return None
    if not model.is_fitted:
        raise RuntimeError("Call fit() before compute_shap_values().")

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer(X[model.feature_cols])
    return shap_values


def evaluate_ahp_vs_ml(
    df: pd.DataFrame,
    ml_model: CheckDamModel,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare AHP baseline vs trained ML model on the same dataset.

    Returns a comparison DataFrame with AUC, precision, recall for both.
    """
    ahp = AHPScorer()
    ahp_scores = ahp.score(df)
    ml_probs   = ml_model.predict_proba(df)
    y_true     = df[LABEL_COL].values

    def metrics(scores, label):
        preds = (scores >= threshold).astype(int)
        return {
            "model":     label,
            "auc":       roc_auc_score(y_true, scores),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall":    recall_score(y_true, preds, zero_division=0),
            "f1":        f1_score(y_true, preds, zero_division=0),
        }

    return pd.DataFrame([
        metrics(ahp_scores, "AHP Baseline"),
        metrics(ml_probs,   f"ML ({ml_model.model_type})"),
    ])
