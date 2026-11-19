"""
ahp.py
──────
Analytic Hierarchy Process (AHP) Suitability Scorer for Check Dam Siting.

Theoretical Background:
-----------------------
AHP is a multi-criteria decision-making framework developed by Thomas Saaty.
Rather than guessing weights arbitrarily:
1. Domain experts build a Pairwise Comparison Matrix A where A[i, j] indicates
   how much more important criterion i is compared to criterion j (on a 1 to 9 scale).
2. The weights vector w is computed as the principal eigenvector corresponding to
   the maximum eigenvalue lambda_max:
       A * w = lambda_max * w, with sum(w) = 1.0.
3. The Consistency Index (CI) and Consistency Ratio (CR) are computed:
       CI = (lambda_max - n) / (n - 1)
       CR = CI / RI(n)
   where RI(n) is Saaty's empirical Random Index.
4. If CR >= 0.10, the matrix is inconsistent and the algorithm FAILS loudly.
5. Physical parameters are converted to [0, 1] fit scores via explicit physical
   functions.
6. The final ahp_score = (sum(w_i * fit_i)) * 100 (range 0 to 100).

CRITICAL RULE:
  AHP weights are determined strictly a priori from engineering domain literature.
  Labels are NEVER used to fit or tune AHP weights or scoring functions.

SYNTHETIC - pipeline test, not a real finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Saaty's Random Consistency Index (RI) table for matrix sizes 1 to 10
SAATY_RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_eigenvector_weights(
    matrix: np.ndarray,
    max_cr: float = 0.10,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute priority weights via the principal eigenvector and verify consistency.

    Parameters
    ----------
    matrix : np.ndarray
        Square, positive pairwise reciprocal matrix of shape (n, n).
    max_cr : float
        Maximum allowable Consistency Ratio (default 0.10).

    Returns
    -------
    weights : np.ndarray
        Normalized priority weight vector (sums to 1.0).
    lambda_max : float
        Principal eigenvalue.
    cr : float
        Consistency ratio.

    Raises
    ------
    ValueError
        If matrix is non-square, n < 1, or CR >= max_cr.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Pairwise matrix must be 2D square. Got shape {matrix.shape}")

    n = matrix.shape[0]
    if n == 1:
        return np.array([1.0]), 1.0, 0.0

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the principal (maximum real) eigenvalue
    real_eigenvalues = np.real(eigenvalues)
    max_idx = int(np.argmax(real_eigenvalues))
    lambda_max = float(real_eigenvalues[max_idx])

    # Extract corresponding eigenvector and normalize to sum to 1.0
    principal_vector = np.real(eigenvectors[:, max_idx])
    # In case principal vector was returned negative, flip signs
    if np.all(principal_vector < 0):
        principal_vector = -principal_vector

    if np.any(principal_vector < 0):
        # Fallback to power iteration to guarantee positive Perron-Frobenius eigenvector
        vec = np.ones(n) / n
        for _ in range(100):
            next_vec = matrix @ vec
            vec = next_vec / np.sum(next_vec)
        principal_vector = vec

    weights = principal_vector / np.sum(principal_vector)

    # Compute Consistency Index (CI) and Consistency Ratio (CR)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = SAATY_RANDOM_INDEX.get(n, 1.49)
    cr = (ci / ri) if ri > 0 else 0.0

    if cr >= max_cr:
        raise ValueError(
            f"AHP Consistency Ratio CR = {cr:.4f} exceeds threshold {max_cr:.2f}! "
            f"The pairwise comparison matrix is inconsistent. Please revise judgements."
        )

    return weights, lambda_max, cr


class AHPScorer:
    """
    AHP Suitability Scorer implementing Saaty's eigenvector method.

    Evaluates suitability independently of any labels.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.ahp_cfg = config.get("ahp", {})
        self.criteria = self.ahp_cfg.get("criteria", [
            "slope_deg",
            "flow_accum",
            "twi",
            "rainfall_mm",
            "soil_texture",
            "lithology",
            "cropland_fraction_upstream",
            "dist_to_road_m",
        ])
        self.raw_matrix = np.array(self.ahp_cfg["pairwise_matrix"], dtype=float)
        self.max_cr = float(self.ahp_cfg.get("max_cr", 0.10))
        self.score_multiplier = float(self.ahp_cfg.get("score_multiplier", 100.0))
        self.fit_params = self.ahp_cfg.get("fit_functions", {})

        # Compute weights and verify CR immediately on initialization
        self.weights, self.lambda_max, self.cr = compute_eigenvector_weights(
            self.raw_matrix, max_cr=self.max_cr
        )
        self.weights_dict = {crit: float(w) for crit, w in zip(self.criteria, self.weights)}

    def compute_fit_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw continuous/categorical features to [0, 1] fit scores.

        Each function reflects verified hydrological / geotechnical principles:
          - slope: trapezoidal fit (optimal 3-8 deg, acceptable 1-15 deg)
          - flow accumulation: log-scaled monotonic increase
          - TWI: normalized linear increase (high wetness = good)
          - rainfall: normalized linear increase (more rainfall = better refill)
          - soil: discrete engineering ratings (clay best, gravelly lowest)
          - lithology: foundation firmness ratings (gneiss/granite best)
          - cropland upstream: linear fraction
          - dist to road: log-scaled inverse (closer to road = better access)
        """
        fit = pd.DataFrame(index=df.index)

        # 1. Slope (optimal 3-8 deg, drops towards 1 deg and 15 deg)
        slope = df["slope_deg"].values
        slope_fit = np.where(
            (slope >= 3.0) & (slope <= 8.0), 1.0,
            np.where(slope < 3.0, 0.5 + 0.5 * (slope - 1.0) / 2.0,
            np.where(slope <= 15.0, 1.0 - 0.8 * (slope - 8.0) / 7.0, 0.1))
        )
        fit["slope_deg"] = np.clip(slope_fit, 0.0, 1.0)

        # 2. Flow accumulation (log-scale normalized)
        fa_log = np.log1p(np.maximum(df["flow_accum"].values, 0.0))
        min_fa_log, max_fa_log = np.log1p(100.0), np.log1p(50000.0)
        fit["flow_accum"] = np.clip((fa_log - min_fa_log) / (max_fa_log - min_fa_log + 1e-9), 0.0, 1.0)

        # 3. TWI (Topographic Wetness Index: higher is wetter)
        twi = df["twi"].values
        fit["twi"] = np.clip((twi - 3.0) / (15.0 - 3.0), 0.0, 1.0)

        # 4. Rainfall (mean annual rainfall mm: higher is better refill)
        rf = df["rainfall_mm"].values
        fit["rainfall_mm"] = np.clip((rf - 500.0) / (1200.0 - 500.0), 0.0, 1.0)

        # 5. Soil texture (clay/loam retain water; sand/gravel leak)
        soil_map = self.fit_params.get("soil_scores", {
            "clay": 1.00,
            "loamy": 0.85,
            "sandy": 0.30,
            "gravelly": 0.20,
        })
        fit["soil_texture"] = df["soil_texture"].map(soil_map).fillna(0.50).values

        # 6. Lithology (gneiss/granite provide stable foundation; alluvium/sandstone seep)
        lith_map = self.fit_params.get("lithology_scores", {
            "gneiss": 0.90,
            "granite": 0.85,
            "basalt": 0.70,
            "sandstone": 0.40,
            "alluvium": 0.30,
        })
        fit["lithology"] = df["lithology"].map(lith_map).fillna(0.50).values

        # 7. Cropland fraction upstream (agricultural water demand)
        fit["cropland_fraction_upstream"] = np.clip(df["cropland_fraction_upstream"].values, 0.0, 1.0)

        # 8. Distance to road (closer = lower construction & transport cost)
        road_log = np.log1p(np.maximum(df["dist_to_road_m"].values, 10.0))
        min_rd_log, max_rd_log = np.log1p(20.0), np.log1p(5000.0)
        fit["dist_to_road_m"] = np.clip(1.0 - (road_log - min_rd_log) / (max_rd_log - min_rd_log + 1e-9), 0.0, 1.0)

        return fit

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute AHP composite suitability scores (0 to 100).

        Notice: This method takes a DataFrame and only accesses the feature columns.
        It NEVER accesses or requires a 'label' column.
        """
        fit_df = self.compute_fit_scores(df)
        weighted_sum = np.zeros(len(df))

        for criterion in self.criteria:
            weighted_sum += self.weights_dict[criterion] * fit_df[criterion].values

        ahp_scores = weighted_sum * self.score_multiplier
        return np.round(ahp_scores, 2)


def main():
    parser = argparse.ArgumentParser(description="Compute AHP suitability scores.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)

    in_candidates = project_root / config["paths"]["candidates_filtered_csv"]
    print("=" * 65)
    print("STAGE 1b - STEP 2: AHP SUITABILITY SCORING")
    print("=" * 65)

    scorer = AHPScorer(config=config)
    print("AHP Pairwise Criteria & Eigenvector Weights:")
    for c, w in scorer.weights_dict.items():
        print(f"  {c:<30} weight = {w:.4f} ({w*100:.1f}%)")

    print(f"\nPrincipal Eigenvalue (lambda_max): {scorer.lambda_max:.4f}")
    print(f"Consistency Ratio (CR):           {scorer.cr:.4f} (Threshold < {scorer.max_cr:.2f})")
    print("Consistency check: PASSED ✅")

    df = pd.read_csv(in_candidates)
    scores = scorer.score(df)
    print(f"\nEvaluated {len(scores)} candidate sites.")
    print(f"AHP Score range: [{scores.min():.2f}, {scores.max():.2f}], Mean: {scores.mean():.2f}")
    print("\n[SYNTHETIC - pipeline test, not a real finding]")


if __name__ == "__main__":
    main()
