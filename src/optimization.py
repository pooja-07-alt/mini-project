"""
optimization.py
───────────────
Multi-objective optimization for check dam site selection using NSGA-II.

Objectives:
  1. Maximize estimated water storage (m³)  → minimize -storage
  2. Minimize cost proxy (slope × dist_to_road)
  3. Minimize estimated cropland submergence (ha)

Constraint:
  Suitability probability ≥ SUITABILITY_THRESHOLD

Requires: pymoo >= 0.6
Install:  pip install pymoo
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False
    warnings.warn("pymoo not installed. Run: pip install pymoo")


# ── Constants ─────────────────────────────────────────────────────────────────
SUITABILITY_THRESHOLD = 0.60   # candidates below this are excluded
DAM_HEIGHT_M          = 3.0    # assumed check dam height for storage estimate
CATCHMENT_COEFF       = 0.003  # storage ≈ coeff × flow_accum × height (proxy)
COST_SCALE            = 1e-4   # normalisation for cost proxy values


# ── Proxy objective functions ─────────────────────────────────────────────────

def estimate_storage_m3(row: pd.Series, dam_height_m: float = DAM_HEIGHT_M) -> float:
    """
    Estimate water storage in m³.

    Proxy formula: storage ≈ CATCHMENT_COEFF × flow_accumulation × dam_height²
    This is a rough proxy — replace with proper curve-number hydrology in Stage 7.

    Higher storage = better. NSGA-II minimises, so we return -storage.
    """
    return CATCHMENT_COEFF * row["flow_accum"] * (dam_height_m ** 2)


def estimate_cost_proxy(row: pd.Series) -> float:
    """
    Estimate relative construction cost.

    Proxy: slope (steeper → more earthwork) × distance to road (farther → more transport)
    Normalised to rough 0–1 range. Lower = cheaper.
    """
    return row["slope_deg"] * row["dist_to_road_m"] * COST_SCALE


def estimate_submergence_ha(row: pd.Series, dam_height_m: float = DAM_HEIGHT_M) -> float:
    """
    Estimate cropland area submerged (ha).

    Proxy: submergence ∝ flow_accum × dam_height × LULC_cropland_fraction
    LULC class 1 = agriculture (most submergence), class 2 = shrub, class 3 = forest.
    """
    lulc_crop_fraction = {1: 0.8, 2: 0.3, 3: 0.1, 4: 0.05}
    crop_frac = lulc_crop_fraction.get(int(row["lulc_class"]), 0.2)
    return (row["flow_accum"] * dam_height_m * crop_frac) / 10_000  # convert m² to ha


def compute_objectives(df: pd.DataFrame, dam_height_m: float = DAM_HEIGHT_M) -> pd.DataFrame:
    """
    Compute all three objective values for every candidate in the DataFrame.

    Parameters
    ----------
    df           : Feature DataFrame with suitability_score column already added
    dam_height_m : Assumed dam height for proxy calculations

    Returns
    -------
    DataFrame with columns: site_id, storage_m3, cost_proxy, submergence_ha,
                            suitability_score (if present)
    """
    df = df.copy()
    df["storage_m3"]      = df.apply(estimate_storage_m3,    axis=1, dam_height_m=dam_height_m)
    df["cost_proxy"]      = df.apply(estimate_cost_proxy,    axis=1)
    df["submergence_ha"]  = df.apply(estimate_submergence_ha, axis=1, dam_height_m=dam_height_m)
    return df


# ── NSGA-II Problem Definition ────────────────────────────────────────────────

class CheckDamSelectionProblem:
    """
    Multi-objective problem wrapper using pymoo.

    This class is used when selecting the BEST SUBSET of k sites from
    all candidates above the suitability threshold.

    For now (Stage 1 warm-up), the problem is simplified to ranking
    individual candidates on their three objectives independently.
    The full combinatorial version (selecting k sites jointly) is Stage 7.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        suitability_threshold: float = SUITABILITY_THRESHOLD,
    ):
        if not _PYMOO_AVAILABLE:
            raise ImportError("pymoo not installed. Run: pip install pymoo")

        self.candidates = candidates.copy()
        self.threshold = suitability_threshold

        # Filter by suitability threshold
        if "suitability_score" not in self.candidates.columns:
            raise ValueError("candidates must have a 'suitability_score' column. "
                             "Run compute_objectives() after the ML model scores them.")
        self.eligible = self.candidates[
            self.candidates["suitability_score"] >= self.threshold
        ].reset_index(drop=True)

        print(f"  Candidates above threshold ({suitability_threshold}): "
              f"{len(self.eligible)} / {len(self.candidates)}")


    def get_pareto_front(self) -> pd.DataFrame:
        """
        Identify Pareto-optimal candidates using non-dominated sorting.

        A site is Pareto-optimal if no other site is strictly better on
        ALL three objectives simultaneously.

        Returns
        -------
        DataFrame of Pareto-optimal sites (subset of eligible candidates)
        """
        obj = self.eligible[["cost_proxy", "submergence_ha"]].copy()
        obj["neg_storage"] = -self.eligible["storage_m3"]  # minimise negative = maximise

        values = obj[["neg_storage", "cost_proxy", "submergence_ha"]].values
        n = len(values)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j is <= i on all objectives and < i on at least one
                if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                    is_pareto[i] = False
                    break

        pareto_df = self.eligible[is_pareto].copy()
        pareto_df["pareto_rank"] = 1
        print(f"  Pareto-optimal sites: {len(pareto_df)}")
        return pareto_df.sort_values("storage_m3", ascending=False).reset_index(drop=True)


def run_optimization(
    scored_candidates: pd.DataFrame,
    suitability_threshold: float = SUITABILITY_THRESHOLD,
    dam_height_m: float = DAM_HEIGHT_M,
) -> pd.DataFrame:
    """
    Full optimization pipeline: compute objectives → Pareto front.

    Parameters
    ----------
    scored_candidates      : DataFrame with suitability_score column
    suitability_threshold  : Minimum suitability to include a site
    dam_height_m           : Dam height for storage/submergence estimates

    Returns
    -------
    DataFrame of Pareto-optimal sites ready to export as GeoJSON
    """
    # Step 1: compute proxy objectives for all candidates
    df_obj = compute_objectives(scored_candidates, dam_height_m=dam_height_m)

    # Step 2: run Pareto selection
    problem = CheckDamSelectionProblem(df_obj, suitability_threshold=suitability_threshold)
    pareto_sites = problem.get_pareto_front()

    return pareto_sites
