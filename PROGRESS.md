# PROGRESS.md — Stage Tracker

> Update this file at the end of every stage.
> Format: what was done, what decisions were made, what comes next.

---

## Current Stage: 0 — Setup ✅

**Date completed:** 2026-10-16  
**Verified:** `setup_stage0.py` ran cleanly; all 4 src modules import correctly.

### What was done
- Created repository skeleton with all folders and documentation files.
- Wrote `README.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `DATA_SOURCES.md`, `LEARNING_NOTES.md`.
- Created `requirements.txt` with all needed packages.
- Created `.gitignore` to exclude raw rasters, credentials, and large files.
- Created `data/tables/synthetic_checkdam_features.csv` — a synthetic dataset for Stage 1 warm-up.
- Created placeholder `docs/index.html` for the Leaflet dashboard.
- Created `src/` module stubs for preprocessing, ML, and optimization.

### Decisions made
- See `DECISIONS.md` for all rationale.

### What comes next (Stage 1)
- Open `notebooks/stage1_ml_warmup.ipynb`.
- Load the synthetic CSV and explore it (shape, columns, statistics, class balance).
- Train a Random Forest and XGBoost model.
- Implement spatial block cross-validation.
- Plot feature importances and SHAP values.
- Record AUC-ROC, precision, recall.
- Write what you learned in `LEARNING_NOTES.md`.

---

## Stage 1 — ML Warm-up ✅

**Date completed:** 2026-10-16

### What was done
- Loaded `synthetic_checkdam_features.csv` (1000 rows, 13 columns, 25% positive class).
- Ran spatial block leave-one-out cross-validation (5 blocks) for both RF and XGBoost.
- Trained final models on all data; computed SHAP values and feature importances.
- Compared AHP baseline vs ML on same dataset.
- Scored all 1000 candidates; saved to `outputs/stage1_all_candidates_scored.csv`.
- Output plots: `stage1_feature_distributions.png`, `stage1_feature_importance.png`, `stage1_shap_summary.png`.

### Key metrics achieved (spatial block CV — mean across 5 blocks)
| Metric | Random Forest | XGBoost | AHP Baseline |
|--------|:------------:|:-------:|:------------:|
| AUC-ROC | **0.992** | **0.993** | 0.921 |
| Precision | 0.911 | 0.924 | 0.325 |
| Recall | 0.933 | 0.933 | 0.996 |
| F1 | 0.921 | 0.928 | 0.491 |

> Note: high AUC on synthetic data is expected — distributions were designed to be separable.
> Real data AUC will be lower. The workflow and spatial CV setup are what matter here.

### Top feature importances (Random Forest)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `slope_deg` | 0.265 |
| 2 | `dist_to_road_m` | 0.205 |
| 3 | `flow_accum` | 0.193 |
| 4 | `twi` | 0.112 |
| 5 | `stream_order` | 0.088 |

### What I learned
*(answer the prompts in LEARNING_NOTES.md in your own words)*

### What comes next (Stage 2)
- Select the sub-watershed study area.
- Register for Google Earth Engine if not done.
- Identify and download all data layers (DEM, LULC, rainfall, soil, geology, roads, dam points).
- Populate `DATA_SOURCES.md` with real entries.

---

## Stage 1b — Pipeline Refactor (AHP + ML Hybrid) 🟡 In Progress

**Branch:** `stage-1b-pipeline-refactor`
**Focus:** Synthetic v2 generation, hard constraints funnel, eigenvector AHP, spatial CV ML models, agreement mapping, and multi-objective optimization ranking.

### Step 0: Audit & Synthetic Data v2 ✅
- Audited Stage 1 v1 synthetic generator: confirmed that class-conditional feature separation and direct alignment with AHP criteria caused circularity and artificially inflated AUC (>0.99).
- Documented findings in `DECISIONS.md` (D-09) and concepts in `LEARNING_NOTES.md`.
- Implemented `generate_synthetic_dataset_v2()` in `src/generate_synthetic.py`:
  - 3,000 candidate stream sites across 8 contiguous geographic spatial blocks.
  - 21 columns including coordinates, terrain, hydrology, soil, geology, land use, and infrastructure distances.
  - Latent data-generating process incorporating an unobserved geotechnical/subsurface factor ($u \sim \mathcal{N}(0, 1)$) and logistic noise to cap achievable model AUC at ~0.75–0.90.
  - Preserved original v1 file `data/tables/synthetic_checkdam_features.csv` intact; generated new table `data/tables/synthetic_checkdam_candidates_v2.csv`.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 1: Hard Constraints & Candidate Filtering ✅
- Implemented `apply_hard_constraints()` in `src/constraints.py`.
- Formulated 7 strict engineering/environmental rules with thresholds configured in `config.yaml` (marked assumed expert-style judgement; literature support UNVERIFIED).
- Evaluated against 3,000 synthetic v2 raw sites:
  - 563 sites passed all constraints (18.8% survival rate).
  - Preserved 331 positive labels (58.8% positive class balance among passed candidates).
  - 566 positives removed from unfeasible terrain (primarily sub-1.0 km² catchments and steep slopes).
- Generated output candidate table: `data/tables/candidates_v2.csv`.
- Generated audit table: `outputs/stage1b_constraint_funnel.csv`.
- Added unit tests for hand-made rows verifying each rule independently.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 2: AHP Scoring with Eigenvector Weights & CR Verification ✅
- Implemented `src/ahp.py` with `AHPScorer` and `compute_eigenvector_weights()`.
- Built an 8-factor pairwise comparison matrix loaded from `config.yaml`:
  - Principal eigenvalue $\lambda_{\max} = 8.0961$.
  - Consistency Ratio $CR = 0.0097 < 0.10$ (fully consistent, passes Saaty's criterion).
  - Derived priority weights: slope (23.5%), flow accumulation (23.5%), TWI (14.0%), rainfall (14.0%), soil (8.2%), lithology (8.2%), cropland upstream (5.1%), road distance (3.4%).
- Converted continuous features to physical [0, 1] fit scores and scaled `ahp_score` to [0, 100].
- Added unit tests verifying:
  - Exact eigenvector weight normalization and CR computation.
  - Exception raised loudly when evaluating an inconsistent matrix.
  - Zero label usage: `ahp_score` executes cleanly when `label` column is removed.
  - Output candidate file with AHP score saved to `data/tables/candidates_v2_ahp.csv`.
  - AHP score distribution: Min: 41.66, Median: 61.60, Max: 78.23, Mean: 61.47.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 3: ML-Only Models with Spatial Cross-Validation ✅
- Implemented `run_spatial_block_cv()` and `run_step3_ml_only()` in `src/ml_model.py`.
- Trained Random Forest and XGBoost across 8 spatial blocks (leave-one-block-out CV):
  - Strictly excluded `ahp_score` from features (ML-only baseline).
  - One-hot encoded categorical variables (`lulc_class`, `soil_texture`, `lithology`) fitted exclusively on training folds.
  - Handled class imbalance using `class_weight="balanced"` (RF) and dynamically calculated `scale_pos_weight` (XGBoost).
  - Selected decision threshold on the training fold by maximizing F1 score, then applied to the held-out test block.
  - Note: These F1-tuned-threshold metrics are *tuned on training predictions - NOT used for method comparison*. Full comparative benchmarking across models using standardized thresholding rules is reserved for Step 5.
- Out-of-fold spatial CV performance (mean ± std across 8 blocks):
  - **Random Forest (ML-only):**
    - AUC-ROC: **0.736 ± 0.061**
    - PR-AUC: **0.782 ± 0.082**
    - Precision: **0.734 ± 0.062**
    - Recall: **0.754 ± 0.135**
    - F1: **0.739 ± 0.079**
  - **XGBoost (ML-only):**
    - AUC-ROC: **0.721 ± 0.049**
    - PR-AUC: **0.766 ± 0.075**
    - Precision: **0.725 ± 0.081**
    - Recall: **0.725 ± 0.114**
    - F1: **0.720 ± 0.075**
- Exported out-of-fold predicted probabilities for every candidate site to `outputs/stage1b_oof_ml_only.csv`.
- Exported per-block class counts to `outputs/stage1b_block_class_counts.csv`.
- Exported per-fold CV metrics to `outputs/stage1b_cv_ml_only.csv`.
- Generated SHAP summary plot at `outputs/stage1b_shap_summary.png`.
- Documented unverified literature and guidelines in `docs/CITATIONS_TO_VERIFY.md` and recorded Decision D-13 (feature double-counting audit) in `DECISIONS.md`.
- Added unit tests verifying:
  - Complete block and site disjointness between train and test sets in all folds.
  - Validity and bounds of out-of-fold probability predictions.
  - Integrity of per-block counts, per-fold CV table, SHAP plot, and citations registry.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 4: Method A — ML with ahp_score as Feature ✅
- Implemented `run_step4_ml_ahp()` in `src/ml_model.py`.
- Evaluated Random Forest and XGBoost across the exact same 8 spatial folds and hyperparameters, with `ahp_score` included as an additional feature.
- Out-of-fold spatial CV performance (mean ± std across 8 blocks):
  - **Random Forest (+ AHP score):**
    - AUC-ROC: **0.754 $\pm$ 0.044** (up from 0.736 in ML-only)
    - PR-AUC: **0.787 $\pm$ 0.072** (up from 0.782 in ML-only)
    - Precision: **0.732 $\pm$ 0.090**
    - Recall: **0.782 $\pm$ 0.125**
    - F1: **0.749 $\pm$ 0.078**
  - **XGBoost (+ AHP score):**
    - AUC-ROC: **0.720 $\pm$ 0.026** (vs 0.721 in ML-only)
    - PR-AUC: **0.752 $\pm$ 0.063** (vs 0.766 in ML-only)
    - Precision: **0.719 $\pm$ 0.081**
    - Recall: **0.738 $\pm$ 0.100**
    - F1: **0.725 $\pm$ 0.073**
- Exported OOF predictions to `outputs/stage1b_oof_ml_ahp.csv` and CV metrics to `outputs/stage1b_cv_ml_ahp.csv`.
- Added unit tests in `tests/test_stage1b.py` verifying file creation, bounds, and metrics schema.
- Documented Decision D-14 in `DECISIONS.md`.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 5: Standardized 5-Method Comparison Benchmark ✅
- Implemented `src/evaluation.py` to benchmark all five suitability approaches across the exact same 8 spatial blocks and 563 candidate sites:
  1. `AHP-only`: Eigenvector composite score [0, 100] (no model training)
  2. `RF ML-only`: Random Forest strictly without AHP feature
  3. `XGB ML-only`: XGBoost strictly without AHP feature
  4. `RF + ahp_score`: Method A — Random Forest with `ahp_score` feature
  5. `XGB + ahp_score`: Method A — XGBoost with `ahp_score` feature
- Standardized Threshold Rule:
  - Eliminated arbitrary 0.50 cutoffs and threshold-tuning leakage.
  - In each held-out fold, classified the top $p\%$ of candidate sites by score, where $p$ is the actual positive prevalence of the training folds ($k_{\text{top}} = \text{round}(p_{\text{train}} \times n_{\text{test}})$).
- Out-of-fold spatial CV performance summary (mean ± std across 8 blocks):
  | Method | AUC-ROC (mean ± std) | PR-AUC (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) | Pooled AUC-ROC | Pooled PR-AUC | Pooled F1 |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **AHP-only** | 0.720 ± 0.064 | 0.776 ± 0.073 | 0.712 ± 0.119 | 0.711 ± 0.075 | 0.706 ± 0.080 | 0.716 | 0.771 | 0.707 |
  | **RF ML-only** | 0.736 ± 0.061 | 0.783 ± 0.082 | 0.720 ± 0.096 | 0.721 ± 0.051 | 0.715 ± 0.039 | 0.737 | 0.779 | 0.716 |
  | **XGB ML-only** | 0.721 ± 0.049 | 0.766 ± 0.075 | 0.723 ± 0.105 | 0.722 ± 0.045 | 0.718 ± 0.051 | 0.725 | 0.762 | 0.719 |
  | **RF + ahp_score** | **0.754 ± 0.044** | **0.788 ± 0.072** | **0.735 ± 0.103** | **0.735 ± 0.037** | **0.730 ± 0.043** | **0.753** | **0.784** | **0.731** |
  | **XGB + ahp_score** | 0.720 ± 0.026 | 0.751 ± 0.062 | 0.743 ± 0.100 | 0.743 ± 0.041 | 0.738 ± 0.043 | 0.721 | 0.745 | 0.740 |
- Paired comparison results:
  - Random Forest: Mean $\Delta$AUC = `+0.0175`, improved in `6 / 8` folds.
  - XGBoost: Mean $\Delta$AUC = `-0.0017`, improved in `2 / 8` folds.
  - Fold-to-fold standard deviation (0.044 to 0.064) exceeds the magnitude of the delta, indicating geographic variability dominates feature variations.
- Exported artifacts:
  - `outputs/stage1b_comparison_table.csv` & `outputs/stage1b_comparison_table.md`
  - `outputs/stage1b_per_fold_metrics.csv`
  - `outputs/stage1b_roc_pr_curves.png` (ROC and PR curves for all 5 methods with prevalence baseline)
  - `outputs/stage1b_comparison_notes.md` (plain-language scientific interpretation)
- Documented Decision D-15 in `DECISIONS.md`.
- Status: SYNTHETIC - pipeline test, not a real finding.

### Step 5b: Synthetic Sensitivity Scenarios (Aligned vs Misaligned) ✅
- Added `scenario` parameter (`"ahp_aligned"` vs `"ahp_misaligned"`) to `src/generate_synthetic.py` and `config.yaml`.
  - Main synthetic candidate datasets (`candidates_v2.csv`, `candidates_v2_ahp.csv`) remain completely untouched.
  - In `ahp_aligned`, terrain slope and drainage area dominate ground truth suitability (matching AHP priorities).
  - In `ahp_misaligned`, rainfall and lithology dominate ground truth suitability, while slope has minor influence. The latent unobserved factor is preserved in both.
- Benchmarked all 5 suitability methods across both scenarios using 8-fold spatial block cross-validation:
  | Scenario | Method | AUC-ROC (mean ± std) | PR-AUC (mean ± std) | Pooled AUC-ROC | Pooled PR-AUC |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | **ahp_aligned** | AHP-only | 0.720 ± 0.064 | 0.776 ± 0.073 | 0.716 | 0.771 |
  | ahp_aligned | RF ML-only | 0.736 ± 0.061 | 0.782 ± 0.082 | 0.737 | 0.779 |
  | ahp_aligned | XGB ML-only | 0.721 ± 0.049 | 0.766 ± 0.075 | 0.725 | 0.762 |
  | ahp_aligned | RF + ahp_score | **0.754 ± 0.044** | **0.787 ± 0.072** | **0.753** | **0.785** |
  | ahp_aligned | XGB + ahp_score | 0.720 ± 0.026 | 0.752 ± 0.063 | 0.721 | 0.745 |
  | **ahp_misaligned** | AHP-only | 0.670 ± 0.055 | 0.785 ± 0.086 | 0.663 | 0.778 |
  | ahp_misaligned | RF ML-only | **0.732 ± 0.061** | 0.830 ± 0.081 | **0.753** | 0.835 |
  | ahp_misaligned | XGB ML-only | 0.714 ± 0.061 | 0.813 ± 0.084 | 0.734 | 0.821 |
  | ahp_misaligned | RF + ahp_score | 0.735 ± 0.045 | **0.835 ± 0.080** | 0.752 | **0.842** |
  | ahp_misaligned | XGB + ahp_score | 0.721 ± 0.061 | 0.812 ± 0.086 | 0.738 | 0.817 |
- Key Finding:
  - When expert rules align with physical terrain reality, AHP provides a valuable inductive bias (+0.018 AUC).
  - When expert rules misjudge terrain reality, AHP-only discrimination degrades significantly (AUC drops from 0.720 to 0.670), whereas machine learning algorithms dynamically adapt to the empirical training patterns (RF ML-only achieves 0.732 / pooled 0.753).
  - Demonstrates why empirical validation on real data in Stage 6 is the true test: the optimal method depends on whether expert weights accurately reflect the study basin.
- Exported artifacts:
  - `outputs/stage1b_scenario_comparison.csv`
  - `outputs/stage1b_scenario_comparison.md`
  - `outputs/stage1b_scenario_notes.md`
- Documented Decision D-16 in `DECISIONS.md`.
- Status: SYNTHETIC - pipeline test, not a real finding.

---

## Stage 2 — Study Area & Data Collection ⬜

**Date completed:** —

*(fill in)*

---

## Stage 3 — Terrain Processing ⬜

**Date completed:** —

*(fill in)*

---

## Stage 4 — Candidates & Feature Table ⬜

**Date completed:** —

*(fill in)*

---

## Stage 5 — Labels ⬜

**Date completed:** —

*(fill in)*

---

## Stage 6 — Real Modelling ⬜

**Date completed:** —

*(fill in)*

---

## Stage 7 — Optimization ⬜

**Date completed:** —

*(fill in)*

---

## Stage 8 — Dashboard ⬜

**Date completed:** —

*(fill in)*

---

## Stage 9 — Evaluation & Report ⬜

**Date completed:** —

*(fill in)*
