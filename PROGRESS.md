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
- Formulated 7 strict engineering/environmental rules with thresholds configured in `config.yaml` (marked ASSUMPTION - verify).
- Evaluated against 3,000 synthetic v2 raw sites:
  - 563 sites passed all constraints (18.8% survival rate).
  - Preserved 206 positive labels (36.6% positive class balance among passed candidates).
  - 658 positives removed from unfeasible terrain (primarily sub-1.0 km² catchments and steep slopes).
- Generated output candidate table: `data/tables/candidates_v2.csv`.
- Generated audit table: `outputs/stage1b_constraint_funnel.csv`.
- Added unit tests for hand-made rows verifying each rule independently.
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
