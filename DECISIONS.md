# DECISIONS.md — Design Choices and Rationale

> Record every significant decision here so future-you and reviewers understand why things are the way they are.
> Format: Decision → Options considered → Choice → Reason

---

## Stage 0 Decisions

---

### D-01: Scope limited to check dams only

**Options considered:**
- A) Check dams + farm ponds + percolation tanks
- B) Check dams only, code extensible for others later

**Choice:** B

**Reason:** Each structure type requires a different ML model, different training labels, and different objective functions for optimization. Building all three correctly in one project multiplies the data labelling problem and the validation burden. For a first ML project, doing one thing well and demonstrating you understand the limitations is far more valuable than three shallow pipelines. Farm ponds and percolation tanks are documented as future work in the README and the code is structured with a `structure_type` parameter to make adding them later straightforward.

---

### D-02: All heavy processing runs offline; dashboard is static HTML

**Options considered:**
- A) Flask/FastAPI backend running ML on demand
- B) Pre-compute everything in Python, export GeoJSON, Leaflet reads files
- C) Streamlit app

**Choice:** B

**Reason:** A backend server requires hosting, authentication, cost, and DevOps knowledge that is out of scope for this project. Streamlit works but cannot be hosted on GitHub Pages for free. Static HTML + Leaflet + pre-computed GeoJSON is the simplest architecture that delivers a real interactive map accessible to anyone with a link, with zero running cost.

---

### D-03: Spatial block cross-validation instead of random split

**Options considered:**
- A) Random 80/20 train/test split
- B) Spatial block cross-validation (split by geographic blocks, not randomly)

**Choice:** B

**Reason:** Neighbouring pixels in a raster have highly correlated feature values (spatial autocorrelation). If you split randomly, your test set contains pixels that are right next to training pixels — the model has effectively "seen" those locations. This gives artificially inflated AUC scores that do not generalize. Spatial CV ensures test blocks are geographically separated from training blocks, giving a realistic estimate of how the model performs on unseen areas. This is standard practice in geospatial ML and is what peer-reviewed reviewers will expect.

---

### D-04: Coordinate system — UTM Zone 43N for processing, WGS84 for output

**Options considered:**
- A) Stay in geographic coordinates (lat/long) throughout
- B) Project to UTM for processing, convert to WGS84 for output

**Choice:** B

**Reason:** Slope, distance, area, and TWI calculations require metric units. Geographic coordinates give meaningless results for these operations because degrees are not equal in length. UTM Zone 43N (EPSG:32643) is the standard projected CRS for Karnataka and most of peninsular India. WGS84 (EPSG:4326) is required by GeoJSON and Leaflet.

---

### D-05: Cost proxy = slope × distance to road (not real cost)

**Options considered:**
- A) Use real construction cost data from Karnataka DWMS tenders
- B) Use a proxy: slope (steeper = harder construction) × distance to road (farther = harder access)

**Choice:** B (initially), with A as future work

**Reason:** Real tender data is not publicly available in a structured form and procuring it would likely take more time than the project allows. The proxy captures the two dominant cost drivers (earthwork difficulty and material transport) in a defensible, reproducible way. The limitation is stated openly in the README and the report.

---

### D-06: NSGA-II for multi-objective optimization

**Options considered:**
- A) Weighted sum of objectives → single score → rank by score
- B) NSGA-II (Non-Dominated Sorting Genetic Algorithm II) via pymoo

**Choice:** B

**Reason:** A weighted sum requires you to decide the relative importance of storage, cost, and submergence — a subjective choice that affects every output. NSGA-II produces a Pareto front: the set of solutions where you genuinely cannot improve one objective without worsening another. This is more honest because it presents the trade-offs to the decision-maker (a water department engineer) rather than hiding them inside weights. pymoo has a clean, well-documented NSGA-II implementation with constraint handling.

---

### D-07: AHP baseline included alongside ML models

**Options considered:**
- A) ML models only
- B) AHP baseline + ML models, compare both

**Choice:** B

**Reason:** An ML model is only meaningful if it beats something simpler. AHP (Analytic Hierarchy Process) is the standard method used by water departments and government agencies for site scoring. Showing that the ML model outperforms AHP (or understanding why it does not) is the core scientific contribution of the project. Without a baseline, you cannot make a credible claim.

---

### D-08: Synthetic dataset for Stage 1 warm-up before real data

**Options considered:**
- A) Start ML directly on real data
- B) First build and validate the entire ML pipeline on a synthetic dataset

**Choice:** B

**Reason:** Real geospatial data requires weeks of acquisition and cleaning. Learning the ML workflow (data loading, class balance, train/test split, spatial CV, SHAP, metrics) on a synthetic dataset with known properties lets you make and correct mistakes on clean data before facing messy real data. When you then apply the same notebook to real data, the bugs are already gone and you can focus on data quality issues.

---

## Stage 1b Decisions (Pipeline Refactor)

---

### D-09: Audit of Stage 1 v1 Synthetic Labels and Redesign for v2

**Audit of v1 Synthetic Label Generation:**
In Stage 1 (`src/generate_synthetic.py` v1), labels were generated through class-conditional sampling:
1. Two separate dictionaries (`pos` for label=1 and `neg` for label=0) drew features from completely separated distributions.
2. For label=1, features were sampled directly around ideal civil engineering/AHP criteria (e.g., slope ~ 8° ± 3°, stream order strictly {1,2,3}, soil texture strictly loamy or clay, low distance to road).
3. For label=0, features were sampled from broader/poorer distributions (e.g., slope ~ 15° ± 7°, stream orders 1–5, soil texture including sandy).
4. **Circularity & Inflation Assessment:**
   Because positive and negative samples were separated *a priori* by manually partitioned parameter boundaries without overlap, realistic noise, or unobserved confounding, ML models achieved near-perfect scores (AUC > 0.992). Furthermore, because the positive sampling aligned directly with AHP preferred criteria, comparing AHP against ML on this data was fundamentally circular—both models were evaluated against an artificial ground truth hand-crafted to reward their specific assumptions.

**v2 Data Generating Process:**
To address this circularity in Stage 1b:
1. **Candidate-first sampling:** ~3,000 stream candidates are generated across realistic Karnataka sub-watershed terrain and climate distributions before assigning labels.
2. **Contiguous Spatial Blocks:** 8 distinct spatial blocks (`block_id` 1 to 8) are constructed via 2D latitude-longitude geographic grid partitioning (2 latitude × 4 longitude blocks) to ensure rigorous spatial cross-validation.
3. **Latent Process with Unobserved Factors:**
   Labels are assigned via a latent utility function:
   $$z = \sum w_i f_i(\text{observed}) + w_{\text{unobs}} \cdot u + \text{penalties} + \epsilon_{\text{logistic}}$$
   where $u \sim \mathcal{N}(0, 1)$ is an **unobserved factor** (representing unmapped subsurface foundation bedrock depth, localized aquifer fracture porosity, or community administrative approval) and $\epsilon \sim \text{Logistic}(0, 0.65)$ is stochastic disturbance.
4. **Target AUC (0.75 – 0.90):** Because the unobserved factor accounts for substantial variance (~35%) and is omitted from the tabular features provided to models, no model can achieve trivial or circular separation.

**Status:** SYNTHETIC - pipeline test, not a real finding. ASSUMPTION - verify latent factors against local field data in Stage 5.

### D-10: Hard Constraints Filtering Before ML and Scoring

**Options considered:**
- A) Let the ML model and AHP score every raw stream pixel directly, including settlements, steep ravines, and water bodies.
- B) Apply hard engineering and environmental constraints upfront to filter unfeasible sites; sample negatives for ML strictly from the passed sites.

**Choice:** B

**Reason:**
1. ML models are designed to learn complex, non-linear relationships, but having them waste model capacity learning that "you cannot build a check dam in the middle of a reservoir or through an existing village" degrades performance on subtle siting distinctions.
2. If negatives are sampled from unfeasible locations (like steep 45° mountain slopes or wide rivers), an ML model gets artificially high accuracy by learning trivial exclusions rather than true siting suitability. Restricting candidate sites and negative samples to physically plausible stream reaches creates a realistic decision boundary.
3. Every threshold is externalized to `config.yaml` and marked `assumed expert-style judgement; literature support UNVERIFIED (IMSD / Karnataka guidelines to be checked)`:
   - `stream_order`: [1, 3] (check dams are built on lower order streams)
   - `slope_deg`: [1.0, 15.0] (avoids standing ponds or high scour risk)
   - `catchment_area_km2`: >= 1.0 km² (ensures sufficient runoff yield)
   - `valley_width_m`: <= 120.0 m (limits masonry span length and cost)
   - `lulc_class`: exclude `built_up`, `water`, and `forest`
   - `protected_area_flag`: exclude protected areas (statutory conservation law)
   - `dist_to_settlement_m`: >= 150.0 m (safety buffer against backwater inundation)

**Output Files:** `data/tables/candidates_v2.csv` and `outputs/stage1b_constraint_funnel.csv`.
**Status:** SYNTHETIC - pipeline test, not a real finding.

### D-11: AHP Pairwise Comparison Matrix and Principal Eigenvector Scoring

**Options considered:**
- A) Ad-hoc arbitrary weight assignment (e.g. 0.20 slope, 0.20 flow, 0.15 TWI...) without formal consistency checks.
- B) Formal Saaty Analytic Hierarchy Process (AHP) using pairwise comparison matrix, principal eigenvector extraction, and Consistency Ratio (CR) verification.

**Choice:** B

**Reason:**
AHP provides a defensible multi-criteria framework rooted in matrix algebra rather than arbitrary guessing.
1. **Pairwise Judgements Rationale (assumed expert-style judgement; literature support UNVERIFIED):**
   - *Slope and Flow Accumulation (Equal Priority, ~23.5% each):* Slope controls hydraulic scouring and ponding geometry; flow accumulation dictates volume of available runoff. They are given equal top weight and judged 2x more important than TWI/rainfall, 3x more than soil/geology, and 5x more than road distance.
   - *TWI and Rainfall (~14.0% each):* Moisture convergence and precipitation supply.
   - *Soil Texture and Lithology (~8.2% each):* Low-permeability soils and bedrock foundations minimize reservoir seepage and prevent dam breach.
   - *Cropland Fraction Upstream (~5.1%):* Demand indicator for irrigation.
   - *Distance to Road (~3.4%):* Economic/accessibility factor. Secondary to physical viability.
2. **Matrix Consistency Check:**
   Principal eigenvalue $\lambda_{\max} = 8.0961$, Consistency Index $CI = 0.0137$, Saaty Random Index $RI(8) = 1.41$, yielding Consistency Ratio $CR = 0.0097 < 0.10$. The code includes an assertion that aborts execution if $CR \ge 0.10$.
3. **Strict A Priori Independence:**
   AHP weights and fit scoring functions are fixed beforehand and NEVER fitted, tuned, or calibrated using training labels.

**Status:** SYNTHETIC - pipeline test, not a real finding.

### D-12: ML-Only Models with Spatial Block Cross-Validation (Step 3)

**Options considered:**
- A) Standard k-fold random cross-validation.
- B) Leave-one-spatial-block-out cross-validation grouped strictly by `block_id` (8 spatial blocks).

**Choice:** B

**Reason:**
1. Random train-test splits on geospatial data cause severe data leakage because adjacent stream cells share terrain, soil, and precipitation features. The model memorizes regional patterns rather than learning transferable hydrological principles.
2. Under leave-one-block-out CV, each of the 8 geographic zones is held out entirely as an unseen test region. The training fold never sees any pixel from the test block, and categorical encoders are fitted strictly on the training fold.
3. Class imbalance is handled via `class_weight="balanced"` in Random Forest and dynamically calculated `scale_pos_weight` ($n_{\text{neg}} / n_{\text{pos}}$ on the training fold) in XGBoost.
4. Threshold selection: To avoid optimistic test evaluation, the probability classification threshold is selected on the training fold by maximizing the F1 score, then frozen and applied to the held-out test fold.
   - **Important Methodological Note:** Existing F1-tuned-threshold metrics from Step 3 are *tuned on training predictions - NOT used for method comparison*. Full comparative benchmarking across models using standardized thresholding rules is reserved for Step 5.
5. **Synthetic Calibration Note:** Synthetic generator coefficients in `src/generate_synthetic.py` were calibrated to reach a realistic target AUC range (~0.72–0.78), so absolute metric values on synthetic data are **design parameters, not empirical findings**.

**Status:** SYNTHETIC - pipeline test, not a real finding.

---

### D-13: Potential Double-Counting of Distance to Road and Cropland Fraction

**Issue Identified:**
- `dist_to_road_m` is currently used as an input feature for suitability scoring (both AHP and ML models) AND it is used in the multi-objective optimization as the primary driver of `cost_proxy = (valley_width * height) * (1 + 0.0005 * dist_to_road_m)`.
- Similarly, `cropland_fraction_upstream` is used in suitability scoring (higher upstream agriculture = higher irrigation demand = higher suitability) AND downstream agricultural land is penalized in multi-objective optimization as `submerged_cropland_ha`.
- This constitutes a potential **double-counting** of features across the suitability and optimization stages.

**Options Considered:**
- **Option A (Keep both):** Retain them in suitability scoring because historical check dams were preferentially built where roads made construction feasible and near farmland. In optimization, they represent operational cost and environmental trade-offs.
- **Option B (Drop from suitability):** Restrict suitability scoring strictly to biophysical/hydrological criteria (terrain slope, flow accumulation, geology, soil, rainfall, TWI) and reserve socio-economic / logistical factors (`dist_to_road_m`, `cropland_fraction_upstream`) entirely for the post-suitability Pareto optimization stage.

**Choice:** Keep Option A for now to preserve pipeline continuity on synthetic tests, but record this design tension. Decision on whether to drop them will be finalized before running on real data in Stage 6/7.

---

### D-14: Method A — Hybrid Model with ahp_score as an Input Feature (Step 4)

**Options considered:**
- A) Separate AHP and ML models without sharing features, comparing them only at the final ranking stage.
- B) Method A: Feed the prior knowledge distilled in `ahp_score` directly into Random Forest and XGBoost as an additional numeric feature alongside the raw terrain variables.

**Choice:** B

**Reason:**
1. Method A tests whether explicit expert domain knowledge (the compound AHP composite score) provides inductive bias that helps tree-based algorithms identify suitable structures in data-constrained regimes.
2. The exact same 8 spatial blocks and leave-one-block-out isolation rules are maintained. The `ahp_score` is computed without labels beforehand, ensuring that no target leakage occurs.
3. Out-of-fold spatial CV results on synthetic data:
   - RF with AHP: AUC-ROC **0.754 $\pm$ 0.044**, PR-AUC **0.787 $\pm$ 0.072** (vs 0.736 / 0.782 ML-only).
   - XGB with AHP: AUC-ROC **0.720 $\pm$ 0.026**, PR-AUC **0.752 $\pm$ 0.063** (vs 0.721 / 0.766 ML-only).
4. These results provide the foundation for the rigorous standardized 5-method comparison in Step 5.

**Status:** SYNTHETIC - pipeline test, not a real finding.

---

## Stage 2 Decisions

*(fill in after completing Stage 2)*

---

*(continue for each stage)*
