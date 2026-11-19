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

---

## Stage 2 Decisions

*(fill in after completing Stage 2)*

---

*(continue for each stage)*
