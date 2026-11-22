# LEARNING_NOTES.md — ML Concepts in Plain Language

> Write these in your own words after each stage.
> The goal is not to copy definitions — it's to prove to yourself (and reviewers) that you actually understand the concepts.
> Leave the Stage 0 template entries as prompts to fill in after Stage 1.

---

## How to Use This File

After each stage, add a section explaining what you learned.
Don't use jargon you don't understand. If you can't explain it simply, you haven't learned it yet.

A good test: could you explain this concept to a civil engineer who builds check dams but has never heard of machine learning?

---

## Stage 1 — ML Warm-up

*(Fill these in after completing Stage 1. The questions below are prompts — answer them in your own words.)*

---

### What is a feature?

*Prompt: In this project, what is an example of one feature? Why does the model need multiple features instead of just one?*

---

### What is a label?

*Prompt: What does label = 1 mean in our dataset? What does label = 0 mean? Why is getting the labels right so important?*

---

### What is a train/test split?

*Prompt: If you trained the model on all your data and then tested it on the same data, what would happen? Why is that a problem?*

---

### What is spatial cross-validation and why is it different from regular cross-validation?

*Prompt: Imagine two pixels that are 10 metres apart on a riverbank. If pixel A is in the training set and pixel B (10 metres away) is in the test set, is the test really "unseen"? What does spatial CV do instead?*

---

### What is overfitting?

*Prompt: Give an example of overfitting in plain language. How would you detect it in this project?*

---

### What is a Random Forest?

*Prompt: What is a "decision tree"? What does a Random Forest do with multiple trees that one tree can't do alone?*

---

### What is XGBoost?

*Prompt: How is XGBoost different from a Random Forest? What does "boosting" mean in plain language?*

---

### What is AUC-ROC?

*Prompt: If you flip a coin to decide whether a site is a check dam or not, what AUC would you expect? Why is a higher AUC better? What does it mean in this project?*

---

### What is precision and recall?

*Prompt: Imagine your model says 20 sites are suitable. Only 15 are actually good. And your model missed 5 real good sites. What is the precision? What is the recall? Which matters more in this project?*

---

### What are SHAP values?

*Prompt: If a site gets a suitability score of 0.82, SHAP tells you why. Give an example: "Site X scored 0.82 because [slope was steep] pushed the score up by +0.12, but [distance to road] pushed it down by −0.05..." Why is this useful for a real water department engineer?*

---

### What is the AHP baseline?

*Prompt: AHP stands for Analytic Hierarchy Process. It's a scoring method used by government engineers. How does it work in 2–3 sentences? Why do we include it in the project even though we have ML?*

---

## Stage 1b — Pipeline Refactor & Methodological Learning

### What is circular validation (and why was v1 synthetic data circular)?

In machine learning, **circular validation** happens when the rule or assumptions used to create the data's labels are the exact same assumptions being tested or scored.
In our Stage 1 v1 synthetic generator:
- Sites with label = 1 were manually assigned slopes of 5–12°, clay/loam soils, and low distances to roads.
- Sites with label = 0 were assigned steeper slopes and poor soils.
When our AHP baseline and Random Forest were tested on this data, they achieved 0.92–0.99 AUC. But this didn't prove the models understood real Karnataka hydrology—it only proved they could rediscover the two statistical distributions we hand-coded!
In real world projects, labels come from nature or historical human construction, which always contain noise, historical preferences, and unobserved variables.

### What is an unobserved factor (confounder)?

In the real world, a satellite cannot see everything. For example, satellite data gives us surface slope, land cover, and estimated rainfall. But it cannot see:
1. How deep solid bedrock is beneath the stream bed (essential for a dam's structural foundation).
2. Local subsurface rock fracture permeability (will the stored water seep underground too fast?).
3. Whether local farmers and gram panchayats consented to build the structure.

In our synthetic v2 generator, we explicitly added a latent `unobserved_factor` that influences whether a check dam exists or succeeds. Because the model cannot see this column, the highest AUC a good model can reach is bounded around 0.75–0.90. This teaches us how to evaluate models under realistic uncertainty.

### Why use hard constraints before machine learning?

Beginners often assume machine learning models should be given 100% of the raw data and allowed to figure out everything on their own. In geospatial engineering, this is a dangerous pitfall:
1. **Trivial Negatives:** If you feed the model stream pixels inside existing towns, inside lakes, or on 50° vertical cliffs, the model will achieve 99% accuracy simply by learning "don't build in a lake" or "don't build in a house". This creates an illusion of high predictive power while failing to distinguish good stream sites from mediocre stream sites.
2. **Search Space Reduction:** Civil engineers follow strict statutory and physical guidelines (such as IMSD guidelines in India — note: specific numeric thresholds are assumed expert-style judgements with literature support UNVERIFIED until primary manuals are checked). By applying hard constraints upfront, we restrict the candidate pool to sites where a check dam is legally and physically possible.
3. **Plausible Negatives:** By ensuring that negative samples (label=0) for ML training come *only* from sites that passed physical feasibility, we force the ML model to learn the genuine hydrological, geological, and terrain subtleties that make one plausible location better than another.

### How does AHP (Analytic Hierarchy Process) work in plain language?

When civil engineers or planners evaluate multiple factors (slope, soil, rainfall, distance to road), they need a way to combine them.
Instead of guessing weights directly out of thin air (e.g. "I think slope is 0.23 and rainfall is 0.14"), AHP breaks the decision down into **pairwise comparisons**:
1. You compare just **two factors at a time**: e.g., "Is slope more important than rainfall?" An engineer might say: "Yes, slope is twice as important as rainfall for check dam stability."
2. You place all these 1-on-1 judgements into an $n \times n$ grid (the pairwise matrix).
3. **The Principal Eigenvector:** A mathematical theorem (the Perron-Frobenius theorem) proves that the unique vector representing the true underlying ranking of all factors simultaneously is the principal eigenvector of this matrix.
4. **Consistency Ratio (CR):** Humans can make logically contradictory judgements (e.g. saying A > B, B > C, but C > A). The Consistency Ratio measures how much contradiction exists in your judgements compared to purely random guesses. Saaty established that if $CR < 0.10$ (less than 10% inconsistency), the judgements are mathematically consistent and trustworthy.

### Why do we tune the decision threshold on the training fold instead of the test fold?

Most beginners evaluate binary classification using a fixed 0.50 cutoff: if predicted probability $\ge 0.50$, predict 1, else 0.
However, when classes are imbalanced (e.g. only 30% of stream candidates are good check dam sites), 0.50 is rarely the optimal threshold.
- If you look at the **test fold** to pick the threshold that gives the highest test F1, you are **cheating (data leakage)**! You used the test answers to pick your decision rule.
- Instead, in rigorous machine learning:
  1. We evaluate different thresholds ($0.10, 0.11, \dots, 0.90$) strictly on the **training fold** to find which threshold maximizes training F1.
  2. We freeze that chosen threshold and apply it blindly to the **held-out test block**.
This guarantees an honest estimate of how the decision rule performs on genuinely unseen terrain.

### How does spatial cross-validation prevent geographical data leakage?

If two candidate sites are 100 metres apart along the same valley, they share almost identical elevation, rainfall, soil type, and distance to roads.
- If you use a random 80/20 train/test split, site A might be in the training set and site B in the test set. The model can get high test accuracy simply by "remembering" what the valley looked like around site A.
- **Spatial CV** solves this by cutting the map into distinct spatial blocks. When Block 3 is tested, the model is trained only on Blocks 1, 2, 4, 5, 6, 7, and 8. The model must predict suitability in an entire unfamiliar region without relying on spatial neighbor memory.

---

## Stage 2 — Study Area & Data

*(fill in after Stage 2)*

---

## Stage 3 — Terrain Processing

*(fill in after Stage 3)*

---

### What is TWI (Topographic Wetness Index)?

*Prompt: TWI = ln(A / tan β). What is A? What is β? What does a high TWI value at a location tell you about that location?*

---

### What is flow accumulation?

*Prompt: Imagine rain falling on every cell of a grid. Water flows downhill. After all the rain has flowed, the "flow accumulation" of a cell is the number of cells that drain into it. Why is this useful for finding stream channels?*

---

### What is Strahler stream order?

*Prompt: Start at the tips of a river network (tiny streams). Those are order 1. When two order-1 streams meet, what order is the new segment? Why do we limit candidates to order 1–3 for check dams?*

---

## Stage 4 — Candidates & Feature Table

*(fill in after Stage 4)*

---

## Stage 5 — Labels

*(fill in after Stage 5)*

---

### Why is the labelling problem so hard?

*Prompt: Your model learns from "existing check dams." But those were built by engineers who already evaluated the sites. What bias does this introduce? What happens if check dams were only ever built in easily accessible places?*

---

## Stage 6 — Real Modelling

*(fill in after Stage 6)*

---

## Stage 7 — Optimization

*(fill in after Stage 7)*

---

### What is a Pareto front?

*Prompt: You have two objectives: maximize water storage and minimize cost. Site A stores 10,000 m³ and costs 5 lakh. Site B stores 8,000 m³ and costs 3 lakh. Site C stores 10,000 m³ and costs 6 lakh. Which sites are on the Pareto front? Which site is dominated and why?*

---

### What is NSGA-II?

*Prompt: NSGA-II is a genetic algorithm. What does "genetic" mean here? How does it find the Pareto front without trying every possible combination of sites?*

---

## Stage 8 — Dashboard

*(fill in after Stage 8)*

---

## Stage 9 — Limitations & Honest Assessment

*(fill in after Stage 9)*
