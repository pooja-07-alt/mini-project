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
