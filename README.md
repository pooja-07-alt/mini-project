# Check Dam Site Selection — Karnataka Sub-Watershed
### ML-Based Geospatial Suitability & Multi-Objective Optimization

[![Stage](https://img.shields.io/badge/Stage-0%20Setup-blue)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## What This Project Does

This system identifies **optimal locations for check dams** in a small sub-watershed in Karnataka using:

1. **Machine Learning (Random Forest / XGBoost)** — scores every candidate stream location for suitability based on terrain, soil, rainfall, and land use features.
2. **Multi-Objective Optimization (NSGA-II)** — among high-scoring candidates, finds Pareto-optimal sites that maximize water storage while minimizing construction cost proxy and farmland submergence.
3. **Interactive Web Dashboard (Leaflet)** — displays recommended sites with explanations, hosted on GitHub Pages.

> **Scope:** This project focuses on **check dams only**. Farm ponds and percolation tanks are designated future work — the code is structured so they can be added later with minimal refactoring.

---

## Key Results

> *Will be filled in after Stage 6 (Real Modelling) and Stage 7 (Optimization).*

- Study area: TBD (sub-watershed, Karnataka)
- Number of candidate sites evaluated: TBD
- Best model AUC-ROC: TBD
- Pareto-optimal sites recommended: TBD

---

## Repository Structure

```
checkdam-siting/
├── README.md              ← You are here
├── PROGRESS.md            ← Stage tracker
├── ARCHITECTURE.md        ← System design
├── DECISIONS.md           ← Design choices and rationale
├── DATA_SOURCES.md        ← Every dataset: source, licence, link
├── LEARNING_NOTES.md      ← ML concepts explained in plain language
├── requirements.txt       ← Python dependencies
├── .gitignore
├── data/
│   ├── raw/               ← Downloaded data (NOT committed — see .gitignore)
│   ├── processed/         ← Derived rasters (NOT committed)
│   └── tables/            ← Feature tables as small CSVs (committed)
├── notebooks/             ← One Jupyter notebook per stage
├── src/                   ← Reusable Python modules
├── outputs/               ← Maps, GeoJSON, plots, metrics
├── docs/                  ← Leaflet dashboard (GitHub Pages root)
└── reports/               ← Final report and presentation
```

---

## How to Run

### 1. Clone and set up the environment
```bash
git clone https://github.com/YOUR_USERNAME/checkdam-siting.git
cd checkdam-siting
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Stage 1 warm-up (synthetic data, no downloads needed)
```bash
jupyter notebook notebooks/stage1_ml_warmup.ipynb
```

### 3. View the dashboard
Open `docs/index.html` in a browser, or visit the GitHub Pages URL (see below).

---

## Stages

| Stage | Goal | Status |
|-------|------|--------|
| 0 | Setup: repo, environment, skeleton | ✅ Done |
| 1 | ML warm-up on synthetic data | ✅ Done |
| 2 | Study area selection & data collection | ⬜ Not started |
| 3 | Terrain processing (DEM → slope, TWI, streams) | ⬜ Not started |
| 4 | Candidate pixels & feature table | ⬜ Not started |
| 5 | Labels (positives & negatives) | ⬜ Not started |
| 6 | Real modelling (AHP baseline vs RF vs XGBoost) | ⬜ Not started |
| 7 | Optimization (NSGA-II, Pareto front) | ⬜ Not started |
| 8 | Dashboard (Leaflet web app) | ⬜ Not started |
| 9 | Evaluation & report | ⬜ Not started |

---

## Tech Stack

| Area | Tool |
|------|------|
| Language | Python 3.11+ |
| GIS | rasterio, geopandas, shapely, pysheds, QGIS |
| ML | scikit-learn, xgboost, shap |
| Optimization | pymoo (NSGA-II) |
| Notebooks | Jupyter |
| Dashboard | HTML + CSS + JS + Leaflet |
| Version control | Git + GitHub, GitHub Pages |

---

## Limitations (Honest Assessment)

- Labels are *built* sites, not proven *optimal* sites — the model learns what has been built, not what is best.
- Construction cost is a **proxy** (slope × distance to road), not a real cost estimate.
- No field validation has been done.
- The study is limited to one small sub-watershed.

---

## Future Work

- Extend the pipeline to **farm ponds** and **percolation tanks** (different models, different labels).
- Incorporate real cost estimates from Karnataka DWMS.
- Add seasonal rainfall variability using CHIRPS time series.
- Field-validate the top-recommended sites.

---

*Project by: [Your Name] | [Institution] | [Year]*
