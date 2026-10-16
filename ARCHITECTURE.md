# ARCHITECTURE.md — System Design

> Keep this file current. Update it whenever the design changes.
> Last updated: Stage 0 (2026-10-16)

---

## Overview

This system identifies optimal check dam locations in a Karnataka sub-watershed using a three-part pipeline:

```
DATA ACQUISITION → PREPROCESSING → FEATURE ENGINEERING
      → ML SUITABILITY MODEL → NSGA-II OPTIMIZATION → DASHBOARD
```

All heavy processing (ML, optimization) runs **offline in Python**. The dashboard only reads pre-computed GeoJSON — no backend server is needed.

---

## Full Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION                                            │
│ Google Earth Engine · Bhuvan · India-WRIS · KSNDMC · OSM       │
│ (DEM, LULC, NDVI, rainfall, soil, geology, roads, dam points)  │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. GEOSPATIAL PREPROCESSING (QGIS + Python)                    │
│ Clip to study area · reproject to UTM Zone 43N                 │
│ Resample all rasters to common 30m grid                        │
│ Derive: slope, aspect, streams, flow accumulation, TWI         │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                                         │
│ Candidate stream pixels (stream order 1–3)                     │
│ Extract all layer values per pixel → one flat table            │
│ Same column schema as synthetic_checkdam_features.csv          │
│ Labels: existing dams = 1, sampled stream pixels = 0           │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. ML SUITABILITY MODEL                                        │
│ Baseline: AHP weighted score (rule-based, explainable)         │
│ Models: Random Forest, XGBoost                                 │
│ Validation: spatial block cross-validation (not random split)  │
│ Explanation: SHAP values per candidate site                    │
│ Output: suitability probability (0–1) per candidate            │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. MULTI-OBJECTIVE OPTIMIZATION (pymoo, NSGA-II)               │
│ Objectives:                                                    │
│   - Maximize estimated water storage (m³)                      │
│   - Minimize cost proxy (slope × distance to road)             │
│   - Minimize submerged cropland area (ha)                      │
│ Constraint: suitability score ≥ 0.60                           │
│ Output: Pareto-optimal sites with lat/long and objective values │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│ 6. OUTPUTS + DASHBOARD                                         │
│ GeoJSON of recommended sites (with all feature values + SHAP)  │
│ CSV of Pareto-optimal sites                                    │
│ Static maps (PNG)                                              │
│ Leaflet web dashboard (docs/index.html → GitHub Pages)         │
└────────────────────────────────────────────────────────────────┘
```

---

## Feature Schema

The feature table (both synthetic and real) uses these columns:

| Column | Type | Description |
|--------|------|-------------|
| `site_id` | int | Unique pixel/candidate identifier |
| `slope_deg` | float | Slope in degrees from DEM |
| `flow_accum` | float | Upstream contributing area (pixels) |
| `twi` | float | Topographic Wetness Index = ln(A / tan β) |
| `stream_order` | int | Strahler stream order (1–5) |
| `annual_rainfall_mm` | float | Mean annual rainfall (CHIRPS) |
| `soil_texture_class` | int | Encoded: 1=Sandy, 2=Loamy, 3=Clay |
| `ndvi` | float | NDVI from Sentinel-2 (−1 to 1) |
| `dist_to_road_m` | float | Distance to nearest road (OpenStreetMap) |
| `lulc_class` | int | Land use class (see DATA_SOURCES.md) |
| `geology_permeability` | int | Encoded: 1=Low, 2=Medium, 3=High |
| `label` | int | 1 = existing check dam, 0 = non-dam stream pixel |

---

## Coordinate System

- **Storage and processing:** UTM Zone 43N (EPSG:32643) — metric units, appropriate for Karnataka
- **Dashboard output:** WGS84 (EPSG:4326) — required for Leaflet/GeoJSON

---

## Data Flow Between Modules

```
src/preprocessing.py    ← clips, reprojects, resamples rasters
src/hydrology.py        ← derives slope, TWI, stream network
src/features.py         ← samples rasters to build the feature table
src/ml_model.py         ← AHP baseline, RF, XGBoost, spatial CV, SHAP
src/optimization.py     ← objective functions, NSGA-II wrapper
src/dashboard_export.py ← exports GeoJSON for Leaflet
```

---

## Design Principles

1. **Offline-first:** all ML and optimization runs locally; dashboard is static HTML.
2. **Modular:** each `src/` module has one responsibility; notebooks call them.
3. **Reproducible:** all random seeds are fixed; all file paths are relative to project root.
4. **Extensible:** `ml_model.py` accepts a `structure_type` parameter — future farm pond or percolation tank models slot in without rewriting the pipeline.
5. **No invented data:** all values in the synthetic CSV are generated from documented distributions based on real Karnataka hydrology literature.

---

## Future Extension Points

| Future feature | Where to add it |
|----------------|----------------|
| Farm pond model | New `FarmPondModel` class in `src/ml_model.py` |
| Percolation tank model | New `PercolationTankModel` class in `src/ml_model.py` |
| Google Earth Engine streaming | New `src/gee_fetch.py` module |
| Real cost estimation | Replace proxy in `src/optimization.py` |
| Field validation upload | New endpoint in a future Flask/FastAPI backend |
