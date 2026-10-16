This directory is the GitHub Pages root.

Structure:
  index.html                  ← Leaflet dashboard (entry point)
  data/
    recommended_sites.geojson ← Pareto-optimal sites (written by dashboard_export.py)
    study_area_boundary.geojson ← Sub-watershed boundary (Stage 2)

To test locally:
  python -m http.server 8080
  Then open: http://localhost:8080/docs/
