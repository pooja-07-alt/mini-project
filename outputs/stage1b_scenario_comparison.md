# Stage 1b Scenario Comparison: AHP-Aligned vs. AHP-Misaligned

> **Notice:** SYNTHETIC - pipeline test, not a real finding.
> Tests sensitivity of hybrid and ML-only approaches to misspecified expert weights.

| Scenario       | Method          | AUC-ROC (mean ± std)   | PR-AUC (mean ± std)   |   Pooled AUC-ROC |   Pooled PR-AUC |
|:---------------|:----------------|:-----------------------|:----------------------|-----------------:|----------------:|
| ahp_aligned    | AHP-only        | 0.720 ± 0.064          | 0.776 ± 0.073         |            0.716 |           0.771 |
| ahp_aligned    | RF ML-only      | 0.736 ± 0.061          | 0.782 ± 0.082         |            0.737 |           0.779 |
| ahp_aligned    | XGB ML-only     | 0.721 ± 0.049          | 0.766 ± 0.075         |            0.725 |           0.762 |
| ahp_aligned    | RF + ahp_score  | 0.754 ± 0.044          | 0.787 ± 0.072         |            0.753 |           0.785 |
| ahp_aligned    | XGB + ahp_score | 0.720 ± 0.026          | 0.752 ± 0.063         |            0.721 |           0.745 |
| ahp_misaligned | AHP-only        | 0.670 ± 0.055          | 0.785 ± 0.086         |            0.663 |           0.778 |
| ahp_misaligned | RF ML-only      | 0.732 ± 0.061          | 0.830 ± 0.081         |            0.753 |           0.835 |
| ahp_misaligned | XGB ML-only     | 0.714 ± 0.061          | 0.813 ± 0.084         |            0.734 |           0.821 |
| ahp_misaligned | RF + ahp_score  | 0.735 ± 0.045          | 0.835 ± 0.080         |            0.752 |           0.842 |
| ahp_misaligned | XGB + ahp_score | 0.721 ± 0.061          | 0.812 ± 0.086         |            0.738 |           0.817 |

## Key Finding & Implications:
- **When expert weights match reality (`ahp_aligned`):**
  - AHP-only achieves competitive baseline ranking (~0.72 AUC).
  - Adding AHP score as a feature into Random Forest provides positive inductive bias (+0.018 AUC).
- **When expert weights mischaracterize reality (`ahp_misaligned`):**
  - AHP-only performance drops substantially because rigid expert assumptions penalize viable sites.
  - Machine learning models (RF and XGBoost) adapt flexibly to empirical patterns discovered from training data, maintaining strong discrimination.
  - Adding a misspecified AHP score can dilute ML tree splits or reduce efficiency.
- **Why Stage 6 on Real Data is the Real Test:**
  - In synthetic tests, we control whether AHP is aligned or misaligned. In real Karnataka watersheds, whether expert AHP weights accurately reflect ground reality is unknown *a priori*.
  - Running this exact pipeline on real geocoded check dams in Stage 6 will empirically determine whether expert AHP priors enhance or constrain machine learning.