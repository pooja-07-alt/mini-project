# Stage 1b Method Comparison Table

> **Note:** Evaluated across 8 leave-one-block-out spatial folds using a standardized threshold rule:
> in each fold, the top p% of sites by score are flagged, where p is the positive prevalence of the training folds.
> **Notice:** SYNTHETIC - pipeline test, not a real finding.

| Method          | AUC-ROC (mean ± std)   | PR-AUC (mean ± std)   | Precision (mean ± std)   | Recall (mean ± std)   | F1 (mean ± std)   |   Pooled AUC-ROC |   Pooled PR-AUC |   Pooled Precision |   Pooled Recall |   Pooled F1 |
|:----------------|:-----------------------|:----------------------|:-------------------------|:----------------------|:------------------|-----------------:|----------------:|-------------------:|----------------:|------------:|
| AHP-only        | 0.720 ± 0.064          | 0.776 ± 0.073         | 0.712 ± 0.119            | 0.711 ± 0.075         | 0.706 ± 0.080     |            0.716 |           0.771 |              0.707 |           0.707 |       0.707 |
| RF ML-only      | 0.736 ± 0.061          | 0.783 ± 0.082         | 0.720 ± 0.096            | 0.721 ± 0.051         | 0.715 ± 0.039     |            0.737 |           0.779 |              0.716 |           0.716 |       0.716 |
| XGB ML-only     | 0.721 ± 0.049          | 0.766 ± 0.075         | 0.723 ± 0.105            | 0.722 ± 0.045         | 0.718 ± 0.051     |            0.725 |           0.762 |              0.719 |           0.719 |       0.719 |
| RF + ahp_score  | 0.754 ± 0.044          | 0.788 ± 0.072         | 0.735 ± 0.103            | 0.735 ± 0.037         | 0.730 ± 0.043     |            0.753 |           0.784 |              0.731 |           0.731 |       0.731 |
| XGB + ahp_score | 0.720 ± 0.026          | 0.751 ± 0.062         | 0.743 ± 0.100            | 0.743 ± 0.041         | 0.738 ± 0.043     |            0.721 |           0.745 |              0.74  |           0.74  |       0.74  |

### Paired Comparison (ML-Only vs ML + ahp_score):

- **Random Forest:** Mean $\Delta$AUC = `+0.0175`, improved in `6 / 8` folds.
- **XGBoost:** Mean $\Delta$AUC = `-0.0017`, improved in `2 / 8` folds.
