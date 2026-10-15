# Cardiovascular ML Workstream — Retraining Results

## Validation status

Nested repeated stratified 5-fold CV (5 outer repeats; 5 inner folds for model selection). Performance values below are held-out outer-fold estimates; model selection was performed only inside each outer-training fold.

## Data audit

- Workbook shape: 303 rows × 59 columns.
- Exact duplicate rows: 0.
- Leakage columns excluded from every model: Cath, LAD, LCX, RCA.
- Constant features dropped: Exertional CP.

## Preliminary CV results

| Target | Most-selected model | ROC-AUC (mean) | Brier (mean) |
|---|---|---:|---:|
| CAD | logistic_regression | 0.919 | 0.104 |
| LAD | random_forest | 0.848 | 0.158 |
| LCX | random_forest | 0.725 | 0.208 |
| RCA | logistic_regression | 0.704 | 0.208 |

PRD ROC-AUC targets are CAD >= 0.85, LAD >= 0.80, LCX >= 0.75, RCA >= 0.75. Nested CV satisfies the internal resampling requirement, but it does not substitute for external validation.

## Remaining limits

- Small single-dataset sample (303 records); estimates are uncertain.
- No external validation dataset, subgroup/fairness analysis, decision-curve analysis, or prospective evaluation was available.
- Sigmoid calibration is fitted for the delivered full-data models; calibration has no independent external assessment.
- This is an educational prototype, not a clinical diagnostic tool.
