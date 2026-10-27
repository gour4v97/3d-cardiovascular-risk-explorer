# Model Card — 3D Cardiovascular Risk Explorer

## Intended use

This is an educational, decision-support prototype for exploring model-estimated likelihood of coronary artery stenosis from structured clinical features. It is **not** a diagnostic device and must not be used to guide individual care, replace angiography, or replace clinical judgement.

## Data

- **Source:** UCI Z-Alizadeh Sani extension
- **Sample:** 303 records and 59 columns
- **Inputs:** 54 features after excluding outcome variables and one constant feature
- **Targets:** CAD (Cath), LAD, LCX and RCA stenosis
- **Data scope:** one dataset only; no external, temporal, or prospective validation

## Training and validation

Candidate logistic-regression and random-forest pipelines include median/mode imputation, scaling for numeric variables, and one-hot encoding for categorical variables. Target columns are excluded before fitting.

Performance is estimated using nested repeated stratified 5-fold cross-validation: five outer repeats and five inner folds. The inner folds select the candidate model; the outer folds estimate performance. Sigmoid calibration is performed inside each outer training set for validation and with five-fold calibration for delivered models.

| Target | Selected model | Mean outer-fold ROC-AUC |
| --- | --- | ---: |
| CAD | Logistic regression | 0.919 |
| LAD | Random forest | 0.848 |
| LCX | Random forest | 0.725 |
| RCA | Logistic regression | 0.704 |

The LCX and RCA models did not meet the project's 0.75 ROC-AUC target and should not be represented as high-performing models.

## Outputs and explanations

The application returns calibrated probabilities, a range across retained calibration-fold models, and SHAP feature contributions. SHAP values describe the fitted model's raw score before sigmoid calibration; they are not causal explanations or clinical advice.

CAD and vessel models are trained independently. A large difference between direct CAD and the largest vessel score is surfaced as a consistency flag; it is not resolved into a diagnosis.

## Known limitations

- Small, single-source sample; results may not generalise.
- No external validation, prospective evaluation, subgroup/fairness analysis, or decision-curve analysis.
- Calibration has no independent external assessment.
- Repeated outer-fold predictions support internal evaluation but are not independent patient observations.
- Threshold 0.35 is a predeclared recall-oriented operating point, not a clinically validated decision threshold.

## Reproducibility

Dependencies are version-pinned in requirements.txt; automated data/pipeline checks run through GitHub Actions. Retrain with python src/train.py.
