# 3D Cardiovascular Risk Explorer - Project Handoff

## Status at handoff

Foundation and the main Core-product ML/API work are implemented. This is an educational decision-support prototype, not a medical device or diagnostic system.

## What was completed

### Data and leakage control

- Preserved the original `data/raw/z_alizadeh_sani_extension.xlsx` workbook unchanged.
- Audited the 303-row, 59-column source dataset.
- Mapped `Cath` to CAD and `LAD`, `LCX`, and `RCA` to binary stenosis targets.
- Excluded `Cath`, `LAD`, `LCX`, and `RCA` from every model input.
- Dropped the constant `Exertional CP` feature, leaving 54 inputs.
- Kept preprocessing, imputation, scaling, and one-hot encoding inside training folds.

### Modeling and validation

- Trained separate CAD, LAD, LCX, and RCA models.
- Compared regularized logistic regression and random forest candidates.
- Used nested repeated stratified 5-fold validation: five outer repeats (25 held-out outer folds) and five inner folds for selection.
- Applied sigmoid calibration inside each outer-fold training set for validation, and saved a five-fold calibrated ensemble for each delivered model.
- Added held-out ROC, PR, calibration-curve, confusion-matrix, Brier, accuracy, precision, recall, and F1 report data.
- Added a documented fixed 0.35 recall-favoring operating threshold for each target. It was not tuned on held-out outer predictions.
- Saved calibrated fold-member probability ranges for each live prediction.

### Final held-out nested-CV results

| Target | Selected model | ROC-AUC mean | Brier mean | PR-AUC |
|---|---|---:|---:|---:|
| CAD | Logistic regression | 0.919 | 0.104 | See `reports/quality_cad.json` |
| LAD | Random forest | 0.848 | 0.158 | See `reports/quality_lad.json` |
| LCX | Random forest | 0.725 | 0.208 | See `reports/quality_lcx.json` |
| RCA | Logistic regression | 0.704 | 0.208 | See `reports/quality_rca.json` |

CAD and LAD meet the PRD ROC-AUC targets. LCX and RCA do not meet their 0.75 targets; this is documented rather than hidden.

### Explainability

- Added grouped, model-specific SHAP summaries in `reports/shap_*.csv`.
- Implemented `LinearExplainer`-equivalent independent-feature linear SHAP values for logistic models and `TreeExplainer` for random forests.
- The explanation API returns top five positive and negative drivers, all grouped feature contributions, the patient value, and a note that SHAP is on the raw pre-calibration model score.
- Explanations are requested on demand for the selected vessel so the probability update is not blocked by four simultaneous explanation calculations.

### API and prototype viewer

- Implemented FastAPI endpoints: `GET /health`, `GET /schema`, `GET /examples`, `GET /metrics`, `POST /predict`, and `POST /explain/{target}`.
- `POST /predict` returns four calibrated probabilities, likelihood bands, operating thresholds, retained-fold ranges, and a direct-CAD versus vessel-risk consistency flag.
- Added a browser prototype with patient form, example patient picker, persistent safety banner, risk cards, selectable 3D LAD/LCX/RCA tubes, orbit/zoom controls, probability coloring, uncertainty display, and selected-vessel SHAP driver display.
- Added a Model quality tab that reads the nested-CV report artifacts and renders per-target ROC, precision-recall, calibration curves, confusion matrix, ROC-AUC, PR-AUC, Brier score, and operating threshold.
- Added what-if comparison: submitting edited inputs shows the probability-point change from the preceding result on every risk card.
- Added a selected-vessel signed-SHAP measurement table covering all 54 inputs with the feature name, entered patient value, source-dataset range or category state, reference-review status, and positive/negative model contribution.
- Added Dockerfile, Makefile, vessel mapping, and an asset/license record. The current heart is procedural geometry; no unlicensed third-party mesh is included.
- Added `config/features.yaml` covering every model input with a display label, PRD form group, unit where known, and reference-status field. Dataset min/max is supplied by the API; clinical reference ranges are intentionally marked as pending clinician review.

## Key files

- `src/train.py` - nested validation, calibration, model artifacts, metrics, quality reports.
- `src/predict.py` - calibrated probability and fold-range helper.
- `src/shap_explain.py` - global and patient-specific SHAP helpers.
- `models/` - four model files, registry, and model card.
- `reports/nested_cv_results.csv` - held-out nested-CV metrics.
- `reports/quality_*.json` - ROC, PR, calibration, confusion matrix, threshold, and operating metrics.
- `api/app.py` - stateless HTTP service.
- `web/` - interactive frontend prototype.
- `tests/` - leakage/data and API-contract tests.

## Verification performed

- Dataset and leakage tests: 3 passed.
- Direct API-function verification: schema/example/prediction/SHAP/metrics path returns all four targets, a CAD fold range, a LinearExplainer explanation, and four quality reports.
- Python compilation of API and source modules passed.

The FastAPI in-process test client hung in this Windows sandbox after the richer calibrated ensemble was added, and local loopback connections are blocked by the sandbox. This is an environment limitation, not a claim that browser or network latency has been fully validated. Run the supplied API tests and browser checks on a normal local machine before demo or deployment.

## How to run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py
python src/shap_explain.py
python -m uvicorn api.app:app --reload --port 8000
```

Then open `http://localhost:8000`.

## Remaining PRD work

1. Have a clinician curate plausible and reference ranges for the 54-feature metadata file; do not substitute dataset min/max for a clinical reference range.
2. Have a clinician curate the reference ranges currently marked as pending review in the full 54-feature measurement table.
4. Add an open-licensed anatomical heart mesh (or retain the documented procedural fallback), record attribution in `assets/ASSETS.md`, and complete browser/GPU-off performance testing.
5. Add browser end-to-end tests and validate API latency on an unrestricted local machine.
6. Produce the six-page documentation and 3-to-10-minute demo video.
7. Deploy only after confirming licensing, public-code requirements, and hosting requirements.
8. Obtain a separate external validation dataset if external validation is required. It cannot be honestly created from the supplied 303-record dataset.

## Safety and limits

- Results are decision support and education only, not a diagnosis or a substitute for angiography or clinical judgment.
- The dataset is small and from a single triage-type population, not general screening.
- There is no external or prospective validation.
- The project predicts vessel-level likelihood only; it does not infer lesion position or use medical images.
