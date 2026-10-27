# 3D Cardiovascular Risk Explorer

Live demo: https://3d-cardiovascular-risk-explorer-cron5njja-pgd4.vercel.app

## Quick start
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# 1. download the dataset (link in Files below) into data/raw/
python src/train.py
python src/shap_explain.py
python -m uvicorn api.app:app --port 8000
```
Training takes several minutes depending on hardware. It creates four calibrated model artifacts and nested-CV reports. Open `http://localhost:8000` after starting the service for the interactive prototype.
The bundled model artifacts were trained on the uploaded dataset in the preparation environment.

## Files
- `data/raw/`: place `z_alizadeh_sani_extension.xlsx` here, downloaded from the UCI Machine Learning Repository: http://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset (CC BY 4.0; the workbook is used unchanged). `data/raw/dataset.csv` is the same data in CSV form, used by the deployed entrypoint.
- `models/*.joblib`: calibrated model artifacts. Not vendored in git (large binaries); regenerated deterministically with fixed seeds by `python src/train.py`. `src/predict.py` also loads LAD/LCX artifacts stored split into `models/<name>_model.joblib/chunk_*` files, as used on the deployment.
- `src/train.py`: label mapping, leakage exclusions, preprocessing, CV model comparison, calibration and artifacts.
- `src/predict.py`: backend-facing prediction helper.
- `src/explain.py`: exploratory permutation-importance generation.
- `src/shap_explain.py`: model-specific SHAP summaries (LinearExplainer for logistic models, TreeExplainer for random forests) and patient-level SHAP helpers.
- `api/app.py`: stateless FastAPI API with `/schema`, `/examples`, `/predict`, `/metrics`, and `/health`, for local use with `uvicorn`.
- `app/main.py`: the same API adapted as the serverless entrypoint for the Vercel deployment (reads `data/raw/dataset.csv` instead of the workbook; `.python-version` is 3.12).
- `web/`: Three.js Foundation-phase viewer, patient form, risk cards, and persistent safety banner.
- `config/vessels.json`: stable API-to-viewer target mapping.
- `assets/ASSETS.md`: asset and licensing record.
- `reports/initial_results.md`: results and caveats.
- `reports/cv_model_comparison.csv`: CV metrics and standard deviations.
- `reports/feature_importance_*.csv`: exploratory in-sample permutation importance summaries (Brier-score increase when a feature is shuffled).
- `tests/test_data.py`: basic data and leakage checks.

## Target mapping
`Cath`: `CAD`=1, `Normal`=0. `LAD`, `LCX`, `RCA`: `Stenotic`=1, `Normal`=0.
Input features exclude `Cath`, `LAD`, `LCX`, and `RCA` as required by the PRD. Constant feature `Exertional CP` is dropped.

## Caveats
The regenerated metrics use nested repeated stratified 5-fold CV (five outer repeats and five inner folds for model selection). This meets the PRD's internal resampling requirement, but not external validation, prospective evaluation, or independent external calibration assessment. The regenerated reports contain the current values and PRD comparison.
Predictions are not medical diagnoses and must not be used for clinical decisions.

## Commands
```bash
make train
make run
make test
```
On Windows without `make`, run the Python commands shown above. Install the pinned dependencies first with `pip install -r requirements.txt`.

## Current implementation status
The API and browser viewer are implemented: entering or loading a dataset patient calls `POST /predict`, updates four risk cards, and colors selectable LAD, LCX, and RCA tube geometry in a lightweight 3D scene. The API is stateless and has no patient storage. Model-specific SHAP contributions (`POST /explain/{target}`), calibrated fold-ensemble uncertainty ranges, documented recall-favoring thresholds (fixed 0.35), and quality charts (ROC, PR, calibration, confusion matrix) are all implemented and served from `reports/quality_*.json`.

Remaining before release: browser end-to-end and GPU-off performance tests on a local machine, clinician-curated reference ranges, an open-licensed anatomical heart mesh (current geometry is procedural), and the demo video. Hosting is done: see the live demo link above.

## License
Code in this repository is MIT licensed (see `LICENSE`). The UCI Z-Alizadeh Sani dataset is CC BY 4.0 and covers only the data.
