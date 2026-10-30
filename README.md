# 3D Cardiovascular Risk Explorer

Live demo: https://3d-cardiovascular-risk-explorer-2onthy24j-pgd4.vercel.app

An educational web app that estimates coronary-stenosis likelihood for CAD, LAD, LCX and RCA from 54 clinical features, shows the three vessels on an anatomical 3D heart, and explains each model output with SHAP contributions. It is not a diagnostic device and must not be used for clinical decision-making.

## Results

Validated with nested repeated stratified 5-fold cross-validation on 303 UCI Z-Alizadeh Sani records (5 outer repeats, 25 outer folds):

| Target | Model | ROC-AUC |
| --- | --- | ---: |
| CAD | Logistic regression | 0.919 |
| LAD | Random forest | 0.848 |
| LCX | Random forest | 0.725 |
| RCA | Logistic regression | 0.704 |

CAD and LAD meet the project's internal targets (0.85 / 0.80). LCX and RCA do not meet their 0.75 target, and are retained to report the full result rather than selectively presenting models. These are internal resampling estimates only: there is no external, temporal, or prospective validation.

## Run locally

Use Python 3.12 or 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python -m uvicorn api.app:app --port 8000
```

Open http://localhost:8000. The training script regenerates model artifacts and reports; the committed model binaries (about 5 MB) are enough to run the app and tests without retraining.

## Test and continuous integration

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

The committed model binaries are enough to run the whole suite without a
training run; `python src/train.py` regenerates them after pipeline changes.

GitHub Actions runs the test suite on Python 3.12 and 3.13 for pushes and pull requests.

## Files

- `MODEL_CARD.md` — intended use, validation design, outputs, and known limitations.
- `api/app.py` — FastAPI service for local use; `app/main.py` is the Vercel entrypoint.
- `src/train.py` — training, nested cross-validation, calibration and reports.
- `src/predict.py` — prediction with fold-ensemble ranges.
- `src/shap_explain.py` — patient-level SHAP explanations.
- `web/` — Three.js viewer, patient form and risk cards.
- `tests/` — data and pipeline regression checks.
- `reports/initial_results.md` — validation metrics and limits.

The API is stateless and stores no patient data. The operating threshold is fixed at 0.35 for all four targets as a recall-oriented exploratory setting; it is not a clinically validated decision threshold.

## Data and licences

- Dataset: UCI Z-Alizadeh Sani extension, 303 rows × 59 columns, in `data/raw/dataset.csv` (CC BY 4.0). `Cath`, `LAD`, `LCX` and `RCA` are targets, not inputs. Constant feature `Exertional CP` is dropped, leaving 54 inputs.
- Heart mesh: BodyParts3D 4.3i in `web/*.json`, CC BY-SA 2.1 JP — see `assets/ASSETS.md` for the source, vessel mapping and conversion script.
- Code: MIT (see `LICENSE`).

## Limits

Small, single-source data make estimates uncertain. There is no external validation, independent calibration assessment, subgroup/fairness analysis, decision-curve analysis, or prospective evaluation. CAD and vessel models are trained independently, so they can disagree; the app flags material discrepancies. SHAP contributions describe model behaviour, not causation. This is an educational prototype, not a diagnosis and not a substitute for angiography or clinical judgement.
