# 3D Cardiovascular Risk Explorer

Live demo: https://3d-cardiovascular-risk-explorer-2onthy24j-pgd4.vercel.app

A web app that predicts coronary stenosis risk for CAD, LAD, LCX and RCA from 54 clinical features, shows the three vessels on an anatomical 3D heart, and explains each prediction with SHAP contributions.

## Results

Validated with nested repeated stratified 5-fold cross-validation on 303 UCI Z-Alizadeh Sani records (5 outer repeats, 25 outer folds):

| Target | Model | ROC-AUC |
| --- | --- | --- |
| CAD | Logistic regression | 0.919 |
| LAD | Random forest | 0.848 |
| LCX | Random forest | 0.725 |
| RCA | Logistic regression | 0.704 |

CAD and LAD clear their targets (0.85 / 0.80). LCX and RCA do not clear theirs (0.75) — that is reported, not hidden. There is no external validation, so these are internal estimates only.

## Run it locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python -m uvicorn api.app:app --port 8000
```

Then open http://localhost:8000. `train.py` regenerates the model artifacts (they are not committed to git, because they are large binaries) and the reports in `reports/`.

`make train`, `make run` and `make test` do the same three steps.

## Files

- `api/app.py` — FastAPI service for local use; `app/main.py` is the same API as the Vercel entrypoint.
- `src/train.py` — training, nested cross-validation, calibration and reports.
- `src/predict.py` — prediction with fold-ensemble ranges.
- `src/shap_explain.py` — patient-level SHAP explanations.
- `src/explain.py` — permutation importance summaries.
- `web/` — Three.js viewer, patient form and risk cards.
- `config/features.yaml` — display labels and groups for the 54 inputs.
- `reports/initial_results.md` — the validation numbers and their limits.
- `reports/quality_*.json` — ROC, precision-recall, calibration and confusion-matrix data for the Model quality tab.

The API is stateless and stores no patient data. The operating threshold is fixed at 0.35 for all four targets, chosen to favour recall.

## Data and licences

- Dataset: UCI Z-Alizadeh Sani extension, 303 rows × 59 columns, already in `data/raw/dataset.csv` (CC BY 4.0). `Cath`, `LAD`, `LCX` and `RCA` are targets, not inputs. Constant feature `Exertional CP` is dropped, leaving 54 inputs.
- Heart mesh: BodyParts3D 4.3i in `web/*.json`, CC BY-SA 2.1 JP — see `assets/ASSETS.md` for the source, the vessel mapping and the conversion script. That licence covers the mesh only.
- Code: MIT (see `LICENSE`).

## Limits

Small, single-dataset sample, so estimates are uncertain. No external or prospective validation, and no independent calibration assessment. This is an educational prototype, not a diagnosis and not a substitute for angiography or clinical judgment.
