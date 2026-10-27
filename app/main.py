from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from yaml import safe_load
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_patient_with_ranges

DISCLAIMER = "Decision support and education only. Not a diagnosis. Not a substitute for angiography or clinical judgment."
TARGETS = ("CAD", "LAD", "LCX", "RCA")
FORBIDDEN = {"Cath", "LAD", "LCX", "RCA"}
GROUPS = {
    "Demographics": {"Age", "Weight", "Length", "Sex", "BMI"},
    "History and symptoms": {"DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang"},
    "Examination and ECG": {"BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"},
    "Labs and echo": {"FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE", "Region RWMA", "VHD"},
}

class PredictionRequest(BaseModel):
    features: dict[str, Any]
    include_explanations: bool = False
    explanation_target: str | None = None

def band(probability: float) -> str:
    if probability < 0.30:
        return "Lower predicted likelihood"
    if probability <= 0.60:
        return "Intermediate predicted likelihood"
    return "Higher predicted likelihood"

@lru_cache
def dataset() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/dataset.csv")

@lru_cache
def feature_columns() -> list[str]:
    artifact = joblib.load(ROOT / "models/cad_model.joblib")
    return artifact["feature_columns"]

@lru_cache
def feature_specs() -> dict[str, dict[str, Any]]:
    specs=safe_load((ROOT / "config/features.yaml").read_text())["features"]
    missing=set(feature_columns())-set(specs)
    unexpected=set(specs)-set(feature_columns())
    if missing or unexpected:
        raise ValueError(f"features.yaml mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return specs

@lru_cache
def thresholds() -> dict[str, float]:
    return json.loads((ROOT / "reports/thresholds.json").read_text())["thresholds"]

def group_for(column: str) -> str:
    return next((group for group, columns in GROUPS.items() if column in columns), "Other")

def explanation_for(target: str, features: dict[str, Any]) -> dict[str, Any]:
    # SHAP imports are intentionally lazy: prediction-only requests should start quickly.
    from shap_explain import explain_features
    return explain_features(target, features)

app = FastAPI(title="Cardiovascular Risk Explorer API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/health")
def health() -> dict[str, Any]:
    card = json.loads((ROOT / "models/model_card.json").read_text())
    return {"status": "ok", "model_card": card, "disclaimer": DISCLAIMER}

@app.get("/schema")
def schema() -> dict[str, Any]:
    df = dataset()
    fields = []
    for column in feature_columns():
        values = df[column].dropna()
        numeric = pd.api.types.is_numeric_dtype(df[column])
        spec=feature_specs()[column]
        item: dict[str, Any] = {"name": column, "label": spec["label"], "group": spec["group"], "unit": spec["unit"], "reference_status": spec["reference_status"], "required": True, "type": "number" if numeric else "category"}
        if numeric:
            item.update({"min": float(values.min()), "max": float(values.max()), "reference": None})
        else:
            item["choices"] = sorted(str(value) for value in values.unique())
        fields.append(item)
    return {"features": fields, "targets": list(TARGETS), "vessel_map": {"LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}, "disclaimer": DISCLAIMER}

@app.get("/examples")
def examples() -> dict[str, Any]:
    df = dataset()
    columns = feature_columns()
    rows = []
    for index in (0, 1, 2):
        row = df.iloc[index]
        rows.append({"id": int(index), "label": f"Dataset row {index + 1}", "features": {column: row[column].item() if hasattr(row[column], "item") else row[column] for column in columns}})
    return {"examples": rows}

@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "nested_cv": pd.read_csv(ROOT / "reports/nested_cv_results.csv").to_dict(orient="records"),
        "model_comparison": pd.read_csv(ROOT / "reports/cv_model_comparison.csv").to_dict(orient="records"),
        "quality": {target: json.loads((ROOT / f"reports/quality_{target.lower()}.json").read_text()) for target in TARGETS},
        "validation_note": "Nested repeated stratified 5-fold CV: 5 outer repeats and 5 inner folds.",
    }

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    expected = set(feature_columns())
    missing = sorted(expected - set(request.features))
    unexpected = sorted(set(request.features) - expected)
    if missing or unexpected:
        raise HTTPException(status_code=422, detail={"missing": missing, "unexpected": unexpected})
    try:
        probabilities = predict_patient_with_ranges(request.features)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    results = {target: {**result, "band": band(result["probability"]), "threshold": thresholds()[target]} for target, result in probabilities.items()}
    direct_cad = results["CAD"]["probability"]
    any_vessel = max(results[vessel]["probability"] for vessel in ("LAD", "LCX", "RCA"))
    if request.include_explanations and request.explanation_target not in TARGETS:
        raise HTTPException(status_code=422, detail="explanation_target must be CAD, LAD, LCX, or RCA when explanations are requested")
    explanations={request.explanation_target:explanation_for(request.explanation_target,request.features)} if request.include_explanations else None
    return {"targets": results, "consistency": {"direct_cad": direct_cad, "max_vessel_probability": any_vessel, "flag": abs(direct_cad - any_vessel) >= 0.30}, "disclaimer": DISCLAIMER, "explanations": explanations, "explanations_available": request.include_explanations}

@app.post("/explain/{target}")
def explain(target: str, request: PredictionRequest) -> dict[str, Any]:
    if target not in TARGETS:
        raise HTTPException(status_code=404, detail="Unknown target")
    missing=sorted(set(feature_columns())-set(request.features))
    if missing:
        raise HTTPException(status_code=422, detail={"missing":missing})
    return {"explanation":explanation_for(target,request.features),"disclaimer":DISCLAIMER}

app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
