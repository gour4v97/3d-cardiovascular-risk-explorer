from pathlib import Path
import io
import joblib,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def _load_artifact(target:str):
    """Load a model artifact, reassembling from chunks if stored split."""
    base=ROOT/"models"/f"{target}_model.joblib"
    if base.is_file():
        return joblib.load(base)
    if base.is_dir():
        buf=io.BytesIO(b"".join(p.read_bytes() for p in sorted(base.glob("chunk_*"))))
        return joblib.load(buf)
    raise FileNotFoundError(f"model artifact missing: {base}")
def predict_patient(features:dict)->dict:
    results={}
    for target in ["cad","lad","lcx","rca"]:
        a=_load_artifact(target)
        missing=sorted(set(a["feature_columns"])-set(features))
        if missing: raise ValueError(f"Missing input features: {missing}")
        row=pd.DataFrame([{c:features[c] for c in a["feature_columns"]}])
        results[target.upper()]=float(a["estimator"].predict_proba(row)[0,1])
    return results

def predict_patient_with_ranges(features:dict)->dict:
    """Return calibrated probability and the spread across retained calibration-fold models."""
    results={}
    for target in ["cad","lad","lcx","rca"]:
        a=_load_artifact(target)
        missing=sorted(set(a["feature_columns"])-set(features))
        if missing: raise ValueError(f"Missing input features: {missing}")
        row=pd.DataFrame([{c:features[c] for c in a["feature_columns"]}])
        ensemble=a["estimator"]
        probability=float(ensemble.predict_proba(row)[0,1])
        fold_probabilities=[float(member.predict_proba(row)[0,1]) for member in ensemble.calibrated_classifiers_]
        results[target.upper()]={"probability":probability,"range":[min(fold_probabilities),max(fold_probabilities)],"fold_count":len(fold_probabilities)}
    return results
