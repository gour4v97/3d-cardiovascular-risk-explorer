"""Model-specific SHAP helpers for API explanations and reproducible summaries.

SHAP is calculated on the selected base model before sigmoid calibration. The API labels
this clearly because calibrated probabilities and raw-model SHAP values differ by design.
"""
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import shap

ROOT=Path(__file__).resolve().parents[1]
TARGETS=("CAD","LAD","LCX","RCA")

@lru_cache
def load_artifact(target: str) -> dict:
    # Reuse predict's chunk-aware loader: lad/lcx artifacts are stored split.
    from predict import _load_artifact
    return _load_artifact(target.lower())

def _pipeline(artifact: dict):
    return artifact["estimator"].calibrated_classifiers_[0].estimator

def _source_feature(transformed_name: str, original_features: list[str]) -> str:
    if transformed_name in original_features:
        return transformed_name
    return next((feature for feature in original_features if transformed_name.startswith(feature+"_")), transformed_name)

@lru_cache
def _bundle(target: str):
    artifact=load_artifact(target); pipeline=_pipeline(artifact)
    model=pipeline.named_steps["model"]
    names=list(pipeline.named_steps["prep"].get_feature_names_out())
    if hasattr(model,"coef_"):
        data=pd.read_csv(ROOT/"data/raw/dataset.csv")[artifact["feature_columns"]].sample(20,random_state=42)
        background=pipeline.named_steps["prep"].transform(data)
        # Exact interventional LinearExplainer values: beta * (x - E[background]).
        # This avoids expensive correlation estimation for the live single-patient path.
        explainer=("linear",np.asarray(background).mean(axis=0))
    else:
        explainer=shap.TreeExplainer(model)
    return artifact,pipeline,names,explainer

def _shap_values(explainer, transformed: np.ndarray) -> np.ndarray:
    if isinstance(explainer,tuple) and explainer[0]=="linear":
        # The model coefficients are attached by _bundle's caller through the tuple.
        raise RuntimeError("Linear SHAP needs model coefficients")
    values=explainer(transformed).values
    if values.ndim==3:
        values=values[:,:,1]
    return np.asarray(values)

def warm_explainers() -> None:
    for target in TARGETS:
        _bundle(target)

def explain_features(target: str, features: dict) -> dict:
    artifact,pipeline,names,explainer=_bundle(target)
    columns=artifact["feature_columns"]
    row=pd.DataFrame([{column:features[column] for column in columns}])
    transformed=pipeline.named_steps["prep"].transform(row)
    if isinstance(explainer,tuple):
        values=(transformed-explainer[1])*pipeline.named_steps["model"].coef_[0]
    else:
        values=_shap_values(explainer,transformed)
    values=values[0]
    grouped={column:0.0 for column in columns}
    for name,value in zip(names,values):
        grouped[_source_feature(name,columns)]=grouped.get(_source_feature(name,columns),0.0)+float(value)
    contributions=[{"feature":column,"value":str(features[column]),"contribution":float(value)} for column,value in grouped.items()]
    contributions.sort(key=lambda item:abs(item["contribution"]),reverse=True)
    positive=sorted((item for item in contributions if item["contribution"]>0),key=lambda item:item["contribution"],reverse=True)[:5]
    negative=sorted((item for item in contributions if item["contribution"]<0),key=lambda item:item["contribution"])[:5]
    return {"target":target,"method":"LinearExplainer" if artifact["model_name"]=="logistic_regression" else "TreeExplainer","scale":"raw model output before sigmoid calibration","top_positive":positive,"top_negative":negative,"contributions":contributions}

def write_global_summaries() -> None:
    df=pd.read_csv(ROOT/"data/raw/dataset.csv")
    reports=ROOT/"reports"
    for target in TARGETS:
        artifact,pipeline,names,explainer=_bundle(target); columns=artifact["feature_columns"]
        sample=df[columns].sample(min(100,len(df)),random_state=42)
        transformed=pipeline.named_steps["prep"].transform(sample)
        if isinstance(explainer,tuple):
            values=(transformed-explainer[1])*pipeline.named_steps["model"].coef_[0]
        else:
            values=_shap_values(explainer,transformed)
        grouped={column:np.zeros(values.shape[0]) for column in columns}
        for index,name in enumerate(names):
            grouped[_source_feature(name,columns)]+=values[:,index]
        pd.DataFrame({"feature":list(grouped),"mean_abs_shap":[float(np.mean(np.abs(v))) for v in grouped.values()],"mean_signed_shap":[float(np.mean(v)) for v in grouped.values()]}).sort_values("mean_abs_shap",ascending=False).to_csv(reports/f"shap_{target.lower()}.csv",index=False)

if __name__=="__main__":
    write_global_summaries()
    print("Saved grouped SHAP summaries for CAD, LAD, LCX, and RCA.")
