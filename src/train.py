from pathlib import Path
import json, numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/raw/dataset.csv"
OUT=ROOT/"models"; REPORTS=ROOT/"reports"
OUT.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
TARGETS={"CAD":"Cath","LAD":"LAD","LCX":"LCX","RCA":"RCA"}
FORBIDDEN={"Cath","LAD","LCX","RCA"}
SEED=42
VALIDATION_NOTE="Nested repeated stratified 5-fold CV (5 outer repeats; 5 inner folds for model selection)"
OPERATING_THRESHOLDS={"CAD":0.35,"LAD":0.35,"LCX":0.35,"RCA":0.35}

def load_data():
    df=pd.read_csv(DATA)
    mappings={"Cath":{"CAD":1,"Normal":0},"LAD":{"Stenotic":1,"Normal":0},
              "LCX":{"Stenotic":1,"Normal":0},"RCA":{"Stenotic":1,"Normal":0}}
    for c,m in mappings.items():
        unknown=set(df[c].dropna().unique())-set(m)
        if unknown: raise ValueError(f"Unexpected labels in {c}: {unknown}")
        df[c]=df[c].map(m)
    return df

def preprocessor(X):
    nums=X.select_dtypes(include=["number","bool"]).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    return ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)
    ],verbose_feature_names_out=False)

def candidates():
    return {
      "logistic_regression":LogisticRegression(max_iter=3000,class_weight="balanced"),
      "random_forest":RandomForestClassifier(n_estimators=80,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1),
    }

SCORING={"roc_auc":"roc_auc","accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","neg_brier":"neg_brier_score"}

def nested_evaluate(X, y):
    """Evaluate selection and testing in separate folds; return comparison and unbiased outer-fold estimates."""
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    inner_scores={name:[] for name in candidates()}
    outer_scores=[]; selections=[]; all_truth=[]; all_probabilities=[]
    for fold,(train_idx,test_idx) in enumerate(outer.split(X,y),start=1):
        X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
        y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
        inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+fold)
        best=None
        for name,est in candidates().items():
            pipe=Pipeline([("prep",preprocessor(X)),("model",est)])
            scores=cross_validate(pipe,X_train,y_train,cv=inner,scoring=SCORING,n_jobs=1,error_score="raise")
            summary={}
            for metric in SCORING:
                values=scores["test_"+metric]
                summary[metric]=float(np.mean(-values if metric=="neg_brier" else values))
            inner_scores[name].append(summary)
            choice=(summary["roc_auc"],-summary["neg_brier"],name,est)
            if best is None or choice[:2]>best[:2]: best=choice
        _,_,name,est=best; selections.append(name)
        fitted=CalibratedClassifierCV(estimator=Pipeline([("prep",preprocessor(X)),("model",est)]),method="sigmoid",cv=5)
        fitted.fit(X_train,y_train)
        prob=fitted.predict_proba(X_test)[:,1]; pred=(prob>=0.5).astype(int)
        all_truth.extend(y_test.tolist()); all_probabilities.extend(prob.tolist())
        outer_scores.append({"roc_auc":roc_auc_score(y_test,prob),"accuracy":accuracy_score(y_test,pred),
                             "precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),
                             "f1":f1_score(y_test,pred,zero_division=0),"brier":brier_score_loss(y_test,prob)})
    comparison=[]
    for name,values in inner_scores.items():
        row={"model":name,"scope":"inner-CV model-selection scores (not final performance)"}
        for metric in SCORING:
            key="brier" if metric=="neg_brier" else metric
            vals=[v[metric] for v in values]
            row[key+"_mean"]=float(np.mean(vals)); row[key+"_std"]=float(np.std(vals,ddof=1))
        comparison.append(row)
    metrics={}
    for key in outer_scores[0]:
        vals=[row[key] for row in outer_scores]
        metrics[key+"_mean"]=float(np.mean(vals)); metrics[key+"_std"]=float(np.std(vals,ddof=1))
    selected=max(set(selections),key=selections.count)
    metrics["selected_model"]=selected; metrics["selection_count"]=selections.count(selected); metrics["outer_folds"]=len(selections)
    return comparison,metrics,{"truth":all_truth,"probability":all_probabilities}

def quality_report(truth, probability, threshold):
    truth=np.asarray(truth); probability=np.asarray(probability)
    fpr,tpr,roc_thresholds=roc_curve(truth,probability)
    precision,recall,pr_thresholds=precision_recall_curve(truth,probability)
    observed,predicted=calibration_curve(truth,probability,n_bins=8,strategy="quantile")
    predicted_class=(probability>=threshold).astype(int)
    return {"threshold":threshold,"threshold_policy":"Fixed 0.35 recall-favoring operating point; not tuned on held-out outer predictions.",
            "pr_auc":float(average_precision_score(truth,probability)),
            "roc_curve":{"fpr":fpr.tolist(),"tpr":tpr.tolist(),"thresholds":roc_thresholds.tolist()},
            "pr_curve":{"precision":precision.tolist(),"recall":recall.tolist(),"thresholds":pr_thresholds.tolist()},
            "calibration_curve":{"mean_predicted_probability":predicted.tolist(),"fraction_positive":observed.tolist()},
            "confusion_matrix":confusion_matrix(truth,predicted_class,labels=[0,1]).tolist(),
            "operating_metrics":{"accuracy":float(accuracy_score(truth,predicted_class)),"precision":float(precision_score(truth,predicted_class,zero_division=0)),"recall":float(recall_score(truth,predicted_class,zero_division=0)),"f1":float(f1_score(truth,predicted_class,zero_division=0)),"brier":float(brier_score_loss(truth,probability))}}

def write_reports(df, X, constants, rows, registry, nested_rows, qualities):
    """Write all non-model reports from the exact data and CV run just completed."""
    pd.DataFrame(rows).to_csv(REPORTS/"cv_model_comparison.csv",index=False)
    pd.DataFrame(nested_rows).to_csv(REPORTS/"nested_cv_results.csv",index=False)
    for target, report in qualities.items():
        (REPORTS/f"quality_{target.lower()}.json").write_text(json.dumps(report,indent=2))
    (REPORTS/"thresholds.json").write_text(json.dumps({"thresholds":OPERATING_THRESHOLDS,"policy":"Fixed 0.35 recall-favoring operating point; not tuned on held-out outer predictions."},indent=2))
    lines=["# Cardiovascular ML Workstream — Retraining Results", "",
           "## Validation status", "",
           f"{VALIDATION_NOTE}. Performance values below are held-out outer-fold estimates; model selection was performed only inside each outer-training fold.",
           "", "## Data audit", "",
           f"- Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.",
           f"- Exact duplicate rows: {int(df.duplicated().sum())}.",
           "- Leakage columns excluded from every model: Cath, LAD, LCX, RCA.",
           f"- Constant features dropped: {', '.join(constants) if constants else 'none'}.",
           f"- Class balance (positive/negative): {', '.join(f'{target} {int(df[c].sum())}/{int((df[c] == 0).sum())}' for target, c in TARGETS.items())}.",
           "", "## Preliminary CV results", "",
           "| Target | Most-selected model | ROC-AUC (mean) | Brier (mean) |", "|---|---|---:|---:|"]
    for target, result in registry.items():
        lines.append(f"| {target} | {result['selected_model']} | {result['cv_mean_roc_auc']:.3f} | {result['cv_mean_brier_score']:.3f} |")
    lines += ["", "PRD ROC-AUC targets are CAD >= 0.85, LAD >= 0.80, LCX >= 0.75, RCA >= 0.75. Nested CV satisfies the internal resampling requirement, but it does not substitute for external validation.",
              "", "## Remaining limits", "", "- Small single-dataset sample (303 records); estimates are uncertain.", "- No external validation dataset, subgroup/fairness analysis, decision-curve analysis, or prospective evaluation was available.", "- Sigmoid calibration is fitted for the delivered full-data models; calibration has no independent external assessment.", "- This is an educational prototype, not a clinical diagnostic tool."]
    (REPORTS/"initial_results.md").write_text("\n".join(lines)+"\n")

def main():
    df=load_data()
    X=df.drop(columns=list(FORBIDDEN))
    if set(X.columns)&FORBIDDEN: raise AssertionError("Leakage columns in X")
    constants=[c for c in X.columns if X[c].nunique(dropna=False)<=1]
    X=X.drop(columns=constants)
    rows=[]; registry={}; nested_rows=[]; qualities={}
    for target,labelcol in TARGETS.items():
        y=df[labelcol].astype(int)
        comparison, nested, predictions=nested_evaluate(X,y)
        for row in comparison:
            row["target"]=target; rows.append(row)
        nested_rows.append({"target":target,**nested})
        qualities[target]=quality_report(predictions["truth"],predictions["probability"],OPERATING_THRESHOLDS[target])
        name=nested["selected_model"]; est=candidates()[name]
        base=Pipeline([("prep",preprocessor(X)),("model",est)])
        calibrated=CalibratedClassifierCV(estimator=base,method="sigmoid",cv=5)
        calibrated.fit(X,y)
        joblib.dump({"target":target,"source_column":labelcol,"model_name":name,"estimator":calibrated,
                     "feature_columns":list(X.columns),"constant_features_dropped":constants,"seed":SEED},
                    OUT/f"{target.lower()}_model.joblib")
        registry[target]={"selected_model":name,"selection_count":nested["selection_count"],"outer_folds":nested["outer_folds"],"cv_mean_roc_auc":nested["roc_auc_mean"],"cv_mean_brier_score":nested["brier_mean"],"operating_threshold":OPERATING_THRESHOLDS[target],"cv_pr_auc":qualities[target]["pr_auc"]}
        print(f"{target}: {name}; nested CV ROC-AUC {nested['roc_auc_mean']:.3f}; Brier {nested['brier_mean']:.3f}")
    write_reports(df, X, constants, rows, registry, nested_rows, qualities)
    (OUT/"model_card.json").write_text(json.dumps({"dataset":"UCI Z-Alizadeh Sani Extension","records":len(df),
      "original_columns":df.shape[1],"input_features":X.shape[1],"targets":list(TARGETS),
      "validation":VALIDATION_NOTE,"seed":SEED,
      "selection":"highest inner-fold mean ROC-AUC; Brier tie-breaker, reported as most frequent choice across 25 outer folds","calibration":"sigmoid calibration inside each outer fold for validation; five-fold sigmoid ensemble for the delivered model","operating_thresholds":OPERATING_THRESHOLDS,
      "preliminary_cv_metrics":registry,
      "limitations":["Small dataset; uncertain estimates","No external validation or prospective evaluation","No independent external calibration assessment","Not for clinical use"]},indent=2))
    print("Saved reports/cv_model_comparison.csv and four calibrated model artifacts.")
if __name__=="__main__": main()
