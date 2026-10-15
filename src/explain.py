from pathlib import Path
import joblib, pandas as pd
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/raw/z_alizadeh_sani_extension.xlsx"
OUT=ROOT/"reports"
def main():
    df=pd.read_excel(DATA); X=df.drop(columns=["Cath","LAD","LCX","RCA"])
    for target in ["CAD","LAD","LCX","RCA"]:
        a=joblib.load(ROOT/"models"/f"{target.lower()}_model.joblib")
        y = df[a["source_column"]].map({"CAD": 1, "Normal": 0, "Stenotic": 1}).astype(int)
        result=permutation_importance(a["estimator"], X[a["feature_columns"]], y,
                                      scoring="neg_brier_score", n_repeats=10,
                                      random_state=42, n_jobs=-1)
        pd.DataFrame({"feature":a["feature_columns"],
                      "brier_score_increase_mean":result.importances_mean,
                      "brier_score_increase_std":result.importances_std}).sort_values(
                          "brier_score_increase_mean",ascending=False).to_csv(
                              OUT/f"feature_importance_{target.lower()}.csv",index=False)
        print("Permutation importance saved for",target)
if __name__=="__main__": main()
