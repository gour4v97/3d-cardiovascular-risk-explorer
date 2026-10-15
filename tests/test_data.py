from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train import load_data, FORBIDDEN

def test_dataset_shape():
    assert load_data().shape == (303,59)

def test_targets_binary():
    df=load_data()
    for col in ['Cath','LAD','LCX','RCA']:
        assert set(df[col].dropna().unique()) <= {0,1}

def test_no_target_leakage():
    df=load_data()
    X=df.drop(columns=list(FORBIDDEN))
    assert not (set(X.columns) & FORBIDDEN)

def test_feature_metadata_covers_model_inputs():
    import yaml
    df=load_data()
    model_inputs=set(df.columns)-FORBIDDEN-{'Exertional CP'}
    specs=yaml.safe_load((ROOT/'config/features.yaml').read_text())['features']
    assert set(specs) == model_inputs
