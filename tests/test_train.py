from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import FORBIDDEN, TARGETS, load_data, preprocessor, quality_report


def test_training_data_has_expected_binary_targets_and_no_leakage():
    data = load_data()
    assert data.shape == (303, 59)
    for column in TARGETS.values():
        assert set(data[column].dropna().unique()) <= {0, 1}

    features = data.drop(columns=FORBIDDEN)
    assert not (set(features.columns) & FORBIDDEN)
    assert "Exertional CP" in features.columns


def test_preprocessor_handles_numeric_and_categorical_features():
    data = load_data()
    features = data.drop(columns=FORBIDDEN).drop(columns=["Exertional CP"])
    transformed = preprocessor(features).fit_transform(features.head(8))
    assert transformed.shape[0] == 8
    assert np.isfinite(transformed).all()


def test_quality_report_uses_the_declared_operating_threshold():
    report = quality_report(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.40, 0.60, 0.95]),
        threshold=0.35,
    )
    assert report["threshold"] == 0.35
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0.0 <= report["pr_auc"] <= 1.0
