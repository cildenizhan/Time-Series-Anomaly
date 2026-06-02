import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics

def test_calculate_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    res = compute_metrics(y_true, y_pred)
    assert res['accuracy'] == 1.0
    assert res['precision'] == 1.0
    assert res['recall'] == 1.0
    assert res['f1_score'] == 1.0

def test_calculate_metrics_all_wrong():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    res = compute_metrics(y_true, y_pred)
    assert res['accuracy'] == 0.0
    assert res['precision'] == 0.0
    assert res['recall'] == 0.0
    assert res['f1_score'] == 0.0

def test_calculate_metrics_no_positive():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    res = compute_metrics(y_true, y_pred)
    assert res['accuracy'] == 1.0
    assert res['precision'] == 0.0
    assert res['recall'] == 0.0
    assert res['f1_score'] == 0.0

def test_calculate_metrics_all_positive_pred():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([1, 1, 1])
    res = compute_metrics(y_true, y_pred)
    assert res['accuracy'] == pytest.approx(0.333, rel=1e-2)
    assert res['precision'] == pytest.approx(0.333, rel=1e-2)
    assert res['recall'] == 1.0
    assert res['f1_score'] == 0.5
