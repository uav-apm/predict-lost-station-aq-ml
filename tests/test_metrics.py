import numpy as np

from src.metrics import r2, rmse


def test_metrics_return_expected_values():
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])

    assert rmse(actual, pred) == 0.0
    assert r2(actual, pred) == 1.0
