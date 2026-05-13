import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.lstm_regressor import LocalLSTMRegressor
from src.pipeline import make_regressor


def test_make_regressor_returns_random_forest():
    assert isinstance(make_regressor("random_forest"), RandomForestRegressor)


def test_make_regressor_returns_extra_trees():
    assert isinstance(make_regressor("extra_trees"), ExtraTreesRegressor)


def test_make_regressor_supports_fully_qualified_class_path():
    model = make_regressor("sklearn.linear_model.LinearRegression", {"fit_intercept": False})

    assert isinstance(model, LinearRegression)
    assert model.fit_intercept is False


def test_make_regressor_supports_local_lstm_class_path():
    model = make_regressor("src.lstm_regressor.LocalLSTMRegressor", {"epochs": 1, "verbose": 0})

    assert isinstance(model, LocalLSTMRegressor)
    assert model.epochs == 1


def test_make_regressor_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        make_regressor("unknown")
