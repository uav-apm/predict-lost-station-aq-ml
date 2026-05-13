"""Model pipeline for station reconstruction and one-step forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from .metrics import r2, rmse


@dataclass
class StationForecastConfig:
    """Minimal model configuration stored with each trained run."""

    target: str
    datetime_col: str
    algorithm: str = "random_forest"
    params: dict | None = None


BUILTIN_REGRESSORS = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
}


def _resolve_regressor_class(algorithm: str):
    """Resolve either a built-in alias or a fully qualified estimator class path."""

    normalized = algorithm.lower().strip()
    if normalized in BUILTIN_REGRESSORS:
        return BUILTIN_REGRESSORS[normalized]

    if "." not in algorithm:
        raise ValueError(
            "Unsupported algorithm alias. Use a built-in alias such as 'random_forest' "
            "or 'extra_trees', or provide a fully qualified class path."
        )

    module_name, class_name = algorithm.rsplit(".", 1)
    module = import_module(module_name)
    estimator_class = getattr(module, class_name)
    return estimator_class


def make_regressor(algorithm: str, params: dict | None = None):
    """Instantiate the configured regressor with optional constructor parameters."""

    estimator_class = _resolve_regressor_class(algorithm)
    estimator_params = {"n_estimators": 100, "random_state": 42} if algorithm.lower().strip() in BUILTIN_REGRESSORS else {}
    if params:
        estimator_params.update(params)

    model = estimator_class(**estimator_params)
    if not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise TypeError("Configured estimator must expose both fit() and predict() methods.")
    return model


class StationForecastPipeline:
    """Thin wrapper around the baseline regressor and its feature metadata."""

    def __init__(self, config: StationForecastConfig):
        self.config = config
        self.model = make_regressor(config.algorithm, config.params)
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y):
        self.feature_names_ = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
            self.model.fit(X, y.astype(float))
        else:
            series = y.astype(float)
            self.target_names_ = [str(getattr(series, "name", "target_value") or "target_value")]
            self.model.fit(X, series)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X[self.feature_names_])

    def predict_scalar(self, values: dict[str, float]):
        frame = pd.DataFrame([values], columns=self.feature_names_)
        pred = self.predict(frame)
        if len(self.target_names_) <= 1:
            return float(pred[0])
        return {
            name: float(value)
            for name, value in zip(self.target_names_, pred[0])
        }

    def evaluate(self, X: pd.DataFrame, y) -> dict:
        preds = self.predict(X)
        if isinstance(y, pd.DataFrame):
            actual = y.astype(float).to_numpy()
            metrics_by_target: dict[str, dict[str, float]] = {}
            for index, target_name in enumerate(list(y.columns)):
                actual_target = actual[:, index]
                pred_target = preds[:, index]
                metrics_by_target[target_name] = {
                    "RMSE": rmse(actual_target, pred_target),
                    "R2": r2(actual_target, pred_target),
                }
            return {
                "prediction": preds,
                "actual": actual,
                "metrics": {"by_target": metrics_by_target},
                "target_names": list(y.columns),
            }

        actual = y.astype(float).to_numpy()
        return {
            "prediction": preds,
            "actual": actual,
            "metrics": {
                "RMSE": rmse(actual, preds),
                "R2": r2(actual, preds),
            },
            "target_names": [str(getattr(y, "name", "target_value") or "target_value")],
        }
