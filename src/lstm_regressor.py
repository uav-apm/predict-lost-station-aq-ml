"""Local sklearn-compatible LSTM regressor used by config class-path loading."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor


class LocalLSTMRegressor(BaseEstimator, RegressorMixin):
    """Minimal LSTM regressor with fit/predict API compatible with sklearn estimators.

    The input is expected to be tabular shape (n_samples, n_features). Internally, features
    are reshaped into (n_samples, timesteps, features_per_step). With `timesteps=1`, this
    works as a drop-in for existing tabular pipelines.
    """

    def __init__(
        self,
        timesteps: int = 1,
        lstm_units: Iterable[int] | None = None,
        bidirectional: bool = False,
        dropout: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        validation_split: float = 0.1,
        patience: int = 8,
        random_state: int = 42,
        verbose: int = 0,
        fallback_on_missing_backend: bool = True,
    ):
        self.timesteps = timesteps
        self.lstm_units = tuple(lstm_units) if lstm_units is not None else (64, 32)
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_on_missing_backend = fallback_on_missing_backend

        self.model_ = None
        self.backend_ = None
        self.backend_note_ = None
        self.device_ = None
        self.n_outputs_ = 1
        self.features_per_step_ = None
        self.history_ = None

    def _import_tf(self):
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorFlow is required for LocalLSTMRegressor. "
                "Install extras with: pip install 'aq-spatial-reconstruction[lstm]'"
            ) from exc
        return tf

    def _try_load_npu_plugins(self) -> list[str]:
        """Attempt to load available NPU acceleration plugins.

        Currently probes Intel Extension for TensorFlow (ITEX), which auto-registers
        Intel XPU/NPU devices with TensorFlow when imported.  Additional plugins can
        be added here as the ecosystem matures.

        Returns a list of successfully loaded plugin name strings (may be empty).
        """
        loaded: list[str] = []
        try:
            import intel_extension_for_tensorflow  # pylint: disable=import-outside-toplevel,unused-import  # noqa: F401
            loaded.append("intel_extension_for_tensorflow")
        except ImportError:
            pass
        return loaded

    def _pick_tf_device(self, tf) -> str:
        """Return the best available TensorFlow device string.

        Priority order: NPU (any kind registered with TF) > GPU > CPU.
        """
        if tf.config.list_physical_devices("NPU"):
            return "/NPU:0"
        if tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        return "/CPU:0"

    def _reshape_X(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("LocalLSTMRegressor expects 2D tabular input (n_samples, n_features).")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be >= 1")
        if X.shape[1] % self.timesteps != 0:
            raise ValueError(
                "Number of input features must be divisible by timesteps. "
                f"Got n_features={X.shape[1]} and timesteps={self.timesteps}."
            )
        features_per_step = X.shape[1] // self.timesteps
        self.features_per_step_ = features_per_step
        return X.reshape((X.shape[0], self.timesteps, features_per_step))

    def _build_model(self, input_shape: tuple[int, int], n_outputs: int):
        tf = self._import_tf()
        tf.keras.utils.set_random_seed(self.random_state)

        model = tf.keras.Sequential(name="local_lstm_regressor")
        model.add(tf.keras.layers.Input(shape=input_shape))

        units = tuple(int(u) for u in self.lstm_units)
        for index, unit in enumerate(units):
            return_sequences = index < len(units) - 1
            lstm_layer = tf.keras.layers.LSTM(unit, return_sequences=return_sequences)
            if self.bidirectional:
                model.add(tf.keras.layers.Bidirectional(lstm_layer))
            else:
                model.add(lstm_layer)
            if self.dropout > 0:
                model.add(tf.keras.layers.Dropout(self.dropout))

        if self.dense_units > 0:
            model.add(tf.keras.layers.Dense(self.dense_units, activation="relu"))
        model.add(tf.keras.layers.Dense(n_outputs, activation="linear"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.n_outputs_ = int(y_arr.shape[1])

        try:
            X_lstm = self._reshape_X(X_arr)
            tf = self._import_tf()
            plugin_notes = self._try_load_npu_plugins()
            device = self._pick_tf_device(tf)
            self.device_ = device
            backend_notes: list[str] = []
            if plugin_notes:
                backend_notes.append(f"plugins: {', '.join(plugin_notes)}")
            backend_notes.append(f"device: {device}")
            self.backend_ = "tensorflow"
            self.backend_note_ = "; ".join(backend_notes)

            with tf.device(device):
                self.model_ = self._build_model((self.timesteps, self.features_per_step_), self.n_outputs_)

                callbacks = []
                if self.patience and self.validation_split and self.validation_split > 0:
                    callbacks.append(
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=int(self.patience),
                            restore_best_weights=True,
                        )
                    )

                self.history_ = self.model_.fit(
                    X_lstm,
                    y_arr,
                    epochs=int(self.epochs),
                    batch_size=int(self.batch_size),
                    validation_split=float(self.validation_split),
                    verbose=int(self.verbose),
                    callbacks=callbacks,
                )
        except ImportError as exc:
            if not self.fallback_on_missing_backend:
                raise

            self.backend_ = "mlp_fallback"
            self.backend_note_ = str(exc)
            hidden_layers = tuple(int(u) for u in self.lstm_units) + ((int(self.dense_units),) if self.dense_units > 0 else tuple())
            self.model_ = MLPRegressor(
                hidden_layer_sizes=hidden_layers or (64, 32),
                learning_rate_init=float(self.learning_rate),
                batch_size=int(self.batch_size),
                max_iter=max(20, int(self.epochs)),
                random_state=int(self.random_state),
                verbose=bool(self.verbose),
                early_stopping=bool(self.validation_split and self.validation_split > 0),
                validation_fraction=float(self.validation_split) if self.validation_split and self.validation_split > 0 else 0.1,
            )
            target = y_arr.reshape(-1) if self.n_outputs_ == 1 else y_arr
            self.model_.fit(X_arr, target)
            self.history_ = None
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        if self.backend_ == "tensorflow":
            tf = self._import_tf()
            X_lstm = self._reshape_X(X_arr)
            device = getattr(self, "device_", "/CPU:0") or "/CPU:0"
            with tf.device(device):
                pred = self.model_.predict(X_lstm, verbose=0)
        else:
            pred = self.model_.predict(X_arr)
        if int(self.n_outputs_) == 1:
            return np.asarray(pred).reshape(-1)
        return pred

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.get("model_")
        state["model_config_"] = None
        state["model_weights_"] = None
        if model is not None and state.get("backend_") == "tensorflow" and hasattr(model, "to_json"):
            state["model_config_"] = model.to_json()
            state["model_weights_"] = model.get_weights()
            state["model_"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model_config = state.get("model_config_")
        model_weights = state.get("model_weights_")
        if self.backend_ != "tensorflow":
            return
        if not model_config:
            self.model_ = None
            return

        tf = self._import_tf()
        self.model_ = tf.keras.models.model_from_json(model_config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model_.compile(optimizer=optimizer, loss="mse")
        if model_weights is not None:
            self.model_.set_weights(model_weights)
