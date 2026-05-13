from pathlib import Path

import joblib
import numpy as np
import pytest

from src.lstm_regressor import LocalLSTMRegressor


@pytest.mark.smoke
def test_local_lstm_regressor_fit_predict_and_joblib_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 6)).astype(np.float32)
    y = (0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]).astype(np.float32)

    model = LocalLSTMRegressor(
        timesteps=1,
        lstm_units=(16,),
        dense_units=8,
        dropout=0.0,
        epochs=2,
        batch_size=8,
        validation_split=0.0,
        patience=0,
        verbose=0,
    )
    model.fit(X, y)
    assert model.backend_ in {"tensorflow", "mlp_fallback"}

    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()

    out_path = Path(tmp_path) / "lstm_smoke.pkl"
    joblib.dump(model, out_path)
    restored = joblib.load(out_path)

    restored_preds = restored.predict(X[:5])
    assert restored_preds.shape == (5,)
    assert np.isfinite(restored_preds).all()
