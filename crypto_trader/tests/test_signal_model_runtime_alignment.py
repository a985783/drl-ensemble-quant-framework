from __future__ import annotations

import numpy as np
import pandas as pd


def test_predict_proba_aligns_to_training_feature_names() -> None:
    from crypto_trader.models.signal_model import SignalPredictor

    predictor = SignalPredictor()

    train_df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5, 6],
            "High": [1, 2, 3, 4, 5, 6],
            "Low": [1, 2, 3, 4, 5, 6],
            "Close": [1, 2, 1, 2, 1, 2],
            "Volume": [10, 10, 10, 10, 10, 10],
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [1, 0, 1, 0, 1, 0],
        }
    )

    X, y = predictor._prepare_features_and_labels(train_df, training=True)
    X_scaled = predictor.scaler.fit_transform(X)
    predictor.model.fit(X_scaled, y)

    infer_df = train_df.copy()
    infer_df["Funding_Rate"] = 0.0001
    infer_df["MACD_Pct"] = 0.1

    probs = predictor.predict_proba(infer_df)

    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(infer_df)
