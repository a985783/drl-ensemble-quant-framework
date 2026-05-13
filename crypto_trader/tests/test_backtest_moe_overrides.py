from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest_moe import apply_data_transform, resolve_execution_frame, resolve_gate_weights


def test_apply_data_transform_signal_delay_and_neutral() -> None:
    import pandas as pd

    df = pd.DataFrame({"Signal_Proba": [0.1, 0.2, 0.3], "Close": [1, 2, 3]})

    delayed = apply_data_transform(df, "signal_delay_1d")
    neutral = apply_data_transform(df, "signal_neutral_0_5")

    assert delayed["Signal_Proba"].tolist() == [0.1, 0.1, 0.2]
    assert neutral["Signal_Proba"].tolist() == [0.5, 0.5, 0.5]


def test_resolve_gate_weights_supports_model_uniform_and_average_modes() -> None:
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    expert_actions = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    model_weights = resolve_gate_weights(logits, expert_actions, "model", 1.0)
    uniform_weights = resolve_gate_weights(logits, expert_actions, "uniform", 1.0)
    average_weights = resolve_gate_weights(logits, expert_actions, "average_experts", 1.0)

    assert np.isclose(model_weights.sum(), 1.0)
    assert np.allclose(uniform_weights, np.ones(3) / 3.0)
    assert np.allclose(average_weights, np.ones(3) / 3.0)


def test_resolve_execution_frame_next_bar_shifts_execution_prices() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 110.0],
            "Signal_Proba": [0.4, 0.6, 0.7],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    shifted, metadata = resolve_execution_frame(df, "next_bar")

    assert shifted["Close"].tolist() == [105.0, 110.0]
    assert shifted["Signal_Proba"].tolist() == [0.6, 0.7]
    assert metadata["execution_mode"] == "next_bar"
    assert metadata["dropped_rows"] == 1
