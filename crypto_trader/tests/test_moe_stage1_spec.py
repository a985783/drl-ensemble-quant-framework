from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_moe_stage1 import _prepare_training_dataframe, build_training_specs, get_algo_registry


def test_algo_registry_supports_expected_algorithms() -> None:
    registry = get_algo_registry()
    assert set(["ppo", "a2c", "sac"]).issubset(set(registry.keys()))


def test_dry_specs_resolve_four_experts() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "moe_experts.yaml"
    specs = build_training_specs(manifest_path)

    assert len(specs) == 4
    assert all(spec.expert_id for spec in specs)
    assert all(spec.algorithm in {"ppo", "a2c", "sac"} for spec in specs)


def test_prepare_training_dataframe_supports_local_csv(tmp_path: Path) -> None:
    from config import get_default_config

    csv_path = tmp_path / "training_sample.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Open": [100 + i for i in range(10)],
            "High": [101 + i for i in range(10)],
            "Low": [99 + i for i in range(10)],
            "Close": [100.5 + i for i in range(10)],
            "Volume": [1000 + i for i in range(10)],
            "ATR": [1.0 + i * 0.01 for i in range(10)],
            "Signal_Proba": [0.5 + i * 0.01 for i in range(10)],
        }
    ).to_csv(csv_path, index=False)

    config = get_default_config()
    df = _prepare_training_dataframe(config, train_data_path=str(csv_path))

    assert len(df) > 0
    assert "Signal_Proba" in df.columns
    assert "ATR" in df.columns
