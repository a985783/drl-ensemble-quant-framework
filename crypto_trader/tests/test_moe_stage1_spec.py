from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_moe_stage1 import _prepare_training_dataframe, build_training_specs, get_algo_registry


def test_algo_registry_supports_expected_algorithms() -> None:
    registry = get_algo_registry()
    assert set(["ppo", "a2c", "sac"]).issubset(set(registry.keys()))


def test_dry_specs_resolve_eight_experts() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "moe_experts.yaml"
    specs = build_training_specs(manifest_path)

    assert len(specs) == 8
    assert all(spec.expert_id for spec in specs)
    assert all(spec.algorithm in {"ppo", "a2c", "sac"} for spec in specs)


def test_prepare_training_dataframe_supports_local_csv() -> None:
    from config import get_default_config

    config = get_default_config()
    df = _prepare_training_dataframe(config, train_data_path="crypto_trader/test_data_ensemble.csv")

    assert len(df) > 0
    assert "Signal_Proba" in df.columns
    assert "ATR" in df.columns
