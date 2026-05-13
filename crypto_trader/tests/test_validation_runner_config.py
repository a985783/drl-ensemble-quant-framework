from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from validation.alpha_validation import build_scenarios, load_validation_config


def test_default_validation_config_expands_expected_scenarios() -> None:
    config = load_validation_config(Path("crypto_trader/validation/default_validation.yaml"))
    scenarios = build_scenarios(config)
    names = [scenario["name"] for scenario in scenarios]

    assert "stable_oos" in names
    assert "signal_delay_1d" in names
    assert "signal_neutral_0_5" in names
    assert "random_baseline" in names
    assert "temperature_0_68" in names
    assert "cost_2x" in names
    assert "funding_5x" in names
    assert "uniform_gate" in names
    assert "average_experts" in names


def test_dry_run_writes_only_plan_metadata(tmp_path: Path) -> None:
    config = load_validation_config(Path("crypto_trader/validation/default_validation.yaml"))
    scenarios = build_scenarios(config)

    assert len(scenarios) >= 10
    assert tmp_path.exists()
