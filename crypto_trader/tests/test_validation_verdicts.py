from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from validation.metrics import bootstrap_confidence_interval
from validation.verdicts import check_gate_collapse, check_bootstrap_significance, evaluate_validation_results


def _scenario(name: str, total_return: float, alpha: float, max_dd: float = 0.1) -> dict:
    return {
        "name": name,
        "status": "ok",
        "metrics": {
            "total_return": total_return,
            "benchmark_return": 0.0,
            "alpha": alpha,
            "max_drawdown": max_dd,
        },
    }


def test_verdict_blocks_missing_artifacts() -> None:
    verdict = evaluate_validation_results(
        scenarios=[],
        walk_forward_summary={"status": "missing"},
        missing_artifacts=["checkpoints/moe/stable/gate/gate_model.zip"],
    )

    assert verdict["status"] == "BLOCKED"
    assert verdict["blocking_items"]


def test_verdict_fails_when_random_baseline_is_profitable() -> None:
    verdict = evaluate_validation_results(
        scenarios=[
            _scenario("stable_oos", 0.4, 0.5),
            _scenario("random_baseline", 0.08, 0.08),
        ],
        walk_forward_summary={"status": "ok", "avg_alpha": 0.05, "folds": 4},
        missing_artifacts=[],
    )

    assert verdict["status"] == "FAIL"
    assert any("random" in item.lower() for item in verdict["failures"])


def test_verdict_warns_when_walk_forward_is_weak() -> None:
    verdict = evaluate_validation_results(
        scenarios=[_scenario("stable_oos", 0.9, 1.3)],
        walk_forward_summary={"status": "ok", "avg_alpha": -0.02, "folds": 4},
        missing_artifacts=[],
    )

    assert verdict["status"] == "WARN"
    assert verdict["warnings"]


def test_verdict_passes_without_blocking_or_warning_conditions() -> None:
    verdict = evaluate_validation_results(
        scenarios=[
            _scenario("stable_oos", 0.2, 0.15),
            _scenario("random_baseline", -0.01, -0.01),
            _scenario("cost_2x", 0.08, 0.05),
        ],
        walk_forward_summary={"status": "ok", "avg_alpha": 0.04, "folds": 4},
        missing_artifacts=[],
    )

    assert verdict["status"] == "PASS"


def test_bootstrap_confidence_interval_basic() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.001, scale=0.02, size=500)
    lower, upper = bootstrap_confidence_interval(returns, n_bootstrap=500, seed=42)
    assert lower < upper, "CI lower bound must be < upper bound"
    assert -0.005 < lower < 0.005, "CI should contain the true mean (~0.001)"


def test_bootstrap_confidence_interval_significant() -> None:
    rng = np.random.default_rng(42)
    # Returns centered near 0 — CI should straddle zero
    noise = rng.normal(loc=0.0, scale=0.02, size=500)
    msg = check_bootstrap_significance(total_return=0.05, returns=noise)
    assert msg is not None, "Should warn when returns are not significantly positive"
    assert "not be significantly positive" in msg

    # Clearly positive returns — CI should be above zero
    positive = rng.normal(loc=0.005, scale=0.01, size=500)
    msg2 = check_bootstrap_significance(total_return=5.0, returns=positive)
    assert msg2 is None, "Clearly positive returns should pass"


def test_check_gate_collapse_detected() -> None:
    rng = np.random.default_rng(42)
    n_steps, n_experts = 100, 4
    # Expert 0 dominates (>80%) for >50% of steps
    weights = rng.dirichlet(alpha=[0.5, 1.0, 1.0, 1.0], size=n_steps)
    for t in range(60):
        weights[t] = [0.85, 0.05, 0.05, 0.05]
    msg = check_gate_collapse(weights, threshold=0.8)
    assert msg is not None, "Gate collapse should be detected"
    assert "Gate collapse detected" in msg


def test_check_gate_collapse_not_detected() -> None:
    n_steps, n_experts = 100, 4
    # Uniform weights across all experts
    weights = np.full((n_steps, n_experts), 0.25)
    msg = check_gate_collapse(weights, threshold=0.8)
    assert msg is None, "Uniform weights should not trigger gate collapse"
