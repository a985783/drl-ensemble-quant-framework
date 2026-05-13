from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


def _metric(scenario: Mapping[str, object], key: str, default: float = 0.0) -> float:
    metrics = scenario.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return default
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _by_name(scenarios: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(s.get("name", "")): s for s in scenarios}


def check_gate_collapse(
    gate_weights_history: np.ndarray,
    threshold: float = 0.8,
) -> Optional[str]:
    """
    Check if any expert's EMA weight exceeds threshold for more than 50% of steps.

    Parameters
    ----------
    gate_weights_history : np.ndarray
        Shape (n_steps, n_experts) — gate weights per step.
    threshold : float
        Weight threshold (default 0.8).

    Returns
    -------
    Optional[str]
        WARN message if collapse detected, None otherwise.
    """
    if gate_weights_history.size == 0 or gate_weights_history.ndim != 2:
        return None
    n_steps = gate_weights_history.shape[0]
    n_experts = gate_weights_history.shape[1]
    half_steps = n_steps * 0.5

    # Compute EMA across time axis (alpha=0.1)
    ema = np.zeros_like(gate_weights_history, dtype=np.float64)
    ema[0] = gate_weights_history[0]
    for t in range(1, n_steps):
        ema[t] = 0.1 * gate_weights_history[t] + 0.9 * ema[t - 1]

    for expert_idx in range(n_experts):
        if np.sum(ema[:, expert_idx] > threshold) > half_steps:
            return (
                f"Gate collapse detected: expert {expert_idx} "
                f"has EMA weight > {threshold:.0%} for >50% of steps"
            )
    return None


def check_bootstrap_significance(
    total_return: float,
    returns: np.ndarray,
    ci: float = 0.95,
) -> Optional[str]:
    """
    Check if total_return is significantly positive via bootstrap CI.

    Parameters
    ----------
    total_return : float
        Observed total return of the strategy.
    returns : np.ndarray
        Per-step returns used for bootstrapping.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    Optional[str]
        WARN message if CI lower bound < 0, None otherwise.
    """
    from crypto_trader.validation.metrics import bootstrap_confidence_interval

    if returns.size < 5:
        return None
    lower, upper = bootstrap_confidence_interval(returns, ci=ci)
    if lower < 0:
        return (
            f"Return may not be significantly positive: "
            f"bootstrap {ci:.0%} CI = [{lower:.6f}, {upper:.6f}], "
            f"lower bound < 0"
        )
    return None


def evaluate_validation_results(
    *,
    scenarios: List[Mapping[str, object]],
    walk_forward_summary: Mapping[str, object],
    missing_artifacts: List[str],
    gate_weights_history: Optional[np.ndarray] = None,
    step_returns: Optional[np.ndarray] = None,
    run_bootstrap: bool = False,
) -> Dict[str, object]:
    blocking_items: List[str] = []
    failures: List[str] = []
    warnings: List[str] = []
    passes: List[str] = []

    if missing_artifacts:
        blocking_items.extend([f"Missing artifact: {item}" for item in missing_artifacts])

    errored = [str(s.get("name", "unknown")) for s in scenarios if s.get("status") == "error"]
    if errored:
        blocking_items.extend([f"Scenario failed to run: {name}" for name in errored])

    scenario_map = _by_name(scenarios)
    stable = scenario_map.get("stable_oos")
    stable_alpha = _metric(stable, "alpha") if stable else 0.0
    stable_dd = _metric(stable, "max_drawdown") if stable else 0.0

    random_baseline = scenario_map.get("random_baseline")
    if random_baseline and _metric(random_baseline, "total_return") > 0.05:
        failures.append("Random baseline total_return > 5%; backtest framework or cost model may be biased.")

    cost_2x = scenario_map.get("cost_2x")
    if cost_2x and stable_alpha > 0:
        cost_alpha = _metric(cost_2x, "alpha")
        cost_dd = _metric(cost_2x, "max_drawdown")
        if cost_alpha < 0 and cost_dd > stable_dd:
            failures.append("Cost 2x stress flips alpha negative while worsening drawdown.")

    signal_delay = scenario_map.get("signal_delay_1d")
    if signal_delay and stable:
        delayed_return = _metric(signal_delay, "total_return")
        stable_return = _metric(stable, "total_return")
        if stable_return > 0 and delayed_return < stable_return * 0.5:
            warnings.append("Signal_Proba one-day delay cuts total_return by more than 50%.")

    temp_returns = [
        _metric(s, "total_return")
        for name, s in scenario_map.items()
        if name.startswith("temperature_")
    ]
    if len(temp_returns) >= 2 and (max(temp_returns) - min(temp_returns)) > 0.5:
        warnings.append("Gate temperature perturbation range exceeds 50 percentage points.")

    wf_status = str(walk_forward_summary.get("status", "missing"))
    if wf_status != "ok":
        warnings.append("Walk-forward metrics are missing or unreadable.")
    else:
        folds = int(walk_forward_summary.get("folds", 0) or 0)
        avg_alpha = float(walk_forward_summary.get("avg_alpha", 0.0) or 0.0)
        if folds < 3:
            warnings.append("Walk-forward evidence has fewer than 3 folds.")
        if avg_alpha <= 0:
            warnings.append("Walk-forward average alpha is not positive.")
        if stable_alpha > 0.5 and avg_alpha < stable_alpha * 0.25:
            warnings.append("Single OOS alpha is materially stronger than walk-forward average alpha.")

    # Gate collapse detection
    if gate_weights_history is not None:
        collapse_msg = check_gate_collapse(gate_weights_history)
        if collapse_msg:
            warnings.append(collapse_msg)

    # Bootstrap significance
    if run_bootstrap and step_returns is not None:
        total_ret = _metric(stable, "total_return") if stable else 0.0
        bootstrap_msg = check_bootstrap_significance(total_ret, step_returns)
        if bootstrap_msg:
            warnings.append(bootstrap_msg)

    if not blocking_items and not failures and not warnings:
        passes.append("No blocking, failure, or warning conditions were triggered.")
    elif not blocking_items and not failures:
        passes.append("No blocking or fail-fast conditions were triggered.")

    if blocking_items:
        status = "BLOCKED"
    elif failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "blocking_items": blocking_items,
        "failures": failures,
        "warnings": warnings,
        "passes": passes,
    }
