from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np


CORE_METRIC_KEYS = [
    "total_return",
    "benchmark_return",
    "alpha",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "turnover",
    "trade_cost",
    "funding_cost",
    "exposure",
    "long_exposure",
    "short_exposure",
    "n_steps",
    "final_net_worth",
]


def _as_float_array(values: Optional[Iterable[float]]) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(list(values), dtype=np.float64)


def _max_drawdown(net_worth: np.ndarray) -> float:
    if net_worth.size == 0:
        return 0.0
    peak = np.maximum.accumulate(net_worth)
    safe_peak = np.maximum(peak, 1e-12)
    return float(np.max((peak - net_worth) / safe_peak))


def _annualized_sharpe(simple_returns: np.ndarray) -> float:
    if simple_returns.size == 0:
        return 0.0
    std = float(np.std(simple_returns))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(simple_returns) / std * np.sqrt(252.0))


def _annualized_sortino(simple_returns: np.ndarray) -> float:
    if simple_returns.size == 0:
        return 0.0
    downside = simple_returns[simple_returns < 0]
    downside_std = float(np.std(downside)) if downside.size else 0.0
    if downside_std <= 1e-12:
        return 0.0
    return float(np.mean(simple_returns) / downside_std * np.sqrt(252.0))


def compute_equity_metrics(
    *,
    net_worth: Iterable[float],
    benchmark_values: Optional[Iterable[float]] = None,
    positions: Optional[Iterable[float]] = None,
    turnovers: Optional[Iterable[float]] = None,
    trade_costs: Optional[Iterable[float]] = None,
    funding_costs: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    nw = _as_float_array(net_worth)
    if nw.size == 0:
        nw = np.asarray([10000.0], dtype=np.float64)

    simple_returns = np.diff(nw) / np.maximum(nw[:-1], 1e-12)
    total_return = float((nw[-1] / max(nw[0], 1e-12)) - 1.0)

    bench = _as_float_array(benchmark_values)
    if bench.size >= 2:
        benchmark_return = float((bench[min(len(bench), len(nw)) - 1] / max(bench[0], 1e-12)) - 1.0)
    else:
        benchmark_return = 0.0

    max_dd = _max_drawdown(nw)
    periods = max(len(nw) - 1, 1)
    annual_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0) if total_return > -1.0 else -1.0
    calmar = float(annual_return / max(max_dd, 1e-12)) if max_dd > 0 else 0.0

    pos = _as_float_array(positions)
    long_exposure = float(np.mean(np.maximum(pos, 0.0))) if pos.size else 0.0
    short_exposure = float(np.mean(np.maximum(-pos, 0.0))) if pos.size else 0.0

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "alpha": float(total_return - benchmark_return),
        "max_drawdown": max_dd,
        "sharpe": _annualized_sharpe(simple_returns),
        "sortino": _annualized_sortino(simple_returns),
        "calmar": calmar,
        "turnover": float(np.sum(_as_float_array(turnovers))),
        "trade_cost": float(np.sum(_as_float_array(trade_costs))),
        "funding_cost": float(np.sum(_as_float_array(funding_costs))),
        "exposure": float(np.mean(np.abs(pos))) if pos.size else 0.0,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "n_steps": float(max(len(nw) - 1, 0)),
        "final_net_worth": float(nw[-1]),
    }


def metrics_from_backtest_result(result: Mapping[str, object]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    history = result.get("history")
    if isinstance(history, Mapping) and history.get("net_worth"):
        computed = compute_equity_metrics(
            net_worth=history.get("net_worth", []),
            benchmark_values=history.get("benchmark_values", []),
            positions=history.get("positions", []),
            turnovers=history.get("turnovers", []),
            trade_costs=history.get("trade_costs", []),
            funding_costs=history.get("funding_costs", []),
        )
        metrics.update(computed)
    else:
        for key in ["total_return", "benchmark_return", "alpha", "max_dd", "final_net_worth"]:
            if key in result:
                out_key = "max_drawdown" if key == "max_dd" else key
                metrics[out_key] = result[key]

    for key in ["gate_usage", "stage2_prior_usage", "expert_contribution", "symbol", "interval"]:
        if key in result:
            metrics[key] = result[key]
    return metrics


def bootstrap_confidence_interval(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean of returns.

    Resamples returns with replacement n_bootstrap times,
    computes the mean return for each bootstrap sample,
    returns (lower_bound, upper_bound) of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        means[i] = float(np.mean(sample))
    alpha = 1.0 - ci
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def flatten_metric_row(scenario: str, metrics: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {"scenario": scenario}
    for key in CORE_METRIC_KEYS:
        if key in metrics:
            row[key] = metrics[key]
    return row
