from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from validation.metrics import compute_equity_metrics, flatten_metric_row


def test_compute_equity_metrics_includes_costs_and_exposure() -> None:
    net_worth = [10000.0, 10100.0, 9900.0, 10400.0]
    benchmark = [10000.0, 10050.0, 9950.0, 10000.0]
    positions = [0.0, 0.5, -0.25]
    turnovers = [0.0, 0.5, 0.75]
    trade_costs = [0.0, 4.0, 6.0]
    funding_costs = [1.0, 1.0, 2.0]

    metrics = compute_equity_metrics(
        net_worth=net_worth,
        benchmark_values=benchmark,
        positions=positions,
        turnovers=turnovers,
        trade_costs=trade_costs,
        funding_costs=funding_costs,
    )

    assert np.isclose(metrics["total_return"], 0.04)
    assert np.isclose(metrics["benchmark_return"], 0.0)
    assert metrics["alpha"] > 0
    assert metrics["max_drawdown"] > 0
    assert metrics["turnover"] == 1.25
    assert metrics["trade_cost"] == 10.0
    assert metrics["funding_cost"] == 4.0
    assert metrics["exposure"] > 0
    assert metrics["long_exposure"] > 0
    assert metrics["short_exposure"] > 0


def test_flatten_metric_row_preserves_scenario_and_core_metrics() -> None:
    row = flatten_metric_row(
        "stress_fee_2x",
        {"total_return": 0.1, "alpha": 0.2, "gate_usage": {"E1": 0.7}},
    )

    assert row["scenario"] == "stress_fee_2x"
    assert row["total_return"] == 0.1
    assert row["alpha"] == 0.2
    assert "gate_usage" not in row
