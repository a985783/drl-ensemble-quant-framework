from pathlib import Path
import json
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest_moe import (
    aggregate_usage,
    estimate_expert_contribution,
    load_stage2_usage,
)


def test_aggregate_usage_returns_mean_weight_per_expert() -> None:
    expert_ids = ["E1", "E2", "E3"]
    weights = np.array(
        [
            [0.5, 0.4, 0.1],
            [0.4, 0.5, 0.1],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    usage = aggregate_usage(weights, expert_ids)

    assert abs(sum(usage.values()) - 1.0) < 1e-6
    assert usage["E1"] > usage["E2"] > usage["E3"]


def test_estimate_expert_contribution_respects_action_alignment() -> None:
    expert_ids = ["E1", "E2"]
    weights = np.array([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]], dtype=np.float32)
    actions = np.array([[0.8, -0.4], [0.7, -0.2], [0.9, -0.1]], dtype=np.float32)
    step_returns = np.array([0.01, 0.02, 0.015], dtype=np.float32)

    contrib = estimate_expert_contribution(weights, actions, step_returns, expert_ids)
    assert contrib["E1"] > 0
    assert contrib["E2"] < contrib["E1"]


def test_load_stage2_usage_from_metadata_or_uniform_fallback(tmp_path: Path) -> None:
    expert_ids = ["E1", "E2", "E3"]

    fallback = load_stage2_usage(tmp_path / "missing", expert_ids)
    assert all(abs(v - (1.0 / 3.0)) < 1e-6 for v in fallback.values())

    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    with open(stage2_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"expert_ids": expert_ids, "usage_ema": [0.2, 0.3, 0.5]}, f)

    loaded = load_stage2_usage(stage2_dir, expert_ids)
    assert loaded["E3"] > loaded["E2"] > loaded["E1"]
