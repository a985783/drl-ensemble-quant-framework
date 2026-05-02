import os
import json
from dataclasses import replace

from crypto_trader.backtest_moe import load_stage2_usage
from crypto_trader.walk_forward.aggregator import Aggregator
from crypto_trader.walk_forward.backtester import FoldResult
from crypto_trader.walk_forward.data_prep import DataPreparer
from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig
from crypto_trader.walk_forward.gate_trainer import coerce_split_timestamp
from crypto_trader.walk_forward.moe_walk_forward import (
    apply_runtime_overrides,
    normalize_fold_arg,
    parse_temperature_candidates,
    should_skip_completed_fold,
)


def test_parse_temperature_candidates_accepts_csv_values():
    assert parse_temperature_candidates("0.5,0.68,1.0") == [0.5, 0.68, 1.0]


def test_normalize_fold_arg_accepts_numeric_alias():
    assert normalize_fold_arg("1") == "fold_1"
    assert normalize_fold_arg("fold_2") == "fold_2"


def test_apply_runtime_overrides_caps_workers_and_threads(monkeypatch):
    cfg = WalkForwardMoEConfig()

    class Args:
        expert_workers = 99
        torch_threads = 2
        gate_timesteps = 123
        expert_timesteps = 456
        temperature_candidates = "0.5,1.0"
        fast_smoke = False
        checkpoint_root = None

    updated = apply_runtime_overrides(cfg, Args(), cpu_count=8)

    assert updated.expert_parallel_workers == 4
    assert updated.torch_num_threads == 2
    assert updated.gate_timesteps == 123
    assert set(updated.expert_timesteps.values()) == {456}
    assert updated.temperature_candidates == [0.5, 1.0]
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_fast_smoke_uses_tiny_training_budget():
    cfg = WalkForwardMoEConfig()

    class Args:
        expert_workers = None
        torch_threads = None
        gate_timesteps = None
        expert_timesteps = None
        temperature_candidates = None
        fast_smoke = True
        checkpoint_root = "tmp/checkpoints"

    updated = apply_runtime_overrides(cfg, Args(), cpu_count=8)

    assert updated.gate_timesteps <= 2_000
    assert max(updated.expert_timesteps.values()) <= 2_000
    assert len(updated.temperature_candidates) <= 2
    assert updated.checkpoint_root == "tmp/checkpoints"


def test_coerce_split_timestamp_matches_timezone_aware_index():
    import pandas as pd

    idx = pd.date_range("2021-01-01", periods=3, tz="UTC")
    split = coerce_split_timestamp("2021-01-02", idx)

    assert split.tzinfo is not None
    assert bool(idx[0] < split)


def test_load_stage2_usage_accepts_dict_metadata(tmp_path):
    metadata = {
        "usage_ema": {
            "E2_PPO_bear_drawdown": 0.25,
            "E4_PPO_highvol_risk": 0.75,
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    usage = load_stage2_usage(
        tmp_path,
        ["E2_PPO_bear_drawdown", "E4_PPO_highvol_risk"],
    )

    assert usage == {
        "E2_PPO_bear_drawdown": 0.25,
        "E4_PPO_highvol_risk": 0.75,
    }


def test_data_preparer_saves_fold_dataframes(tmp_path):
    cfg = WalkForwardMoEConfig()
    cfg.results_root = str(tmp_path)
    fold = cfg.folds[0]

    train_df = __import__("pandas").DataFrame({"Signal_Proba": [0.6], "Close": [101.0]})
    test_df = __import__("pandas").DataFrame({"Signal_Proba": [0.7], "Close": [102.0]})

    paths = DataPreparer(cfg, fold).save_fold_data(train_df, test_df)

    assert paths["train"].exists()
    assert paths["test"].exists()
    assert paths["metadata"].exists()


def test_aggregator_can_load_saved_fold_metrics(tmp_path):
    cfg = WalkForwardMoEConfig()
    cfg.results_root = str(tmp_path)
    result = FoldResult(
        fold_id="fold_1",
        train_window="2020-01-01 to 2021-12-31",
        test_window="2022-01-01 to 2022-12-31",
        total_return=0.1,
        benchmark_return=-0.1,
        alpha=0.2,
        max_drawdown=0.05,
        sharpe=1.0,
        sortino=1.5,
        gate_usage={},
        expert_contribution={},
        selected_temperature=0.68,
        pass_fold=True,
    )
    Aggregator.save_fold_result(result, tmp_path / "fold_1")

    loaded = Aggregator(cfg).load_fold_results(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].fold_id == "fold_1"
    assert loaded[0].alpha == 0.2


def test_should_skip_completed_fold_requires_resume_and_metrics(tmp_path):
    cfg = WalkForwardMoEConfig()
    cfg.results_root = str(tmp_path)
    fold = cfg.folds[0]

    assert not should_skip_completed_fold(cfg, fold, resume=False)
    assert not should_skip_completed_fold(cfg, fold, resume=True)

    (tmp_path / "fold_1").mkdir()
    (tmp_path / "fold_1" / "metrics.json").write_text("{}", encoding="utf-8")

    assert should_skip_completed_fold(cfg, fold, resume=True)
