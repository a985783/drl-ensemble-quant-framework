"""
FoldBacktester: runs next-bar MoE backtest on a single walk-forward fold.

Each fold has its own experts + gate checkpoints under
  {checkpoint_root}/{fold_id}/experts/   and   {checkpoint_root}/{fold_id}/gate/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from crypto_trader.backtest_moe import backtest_moe
    from crypto_trader.train_moe_stage2_gate import (
        build_gate_artifacts,
        validate_stage1_artifacts,
    )
    from crypto_trader.walk_forward.moe_config import FoldSpec, WalkForwardMoEConfig
except ImportError:
    from backtest_moe import backtest_moe
    from train_moe_stage2_gate import (
        build_gate_artifacts,
        validate_stage1_artifacts,
    )
    from walk_forward.moe_config import FoldSpec, WalkForwardMoEConfig


@dataclass
class FoldResult:
    """Metrics and pass/fail verdict for a single fold's backtest."""

    fold_id: str
    train_window: str
    test_window: str
    total_return: float
    benchmark_return: float
    alpha: float
    max_drawdown: float
    sharpe: float
    sortino: float
    gate_usage: dict
    expert_contribution: dict
    selected_temperature: float
    pass_fold: bool


class FoldBacktester:
    """Loads fold-specific experts + gate and runs a next-bar backtest."""

    def __init__(self, config: WalkForwardMoEConfig, fold_config: FoldSpec):
        self.config = config
        self.fold = fold_config

    def run(
        self,
        test_df: pd.DataFrame,
        best_temperature: float = 0.68,
    ) -> FoldResult:
        """Execute a full MoE backtest for this fold.

        Parameters
        ----------
        test_df : pd.DataFrame
            OOS test-period DataFrame with datetime index and all feature
            columns including ``Signal_Proba``.
        best_temperature : float
            Gate softmax temperature selected during validation.
        """
        expert_root = Path(self.config.checkpoint_root) / self.fold.fold_id / "experts"
        gate_root = Path(self.config.checkpoint_root) / self.fold.fold_id / "gate"

        missing = self._check_checkpoints(expert_root, gate_root)
        if missing:
            return self._error_result(best_temperature, missing)

        tmp_data_path = self._save_test_data(test_df)

        try:
            raw_result = backtest_moe(
                manifest_path=Path(self.config.manifest_path),
                stage1_root=str(expert_root),
                stage2_root=str(gate_root),
                data_path=tmp_data_path,
                gate_temperature=best_temperature,
                symbol=self.config.symbol,
                env_overrides={"tau": 0.25},
                execution_mode=self.config.execution_mode,
                active_expert_ids=self.config.expert_ids,
                return_history=True,
            )

            if "error" in raw_result:
                return self._error_result(
                    best_temperature, raw_result.get("error", "unknown")
                )

            alpha = self._extract_alpha(raw_result)
            sharpe_val, sortino_val = self._compute_risk_metrics(raw_result)

            result = FoldResult(
                fold_id=self.fold.fold_id,
                train_window=f"{self.fold.train_start} to {self.fold.train_end}",
                test_window=f"{self.fold.test_start} to {self.fold.test_end}",
                total_return=float(raw_result.get("total_return", 0)),
                benchmark_return=float(raw_result.get("benchmark_return", 0)),
                alpha=alpha,
                max_drawdown=float(raw_result.get("max_dd", 0)),
                sharpe=sharpe_val,
                sortino=sortino_val,
                gate_usage=raw_result.get("gate_usage", {}),
                expert_contribution=raw_result.get("expert_contribution", {}),
                selected_temperature=best_temperature,
                pass_fold=alpha > self.config.min_fold_alpha,
            )
            self._save_fold_result(result)
            return result

        finally:
            self._cleanup_test_data(tmp_data_path)

    def _check_checkpoints(self, expert_root: Path, gate_root: Path) -> list:
        artifacts = build_gate_artifacts(
            Path(self.config.manifest_path),
            str(expert_root),
            active_expert_ids=self.config.expert_ids,
        )
        missing = validate_stage1_artifacts(artifacts)

        gate_model = gate_root / "gate_model.zip"
        gate_vecnorm = gate_root / "gate_vec_normalize.pkl"
        if not gate_model.exists():
            missing.append(str(gate_model))
        if not gate_vecnorm.exists():
            missing.append(str(gate_vecnorm))

        return missing

    def _save_test_data(self, df: pd.DataFrame) -> str:
        results_dir = Path(self.config.results_root) / "tmp"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{self.fold.fold_id}_test_data.csv"
        df.to_csv(path)
        return str(path)

    def _cleanup_test_data(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            p.unlink(missing_ok=True)

    @staticmethod
    def _extract_alpha(raw: dict) -> float:
        if "alpha" in raw:
            return float(raw["alpha"])
        tr = float(raw.get("total_return", 0))
        br = float(raw.get("benchmark_return", 0))
        return tr - br

    @staticmethod
    def _compute_risk_metrics(raw: dict) -> tuple[float, float]:
        """Annualised sharpe and sortino from net-worth history."""
        history = raw.get("history")
        if not history:
            return 0.0, 0.0

        nw = np.asarray(history.get("net_worth", []), dtype=np.float64)
        if len(nw) < 2:
            return 0.0, 0.0

        rets = np.diff(nw) / nw[:-1]
        if len(rets) == 0:
            return 0.0, 0.0

        mu = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else 0.0

        downside = rets[rets < 0]
        down_sigma = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (mu / down_sigma) * np.sqrt(252) if down_sigma > 1e-12 else 0.0

        return sharpe, sortino

    def _error_result(self, temperature: float, missing: object) -> FoldResult:
        msg = (
            json.dumps({"error": missing})
            if isinstance(missing, str)
            else json.dumps({"missing": list(missing)})
        )
        result = FoldResult(
            fold_id=self.fold.fold_id,
            train_window=f"{self.fold.train_start} to {self.fold.train_end}",
            test_window=f"{self.fold.test_start} to {self.fold.test_end}",
            total_return=0.0,
            benchmark_return=0.0,
            alpha=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
            sortino=0.0,
            gate_usage={"error": msg},
            expert_contribution={},
            selected_temperature=temperature,
            pass_fold=False,
        )
        try:
            self._save_fold_result(result)
        except Exception:
            pass
        return result

    def _save_fold_result(self, result: FoldResult) -> Path:
        output_dir = Path(self.config.results_root) / self.fold.fold_id
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.json"
        path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
