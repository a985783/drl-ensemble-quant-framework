"""
GateTrainer: MoE Gate PPO training with per-fold temperature selection.

Two-phase workflow:
  A. Temperature scan — train Gate on temporal 80/20 train/val split for
     each candidate temperature, pick the one maximizing validation Sharpe.
  B. Final training — retrain Gate on full training data with best temperature.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_trader.config import BaseConfig, DataConfig, ModelConfig, RiskConfig, SeedConfig
from crypto_trader.envs.trading_env import TradingEnv
from crypto_trader.risk_manager import RiskManager
from crypto_trader.train_moe_stage2_gate import (
    GateExpertArtifact,
    GateRoutingEnv,
    LoadedExpert,
    _env_kwargs_for_symbol,
    _load_experts,
    softmax_weights,
)
from crypto_trader.walk_forward.folding import FoldConfig
from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig


@dataclass
class CandidateResult:
    """Validation metrics for a single temperature candidate."""

    temperature: float
    val_return: float
    val_sharpe: float
    val_alpha: float


def coerce_split_timestamp(value: str, index: pd.Index) -> pd.Timestamp:
    """Return a split timestamp compatible with ``index`` timezone awareness."""
    split = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None:
        if split.tzinfo is None:
            return split.tz_localize(tz)
        return split.tz_convert(tz)
    if split.tzinfo is not None:
        return split.tz_localize(None)
    return split


class GateTrainer:
    """Trains the Gate PPO network with per-fold temperature selection.

    Phase A (select_temperature):
        Split train_df at fold.val_split_date → gate_train_df (80%) + gate_val_df (20%).
        For each temperature in config.temperature_candidates, train Gate on gate_train_df
        then measure validation Sharpe on gate_val_df. Select temperature with highest
        validation Sharpe (tiebreak on alpha).

    Phase B (train_final):
        Retrain Gate on the FULL train_df using the selected temperature.
        Save checkpoint and metadata to {fold_checkpoint}/gate/.
    """

    def __init__(
        self,
        fold: FoldConfig,
        config: WalkForwardMoEConfig,
        checkpoint_root: str = "crypto_trader/walk_forward/checkpoints/walk_forward_moe",
    ):
        self.fold = fold
        self.config = config
        self.checkpoint_root = Path(checkpoint_root)
        self.fold_checkpoint = self.checkpoint_root / fold.fold_id

        self._base_config = BaseConfig(
            seed=SeedConfig(global_seed=42),
            data=DataConfig(symbol=config.symbol, interval=config.interval),
            model=ModelConfig(
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                ent_coef=config.ent_coef,
                clip_range=config.clip_range,
                gae_lambda=config.gae_lambda,
            ),
            risk=RiskConfig(max_drawdown_limit=0.15, freeze_period_steps=1),
        )

    def _build_risk_manager(self) -> RiskManager:
        cfg = self._base_config.risk
        return RiskManager(
            max_drawdown_limit=cfg.max_drawdown_limit,
            freeze_period_steps=cfg.freeze_period_steps,
            tier1_drawdown=cfg.tier1_drawdown,
            tier1_limit=cfg.tier1_limit,
            tier2_drawdown=cfg.tier2_drawdown,
            tier2_limit=cfg.tier2_limit,
            survival_drawdown=cfg.survival_drawdown,
            survival_limit=cfg.survival_limit,
        )

    @staticmethod
    def _mask_obs(obs: np.ndarray, feature_mask: Optional[List[int]]) -> np.ndarray:
        if feature_mask is None:
            return obs
        out = np.zeros_like(obs)
        out[feature_mask] = obs[feature_mask]
        return out

    def _make_trading_env(
        self,
        df: pd.DataFrame,
        feature_mask: Optional[List[int]] = None,
        reward_profile: Optional[Dict[str, float]] = None,
    ) -> TradingEnv:
        env_kwargs = _env_kwargs_for_symbol(self.config.symbol, interval=self.config.interval)
        rm = self._build_risk_manager()
        return TradingEnv(
            df,
            risk_manager=rm,
            **env_kwargs,
            feature_mask=feature_mask,
            reward_profile=reward_profile or {},
        )

    @staticmethod
    def _compute_metrics(
        net_worths: np.ndarray, close_series: pd.Series
    ) -> Dict[str, float]:
        """Compute total_return, sharpe (annualized), alpha, benchmark_return."""
        total_return = float((net_worths[-1] - 10000.0) / 10000.0)

        bench = (close_series / close_series.iloc[0]) * 10000.0
        bench_idx = min(len(bench) - 1, len(net_worths) - 1)
        bench_ret = float((bench.iloc[bench_idx] - 10000.0) / 10000.0)
        alpha = total_return - bench_ret

        step_rets = np.diff(net_worths) / (net_worths[:-1] + 1e-10)
        if len(step_rets) > 1:
            std = float(np.std(step_rets))
            sharpe = float(np.mean(step_rets) / std * np.sqrt(252)) if std > 1e-10 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "alpha": alpha,
            "benchmark_return": bench_ret,
        }

    def _run_val_backtest(
        self,
        gate_model: PPO,
        gate_vecnorm: VecNormalize,
        experts: List[LoadedExpert],
        val_df: pd.DataFrame,
        temperature: float,
    ) -> np.ndarray:
        val_experts: List[LoadedExpert] = []
        for expert in experts:
            temp_vec = DummyVecEnv(
                [
                    lambda a=expert.artifact: self._make_trading_env(
                        val_df,
                        feature_mask=a.feature_mask,
                        reward_profile=a.reward_profile,
                    )
                ]
            )
            vn = VecNormalize.load(str(expert.artifact.vecnorm_path), temp_vec)
            vn.training = False
            vn.norm_reward = False
            val_experts.append(
                LoadedExpert(artifact=expert.artifact, model=expert.model, vecnorm=vn)
            )

        gate_temp_vec = DummyVecEnv([lambda: self._make_trading_env(val_df)])
        gate_vn = VecNormalize.load(
            str(self.fold_checkpoint / "gate" / "gate_vec_normalize.pkl"),
            gate_temp_vec,
        )
        gate_vn.training = False
        gate_vn.norm_reward = False

        main_env = DummyVecEnv([lambda: self._make_trading_env(val_df)])
        obs = main_env.reset()
        net_worths = [10000.0]

        for _ in range(len(val_df) - 1):
            gate_obs = gate_vn.normalize_obs(obs)
            logits, _ = gate_model.predict(gate_obs, deterministic=True)
            logits = np.asarray(logits, dtype=np.float32).reshape(-1)
            weights = softmax_weights(logits, temperature=temperature)

            raw_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)[0]
            expert_actions = []
            for ve in val_experts:
                masked = self._mask_obs(raw_obs, ve.artifact.feature_mask)
                norm_obs = ve.vecnorm.normalize_obs(masked.reshape(1, -1))
                action, _ = ve.model.predict(norm_obs, deterministic=True)
                expert_actions.append(float(np.asarray(action).reshape(-1)[0]))

            mixed_action = float(
                np.clip(
                    np.dot(np.asarray(expert_actions, dtype=np.float32), weights),
                    -1.0,
                    1.0,
                )
            )
            obs, _, _, info = main_env.step(
                np.array([[mixed_action]], dtype=np.float32)
            )
            net_worths.append(float(info[0]["net_worth"]))

        tmp_gate_model = self.fold_checkpoint / "gate" / "gate_model.zip"
        tmp_gate_norm = self.fold_checkpoint / "gate" / "gate_vec_normalize.pkl"
        for p in (tmp_gate_model, tmp_gate_norm):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        return np.asarray(net_worths, dtype=np.float64)

    def select_temperature(
        self,
        train_df: pd.DataFrame,
        expert_artifacts: List[GateExpertArtifact],
    ) -> tuple[float, List[CandidateResult]]:
        """Scan candidate temperatures on an 80/20 temporal split.

        Returns (best_temperature, list of per-candidate results).
        """
        val_split = coerce_split_timestamp(self.fold.val_split_date, train_df.index)
        gate_train_df = train_df[train_df.index < val_split].copy()
        gate_val_df = train_df[train_df.index >= val_split].copy()

        if len(gate_train_df) < 100:
            raise ValueError(
                f"gate_train_df too small ({len(gate_train_df)} rows) "
                f"for fold {self.fold.fold_id}"
            )
        if len(gate_val_df) < 50:
            raise ValueError(
                f"gate_val_df too small ({len(gate_val_df)} rows) "
                f"for fold {self.fold.fold_id}"
            )

        train_experts = _load_experts(expert_artifacts, gate_train_df, self._base_config)

        candidates = self.config.temperature_candidates
        results: List[CandidateResult] = []
        best_temp: float = candidates[0]
        best_sharpe: float = -float("inf")
        best_alpha: float = -float("inf")

        for temp in candidates:
            started = time.time()
            print(f"    - gate temp={temp}: train {self.config.gate_timesteps:,} steps", flush=True)
            def make_env():
                return GateRoutingEnv(
                    df=gate_train_df,
                    experts=train_experts,
                    config=self._base_config,
                    load_balance_coef=self.config.gate_load_balance_coef,
                    diversity_coef=self.config.gate_diversity_coef,
                    gate_temperature=temp,
                )

            vec_env = DummyVecEnv([make_env])
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0,
            )

            gate_model = PPO(
                "MlpPolicy", vec_env, verbose=0,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                n_steps=min(self.config.n_steps, 1024),
                batch_size=min(self.config.batch_size, 128),
                ent_coef=max(self.config.ent_coef, 0.01),
                clip_range=self.config.clip_range,
                gae_lambda=self.config.gae_lambda,
                seed=42,
            )
            gate_model.learn(total_timesteps=self.config.gate_timesteps)

            gate_dir = self.fold_checkpoint / "gate"
            gate_dir.mkdir(parents=True, exist_ok=True)
            gate_model.save(str(gate_dir / "gate_model"))
            vec_env.save(str(gate_dir / "gate_vec_normalize.pkl"))

            gate_model_reloaded = PPO.load(str(gate_dir / "gate_model"))
            net_worths = self._run_val_backtest(
                gate_model=gate_model_reloaded,
                gate_vecnorm=vec_env,
                experts=train_experts,
                val_df=gate_val_df,
                temperature=temp,
            )

            metrics = self._compute_metrics(net_worths, gate_val_df["Close"])
            result = CandidateResult(
                temperature=temp,
                val_return=metrics["total_return"],
                val_sharpe=metrics["sharpe"],
                val_alpha=metrics["alpha"],
            )
            results.append(result)
            elapsed = (time.time() - started) / 60.0
            print(
                f"    - gate temp={temp}: val_return={result.val_return:+.2%}, "
                f"val_alpha={result.val_alpha:+.2%}, sharpe={result.val_sharpe:+.2f}, "
                f"elapsed={elapsed:.1f}m",
                flush=True,
            )

            if metrics["sharpe"] > best_sharpe or (
                abs(metrics["sharpe"] - best_sharpe) < 1e-8
                and metrics["alpha"] > best_alpha
            ):
                best_sharpe = metrics["sharpe"]
                best_alpha = metrics["alpha"]
                best_temp = temp

        return best_temp, results

    def train_final(
        self,
        train_df: pd.DataFrame,
        expert_artifacts: List[GateExpertArtifact],
        best_temp: float,
        candidate_results: List[CandidateResult],
    ) -> str:
        """Retrain Gate on full *train_df* with *best_temp*, save checkpoint.

        Returns path to the saved gate checkpoint directory.
        """
        gate_dir = self.fold_checkpoint / "gate"
        gate_dir.mkdir(parents=True, exist_ok=True)

        experts = _load_experts(expert_artifacts, train_df, self._base_config)

        def make_env():
            return GateRoutingEnv(
                df=train_df,
                experts=experts,
                config=self._base_config,
                load_balance_coef=self.config.gate_load_balance_coef,
                diversity_coef=self.config.gate_diversity_coef,
                gate_temperature=best_temp,
            )

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
        )

        gate_model = PPO(
            "MlpPolicy", vec_env, verbose=0,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            n_steps=min(self.config.n_steps, 1024),
            batch_size=min(self.config.batch_size, 128),
            ent_coef=max(self.config.ent_coef, 0.01),
            clip_range=self.config.clip_range,
            gae_lambda=self.config.gae_lambda,
            seed=42,
        )
        print(f"    - final gate: train {self.config.gate_timesteps:,} steps at temp={best_temp}", flush=True)
        gate_model.learn(total_timesteps=self.config.gate_timesteps)

        gate_model.save(str(gate_dir / "gate_model"))
        vec_env.save(str(gate_dir / "gate_vec_normalize.pkl"))

        raw_env = getattr(getattr(vec_env, "venv", vec_env), "envs", [None])[0]
        k = len(experts)
        usage_list = (
            raw_env.usage_ema.tolist()
            if raw_env is not None
            else [1.0 / k] * k
        )
        usage_ema = {
            e.artifact.expert_id: float(w)
            for e, w in zip(experts, usage_list)
        }

        metadata = {
            "selected_temperature": best_temp,
            "candidate_results": [
                {
                    "temp": r.temperature,
                    "val_sharpe": r.val_sharpe,
                    "val_return": r.val_return,
                    "val_alpha": r.val_alpha,
                }
                for r in candidate_results
            ],
            "usage_ema": usage_ema,
            "fold_id": self.fold.fold_id,
            "gate_timesteps": self.config.gate_timesteps,
        }

        with open(gate_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return str(gate_dir)
