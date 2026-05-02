"""
Expert trainer for walk-forward MoE Stage 1.
Trains 4 experts (E5, E2, E4, E7) on a single fold's training data.

Uses regime masks computed from training data only, with frozen reward profiles
and training controls from the existing pipeline.

All expert models are saved to: {fold_checkpoint_dir}/experts/{expert_id}/
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_trader.asset_profile import get_asset_profile
from crypto_trader.config import BaseConfig, get_default_config, load_config
from crypto_trader.envs.trading_env import TradingEnv
from crypto_trader.moe.manifest import (
    FEATURE_MASKS,
    ExpertSpec,
    load_expert_manifest,
    resolve_feature_mask,
)
from crypto_trader.risk_manager import RiskManager
from crypto_trader.train_moe_stage1 import (
    TrainingSpec,
    _algo_kwargs,
    _build_regime_mask,
    _training_controls_for_expert,
    get_algo_registry,
)

logger = logging.getLogger(__name__)

WF_EXPERT_IDS = [
    "E5_PPO_lowvol_carry",
    "E2_PPO_bear_drawdown",
    "E4_PPO_highvol_risk",
    "E7_SAC_fast_adapt",
]


class ExpertTrainer:
    """Trains 4 MoE experts on a single fold's training data.

    Uses regime masks computed from training data only, with frozen reward
    profiles and training controls from the existing pipeline.  Expert
    definitions are loaded from ``moe_experts.yaml`` and filtered to the
    walk-forward subset.

    Parameters
    ----------
    config : BaseConfig
        Frozen hyper-parameter source (model, risk, data configs).
    fold_config : Any
        Object with at minimum ``fold_id`` (str), ``train_start`` (str),
        ``train_end`` (str) attributes.  Used for metadata / logging.
    train_df : pd.DataFrame
        Training data for this fold (must include ``Signal_Proba`` column).
    manifest_path : Path | None
        Override for the manifest YAML path (default:
        ``crypto_trader/configs/moe_experts.yaml``).
    """

    def __init__(
        self,
        config: BaseConfig,
        fold_config: Any,
        train_df: pd.DataFrame,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.fold_config = fold_config
        self.train_df = train_df
        self._manifest = None
        self._manifest_path = manifest_path

    def train_all(self, fold_checkpoint_dir: Path) -> List[str]:
        """Train all 4 experts and save to *fold_checkpoint_dir*.

        Model files:
          ``{fold_checkpoint_dir}/experts/{expert_id}/model.zip``
          ``{fold_checkpoint_dir}/experts/{expert_id}/vec_normalize.pkl``

        Returns
        -------
        List[str]
            Expert IDs that were trained successfully.
        """
        fold_checkpoint_dir = Path(fold_checkpoint_dir)
        experts_dir = fold_checkpoint_dir / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

        algo_registry = get_algo_registry(load_classes=True)
        env_cfg = get_asset_profile(
            self.config.symbol, interval=self.config.interval
        ).env

        specs = self._get_expert_specs()
        workers = int(getattr(self.config, "expert_parallel_workers", 1) or 1)
        workers = max(1, min(workers, len(specs)))

        if workers == 1:
            trained: List[str] = []
            for spec in specs:
                print(f"    - {spec.expert_id}: start {spec.timesteps:,} steps", flush=True)
                logger.info("Training expert: %s", spec.expert_id)
                expert_dir = experts_dir / spec.expert_id
                expert_dir.mkdir(parents=True, exist_ok=True)

                self._train_one(spec, algo_registry, env_cfg, expert_dir)
                trained.append(spec.expert_id)
                print(f"    - {spec.expert_id}: done", flush=True)
                logger.info("✓ Expert %s trained (%.0f steps)", spec.expert_id, spec.timesteps)

            return trained

        print(f"    Parallel expert training: {workers} workers", flush=True)
        trained_by_id: Dict[str, str] = {}
        fold_meta = {
            "fold_id": getattr(self.fold_config, "fold_id", "unknown"),
            "train_start": getattr(self.fold_config, "train_start", ""),
            "train_end": getattr(self.fold_config, "train_end", ""),
        }
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for spec in specs:
                expert_dir = experts_dir / spec.expert_id
                expert_dir.mkdir(parents=True, exist_ok=True)
                print(f"    - {spec.expert_id}: queued {spec.timesteps:,} steps", flush=True)
                fut = pool.submit(
                    _train_expert_worker,
                    spec,
                    self.train_df,
                    self.config,
                    str(expert_dir),
                    fold_meta,
                )
                futures[fut] = spec.expert_id

            for fut in as_completed(futures):
                expert_id = futures[fut]
                trained_id = fut.result()
                trained_by_id[expert_id] = trained_id
                print(f"    - {trained_id}: done", flush=True)

        return [s.expert_id for s in specs if s.expert_id in trained_by_id]

    def _train_one(
        self,
        spec: TrainingSpec,
        algo_registry: Dict[str, Any],
        env_cfg: Any,
        output_dir: Path,
    ) -> None:
        algo_cls = algo_registry[spec.algorithm]
        regime_mask = _build_regime_mask(self.train_df, spec.data_slice)
        controls = _training_controls_for_expert(spec)
        short_squeeze_threshold = controls["short_squeeze_threshold"]
        if short_squeeze_threshold < 0:
            short_squeeze_threshold = None

        rm = RiskManager()

        def make_env():
            return TradingEnv(
                self.train_df,
                risk_manager=rm,
                atr_floor=env_cfg.atr_floor,
                vol_scale_min=env_cfg.vol_scale_min,
                vol_scale_max=env_cfg.vol_scale_max,
                symbol=self.config.symbol,
                target_atr_pct=env_cfg.target_atr_pct,
                tau=env_cfg.tau,
                delta_max=env_cfg.delta_max,
                cooldown_n=env_cfg.cooldown_n,
                k_single=env_cfg.k_single,
                funding_daily=env_cfg.funding_daily,
                feature_mask=spec.feature_mask,
                reward_profile=spec.reward_profile,
                regime_mask=regime_mask,
                regime_main_reward_weight=controls["regime_main_reward_weight"],
                regime_off_reward_weight=controls["regime_off_reward_weight"],
                sat_penalty_coef=controls["sat_penalty_coef"],
                sat_threshold=controls["sat_threshold"],
                directional_bias_coef=controls["directional_bias_coef"],
                directional_bias_alpha=controls["directional_bias_alpha"],
                action_cap=controls["action_cap"],
                funding_cost_multiplier=controls["funding_cost_multiplier"],
                short_squeeze_threshold=short_squeeze_threshold,
                short_squeeze_max_short=controls["short_squeeze_max_short"],
            )

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
        )

        model = algo_cls(
            "MlpPolicy", vec_env, verbose=0,
            **_algo_kwargs(spec.algorithm, self.config, spec.seed),
        )
        model.learn(total_timesteps=spec.timesteps)

        model.save(str(output_dir / "model"))
        vec_env.save(str(output_dir / "vec_normalize.pkl"))

        payload = asdict(spec)
        payload["training_controls"] = controls
        payload["regime_coverage"] = float(np.mean(regime_mask > 0.5))
        payload["fold_id"] = getattr(self.fold_config, "fold_id", "unknown")
        payload["train_start"] = getattr(self.fold_config, "train_start", "")
        payload["train_end"] = getattr(self.fold_config, "train_end", "")

        with open(output_dir / "spec.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _get_expert_specs(self) -> List[TrainingSpec]:
        manifest = self._load_manifest()
        specs: List[TrainingSpec] = []
        for expert in manifest.experts:
            if expert.expert_id not in WF_EXPERT_IDS:
                continue
            timesteps = int(
                getattr(self.config, "expert_timesteps", {}).get(
                    expert.expert_id, expert.timesteps
                )
            )
            specs.append(
                TrainingSpec(
                    expert_id=expert.expert_id,
                    algorithm=expert.algorithm,
                    seed=expert.seed,
                    data_slice=expert.data_slice,
                    feature_mask=expert.feature_mask,
                    reward_profile=expert.reward_profile or {},
                    timesteps=timesteps,
                    output_dir="",
                )
            )
        specs.sort(key=lambda s: WF_EXPERT_IDS.index(s.expert_id))
        return specs

    def _load_manifest(self):
        if self._manifest is None:
            path = self._manifest_path or Path("crypto_trader/configs/moe_experts.yaml")
            self._manifest = load_expert_manifest(path)
        return self._manifest


def _train_expert_worker(
    spec: TrainingSpec,
    train_df: pd.DataFrame,
    config: BaseConfig,
    output_dir: str,
    fold_meta: Dict[str, str],
) -> str:
    """Process-pool entrypoint for one expert.

    Each child process limits BLAS/OpenMP/PyTorch threads so four experts can
    run concurrently without oversubscribing the CPU.
    """
    thread_count = str(int(getattr(config, "torch_num_threads", 1) or 1))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = thread_count
    try:
        import torch

        torch.set_num_threads(int(thread_count))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    algo_registry = get_algo_registry(load_classes=True)
    env_cfg = get_asset_profile(config.symbol, interval=config.interval).env

    class _Fold:
        fold_id = fold_meta.get("fold_id", "unknown")
        train_start = fold_meta.get("train_start", "")
        train_end = fold_meta.get("train_end", "")

    trainer = ExpertTrainer(config, _Fold(), train_df)
    trainer._train_one(spec, algo_registry, env_cfg, Path(output_dir))
    return spec.expert_id
