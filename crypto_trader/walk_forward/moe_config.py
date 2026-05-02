"""
Walk-Forward MoE configuration dataclasses.

5-fold anchored MoE walk-forward validation config:
  - All folds share the same train_start (2020-01-01)
  - Each fold extends train_end by 1 year
  - Test period is the year immediately following train_end
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FoldSpec:
    """A single walk-forward fold with anchored training start."""

    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class ExpertSpec:
    """Metadata for a single MoE expert in walk-forward training."""

    expert_id: str
    algorithm: str = "PPO"
    timesteps: int = 150_000


@dataclass
class GateSearchSpec:
    """Temperature grid-search configuration for the gate network."""

    temperature_candidates: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.68, 0.8, 1.0, 1.5, 2.0])
    gate_val_ratio: float = 0.2


@dataclass
class PassCriteria:
    """Criteria for a walk-forward fold result to be considered passing."""

    min_fold_alpha: float = 0.0
    target_avg_alpha: float = 0.20


class _ModelConfigProxy:
    """Compatibility wrapper for existing PPO training code."""
    def __init__(self, cfg):
        self.learning_rate = cfg.learning_rate
        self.gamma = cfg.gamma
        self.n_steps = cfg.n_steps
        self.batch_size = cfg.batch_size
        self.ent_coef = cfg.ent_coef
        self.clip_range = cfg.clip_range
        self.gae_lambda = cfg.gae_lambda


class _DataConfigProxy:
    """Compatibility wrapper for existing data config code."""
    def __init__(self, cfg):
        self.symbol = cfg.symbol
        self.interval = cfg.interval
        self.train_split_ratio = 0.8


class _RiskConfigProxy:
    """Compatibility wrapper with RiskManager defaults."""
    max_drawdown_limit: float = 0.15
    freeze_period_steps: int = 1
    tier1_drawdown: float = 0.05
    tier1_limit: float = 0.8
    tier2_drawdown: float = 0.10
    tier2_limit: float = 0.5
    survival_drawdown: float = 0.15
    survival_limit: float = 0.2


@dataclass
class WalkForwardMoEConfig:
    """Top-level configuration for 5-fold anchored MoE walk-forward validation."""

    # --- Instrument ---
    symbol: str = "ETH/USDT:USDT"
    interval: str = "1d"

    # --- 5 anchored folds (all start from 2020-01-01) ---
    folds: List[FoldSpec] = field(default_factory=lambda: [
        FoldSpec("fold_1", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        FoldSpec("fold_2", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        FoldSpec("fold_3", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        FoldSpec("fold_4", "2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        FoldSpec("fold_5", "2020-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
    ])

    # --- 4 experts (subset of the full 8) ---
    expert_ids: List[str] = field(default_factory=lambda: [
        "E2_PPO_bear_drawdown",
        "E4_PPO_highvol_risk",
        "E5_PPO_lowvol_carry",
        "E7_SAC_fast_adapt",
    ])

    # --- Per-expert frozen training timesteps ---
    expert_timesteps: Dict[str, int] = field(default_factory=lambda: {
        "E2_PPO_bear_drawdown": 150_000,
        "E4_PPO_highvol_risk": 150_000,
        "E5_PPO_lowvol_carry": 150_000,
        "E7_SAC_fast_adapt": 180_000,
    })
    gate_timesteps: int = 120_000
    expert_parallel_workers: int = 4
    torch_num_threads: int = 1

    # --- PPO defaults (frozen across all folds) ---
    learning_rate: float = 3e-4
    gamma: float = 0.995
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.005
    clip_range: float = 0.2
    gae_lambda: float = 0.95

    # --- Compatibility wrappers for existing training code ---
    @property
    def model(self):
        """ModelConfig-compatible wrapper for PPO hyperparameters."""
        return _ModelConfigProxy(self)

    @property
    def data(self):
        """DataConfig-compatible wrapper."""
        return _DataConfigProxy(self)

    @property
    def risk(self):
        """RiskConfig-compatible wrapper."""
        return _RiskConfigProxy()

    # --- Gate routing defaults ---
    gate_load_balance_coef: float = 0.02
    gate_diversity_coef: float = 0.01

    # --- Temperature search ---
    temperature_candidates: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.68, 0.8, 1.0, 1.5, 2.0])
    gate_val_ratio: float = 0.2

    # --- Pass criteria ---
    min_fold_alpha: float = 0.0
    target_avg_alpha: float = 0.20

    # --- Paths ---
    manifest_path: str = "crypto_trader/configs/moe_experts.yaml"
    checkpoint_root: str = "crypto_trader/walk_forward/checkpoints/walk_forward_moe"
    results_root: str = "crypto_trader/walk_forward/results/walk_forward_moe"

    # --- Execution ---
    execution_mode: str = "next_bar"
