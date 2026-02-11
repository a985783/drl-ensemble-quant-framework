"""
config.py - Unified configuration with type checking

Provides:
- BaseConfig, TrainConfig, BacktestConfig, LiveConfig dataclasses
- load_config(path): Load config from YAML file
- save_config(config, path): Save config to YAML file
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
import os


@dataclass
class SeedConfig:
    """Random seed configuration."""
    global_seed: int = 42
    ensemble_seeds: List[int] = field(default_factory=lambda: [
        42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
    ])


@dataclass
class DataConfig:
    """Data loading configuration."""
    symbol: str = "ETH/USDT:USDT"
    interval: str = "1d"
    train_start: str = "2020-01-01"
    train_end: str = "2025-12-15"
    train_split_ratio: float = 0.8


@dataclass
class ModelConfig:
    """PPO model hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.995
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.005
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    total_timesteps: int = 150000


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown_limit: float = 0.15
    freeze_period_steps: int = 1
    # Tiered risk management thresholds
    tier1_drawdown: float = 0.05
    tier1_limit: float = 0.8
    tier2_drawdown: float = 0.10
    tier2_limit: float = 0.5
    # Volatility scaling guards
    atr_floor: float = 0.005
    vol_scale_min: float = 0.1
    vol_scale_max: float = 2.0
    # Slippage guard (risk-on only)
    max_slippage_risk_on: float = 0.03


@dataclass
class PathConfig:
    """Path configuration."""
    checkpoints_dir: str = "checkpoints/ensemble"
    results_dir: str = "results"
    runs_dir: str = "runs"


@dataclass
class BaseConfig:
    """Complete configuration."""
    seed: SeedConfig = field(default_factory=SeedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Run metadata
    run_id: Optional[str] = None
    mode: str = "backtest"  # backtest, train, live


def load_config(path: Union[str, Path]) -> BaseConfig:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        BaseConfig instance
    """
    import yaml
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # Build config from nested dict
    config = BaseConfig(
        seed=SeedConfig(**data.get("seed", {})),
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        risk=RiskConfig(**data.get("risk", {})),
        paths=PathConfig(**data.get("paths", {})),
        run_id=data.get("run_id"),
        mode=data.get("mode", "backtest")
    )
    
    return config


def save_config(config: BaseConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: BaseConfig instance
        path: Path to save YAML file
    """
    import yaml
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = asdict(config)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_default_config() -> BaseConfig:
    """Get default configuration."""
    return BaseConfig()
