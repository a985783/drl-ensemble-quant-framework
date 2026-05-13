"""
rollout_controller.py - Gradual Rollout & Model Registry

Provides:
- Rollout level management (0.25 -> 0.5 -> 1.0)
- Stable model registry (tracking stable vs candidate models)
- Automatic rollback when KPIs not met
- Daily summary generation

Design Principle:
    New models start at low rollout, gradually increase if KPIs pass.
    Never force liquidation - only adjust future position sizing.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
import csv

# Constants
REGISTRY_FILE = Path(__file__).parent.parent / "stable_model_registry.json"
SUMMARY_DIR = Path(__file__).parent.parent / "daily_summaries"

# Rollout levels
ROLLOUT_LEVELS = [0.25, 0.5, 1.0]
MIN_DAYS_PER_LEVEL = 3
MIN_TRADES_PER_LEVEL = 6  # ~6 trades = 1 day of 4H bars

# KPI thresholds
MAX_SLIPPAGE_THRESHOLD = 0.005  # 0.5%
MIN_FILL_RATE = 0.9  # 90% of orders should fill
MAX_RECONCILE_ERRORS = 2  # Max reconciliation errors before demote


@dataclass
class RolloutConfig:
    """Configuration for rollout behavior."""
    levels: List[float] = field(default_factory=lambda: ROLLOUT_LEVELS)
    min_days_per_level: int = MIN_DAYS_PER_LEVEL
    min_trades_per_level: int = MIN_TRADES_PER_LEVEL
    max_slippage: float = MAX_SLIPPAGE_THRESHOLD
    min_fill_rate: float = MIN_FILL_RATE
    max_reconcile_errors: int = MAX_RECONCILE_ERRORS


@dataclass
class ModelRegistry:
    """Registry tracking stable and candidate models."""
    # Stable model (currently in production)
    stable_run_id: Optional[str] = None
    stable_model_path: str = "checkpoints/moe/stable"
    
    # Candidate model (being tested)
    candidate_run_id: Optional[str] = None
    candidate_model_path: Optional[str] = None
    
    # Rollout state
    rollout_level: float = 1.0  # 1.0 = full rollout (stable only)
    rollout_level_index: int = -1  # -1 = stable, 0/1/2 = levels
    rollout_started_at: Optional[float] = None
    rollout_trades_count: int = 0
    
    # KPI tracking
    total_trades: int = 0
    filled_trades: int = 0
    total_slippage: float = 0.0
    reconcile_errors: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


def load_registry() -> ModelRegistry:
    """Load model registry from file."""
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, 'r') as f:
                data = json.load(f)
            return ModelRegistry(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARN] Failed to load registry: {e}")
    return ModelRegistry()


def save_registry(registry: ModelRegistry) -> bool:
    """Save model registry to file."""
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save registry: {e}")
        return False


def get_rollout_level(registry: Optional[ModelRegistry] = None) -> float:
    """Get current rollout level multiplier."""
    if registry is None:
        registry = load_registry()
    return registry.rollout_level


def get_active_model_path(registry: Optional[ModelRegistry] = None) -> str:
    """Get path to currently active model (candidate or stable)."""
    if registry is None:
        registry = load_registry()
    
    # If we have a candidate and rollout is in progress
    if registry.candidate_model_path and registry.rollout_level_index >= 0:
        return registry.candidate_model_path
    
    return registry.stable_model_path


def start_rollout(candidate_run_id: str, candidate_model_path: str) -> ModelRegistry:
    """
    Start rollout of a new candidate model.
    
    Args:
        candidate_run_id: Run ID of the new model
        candidate_model_path: Path to the new model checkpoint
    
    Returns:
        Updated registry
    """
    registry = load_registry()
    
    registry.candidate_run_id = candidate_run_id
    registry.candidate_model_path = candidate_model_path
    registry.rollout_level_index = 0
    registry.rollout_level = ROLLOUT_LEVELS[0]
    registry.rollout_started_at = time.time()
    registry.rollout_trades_count = 0
    
    # Reset KPI counters for new rollout
    registry.total_trades = 0
    registry.filled_trades = 0
    registry.total_slippage = 0.0
    registry.reconcile_errors = 0
    
    save_registry(registry)
    print(f"[Rollout] Started rollout for {candidate_run_id} at level {registry.rollout_level}")
    
    return registry


def record_trade(
    filled: bool,
    slippage: float,
    reconcile_ok: bool,
    registry: Optional[ModelRegistry] = None
) -> ModelRegistry:
    """
    Record a trade execution for KPI tracking.
    
    Args:
        filled: Whether the order was filled
        slippage: Actual slippage (0.001 = 0.1%)
        reconcile_ok: Whether reconciliation passed
    """
    if registry is None:
        registry = load_registry()
    
    registry.total_trades += 1
    registry.rollout_trades_count += 1
    
    if filled:
        registry.filled_trades += 1
    registry.total_slippage += abs(slippage)
    
    if not reconcile_ok:
        registry.reconcile_errors += 1
    
    save_registry(registry)
    return registry


def check_kpis(registry: ModelRegistry, config: RolloutConfig = None) -> tuple[bool, str]:
    """
    Check if current rollout meets KPI thresholds.
    
    Returns:
        (passed, reason)
    """
    if config is None:
        config = RolloutConfig()
    
    if registry.total_trades == 0:
        return True, "no_trades_yet"
    
    # Fill rate
    fill_rate = registry.filled_trades / registry.total_trades
    if fill_rate < config.min_fill_rate:
        return False, f"fill_rate_low: {fill_rate:.2%} < {config.min_fill_rate:.2%}"
    
    # Average slippage
    avg_slippage = registry.total_slippage / registry.total_trades
    if avg_slippage > config.max_slippage:
        return False, f"slippage_high: {avg_slippage:.4f} > {config.max_slippage}"
    
    # Reconciliation errors
    if registry.reconcile_errors > config.max_reconcile_errors:
        return False, f"reconcile_errors: {registry.reconcile_errors} > {config.max_reconcile_errors}"
    
    return True, "kpis_ok"


def maybe_promote(registry: Optional[ModelRegistry] = None, 
                  config: RolloutConfig = None) -> ModelRegistry:
    """
    Check if rollout should be promoted to next level.
    
    Conditions:
    - Minimum trades executed
    - Minimum days passed
    - KPIs met
    """
    if registry is None:
        registry = load_registry()
    if config is None:
        config = RolloutConfig()
    
    # Not in rollout mode
    if registry.rollout_level_index < 0:
        return registry
    
    # Already at max level
    if registry.rollout_level_index >= len(config.levels) - 1:
        return registry
    
    # Check minimum trades
    if registry.rollout_trades_count < config.min_trades_per_level:
        return registry
    
    # Check minimum days
    if registry.rollout_started_at:
        days_elapsed = (time.time() - registry.rollout_started_at) / 86400
        if days_elapsed < config.min_days_per_level:
            return registry
    
    # Check KPIs
    passed, reason = check_kpis(registry, config)
    if not passed:
        print(f"[Rollout] Cannot promote: {reason}")
        return registry
    
    # Promote to next level
    registry.rollout_level_index += 1
    registry.rollout_level = config.levels[registry.rollout_level_index]
    registry.rollout_trades_count = 0  # Reset for next level
    
    save_registry(registry)
    print(f"[Rollout] Promoted to level {registry.rollout_level}")
    
    # Check if fully promoted
    if registry.rollout_level >= 1.0:
        finalize_rollout(registry)
    
    return registry


def demote_rollout(reason: str, registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """
    Demote rollout (reduce level or rollback to stable).
    """
    if registry is None:
        registry = load_registry()
    
    if registry.rollout_level_index <= 0:
        # Already at lowest level or stable, rollback completely
        return rollback_to_stable(reason, registry)
    
    # Demote one level
    registry.rollout_level_index -= 1
    registry.rollout_level = ROLLOUT_LEVELS[registry.rollout_level_index]
    registry.rollout_trades_count = 0
    
    save_registry(registry)
    print(f"[Rollout] Demoted to level {registry.rollout_level}: {reason}")
    
    return registry


def rollback_to_stable(reason: str, registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """
    Rollback to stable model completely.
    
    Does NOT force liquidation - only affects future decisions.
    """
    if registry is None:
        registry = load_registry()
    
    print(f"[Rollout] Rolling back to stable model: {reason}")
    
    # Clear candidate
    registry.candidate_run_id = None
    registry.candidate_model_path = None
    registry.rollout_level_index = -1
    registry.rollout_level = 1.0  # Full weight on stable
    registry.rollout_started_at = None
    registry.rollout_trades_count = 0
    
    save_registry(registry)
    return registry


def finalize_rollout(registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """
    Finalize rollout: candidate becomes new stable.
    """
    if registry is None:
        registry = load_registry()
    
    if not registry.candidate_run_id:
        return registry
    
    print(f"[Rollout] Finalizing: {registry.candidate_run_id} is now stable")
    
    # Candidate becomes stable
    registry.stable_run_id = registry.candidate_run_id
    registry.stable_model_path = registry.candidate_model_path
    
    # Clear candidate state
    registry.candidate_run_id = None
    registry.candidate_model_path = None
    registry.rollout_level_index = -1
    registry.rollout_level = 1.0
    registry.rollout_started_at = None
    
    save_registry(registry)
    return registry


def generate_daily_summary(trade_logs_path: str = None) -> Dict[str, Any]:
    """
    Generate daily summary from trade logs.
    
    Returns dict with:
    - rollout_level
    - total_trades, fill_rate, avg_slippage
    - safe_mode_count
    - reconcile_errors
    """
    registry = load_registry()
    
    summary = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": time.time(),
        "rollout_level": registry.rollout_level,
        "rollout_level_index": registry.rollout_level_index,
        "active_model": registry.candidate_run_id or registry.stable_run_id,
        "total_trades": registry.total_trades,
        "filled_trades": registry.filled_trades,
        "fill_rate": registry.filled_trades / max(1, registry.total_trades),
        "avg_slippage": registry.total_slippage / max(1, registry.total_trades),
        "reconcile_errors": registry.reconcile_errors,
        "safe_mode_count": 0  # TODO: track from execution_safety
    }
    
    # Save summary
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = SUMMARY_DIR / f"daily_summary_{summary['date']}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Summary] Saved to {summary_file}")
    return summary


def get_status() -> Dict[str, Any]:
    """Get current rollout status."""
    registry = load_registry()
    passed, reason = check_kpis(registry)
    
    return {
        "rollout_level": registry.rollout_level,
        "rollout_level_index": registry.rollout_level_index,
        "candidate": registry.candidate_run_id,
        "stable": registry.stable_run_id,
        "trades_at_level": registry.rollout_trades_count,
        "kpis_passed": passed,
        "kpis_reason": reason
    }
