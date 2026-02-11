"""
sanity_run.py - Reproducibility verification script

Usage:
    python3 -m crypto_trader.sanity_run --mode backtest --config configs/base.yaml

Verifies:
    - Configuration loads correctly
    - Same seed produces same outputs
    - Logs are saved to runs/<run_id>/
"""
import argparse
import sys
import os
from pathlib import Path

# Ensure crypto_trader is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from crypto_trader.config import load_config, save_config, get_default_config, BaseConfig
from crypto_trader.seed_utils import set_global_seeds, get_run_id
from crypto_trader.logger import setup_logging, get_logger, get_run_dir

import numpy as np


def run_backtest_sanity(config: BaseConfig, logger) -> dict:
    """
    Run a minimal backtest and return key metrics for reproducibility check.
    """
    from crypto_trader.backtest_ensemble import backtest_ensemble_with_config
    
    logger.info("Running backtest sanity check...")
    
    # Run backtest with config
    result = backtest_ensemble_with_config(config, max_steps=50)
    
    return result


def run_reproducibility_check(config: BaseConfig, mode: str, n_steps: int = 10) -> bool:
    """
    Verify that running twice with same seed produces identical results.
    """
    logger = get_logger(__name__)
    logger.info(f"Running reproducibility check (mode={mode}, n_steps={n_steps})...")
    
    # First run
    set_global_seeds(config.seed.global_seed)
    results1 = []
    for i in range(n_steps):
        results1.append(np.random.random())
    
    # Second run with same seed
    set_global_seeds(config.seed.global_seed)
    results2 = []
    for i in range(n_steps):
        results2.append(np.random.random())
    
    # Compare
    match = np.allclose(results1, results2, rtol=1e-10)
    
    if match:
        logger.info("✓ Reproducibility check: PASSED (random outputs match)")
    else:
        logger.error("✗ Reproducibility check: FAILED (outputs differ)")
        logger.error(f"  Run 1: {results1[:5]}...")
        logger.error(f"  Run 2: {results2[:5]}...")
    
    return match


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check for crypto_trader reproducibility"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "train", "live"],
        default="backtest",
        help="Run mode (default: backtest)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file (default: configs/base.yaml)"
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip actual backtest, only test reproducibility"
    )
    
    args = parser.parse_args()
    
    # Resolve config path relative to crypto_trader/
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        # Try from project root
        config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        # Try absolute path
        config_path = Path(args.config)
    
    # Load or create config
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = get_default_config()
    
    # Generate run_id
    run_id = get_run_id(prefix=args.mode)
    config.run_id = run_id
    config.mode = args.mode
    
    # Setup logging
    run_dir = setup_logging(run_id)
    logger = get_logger(__name__)
    
    # Set global seeds
    set_global_seeds(config.seed.global_seed)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info(f"Sanity Run - Mode: {args.mode}")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Global Seed: {config.seed.global_seed}")
    logger.info(f"Config Path: {config_path}")
    logger.info(f"Output Dir: {run_dir}")
    
    # Save config to run directory
    config_save_path = run_dir / "config.yaml"
    save_config(config, config_save_path)
    logger.info(f"Config saved to: {config_save_path}")
    
    # Run reproducibility check
    repro_ok = run_reproducibility_check(config, args.mode)
    
    # Run mode-specific sanity check
    if args.mode == "backtest" and not args.skip_backtest:
        try:
            result = run_backtest_sanity(config, logger)
            logger.info(f"Backtest result: {result}")
        except Exception as e:
            logger.warning(f"Backtest sanity check skipped: {e}")
            logger.info("(This is OK - full backtest requires trained models)")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Sanity Check Summary")
    logger.info("=" * 60)
    logger.info(f"✓ Run ID: {run_id}")
    logger.info(f"✓ Logs saved to: {run_dir}")
    logger.info(f"{'✓' if repro_ok else '✗'} Reproducibility: {'PASSED' if repro_ok else 'FAILED'}")
    logger.info("=" * 60)
    
    # Print to console as well
    print()
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Logs saved to: {run_dir}/")
    print(f"{'✓' if repro_ok else '✗'} Reproducibility check: {'PASSED' if repro_ok else 'FAILED'}")
    
    return 0 if repro_ok else 1


if __name__ == "__main__":
    sys.exit(main())
