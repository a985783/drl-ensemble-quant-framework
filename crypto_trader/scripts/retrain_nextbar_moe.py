"""
retrain_nextbar_moe.py

Retrain MoE pipeline from scratch under next-bar execution.
Trains 4 candidate experts (E2, E4, E5, E7) with next-bar-shifted data,
then gate, optional joint finetune, and OOS backtest.

Stage 1: Train 4 experts on next-bar-shifted train80 data with regime slices
Stage 2: Train gate routing between the 4 experts
Stage 3: (Optional) Alternating joint finetune
OOS Backtest: Evaluate on oos20 data with next_bar execution

Output:
  checkpoints/moe/candidate/experts/<expert_id>/model.zip + vec_normalize.pkl
  checkpoints/moe/candidate/gate/gate_model.zip + gate_vec_normalize.pkl
  checkpoints/moe/candidate/candidate_manifest.json
  results/candidates/retrained/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import contextmanager

import crypto_trader.asset_profile as asset_profile_mod
from crypto_trader.backtest_moe import (
    _env_kwargs_for_symbol,
    resolve_execution_frame,
)
from crypto_trader.config import BaseConfig, get_default_config
from crypto_trader.moe.manifest import FEATURE_MASKS, load_expert_manifest

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CANDIDATE_MANIFEST_PATH = _PROJECT_ROOT / "configs" / "moe_experts_candidate.yaml"
TRAIN_CSV = _PROJECT_ROOT / "data_moe_20200101_20260216_train80.csv"
OOS_CSV = _PROJECT_ROOT / "data_moe_20200101_20260216_oos20.csv"

STAGE1_ROOT = _PROJECT_ROOT.parent / "checkpoints" / "moe" / "candidate" / "experts"
STAGE2_ROOT = _PROJECT_ROOT.parent / "checkpoints" / "moe" / "candidate" / "gate"
STAGE3_ROOT = _PROJECT_ROOT.parent / "checkpoints" / "moe" / "candidate" / "stage3"
RESULTS_DIR = _PROJECT_ROOT.parent / "results" / "candidates" / "retrained"

SYMBOL = "ETH/USDT:USDT"
INTERVAL = "1d"
GATE_TEMPERATURE = 0.68
TAU_OVERRIDE = 0.12


# ──────────────────────────────────────────────────────────────────────────────
# Tau override for training consistency
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _tau_override(tau: float):
    """Temporarily override AssetProfile tau so training matches evaluation."""
    original_get = asset_profile_mod.get_asset_profile

    def patched_get_asset_profile(symbol, interval="1d"):
        profile = original_get(symbol, interval=interval)
        return asset_profile_mod.AssetProfile(
            key=profile.key,
            feature=profile.feature,
            env=asset_profile_mod.EnvProfile(
                atr_floor=profile.env.atr_floor,
                vol_scale_min=profile.env.vol_scale_min,
                vol_scale_max=profile.env.vol_scale_max,
                target_atr_pct=profile.env.target_atr_pct,
                tau=tau,
                delta_max=profile.env.delta_max,
                cooldown_n=profile.env.cooldown_n,
                k_single=profile.env.k_single,
                funding_daily=profile.env.funding_daily,
            ),
        )

    asset_profile_mod.get_asset_profile = patched_get_asset_profile
    try:
        yield
    finally:
        asset_profile_mod.get_asset_profile = original_get


# ──────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_and_shift(csv_path: Path) -> pd.DataFrame:
    """Load CSV, set datetime index, apply next_bar OHLCV shift."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    shifted, meta = resolve_execution_frame(df, execution_mode="next_bar")
    assert meta["execution_mode"] == "next_bar", "Execution mode mismatch"
    print(f"[Data] Loaded {len(shifted)} rows from {csv_path.name} (next_bar shift, dropped {meta['dropped_rows']})")
    return shifted


def save_shifted_data(df: pd.DataFrame, out_path: Path) -> Path:
    """Save shifted dataframe to CSV for consumption by existing training pipeline."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"[Data] Saved shifted data ({len(df)} rows) → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Expert Training
# ──────────────────────────────────────────────────────────────────────────────

def run_stage1_nextbar(
    shifted_train_csv: Path,
    output_root: str,
    config: BaseConfig,
) -> None:
    """Train all 4 experts using the existing stage1 pipeline with next-bar data."""
    from crypto_trader.train_moe_stage1 import run_stage1

    print("\n" + "=" * 70)
    print("STAGE 1: Expert Pre-training (next-bar execution)")
    print("=" * 70)

    specs = run_stage1(
        manifest_path=CANDIDATE_MANIFEST_PATH,
        dry_run=False,
        output_root=output_root,
        train_data_path=str(shifted_train_csv),
        symbol=SYMBOL,
    )
    print(f"\n[Stage1] Completed. {len(specs)} experts trained → {output_root}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Gate Training
# ──────────────────────────────────────────────────────────────────────────────

def run_stage2_nextbar(
    shifted_train_csv: Path,
    stage1_root: str,
    output_dir: str,
    config: BaseConfig,
) -> None:
    """Train gate on next-bar data routing between the 4 experts."""
    from crypto_trader.train_moe_stage2_gate import run_stage2

    print("\n" + "=" * 70)
    print("STAGE 2: Gate Routing Training (next-bar execution)")
    print("=" * 70)

    artifacts = run_stage2(
        manifest_path=CANDIDATE_MANIFEST_PATH,
        stage1_root=stage1_root,
        output_dir=output_dir,
        dry_run=False,
        total_timesteps=300_000,
        load_balance_coef=0.02,
        diversity_coef=0.01,
        gate_temperature=GATE_TEMPERATURE,
        train_data_path=str(shifted_train_csv),
        symbol=SYMBOL,
    )
    print(f"\n[Stage2] Completed. Gate + {len(artifacts)} experts → {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Joint Finetune (Optional)
# ──────────────────────────────────────────────────────────────────────────────

def run_stage3_nextbar(
    shifted_train_csv: Path,
    stage1_root: str,
    stage2_root: str,
    output_root: str,
    config: BaseConfig,
    rounds: int = 2,
) -> None:
    """Alternating joint finetune on next-bar data."""
    from crypto_trader.train_moe_stage3_joint import run_stage3

    print("\n" + "=" * 70)
    print("STAGE 3: Joint Finetune (next-bar execution)")
    print("=" * 70)

    run_stage3(
        manifest_path=CANDIDATE_MANIFEST_PATH,
        stage1_root=stage1_root,
        stage2_root=stage2_root,
        output_root=output_root,
        rounds=rounds,
        base_expert_timesteps=60_000,
        gate_timesteps=80_000,
        load_balance_coef=0.02,
        diversity_coef=0.01,
        gate_temperature=GATE_TEMPERATURE,
        dry_run=False,
        train_data_path=str(shifted_train_csv),
        symbol=SYMBOL,
    )
    print(f"\n[Stage3] Completed. {rounds} rounds → {output_root}")


# ──────────────────────────────────────────────────────────────────────────────
# OOS Backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_oos_backtest(
    stage1_root: str,
    stage2_root: str,
    output_dir: Path,
) -> Dict[str, object]:
    """Run OOS backtest with next_bar execution and tau=0.12."""
    from crypto_trader.backtest_moe import backtest_moe

    print("\n" + "=" * 70)
    print("OOS BACKTEST (next_bar, tau=0.12, temperature=0.68)")
    print("=" * 70)

    # Load unshifted OOS data — backtest_moe applies its own resolve_execution_frame
    oos_df_raw = pd.read_csv(OOS_CSV, index_col=0, parse_dates=True)
    print(f"[Backtest] OOS data: {len(oos_df_raw)} rows before shift, "
          f"period: {oos_df_raw.index.min().date()} → {oos_df_raw.index.max().date()}")

    result = backtest_moe(
        manifest_path=CANDIDATE_MANIFEST_PATH,
        stage1_root=stage1_root,
        stage2_root=stage2_root,
        data_path=str(OOS_CSV),
        gate_temperature=GATE_TEMPERATURE,
        symbol=SYMBOL,
        enable_kill_switch=False,
        execution_mode="next_bar",
        env_overrides={"tau": TAU_OVERRIDE},
        return_history=True,
    )

    if "error" in result:
        print(f"[Backtest] ERROR: {result['error']}")
        if "missing" in result:
            for m in result["missing"]:
                print(f"  Missing: {m}")
    else:
        print(f"[Backtest] Total Return: {result.get('total_return', 'N/A')}")
        print(f"[Backtest] Max Drawdown: {result.get('max_drawdown', 'N/A')}")
        print(f"[Backtest] Sharpe: {result.get('sharpe', 'N/A')}")
        print(f"[Backtest] Sortino: {result.get('sortino', 'N/A')}")
        print(f"[Backtest] Calmar: {result.get('calmar', 'N/A')}")
        print(f"[Backtest] Turnover: {result.get('turnover', 'N/A')}")

    # Save metrics JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_dict = {k: v for k, v in result.items() if k != "history"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Backtest] Metrics saved → {metrics_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Candidate Manifest
# ──────────────────────────────────────────────────────────────────────────────

def create_candidate_manifest(
    experts_root: str,
    gate_root: str,
    oos_data_path: Path,
    backtest_result: Dict[str, object],
) -> Path:
    """Create candidate_manifest.json summarizing the retrained MoE."""
    manifest = {
        "name": "candidate_moe_nextbar_v1",
        "retrained_at": datetime.now().isoformat(),
        "experts": [
            "E5_PPO_lowvol_carry",
            "E2_PPO_bear_drawdown",
            "E4_PPO_highvol_risk",
            "E7_SAC_fast_adapt",
        ],
        "tau": TAU_OVERRIDE,
        "gate_temperature": GATE_TEMPERATURE,
        "execution_mode": "next_bar",
        "oos_metrics": {
            "total_return": backtest_result.get("total_return"),
            "max_drawdown": backtest_result.get("max_drawdown"),
            "sharpe": backtest_result.get("sharpe"),
            "sortino": backtest_result.get("sortino"),
            "calmar": backtest_result.get("calmar"),
            "turnover": backtest_result.get("turnover"),
        },
        "checkpoints": {
            "experts_root": experts_root,
            "gate_root": gate_root,
        },
    }

    out_path = Path(experts_root).parent / "candidate_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[Manifest] Created → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain MoE from scratch under next-bar execution")
    parser.add_argument("--skip-stage3", action="store_true", help="Skip Stage 3 joint finetune")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip OOS backtest")
    parser.add_argument("--stage3-rounds", type=int, default=2, help="Number of Stage 3 rounds (default: 2)")
    parser.add_argument("--skip-all", action="store_true", help="Skip everything (dry-run mode)")
    args = parser.parse_args()

    if args.skip_all:
        print("Dry-run mode: nothing to do.")
        return

    # ── Prepare data ──
    print("=" * 70)
    print("RETRAIN MoE FROM SCRATCH — NEXT-BAR EXECUTION")
    print(f"Manifest: {CANDIDATE_MANIFEST_PATH}")
    print(f"Train data: {TRAIN_CSV}")
    print(f"OOS data: {OOS_CSV}")
    print(f"Gate temperature: {GATE_TEMPERATURE}")
    print(f"Tau override: {TAU_OVERRIDE}")
    print("=" * 70)

    # Load and shift training data
    print("\n[Prep] Loading & shifting training data with next_bar execution...")
    train_df = load_and_shift(TRAIN_CSV)

    # Save shifted data to temp file for the training pipelines
    # (The training pipelines don't have a way to accept raw DataFrames directly)
    with tempfile.TemporaryDirectory() as tmpdir:
        shifted_train_csv = Path(tmpdir) / "train_nextbar_shifted.csv"
        save_shifted_data(train_df, shifted_train_csv)

        # Build config with train_split_ratio=1.0 since data is already train-only
        config = get_default_config()
        config.data.symbol = SYMBOL
        config.data.train_split_ratio = 1.0  # Don't re-split — data is already train80

        # ── Stage 1 ──
        with _tau_override(TAU_OVERRIDE):
            run_stage1_nextbar(
                shifted_train_csv=shifted_train_csv,
                output_root=str(STAGE1_ROOT),
                config=config,
            )

        # Validate stage1 outputs
        for expert_id in ["E5_PPO_lowvol_carry", "E2_PPO_bear_drawdown",
                           "E4_PPO_highvol_risk", "E7_SAC_fast_adapt"]:
            model_path = STAGE1_ROOT / expert_id / "model.zip"
            vecnorm_path = STAGE1_ROOT / expert_id / "vec_normalize.pkl"
            assert model_path.exists(), f"Missing: {model_path}"
            assert vecnorm_path.exists(), f"Missing: {vecnorm_path}"
            print(f"[Verify] ✓ {expert_id}")

        # ── Stage 2 ──
        with _tau_override(TAU_OVERRIDE):
            run_stage2_nextbar(
                shifted_train_csv=shifted_train_csv,
                stage1_root=str(STAGE1_ROOT),
                output_dir=str(STAGE2_ROOT),
                config=config,
            )

        # Validate stage2 outputs
        assert (STAGE2_ROOT / "gate_model.zip").exists(), "Missing gate model"
        assert (STAGE2_ROOT / "gate_vec_normalize.pkl").exists(), "Missing gate vecnorm"
        print(f"[Verify] ✓ Gate model")

        # ── Stage 3 (optional) ──
        if not args.skip_stage3:
            with _tau_override(TAU_OVERRIDE):
                run_stage3_nextbar(
                    shifted_train_csv=shifted_train_csv,
                    stage1_root=str(STAGE1_ROOT),
                    stage2_root=str(STAGE2_ROOT),
                    output_root=str(STAGE3_ROOT),
                    config=config,
                    rounds=args.stage3_rounds,
                )
            # After stage3, use the final round's outputs for backtest
            final_round_dir = STAGE3_ROOT / f"round{args.stage3_rounds}"
            backtest_experts_root = str(final_round_dir / "experts")
            backtest_gate_root = str(final_round_dir / "gate")
        else:
            backtest_experts_root = str(STAGE1_ROOT)
            backtest_gate_root = str(STAGE2_ROOT)

        # ── OOS Backtest ──
        backtest_result = {}
        if not args.skip_backtest:
            backtest_result = run_oos_backtest(
                stage1_root=backtest_experts_root,
                stage2_root=backtest_gate_root,
                output_dir=RESULTS_DIR,
            )

        # ── Candidate Manifest ──
        manifest_path = create_candidate_manifest(
            experts_root=str(STAGE1_ROOT),
            gate_root=str(STAGE2_ROOT),
            oos_data_path=OOS_CSV,
            backtest_result=backtest_result,
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)
    print(f"Experts: {STAGE1_ROOT}")
    print(f"Gate:    {STAGE2_ROOT}")
    if not args.skip_stage3:
        print(f"Stage3:  {STAGE3_ROOT}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Manifest: {manifest_path}")
    if backtest_result and "total_return" in backtest_result:
        tr = backtest_result.get("total_return", 0)
        dd = backtest_result.get("max_drawdown", 0)
        sr = backtest_result.get("sharpe", 0)
        print(f"\nOOS Metrics:")
        print(f"  Total Return: {tr}")
        print(f"  Max Drawdown: {dd}")
        print(f"  Sharpe:       {sr}")
        if isinstance(tr, (int, float)):
            status = "✓" if float(tr) > 0 else "✗"
            print(f"  Status:       {status} (target: total_return > 0)")
    print("\nDone.")


if __name__ == "__main__":
    main()
