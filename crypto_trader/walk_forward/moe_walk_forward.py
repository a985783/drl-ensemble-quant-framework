"""MoE Anchored Walk-Forward Validation Orchestrator.

5-fold expanding-window validation:
  fold_1: train 2020-2021 → test 2022
  fold_2: train 2020-2022 → test 2023
  fold_3: train 2020-2023 → test 2024
  fold_4: train 2020-2024 → test 2025
  fold_5: train 2020-2025 → test 2026

Each fold independently: XGBoost → 4 experts → Gate (temp scan) → backtest.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig
from crypto_trader.walk_forward.folding import FoldingManager, FoldConfig
from crypto_trader.walk_forward.data_prep import DataPreparer
from crypto_trader.walk_forward.expert_trainer import ExpertTrainer
from crypto_trader.walk_forward.gate_trainer import GateTrainer
from crypto_trader.walk_forward.backtester import FoldBacktester
from crypto_trader.walk_forward.aggregator import Aggregator
from crypto_trader.train_moe_stage2_gate import build_gate_artifacts


def parse_temperature_candidates(raw):
    if raw is None:
        return None
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--temperature-candidates must include at least one value")
    return values


def normalize_fold_arg(value):
    if value is None:
        return None
    raw = str(value).strip()
    if raw.isdigit():
        return f"fold_{raw}"
    return raw


def should_skip_completed_fold(config, fold_config, resume=False):
    if not resume:
        return False
    metrics_path = Path(config.results_root) / fold_config.fold_id / "metrics.json"
    return metrics_path.exists()


def _default_expert_workers(cpu_count=None):
    cores = int(cpu_count or os.cpu_count() or 1)
    return max(1, min(4, cores // 2 if cores > 2 else 1))


def apply_runtime_overrides(config, args, cpu_count=None):
    if getattr(args, "fast_smoke", False):
        config.expert_timesteps = {eid: 1_000 for eid in config.expert_ids}
        config.gate_timesteps = 1_000
        config.temperature_candidates = [0.68]

    if getattr(args, "expert_timesteps", None) is not None:
        steps = int(args.expert_timesteps)
        config.expert_timesteps = {eid: steps for eid in config.expert_ids}

    if getattr(args, "gate_timesteps", None) is not None:
        config.gate_timesteps = int(args.gate_timesteps)

    checkpoint_root = getattr(args, "checkpoint_root", None)
    if checkpoint_root:
        config.checkpoint_root = str(checkpoint_root)

    temps = parse_temperature_candidates(getattr(args, "temperature_candidates", None))
    if temps is not None:
        config.temperature_candidates = temps

    workers = getattr(args, "expert_workers", None)
    if workers is None:
        workers = _default_expert_workers(cpu_count)
    config.expert_parallel_workers = max(1, min(int(workers), int(cpu_count or os.cpu_count() or 1), 4))

    torch_threads = getattr(args, "torch_threads", None)
    if torch_threads is None:
        torch_threads = 1
    config.torch_num_threads = max(1, int(torch_threads))

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(config.torch_num_threads)

    try:
        import torch

        torch.set_num_threads(config.torch_num_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    return config


def run_single_fold(config, fold_config, skip_train=False):
    started = time.time()
    print(f"\n{'='*60}")
    print(f"  {fold_config.fold_id}: train {fold_config.train_start}→{fold_config.train_end}")
    print(f"  test {fold_config.test_start}→{fold_config.test_end}")
    print(
        f"  workers={config.expert_parallel_workers}, torch_threads={config.torch_num_threads}, "
        f"expert_steps={config.expert_timesteps}, gate_steps={config.gate_timesteps}, "
        f"temps={config.temperature_candidates}",
        flush=True,
    )
    print(f"{'='*60}")

    # 1. Data preparation (XGBoost per-fold + next-bar shift)
    print("[1/4] Preparing data (XGBoost + next-bar shift)...", flush=True)
    prep = DataPreparer(config, fold_config)
    train_df, test_df = prep.prepare()
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows", flush=True)

    if skip_train:
        print("[SKIP] Training skipped. Loading existing checkpoints if available.", flush=True)
        expert_root = Path(config.checkpoint_root) / fold_config.fold_id / "experts"
        gate_root = Path(config.checkpoint_root) / fold_config.fold_id / "gate"
        if not expert_root.exists() or not gate_root.exists():
            raise FileNotFoundError(f"Checkpoints not found for {fold_config.fold_id}")
    else:
        # 2. Train experts (Stage1)
        print("[2/4] Training 4 experts (Stage1)...", flush=True)
        chk_root = Path(config.checkpoint_root) / fold_config.fold_id
        trainer = ExpertTrainer(config, fold_config, train_df)
        trainer.train_all(chk_root)

        # 3. Train gate + temperature selection (Stage2)
        print("[3/4] Training Gate with temperature scan...", flush=True)
        expert_root = Path(config.checkpoint_root) / fold_config.fold_id / "experts"
        artifacts = build_gate_artifacts(
            config.manifest_path,
            str(expert_root),
            active_expert_ids=config.expert_ids,
        )
        gate_trainer = GateTrainer(fold_config, config)
        best_temp, candidate_results = gate_trainer.select_temperature(train_df, artifacts)
        print(f"  Selected temperature: {best_temp}", flush=True)
        gate_trainer.train_final(train_df, artifacts, best_temp, candidate_results)

    # Load best_temp from metadata if skipping training
    if skip_train:
        gate_meta_path = Path(config.checkpoint_root) / fold_config.fold_id / "gate" / "metadata.json"
        with open(gate_meta_path) as f:
            best_temp = json.load(f)["selected_temperature"]

    # 4. Backtest on test period
    print("[4/4] Running backtest on test period...", flush=True)
    backtester = FoldBacktester(config, fold_config)
    result = backtester.run(test_df, best_temp)

    elapsed = (time.time() - started) / 60.0
    print(
        f"  Result: return={result.total_return:.2%}, alpha={result.alpha:.2%}, "
        f"dd={result.max_drawdown:.2%}, pass={result.pass_fold}, elapsed={elapsed:.1f}m",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="MoE Anchored Walk-Forward Validation")
    parser.add_argument("--dry-run", action="store_true", help="Print fold plan without executing")
    parser.add_argument("--fold", type=str, default=None, help="Run only a specific fold (e.g., fold_1)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoints")
    parser.add_argument("--output", type=str, default=None, help="Override results directory")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Skip folds that already have metrics.json")
    parser.add_argument("--expert-workers", type=int, default=None, help="Parallel expert training workers (default: auto, max 4)")
    parser.add_argument("--torch-threads", type=int, default=None, help="Threads per worker for PyTorch/BLAS (default: 1)")
    parser.add_argument("--expert-timesteps", type=int, default=None, help="Override timesteps for every expert")
    parser.add_argument("--gate-timesteps", type=int, default=None, help="Override gate PPO timesteps")
    parser.add_argument("--temperature-candidates", type=str, default=None, help="Comma-separated gate temperatures, e.g. 0.5,0.68,1.0")
    parser.add_argument("--fast-smoke", action="store_true", help="Tiny training budget for plumbing checks only")
    args = parser.parse_args()

    config = WalkForwardMoEConfig()
    if args.output:
        config.results_root = args.output
    config = apply_runtime_overrides(config, args)

    fm = FoldingManager(config)
    folds = fm.build_folds()
    fm.validate_folds(folds)

    if args.dry_run:
        print(fm.describe(folds))
        print(
            f"Runtime: expert_workers={config.expert_parallel_workers}, "
            f"torch_threads={config.torch_num_threads}, gate_timesteps={config.gate_timesteps}, "
            f"temperatures={config.temperature_candidates}"
        )
        return

    # Filter to specific fold if requested
    if args.fold:
        wanted_fold = normalize_fold_arg(args.fold)
        folds = [f for f in folds if f.fold_id == wanted_fold]
        if not folds:
            print(f"Fold '{args.fold}' not found. Available: fold_1, fold_2, fold_3, fold_4, fold_5")
            sys.exit(1)

    results = []
    for fold in folds:
        if should_skip_completed_fold(config, fold, resume=args.resume):
            print(f"[SKIP] {fold.fold_id}: existing metrics.json found", flush=True)
            continue
        try:
            result = run_single_fold(config, fold, skip_train=args.skip_train)
            results.append(result)
        except Exception as e:
            print(f"  ❌ {fold.fold_id} FAILED: {e}")
            import traceback
            traceback.print_exc()

    if args.resume:
        existing_results = Aggregator(config).load_fold_results(Path(config.results_root))
        seen = {r.fold_id for r in results}
        results = [r for r in existing_results if r.fold_id not in seen] + results

    if not results:
        print("No folds completed.")
        sys.exit(1)

    # Aggregate
    agg = Aggregator(config)
    verdict = agg.evaluate(results)
    output_dir = Path(config.results_root) / "summary"
    agg.save_verdict(verdict, output_dir)
    agg.generate_report(verdict, output_dir / "final_report.md")

    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict['status']}")
    print(f"  Folds: {verdict['folds_passed']}/{verdict['folds_completed']} passed")
    print(f"  Avg Alpha: {verdict['avg_alpha']:.2%} (target: {verdict['target_avg_alpha']:.0%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
