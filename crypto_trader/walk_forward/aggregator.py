"""
Aggregator: collects fold results and produces pass/fail verdict.

Usage:
    agg = Aggregator(config)
    verdict = agg.evaluate(fold_results)
    agg.save_verdict(verdict, output_dir)
    agg.generate_report(verdict, output_path)
"""

from __future__ import annotations

import json
import statistics
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

try:
    from crypto_trader.walk_forward.backtester import FoldResult
    from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig
except ImportError:
    from walk_forward.backtester import FoldResult
    from walk_forward.moe_config import WalkForwardMoEConfig


class Aggregator:
    """Collects fold results against pass criteria and emits verdict artifacts."""

    def __init__(self, config: WalkForwardMoEConfig):
        self.config = config

    def evaluate(self, results: List[FoldResult]) -> dict:
        per_fold_pass = [r.pass_fold for r in results]
        alphas = [r.alpha for r in results]

        avg_alpha = statistics.mean(alphas) if alphas else 0.0

        all_pass = all(per_fold_pass)
        avg_pass = avg_alpha >= self.config.target_avg_alpha

        return {
            "status": "PASS" if (all_pass and avg_pass) else "FAIL",
            "folds_completed": len(results),
            "folds_passed": sum(per_fold_pass),
            "avg_alpha": avg_alpha,
            "target_avg_alpha": self.config.target_avg_alpha,
            "all_folds_pass": all_pass,
            "avg_alpha_pass": avg_pass,
            "fold_details": [
                {
                    "fold_id": r.fold_id,
                    "train_window": r.train_window,
                    "test_window": r.test_window,
                    "alpha": r.alpha,
                    "total_return": r.total_return,
                    "benchmark_return": r.benchmark_return,
                    "max_drawdown": r.max_drawdown,
                    "sharpe": r.sharpe,
                    "sortino": r.sortino,
                    "pass": r.pass_fold,
                    "temperature": r.selected_temperature,
                }
                for r in results
            ],
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def save_fold_result(result: FoldResult, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.json"
        path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_fold_results(self, results_root: Path | None = None) -> List[FoldResult]:
        root = Path(results_root or self.config.results_root)
        results: List[FoldResult] = []
        for path in sorted(root.glob("fold_*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            results.append(FoldResult(**payload))
        return results

    def save_verdict(self, verdict: dict, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)

        rows = list(verdict["fold_details"])
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary.csv", index=False)

        return verdict

    def generate_report(self, verdict: dict, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Walk-Forward MoE Validation Report",
            "",
            f"- **Generated**: {verdict['generated_at']}",
            f"- **Verdict**: **{verdict['status']}**",
            f"- **Folds completed**: {verdict['folds_completed']}",
            f"- **Folds passed**: {verdict['folds_passed']}/{verdict['folds_completed']}",
            f"- **Average Alpha**: {verdict['avg_alpha']:.2%} (target: {verdict['target_avg_alpha']:.0%})",
            "",
            "## Fold Details",
            "",
            "| Fold | Train Window | Test Window | Alpha | Return | BM Return | MaxDD | Sharpe | Sortino | Pass |",
            "|------|-------------|------------|------:|------:|------:|------:|------:|------:|------|",
        ]

        for d in verdict["fold_details"]:
            lines.append(
                f"| {d['fold_id']} "
                f"| {d['train_window']} "
                f"| {d['test_window']} "
                f"| {d['alpha']:.2%} "
                f"| {d['total_return']:.2%} "
                f"| {d['benchmark_return']:.2%} "
                f"| {d['max_drawdown']:.2%} "
                f"| {d['sharpe']:.2f} "
                f"| {d['sortino']:.2f} "
                f"| {'✅' if d['pass'] else '❌'} |"
            )

        lines += [
            "",
            "## Criteria",
            f"- Per-fold alpha > {self.config.min_fold_alpha:.0%}: "
            f"{'✅' if verdict['all_folds_pass'] else '❌'}",
            f"- Average alpha ≥ {self.config.target_avg_alpha:.0%}: "
            f"{'✅' if verdict['avg_alpha_pass'] else '❌'}",
        ]

        with open(output_path, "w") as f:
            f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MoE walk-forward fold metrics")
    parser.add_argument("--results-root", default=None, help="Results root containing fold_*/metrics.json")
    args = parser.parse_args()

    config = WalkForwardMoEConfig()
    if args.results_root:
        config.results_root = args.results_root

    agg = Aggregator(config)
    results = agg.load_fold_results(Path(config.results_root))
    if not results:
        raise SystemExit(f"No fold metrics found under {config.results_root}")

    verdict = agg.evaluate(results)
    output_dir = Path(config.results_root) / "summary"
    agg.save_verdict(verdict, output_dir)
    agg.generate_report(verdict, output_dir / "final_report.md")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
