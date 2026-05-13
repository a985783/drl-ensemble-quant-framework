from __future__ import annotations

import argparse
import csv
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import yaml

try:
    from crypto_trader.asset_profile import get_asset_profile
    from crypto_trader.backtest_moe import backtest_moe
    from crypto_trader.validation.metrics import metrics_from_backtest_result
except ImportError:
    from asset_profile import get_asset_profile
    from backtest_moe import backtest_moe
    from validation.metrics import metrics_from_backtest_result

SWEEP_CONFIG_PATH = Path("crypto_trader/validation/param_sweep.yaml")


def _build_run_id(dimension: str, value: Any, suffix: str = "") -> str:
    """Create a concise run_id for a sweep combination."""
    label = str(value).replace(".", "_").replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace("'", "")
    if suffix:
        return f"{dimension}_{label}_{suffix}"
    return f"{dimension}_{label}"


def _make_combo_from_baseline(
    baseline: Dict[str, Any],
    base_k_single: float,
    base_funding_daily: float,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single combo dict by applying *updates* on top of *baseline*.

    Multipliers are resolved to absolute k_single / funding_daily values.
    """
    combo = dict(baseline)
    combo.update(updates)

    k_mult = float(combo.pop("k_single_multiplier", 1.0))
    f_mult = float(combo.pop("funding_multiplier", 1.0))
    combo["k_single"] = round(base_k_single * k_mult, 8)
    combo["funding_daily"] = round(base_funding_daily * f_mult, 8)
    combo["execution_mode"] = "next_bar"

    if "run_id" not in combo:
        combo["run_id"] = "baseline"
    return combo


def _combo_fingerprint(combo: Dict[str, Any]) -> str:
    """Deterministic fingerprint for deduplication.

    Excludes run_id so that two combos with identical parameters but different
    names are recognised as the same.
    """
    relevant = {k: v for k, v in combo.items() if k != "run_id"}
    items = sorted(relevant.items(), key=lambda kv: str(kv[0]))
    return str(items)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sweep_config(path: Path = SWEEP_CONFIG_PATH) -> Dict[str, Any]:
    """Load the sweep YAML config from *path*."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def generate_sweep_combos(config: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Generate ≥50 parameter combinations for the next-bar sweep.

    Three sources of combos, all deduplicated at the end:

    1. **Baseline** – the reference (stable) parameter set.
    2. **Single-param sweeps** – for each dimension, one combo per value that
       differs from the baseline value.
    3. **Interaction grids** – cartesian products over selected parameter
       pairs to capture interaction effects.

    Returns

        List of combo dicts.  Each dict has keys:
            run_id, tau, temperature, delta_max, cooldown_n,
            k_single, funding_daily, gate_mode, disabled_experts,
            execution_mode
    """
    if config is None:
        config = load_sweep_config()

    defaults = dict(config.get("defaults", {}))
    baseline = dict(defaults.get("baseline", {}))
    base_k_single = float(defaults.get("base_k_single", 0.0008))
    base_funding_daily = float(defaults.get("base_funding_daily", 0.0003))
    dimensions: Dict[str, list] = dict(config.get("dimensions", {}))
    interactions: List[Dict[str, Any]] = list(config.get("interactions", []))

    combos: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------------------------
    combos.append(
        _make_combo_from_baseline(baseline, base_k_single, base_funding_daily, {"run_id": "baseline"})
    )

    # -----------------------------------------------------------------------
    # 2. Single-parameter sweeps
    # -----------------------------------------------------------------------
    for dim_name, values in dimensions.items():
        bl_value = baseline.get(dim_name)
        for val in values:
            if val == bl_value:
                continue
            rid = _build_run_id(dim_name, val)
            combos.append(
                _make_combo_from_baseline(
                    baseline,
                    base_k_single,
                    base_funding_daily,
                    {"run_id": rid, dim_name: val},
                )
            )

    # -----------------------------------------------------------------------
    # 3. Multi-parameter interaction grids
    # -----------------------------------------------------------------------
    for interaction in interactions:
        name = interaction.get("name", "interaction")
        grid: Dict[str, list] = dict(interaction.get("grid", {}))
        if not grid:
            continue
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        for values in itertools.product(*value_lists):
            updates = dict(zip(keys, values))
            rid_values = {k: v for k, v in updates.items() if k != "disabled_experts"}
            rid = _build_run_id(name, "", suffix="_".join(f"{k}={v}" for k, v in rid_values.items()))
            combos.append(
                _make_combo_from_baseline(baseline, base_k_single, base_funding_daily, {"run_id": rid, **updates})
            )

    # -----------------------------------------------------------------------
    # Deduplicate (keep first occurrence)
    # -----------------------------------------------------------------------
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for combo in combos:
        fp = _combo_fingerprint(combo)
        if fp not in seen:
            seen.add(fp)
            unique.append(combo)

    return unique


def combos_to_csv(combos: List[Dict[str, Any]], path: Optional[Path] = None) -> str:
    """Render the list of combos as CSV and optionally write to *path*.

    Returns the CSV text.
    """
    fieldnames = [
        "run_id",
        "tau",
        "temperature",
        "delta_max",
        "cooldown_n",
        "k_single",
        "funding_daily",
        "gate_mode",
        "disabled_experts",
        "execution_mode",
    ]
    lines = []
    for combo in combos:
        row = {k: combo.get(k, "") for k in fieldnames}
        if isinstance(row["disabled_experts"], list):
            row["disabled_experts"] = "|".join(row["disabled_experts"])
        lines.append(row)

    if sys.version_info >= (3, 9):
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
        csv_text = buf.getvalue()
    else:
        csv_text = ",".join(fieldnames) + "\n"
        for row in lines:
            csv_text += ",".join(str(row.get(f, "")) for f in fieldnames) + "\n"

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(csv_text)

    return csv_text


def run_sweep(
    combos: List[Dict[str, Any]],
    output_dir: str = "results/sweep",
    config_path: Path = SWEEP_CONFIG_PATH,
) -> List[Dict[str, Any]]:
    """Execute backtest_moe for every combo and collect results.

    Each combo is run with ``execution_mode="next_bar"``.  Results are
    returned as a list of dicts with keys ``run_id``, ``status``, and
    ``metrics`` (on success).

    Parameters
    ----------
    combos:
        List of combo dicts from :func:`generate_sweep_combos`.
    output_dir:
        Root directory for sweep outputs (plots + results CSV).
    config_path:
        Path to the sweep YAML (used for model/data paths).
    """
    sweep_config = load_sweep_config(config_path)
    defaults = dict(sweep_config.get("defaults", {}))

    manifest = Path(str(defaults.get("manifest", "crypto_trader/configs/moe_experts.yaml")))
    stage1_root = str(defaults.get("stage1_root", "checkpoints/moe/stable/experts"))
    stage2_root = str(defaults.get("stage2_root", "checkpoints/moe/stable/gate"))
    data_path = str(defaults.get("data_path", "crypto_trader/data_moe_20200101_20260216_oos20.csv"))
    symbol = str(defaults.get("symbol", "ETH/USDT:USDT"))

    out_dir = Path(output_dir)
    run_id_base = datetime.now(timezone.utc).strftime("sweep_%Y%m%dT%H%M%SZ")
    run_output_dir = out_dir / run_id_base
    run_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for combo in combos:
        rid = combo["run_id"]

        env_overrides: Dict[str, float] = {}
        for param in ("tau", "delta_max", "cooldown_n", "k_single", "funding_daily"):
            if param in combo and combo[param] is not None:
                env_overrides[param] = combo[param]

        result = backtest_moe(
            manifest_path=manifest,
            stage1_root=stage1_root,
            stage2_root=stage2_root,
            data_path=data_path,
            plot_path=str(run_output_dir / f"{rid}.png"),
            gate_temperature=float(combo.get("temperature", 0.68)),
            symbol=symbol,
            env_overrides=env_overrides if env_overrides else None,
            gate_mode=str(combo.get("gate_mode", "model")),
            disabled_experts=combo.get("disabled_experts"),
            execution_mode=str(combo.get("execution_mode", "next_bar")),
            return_history=False,
        )

        if "error" in result:
            entry = {"run_id": rid, "status": "error", "error": result}
            print(f"  [{rid}] ERROR: {result.get('error', 'unknown')}")
        else:
            metrics = metrics_from_backtest_result(result)
            entry = {"run_id": rid, "status": "ok", "metrics": metrics}
            print(f"  [{rid}] total_return={metrics.get('total_return', '?'):>+.2%}  alpha={metrics.get('alpha', '?'):>+.2%}")

        results.append(entry)

    # Write summary CSV
    summary_path = run_output_dir / "sweep_results.csv"
    _write_results_csv(results, summary_path)
    print(f"\nSweep complete: {len(results)} combos -> {summary_path}")

    return results


def _write_results_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Write a flat CSV of sweep results with one row per run."""
    rows = []
    for entry in results:
        row: Dict[str, Any] = {"run_id": entry["run_id"], "status": entry["status"]}
        if entry["status"] == "ok":
            metrics = entry.get("metrics", {})
            for key in (
                "total_return", "benchmark_return", "alpha", "max_drawdown",
                "sharpe", "sortino", "calmar", "turnover", "trade_cost",
                "funding_cost", "exposure", "n_steps",
            ):
                row[key] = metrics.get(key, "")
        rows.append(row)

    fieldnames = [
        "run_id", "status",
        "total_return", "benchmark_return", "alpha", "max_drawdown",
        "sharpe", "sortino", "calmar", "turnover", "trade_cost",
        "funding_cost", "exposure", "n_steps",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE Parameter Sweep Runner (next-bar execution)")
    parser.add_argument("--config", type=str, default=str(SWEEP_CONFIG_PATH),
                        help="Path to sweep YAML config")
    parser.add_argument("--output-dir", type=str, default="results/sweep",
                        help="Root output directory for sweep results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated combos as CSV without running backtests")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_sweep_config(Path(args.config))
    combos = generate_sweep_combos(config)

    print(f"Generated {len(combos)} parameter combinations\n")

    if args.dry_run:
        csv_text = combos_to_csv(combos, None)
        print(csv_text)
        return

    run_sweep(combos, output_dir=args.output_dir, config_path=Path(args.config))


if __name__ == "__main__":
    main()
