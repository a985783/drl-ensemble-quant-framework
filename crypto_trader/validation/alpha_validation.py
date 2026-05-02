from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from crypto_trader.asset_profile import get_asset_profile
    from crypto_trader.backtest_moe import backtest_moe
    from crypto_trader.backtest_sanity import method_b_random_baseline_test
    from crypto_trader.validation.metrics import metrics_from_backtest_result
    from crypto_trader.validation.report import write_report_bundle
    from crypto_trader.validation.verdicts import evaluate_validation_results
except ImportError:  # pragma: no cover - supports direct crypto_trader path imports
    from asset_profile import get_asset_profile
    from backtest_moe import backtest_moe
    from backtest_sanity import method_b_random_baseline_test
    from validation.metrics import metrics_from_backtest_result
    from validation.report import write_report_bundle
    from validation.verdicts import evaluate_validation_results


DEFAULT_CONFIG_PATH = Path("crypto_trader/validation/default_validation.yaml")


def load_validation_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, object]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("defaults", {})
    data.setdefault("scenarios", {})
    return data


def _scenario_enabled(config: Mapping[str, object], key: str) -> bool:
    scenarios = config.get("scenarios", {})
    if not isinstance(scenarios, Mapping):
        return False
    value = scenarios.get(key, {})
    return bool(isinstance(value, Mapping) and value.get("enabled", False))


def _scenario_config(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    scenarios = config.get("scenarios", {})
    if not isinstance(scenarios, Mapping):
        return {}
    value = scenarios.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _merge_env_overrides(default_overrides: Optional[Mapping[str, object]], scenario_overrides: Optional[Mapping[str, object]]) -> Optional[Dict[str, float]]:
    merged: Dict[str, float] = {}
    if default_overrides:
        for k, v in default_overrides.items():
            merged[str(k)] = float(v)
    if scenario_overrides:
        for k, v in scenario_overrides.items():
            merged[str(k)] = float(v)
    return merged or None


def _inject_default_scenario_params(scenario: Dict[str, object], defaults: Mapping[str, object]) -> Dict[str, object]:
    s = dict(scenario)
    de = defaults.get("disabled_experts")
    if de is not None and "disabled_experts" not in s:
        s["disabled_experts"] = list(de)
    em = defaults.get("execution_mode")
    if em is not None and "execution_mode" not in s:
        s["execution_mode"] = str(em)
    default_eo = defaults.get("env_overrides")
    if default_eo and isinstance(default_eo, Mapping):
        s["env_overrides"] = _merge_env_overrides(default_eo, s.get("env_overrides"))
    return s


def build_scenarios(config: Mapping[str, object]) -> List[Dict[str, object]]:
    defaults = config.get("defaults", {})
    if not isinstance(defaults, Mapping):
        defaults = {}

    symbol = str(defaults.get("symbol", "ETH/USDT:USDT"))
    profile = get_asset_profile(symbol).env
    gate_temperature = float(defaults.get("gate_temperature", 0.68))
    default_eo = defaults.get("env_overrides")
    effective_tau = float(default_eo.get("tau", profile.tau)) if default_eo and isinstance(default_eo, Mapping) else profile.tau
    scenarios: List[Dict[str, object]] = []

    if _scenario_enabled(config, "stable_oos"):
        scenarios.append(_inject_default_scenario_params(
            {"name": "stable_oos", "kind": "moe", "gate_temperature": gate_temperature}, defaults))

    if _scenario_enabled(config, "signal_delay_1d"):
        scenarios.append(_inject_default_scenario_params(
            {"name": "signal_delay_1d", "kind": "moe", "gate_temperature": gate_temperature,
             "data_transform": "signal_delay_1d"}, defaults))

    if _scenario_enabled(config, "signal_neutral_0_5"):
        scenarios.append(_inject_default_scenario_params(
            {"name": "signal_neutral_0_5", "kind": "moe", "gate_temperature": gate_temperature,
             "data_transform": "signal_neutral_0_5"}, defaults))

    if _scenario_enabled(config, "random_baseline"):
        random_cfg = _scenario_config(config, "random_baseline")
        scenarios.append(
            {"name": "random_baseline", "kind": "random_baseline",
             "n_runs": int(random_cfg.get("n_runs", 10)),
             "seed": int(random_cfg.get("seed", 42))})

    temp_cfg = _scenario_config(config, "gate_temperatures")
    if _scenario_enabled(config, "gate_temperatures"):
        for value in temp_cfg.get("values", []):
            temp = float(value)
            label = str(value).replace(".", "_")
            scenarios.append(_inject_default_scenario_params(
                {"name": f"temperature_{label}", "kind": "moe", "gate_temperature": temp}, defaults))

    cost_cfg = _scenario_config(config, "cost_stress")
    if _scenario_enabled(config, "cost_stress"):
        for value in cost_cfg.get("multipliers", []):
            mult = float(value)
            label = str(value).replace(".0", "").replace(".", "_")
            scenarios.append(_inject_default_scenario_params(
                {"name": f"cost_{label}x", "kind": "moe", "gate_temperature": gate_temperature,
                 "env_overrides": {"k_single": profile.k_single * mult}}, defaults))

    funding_cfg = _scenario_config(config, "funding_stress")
    if _scenario_enabled(config, "funding_stress"):
        for value in funding_cfg.get("multipliers", []):
            mult = float(value)
            label = str(value).replace(".0", "").replace(".", "_")
            scenarios.append(_inject_default_scenario_params(
                {"name": f"funding_{label}x", "kind": "moe", "gate_temperature": gate_temperature,
                 "env_overrides": {"funding_cost_multiplier": mult}}, defaults))

    exec_cfg = _scenario_config(config, "execution_perturbations")
    if _scenario_enabled(config, "execution_perturbations"):
        for value in exec_cfg.get("tau_multipliers", []):
            mult = float(value)
            label = str(value).replace(".", "_")
            scenarios.append(_inject_default_scenario_params(
                {"name": f"tau_{label}x", "kind": "moe", "gate_temperature": gate_temperature,
                 "env_overrides": {"tau": effective_tau * mult}}, defaults))
        for value in exec_cfg.get("delta_max_multipliers", []):
            mult = float(value)
            label = str(value).replace(".", "_")
            scenarios.append(_inject_default_scenario_params(
                {"name": f"delta_max_{label}x", "kind": "moe", "gate_temperature": gate_temperature,
                 "env_overrides": {"delta_max": profile.delta_max * mult}}, defaults))
        for value in exec_cfg.get("cooldown_values", []):
            cooldown = int(value)
            scenarios.append(_inject_default_scenario_params(
                {"name": f"cooldown_{cooldown}", "kind": "moe", "gate_temperature": gate_temperature,
                 "env_overrides": {"cooldown_n": cooldown}}, defaults))

    ablation_cfg = _scenario_config(config, "ablations")
    if _scenario_enabled(config, "ablations"):
        if ablation_cfg.get("uniform_gate", False):
            scenarios.append(_inject_default_scenario_params(
                {"name": "uniform_gate", "kind": "moe", "gate_temperature": gate_temperature,
                 "gate_mode": "uniform"}, defaults))
        if ablation_cfg.get("average_experts", False):
            scenarios.append(_inject_default_scenario_params(
                {"name": "average_experts", "kind": "moe", "gate_temperature": gate_temperature,
                 "gate_mode": "average_experts"}, defaults))
        if ablation_cfg.get("drop_top_contributor", False):
            scenarios.append(_inject_default_scenario_params(
                {"name": "drop_top_contributor", "kind": "drop_top_contributor",
                 "gate_temperature": gate_temperature}, defaults))

    return scenarios


def _missing_artifacts(defaults: Mapping[str, object]) -> List[str]:
    paths = [
        defaults.get("manifest"),
        Path(str(defaults.get("stage2_root", ""))) / "gate_model.zip",
        Path(str(defaults.get("stage2_root", ""))) / "gate_vec_normalize.pkl",
        defaults.get("data_path"),
        defaults.get("registry_path"),
    ]
    missing = []
    for raw in paths:
        path = Path(str(raw))
        if not path.exists():
            missing.append(str(path))
    return missing


def summarize_walk_forward(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        return {"status": "error", "path": str(path), "error": str(exc)}
    if df.empty:
        return {"status": "missing", "path": str(path), "reason": "empty"}
    return {
        "status": "ok",
        "path": str(path),
        "folds": int(len(df)),
        "avg_alpha": float(df["alpha"].mean()) if "alpha" in df.columns else 0.0,
        "max_drawdown": float(df["max_drawdown"].max()) if "max_drawdown" in df.columns else 0.0,
        "positive_alpha_fraction": float((df["alpha"] > 0).mean()) if "alpha" in df.columns else 0.0,
    }


def _run_moe_scenario(scenario: Mapping[str, object], defaults: Mapping[str, object]) -> Dict[str, object]:
    result = backtest_moe(
        manifest_path=Path(str(defaults.get("manifest"))),
        stage1_root=str(defaults.get("stage1_root")),
        stage2_root=str(defaults.get("stage2_root")),
        data_path=str(defaults.get("data_path")),
        plot_path=str(Path(str(defaults.get("output_dir", "results/validation"))) / f"{scenario['name']}.png"),
        gate_temperature=float(scenario.get("gate_temperature", defaults.get("gate_temperature", 0.68))),
        symbol=str(defaults.get("symbol", "ETH/USDT:USDT")),
        env_overrides=scenario.get("env_overrides"),
        gate_mode=str(scenario.get("gate_mode", "model")),
        disabled_experts=scenario.get("disabled_experts"),
        data_transform=scenario.get("data_transform"),
        execution_mode=str(scenario.get("execution_mode", str(defaults.get("execution_mode", "next_bar")))),
        return_history=True,
    )
    if "error" in result:
        return {"name": scenario["name"], "status": "error", "error": result}
    scenario_result: Dict[str, object] = {
        "name": scenario["name"],
        "status": "ok",
        "metrics": metrics_from_backtest_result(result),
    }
    hist = result.get("history", {})
    if isinstance(hist, Mapping):
        weights = hist.get("weights", [])
        if weights:
            scenario_result["gate_weights_history"] = np.asarray(weights, dtype=np.float64)
        net_worth = hist.get("net_worth", [])
        if len(net_worth) >= 2:
            nw_arr = np.asarray(list(net_worth), dtype=np.float64)
            step_returns = np.diff(nw_arr) / np.maximum(nw_arr[:-1], 1e-12)
            scenario_result["step_returns"] = step_returns
    return scenario_result


def _run_random_scenario(scenario: Mapping[str, object], defaults: Mapping[str, object]) -> Dict[str, object]:
    result = method_b_random_baseline_test(
        data_path=str(defaults.get("data_path")),
        symbol=str(defaults.get("symbol", "ETH/USDT:USDT")),
        n_runs=int(scenario.get("n_runs", 10)),
        seed=int(scenario.get("seed", 42)),
    )
    if result.get("status") != "ok":
        return {"name": scenario["name"], "status": "error", "error": result}
    mean_return = float(result.get("mean_return", 0.0))
    return {
        "name": scenario["name"],
        "status": "ok",
        "metrics": {
            "total_return": mean_return,
            "benchmark_return": 0.0,
            "alpha": mean_return,
            "max_drawdown": 0.0,
            "n_runs": int(result.get("n_runs", 0)),
            "std_return": float(result.get("std_return", 0.0)),
            "positive_fraction": float(result.get("positive_fraction", 0.0)),
        },
    }


def run_validation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    include_training_plan: bool = False,
    bootstrap: bool = False,
) -> Dict[str, object]:
    config = load_validation_config(config_path)
    defaults = dict(config.get("defaults", {}))
    run_id = run_id or datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%SZ")
    output_root = Path(str(defaults.get("output_root", "results/validation")))
    output_dir = output_root / run_id
    defaults["output_dir"] = str(output_dir)

    scenarios = build_scenarios(config)
    missing = _missing_artifacts(defaults)
    walk_forward = summarize_walk_forward(Path(str(defaults.get("walk_forward_metrics", ""))))

    notes = [
        "Validation framework is sidecar-only; it does not update stable registries or live trading entrypoints.",
        "Anchored walk-forward retraining is not executed by default.",
    ]
    if include_training_plan:
        notes.append(
            "Optional training plan: train XGBoost, experts, and gate inside each anchored fold, then evaluate frozen fold artifacts."
        )

    if dry_run:
        scenario_results = [
            {"name": str(s["name"]), "status": "planned", "metrics": {}} for s in scenarios
        ]
    elif missing:
        scenario_results = []
    else:
        scenario_results = []
        stable_result: Optional[Dict[str, object]] = None
        for scenario in scenarios:
            if scenario["kind"] == "random_baseline":
                scenario_results.append(_run_random_scenario(scenario, defaults))
                continue

            if scenario["kind"] == "drop_top_contributor":
                if stable_result is None:
                    stable_oos_scenario = _inject_default_scenario_params(
                        {"name": "stable_oos", "kind": "moe", "gate_temperature": defaults.get("gate_temperature", 0.68)},
                        defaults,
                    )
                    stable_result = _run_moe_scenario(stable_oos_scenario, defaults)
                contribution = stable_result.get("metrics", {}).get("expert_contribution", {})
                existing_disabled = set(scenario.get("disabled_experts") or [])
                disabled = []
                if isinstance(contribution, Mapping) and contribution:
                    for eid in sorted(contribution, key=lambda k: abs(float(contribution[k])), reverse=True):
                        if eid not in existing_disabled and len(disabled) < 1:
                            disabled.append(eid)
                drop_scenario = dict(scenario)
                drop_scenario["kind"] = "moe"
                drop_scenario["disabled_experts"] = list(existing_disabled) + disabled
                drop_scenario = _inject_default_scenario_params(drop_scenario, defaults)
                scenario_results.append(_run_moe_scenario(drop_scenario, defaults))
                continue

            result = _run_moe_scenario(scenario, defaults)
            if scenario["name"] == "stable_oos":
                stable_result = result
            scenario_results.append(result)

    # Extract gate_weights_history and step_returns from stable scenario
    stable_gate_weights: Optional[np.ndarray] = None
    stable_step_returns: Optional[np.ndarray] = None
    for s in scenario_results:
        if s.get("name") == "stable_oos":
            gw = s.get("gate_weights_history")
            if gw is not None:
                stable_gate_weights = np.asarray(gw, dtype=np.float64)
            sr = s.get("step_returns")
            if sr is not None:
                stable_step_returns = np.asarray(sr, dtype=np.float64)
            break

    verdict = evaluate_validation_results(
        scenarios=scenario_results,
        walk_forward_summary=walk_forward,
        missing_artifacts=missing,
        gate_weights_history=stable_gate_weights,
        step_returns=stable_step_returns,
        run_bootstrap=bootstrap,
    )
    report = {
        "run_id": run_id,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "defaults": defaults,
        "missing_artifacts": missing,
        "walk_forward": walk_forward,
        "scenarios": scenario_results,
        "verdict": verdict,
        "notes": notes,
    }
    write_report_bundle(output_dir, report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sidecar alpha validation audit.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-training-plan", action="store_true")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CI for return significance check")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_validation(
        config_path=Path(args.config),
        run_id=args.run_id,
        dry_run=args.dry_run,
        include_training_plan=args.include_training_plan,
        bootstrap=args.bootstrap,
    )
    output_dir = Path(str(report["defaults"]["output_dir"]))
    print(f"Validation report written to {output_dir}")
    print(f"Verdict: {report['verdict']['status']}")


if __name__ == "__main__":
    main()
