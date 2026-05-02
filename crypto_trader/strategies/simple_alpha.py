from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimpleAlphaParams:
    signal_weight: float = 0.6
    trend_weight: float = 0.4
    threshold: float = 0.08
    max_position: float = 0.8
    target_vol: float = 0.035
    min_vol_scale: float = 0.2
    max_vol_scale: float = 1.2
    tau: float = 0.15
    delta_max: float = 0.25
    fee_rate: float = 0.0008
    funding_daily: float = 0.0003


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _bounded_score(value: float, scale: float) -> float:
    return float(np.tanh(_safe_float(value) / max(scale, 1e-12)))


def compute_target_position(row: pd.Series, params: SimpleAlphaParams, vol_cap: float | None = None) -> float:
    signal_score = float(np.clip(2.0 * (_safe_float(row.get("Signal_Proba"), 0.5) - 0.5), -1.0, 1.0))
    trend_score = _bounded_score(row.get("Dist_SMA_200", 0.0), 0.12)
    macd_score = _bounded_score(_safe_float(row.get("MACD"), 0.0) / max(_safe_float(row.get("ATR"), 1.0), 1e-12), 0.08)
    combined_trend = 0.7 * trend_score + 0.3 * macd_score
    raw_score = params.signal_weight * signal_score + params.trend_weight * combined_trend

    if abs(raw_score) < params.threshold:
        return 0.0

    rolling_vol = max(_safe_float(row.get("Rolling_Vol"), params.target_vol), 1e-6)
    cap = vol_cap if vol_cap is not None else params.target_vol
    vol_scale = float(np.clip(cap / rolling_vol, params.min_vol_scale, params.max_vol_scale))
    return float(np.clip(raw_score * params.max_position * vol_scale, -params.max_position, params.max_position))


def apply_position_constraints(target: float, current: float, params: SimpleAlphaParams) -> float:
    if abs(target - current) < params.tau:
        return current
    delta = float(np.clip(target - current, -params.delta_max, params.delta_max))
    return float(np.clip(current + delta, -params.max_position, params.max_position))


def _max_drawdown(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max((peak - arr) / np.maximum(peak, 1e-12)))


def _sharpe(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    std = float(np.std(rets))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252.0))


def backtest_simple_alpha(
    df: pd.DataFrame,
    params: SimpleAlphaParams,
    initial_balance: float = 10000.0,
) -> Dict[str, object]:
    required = {"Close", "Signal_Proba", "Dist_SMA_200", "MACD", "ATR", "Rolling_Vol"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(df) < 2:
        raise ValueError("backtest requires at least 2 rows")

    net_worth = [float(initial_balance)]
    benchmark = [float(initial_balance)]
    positions: List[float] = []
    turnovers: List[float] = []
    trade_costs: List[float] = []
    funding_costs: List[float] = []
    current_pos = 0.0
    equity = float(initial_balance)
    bench_equity = float(initial_balance)

    for i in range(len(df) - 1):
        row = df.iloc[i]
        target = compute_target_position(row, params)
        exec_pos = apply_position_constraints(target, current_pos, params)
        turnover = abs(exec_pos - current_pos)
        trade_cost = turnover * equity * params.fee_rate
        funding_rate = _safe_float(row.get("Funding_Rate"), params.funding_daily)
        funding_cost = abs(exec_pos) * equity * funding_rate

        close_t = max(_safe_float(df["Close"].iloc[i]), 1e-12)
        close_next = max(_safe_float(df["Close"].iloc[i + 1]), 1e-12)
        next_return = close_next / close_t - 1.0

        equity = equity * (1.0 + exec_pos * next_return) - trade_cost - funding_cost
        equity = max(equity, 1e-8)
        bench_equity = bench_equity * (1.0 + next_return)

        net_worth.append(float(equity))
        benchmark.append(float(bench_equity))
        positions.append(exec_pos)
        turnovers.append(turnover)
        trade_costs.append(trade_cost)
        funding_costs.append(funding_cost)
        current_pos = exec_pos

    total_return = net_worth[-1] / net_worth[0] - 1.0
    benchmark_return = benchmark[-1] / benchmark[0] - 1.0
    return {
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "alpha": float(total_return - benchmark_return),
        "max_drawdown": _max_drawdown(net_worth),
        "sharpe": _sharpe(net_worth),
        "final_net_worth": float(net_worth[-1]),
        "net_worth": net_worth,
        "benchmark": benchmark,
        "positions": positions,
        "turnover": float(np.sum(turnovers)),
        "trade_cost": float(np.sum(trade_costs)),
        "funding_cost": float(np.sum(funding_costs)),
        "params": asdict(params),
    }


def candidate_grid() -> List[SimpleAlphaParams]:
    out: List[SimpleAlphaParams] = []
    for signal_weight in [0.4, 0.6, 0.8, 1.0]:
        trend_weight = 1.0 - signal_weight
        for threshold in [0.04, 0.08, 0.12, 0.18]:
            for max_position in [0.35, 0.5, 0.65, 0.8]:
                for tau in [0.10, 0.18, 0.25]:
                    out.append(
                        SimpleAlphaParams(
                            signal_weight=signal_weight,
                            trend_weight=trend_weight,
                            threshold=threshold,
                            max_position=max_position,
                            tau=tau,
                        )
                    )
    return out


def _selection_score(result: Dict[str, object]) -> float:
    max_dd = max(float(result["max_drawdown"]), 1e-6)
    return float(result["total_return"]) / max_dd + 0.25 * float(result["alpha"])


def choose_params(df: pd.DataFrame) -> Tuple[SimpleAlphaParams, Dict[str, object]]:
    best_params = None
    best_result = None
    best_score = -np.inf
    for params in candidate_grid():
        result = backtest_simple_alpha(df, params)
        score = _selection_score(result)
        if result["max_drawdown"] > 0.35:
            score -= 10.0
        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
    if best_params is None or best_result is None:
        raise RuntimeError("No candidate params evaluated")
    return best_params, best_result


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.set_index("Timestamp")
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def write_equity_curve(path: Path, result: Dict[str, object], index: pd.Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(index[: len(result["net_worth"])], result["net_worth"], result["benchmark"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "NetWorth", "Benchmark"])
        writer.writerows(rows)


def run_cli() -> Dict[str, object]:
    parser = argparse.ArgumentParser(description="Backtest low-dimensional simple alpha strategy.")
    parser.add_argument("--train-data", default="crypto_trader/data_moe_20200101_20260216_train80.csv")
    parser.add_argument("--test-data", default="crypto_trader/data_moe_20200101_20260216_oos20.csv")
    parser.add_argument("--output-dir", default="results/simple_alpha")
    args = parser.parse_args()

    train_df = load_dataset(args.train_data)
    test_df = load_dataset(args.test_data)
    params, train_result = choose_params(train_df)
    test_result = backtest_simple_alpha(test_df, params)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_equity_curve(output_dir / "equity_curve.csv", test_result, test_df.index)

    payload = {
        "strategy": "simple_alpha_v1",
        "selection": "train80_grid_only_next_bar",
        "train": {k: v for k, v in train_result.items() if k not in {"net_worth", "benchmark", "positions"}},
        "test": {k: v for k, v in test_result.items() if k not in {"net_worth", "benchmark", "positions"}},
        "params": asdict(params),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run_cli()
