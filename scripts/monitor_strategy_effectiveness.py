#!/usr/bin/env python3
"""
Strategy effectiveness monitoring for locked stable MoE.

Outputs:
- runs/monitoring/effectiveness_latest.json
- runs/monitoring/effectiveness_history.jsonl
- runs/monitoring/effectiveness_latest.md
- runs/monitoring/weekly_report_YYYYMMDD.md (weekly mode)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_MONITORING = PROJECT_ROOT / "runs" / "monitoring"
DEFAULT_OOS_DATA = PROJECT_ROOT / "crypto_trader" / "data_moe_20200101_20260216_oos20.csv"
sys.path.insert(0, str(PROJECT_ROOT))
from crypto_trader.monitoring_utils import parse_trade_log_for_monitoring


@dataclass
class Thresholds:
    max_drawdown: float = 0.25
    min_alpha: float = 0.0
    min_fill_rate_7d: float = 0.85
    max_slippage_7d: float = 0.01
    max_reconcile_errors_7d: int = 2
    min_return_drift_vs_lock: float = -0.25


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(moe_registry: Dict, data_path_arg: Optional[str]) -> Tuple[Path, Path, Path]:
    manifest = PROJECT_ROOT / moe_registry.get("stable_manifest_path", "crypto_trader/configs/moe_experts.yaml")
    experts = PROJECT_ROOT / moe_registry.get("stable_experts_path", "checkpoints/moe/stable/experts")
    gate = PROJECT_ROOT / moe_registry.get("stable_gate_path", "checkpoints/moe/stable/gate")

    if data_path_arg:
        data_path = PROJECT_ROOT / data_path_arg if not Path(data_path_arg).is_absolute() else Path(data_path_arg)
    else:
        data_path = DEFAULT_OOS_DATA

    return manifest, experts, gate, data_path


def _run_backtest(
    manifest: Path,
    experts: Path,
    gate: Path,
    data_path: Path,
    gate_temperature: float,
) -> Dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    from crypto_trader.backtest_moe import backtest_moe

    result = backtest_moe(
        manifest_path=manifest,
        stage1_root=str(experts),
        stage2_root=str(gate),
        data_path=str(data_path),
        gate_temperature=float(gate_temperature),
        symbol="ETH/USDT:USDT",
        plot_path=str(RUNS_MONITORING / "latest_backtest_curve.png"),
    )
    return result


def _live_metrics_from_trade_logs() -> Dict:
    trade_log = PROJECT_ROOT / "trade_logs.csv"
    if not trade_log.exists():
        return {"status": "missing_trade_logs"}

    df = parse_trade_log_for_monitoring(trade_log)
    if df.empty:
        return {"status": "empty_trade_logs"}

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    else:
        return {"status": "invalid_trade_logs_no_timestamp"}

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.tz_localize(None)
    cutoff = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=7))
    week_df = df[df["Timestamp"] >= cutoff]

    fill_rate = None
    if not week_df.empty and "Action" in week_df.columns and "Contracts" in week_df.columns:
        trade_rows = week_df[week_df["Action"].astype(str).str.lower() == "trade"]
        if len(trade_rows) > 0:
            fill_rate = float((pd.to_numeric(trade_rows["Contracts"], errors="coerce").fillna(0) > 0).mean())

    mean_slippage_7d = None
    if "Slippage" in week_df.columns and not week_df.empty:
        mean_slippage_7d = float(pd.to_numeric(week_df["Slippage"], errors="coerce").fillna(0.0).mean())

    reconcile_errors_7d = 0
    if "Reconcile_Diff" in week_df.columns and not week_df.empty:
        reconcile_errors_7d = int((week_df["Reconcile_Diff"].astype(str) != "OK").sum())

    latest_row = df.iloc[-1]
    return {
        "status": "ok",
        "last_trade_at": str(latest_row["Timestamp"]),
        "rows_total": int(len(df)),
        "rows_7d": int(len(week_df)),
        "fill_rate_7d": fill_rate,
        "mean_slippage_7d": mean_slippage_7d,
        "reconcile_errors_7d": reconcile_errors_7d,
        "last_net_worth": float(pd.to_numeric(latest_row.get("Net_Worth"), errors="coerce")),
        "last_exec_pos": float(pd.to_numeric(latest_row.get("Exec_Pos"), errors="coerce")),
    }


def _evaluate_alerts(backtest: Dict, live: Dict, locked: Dict, th: Thresholds) -> List[str]:
    alerts: List[str] = []
    if "error" in backtest:
        alerts.append(f"backtest_error:{backtest['error']}")
        return alerts

    if float(backtest.get("max_dd", 0.0)) > th.max_drawdown:
        alerts.append(f"max_dd_exceeded:{backtest.get('max_dd'):.4f}>{th.max_drawdown:.4f}")
    if float(backtest.get("alpha", 0.0)) < th.min_alpha:
        alerts.append(f"alpha_below_threshold:{backtest.get('alpha'):.4f}<{th.min_alpha:.4f}")

    locked_return = float(((locked or {}).get("oos20_metrics") or {}).get("total_return", 0.0))
    curr_return = float(backtest.get("total_return", 0.0))
    return_drift = curr_return - locked_return
    if return_drift < th.min_return_drift_vs_lock:
        alerts.append(
            f"return_drift_vs_lock:{return_drift:.4f}<{th.min_return_drift_vs_lock:.4f}"
        )

    if live.get("status") == "ok":
        fr = live.get("fill_rate_7d")
        sl = live.get("mean_slippage_7d")
        rc = int(live.get("reconcile_errors_7d", 0))
        if fr is not None and fr < th.min_fill_rate_7d:
            alerts.append(f"fill_rate_7d_low:{fr:.4f}<{th.min_fill_rate_7d:.4f}")
        if sl is not None and sl > th.max_slippage_7d:
            alerts.append(f"slippage_7d_high:{sl:.4f}>{th.max_slippage_7d:.4f}")
        if rc > th.max_reconcile_errors_7d:
            alerts.append(f"reconcile_errors_7d_high:{rc}>{th.max_reconcile_errors_7d}")

    return alerts


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_latest_md(path: Path, payload: Dict) -> None:
    bt = payload.get("backtest", {})
    lv = payload.get("live", {})
    alerts = payload.get("alerts", [])
    lines = [
        "# Strategy Effectiveness Snapshot",
        f"- generated_at_utc: {payload.get('generated_at_utc')}",
        f"- stable_run_id: {payload.get('stable_run_id')}",
        "",
        "## Backtest Recheck",
        f"- total_return: {bt.get('total_return')}",
        f"- alpha: {bt.get('alpha')}",
        f"- max_dd: {bt.get('max_dd')}",
        "",
        "## Live 7D",
        f"- fill_rate_7d: {lv.get('fill_rate_7d')}",
        f"- mean_slippage_7d: {lv.get('mean_slippage_7d')}",
        f"- reconcile_errors_7d: {lv.get('reconcile_errors_7d')}",
        "",
        "## Alerts",
    ]
    if alerts:
        lines.extend([f"- {x}" for x in alerts])
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_recent_history(path: Path, days: int = 7) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ts = datetime.fromisoformat(row.get("generated_at_utc", "1970-01-01T00:00:00+00:00"))
        if ts >= cutoff:
            out.append(row)
    return out


def _write_weekly_report(path: Path, recent: List[Dict]) -> None:
    lines = [
        "# Weekly Strategy Effectiveness Report",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- snapshots_included: {len(recent)}",
        "",
    ]
    if not recent:
        lines.append("No snapshots in last 7 days.")
    else:
        bt = [r.get("backtest", {}) for r in recent if "error" not in r.get("backtest", {})]
        if bt:
            avg_ret = sum(float(x.get("total_return", 0.0)) for x in bt) / len(bt)
            avg_alpha = sum(float(x.get("alpha", 0.0)) for x in bt) / len(bt)
            worst_dd = max(float(x.get("max_dd", 0.0)) for x in bt)
            lines.extend(
                [
                    "## Weekly Averages",
                    f"- avg_total_return_recheck: {avg_ret:.6f}",
                    f"- avg_alpha_recheck: {avg_alpha:.6f}",
                    f"- worst_max_dd_recheck: {worst_dd:.6f}",
                    "",
                ]
            )
        alert_counts: Dict[str, int] = {}
        for r in recent:
            for a in r.get("alerts", []):
                alert_counts[a] = alert_counts.get(a, 0) + 1
        lines.append("## Alert Frequency")
        if alert_counts:
            lines.extend([f"- {k}: {v}" for k, v in sorted(alert_counts.items(), key=lambda kv: -kv[1])])
        else:
            lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor strategy effectiveness for stable MoE")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--data-path", type=str, default=None, help="Optional OOS CSV path")
    parser.add_argument("--send-alert", action="store_true", help="Send alert if violations found")
    args = parser.parse_args()

    stable_registry = _load_json(PROJECT_ROOT / "stable_model_registry.json")
    moe_registry = _load_json(PROJECT_ROOT / "moe_model_registry.json")
    manifest, experts, gate, data_path = _resolve_paths(moe_registry, args.data_path)
    gate_temperature = float(moe_registry.get("stable_gate_temperature", 0.68))

    backtest = _run_backtest(
        manifest=manifest,
        experts=experts,
        gate=gate,
        data_path=data_path,
        gate_temperature=gate_temperature,
    )
    live = _live_metrics_from_trade_logs()
    alerts = _evaluate_alerts(backtest, live, moe_registry, Thresholds())

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "stable_run_id": stable_registry.get("stable_run_id"),
        "stable_model_path": stable_registry.get("stable_model_path"),
        "data_path": str(data_path),
        "backtest": backtest,
        "live": live,
        "alerts": alerts,
    }

    latest_json = RUNS_MONITORING / "effectiveness_latest.json"
    history_jsonl = RUNS_MONITORING / "effectiveness_history.jsonl"
    latest_md = RUNS_MONITORING / "effectiveness_latest.md"

    _write_json(latest_json, payload)
    _append_jsonl(history_jsonl, payload)
    _write_latest_md(latest_md, payload)

    if args.mode == "weekly":
        recent = _load_recent_history(history_jsonl, days=7)
        weekly_md = RUNS_MONITORING / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.md"
        _write_weekly_report(weekly_md, recent)
        print(f"weekly_report={weekly_md}")

    if args.send_alert and alerts:
        sys.path.insert(0, str(PROJECT_ROOT))
        from crypto_trader.alerting import AlertManager
        AlertManager().send("WARN", "【策略有效性告警】\n" + "\n".join(alerts))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
