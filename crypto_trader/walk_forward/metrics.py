"""Utilities for exporting walk-forward metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_test_period(test_period: str) -> Tuple[str, str]:
    if not test_period:
        return "", ""
    parts = test_period.split("~")
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def build_metrics_rows(results: Iterable[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        test_period = result.get("test_period", "")
        test_start, test_end = _parse_test_period(test_period)
        net_worths = result.get("net_worths", [])
        rows.append(
            {
                "fold": result.get("fold", ""),
                "test_start": test_start,
                "test_end": test_end,
                "total_return": float(result.get("total_return", 0.0)),
                "benchmark_return": float(result.get("benchmark", 0.0)),
                "alpha": float(result.get("alpha", 0.0)),
                "max_drawdown": float(result.get("max_dd", 0.0)),
                "num_points": int(len(net_worths)),
            }
        )
    return rows


def write_metrics_csv(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
