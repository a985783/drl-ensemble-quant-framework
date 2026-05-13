"""Utilities for maintaining experiment_log.csv in quant_docs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


def append_experiment_log(row: Dict[str, str], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date",
        "hypothesis",
        "data_version",
        "strategy_version",
        "params",
        "results",
        "risk_notes",
        "decision",
        "owner",
    ]

    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
