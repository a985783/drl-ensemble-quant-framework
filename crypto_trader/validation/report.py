from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import flatten_metric_row


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def write_metrics_csv(path: Path, scenarios: Iterable[Mapping[str, object]]) -> None:
    rows = []
    for scenario in scenarios:
        metrics = scenario.get("metrics", {})
        if isinstance(metrics, Mapping):
            rows.append(flatten_metric_row(str(scenario.get("name", "")), metrics))

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_summary(report: Mapping[str, object]) -> str:
    verdict = report.get("verdict", {})
    if not isinstance(verdict, Mapping):
        verdict = {}

    lines = [
        "# Alpha Validation Summary",
        "",
        f"- Run ID: `{report.get('run_id', '')}`",
        f"- Verdict: **{verdict.get('status', 'UNKNOWN')}**",
        f"- Scenarios: {len(report.get('scenarios', []))}",
        "",
        "## Blocking Items",
    ]

    for item in verdict.get("blocking_items", []) or ["None"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Failures")
    for item in verdict.get("failures", []) or ["None"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Warnings")
    for item in verdict.get("warnings", []) or ["None"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Passed Checks")
    for item in verdict.get("passes", []) or ["None"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Scenario Metrics")
    lines.append("")
    lines.append("| Scenario | Return | Alpha | Max DD | Status |")
    lines.append("|---|---:|---:|---:|---|")
    for scenario in report.get("scenarios", []):
        if not isinstance(scenario, Mapping):
            continue
        metrics = scenario.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        lines.append(
            "| {name} | {ret:.2%} | {alpha:.2%} | {dd:.2%} | {status} |".format(
                name=scenario.get("name", ""),
                ret=float(metrics.get("total_return", 0.0) or 0.0),
                alpha=float(metrics.get("alpha", 0.0) or 0.0),
                dd=float(metrics.get("max_drawdown", 0.0) or 0.0),
                status=scenario.get("status", ""),
            )
        )

    lines.append("")
    lines.append("## Notes")
    for item in report.get("notes", []) or ["No additional notes."]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def write_report_bundle(output_dir: Path, report: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "verdict.json", report.get("verdict", {}))
    write_metrics_csv(output_dir / "metrics.csv", report.get("scenarios", []))
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(render_markdown_summary(report))
