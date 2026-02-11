import csv
import tempfile
from pathlib import Path
import sys

# Ensure crypto_trader on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment_log import append_experiment_log


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / "experiment_log.csv"

    row = {
        "date": "2026-02-03",
        "hypothesis": "unit test",
        "data_version": "hash123",
        "strategy_version": "repo-local",
        "params": "p1=1",
        "results": "ok",
        "risk_notes": "none",
        "decision": "continue",
        "owner": "self",
    }

    append_experiment_log(row, out_path)

    with out_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["hypothesis"] == "unit test"


if __name__ == "__main__":
    main()
