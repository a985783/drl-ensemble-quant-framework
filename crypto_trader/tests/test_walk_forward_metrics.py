import csv
import tempfile
from pathlib import Path
import sys

# Add crypto_trader to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from walk_forward.metrics import build_metrics_rows, write_metrics_csv


def main():
    results = [
        {
            "fold": "fold1_test2022",
            "test_period": "2022-01-01 ~ 2022-12-31",
            "total_return": 0.10,
            "benchmark": 0.05,
            "alpha": 0.05,
            "max_dd": 0.12,
            "net_worths": [10000, 10500, 11000],
        }
    ]

    rows = build_metrics_rows(results)
    assert len(rows) == 1
    row = rows[0]
    assert row["fold"] == "fold1_test2022"
    assert row["test_start"] == "2022-01-01"
    assert row["test_end"] == "2022-12-31"
    assert abs(row["total_return"] - 0.10) < 1e-9
    assert abs(row["benchmark_return"] - 0.05) < 1e-9
    assert abs(row["alpha"] - 0.05) < 1e-9
    assert abs(row["max_drawdown"] - 0.12) < 1e-9
    assert row["num_points"] == 3

    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / "metrics.csv"
    write_metrics_csv(rows, out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    assert len(data) == 1
    assert data[0]["fold"] == "fold1_test2022"


if __name__ == "__main__":
    main()
