import csv
import tempfile
from pathlib import Path
import sys

import pandas as pd

# Ensure crypto_trader is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_versioning import build_data_version_record, append_data_version_record


def main():
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    record = build_data_version_record(
        df,
        symbol="ETH/USDT:USDT",
        interval="1d",
        requested_start="2020-01-01",
        requested_end="2020-01-02",
        source="okx",
        note="unit-test",
    )

    assert record["symbol"] == "ETH/USDT:USDT"
    assert record["interval"] == "1d"
    assert record["data_start"] == "2020-01-01"
    assert record["data_end"] == "2020-01-02"
    assert record["rows"] == 2
    assert record["cols"] == 5
    assert len(record["hash_sha256"]) == 64

    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / "data_versions.csv"
    append_data_version_record(record, out_path)

    with out_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETH/USDT:USDT"


if __name__ == "__main__":
    main()
