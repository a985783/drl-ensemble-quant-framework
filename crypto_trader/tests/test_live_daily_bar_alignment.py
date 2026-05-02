from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from live_trading_okx import drop_incomplete_daily_bar, latest_completed_daily_start


def test_latest_completed_daily_start_excludes_current_utc_day() -> None:
    now = datetime(2026, 5, 2, 0, 5, tzinfo=timezone.utc)

    completed = latest_completed_daily_start(now)

    assert completed == pd.Timestamp("2026-05-01T00:00:00Z")


def test_drop_incomplete_daily_bar_removes_current_bar_start() -> None:
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(
            ["2026-04-30T00:00:00Z", "2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z"]
        ),
    )
    now = datetime(2026, 5, 2, 0, 5, tzinfo=timezone.utc)

    filtered = drop_incomplete_daily_bar(df, now)

    assert list(filtered.index) == [pd.Timestamp("2026-04-30T00:00:00Z"), pd.Timestamp("2026-05-01T00:00:00Z")]
