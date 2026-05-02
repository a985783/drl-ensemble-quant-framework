from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def parse_trade_log_for_monitoring(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse trade_logs.csv with tolerance for schema drift.

    Historical rows may have different column counts as fields were added.
    For monitoring we only need a stable subset, extracted by relative position.
    """
    trade_log = Path(path)
    if not trade_log.exists():
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    with trade_log.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        # Header may be stale compared with row schema; skip and parse rows directly.
        _ = next(reader, None)
        for raw in reader:
            if not raw:
                continue
            # Need at least enough fields to index the stable tail subset.
            if len(raw) < 14:
                continue

            rows.append(
                {
                    "Timestamp": raw[0].strip(),
                    "Net_Worth": raw[1].strip() if len(raw) > 1 else "",
                    "Exec_Pos": raw[-14].strip(),
                    "Action": raw[-13].strip(),
                    "Contracts": raw[-12].strip(),
                    "Slippage": raw[-9].strip(),
                    "Reconcile_Diff": raw[-3].strip(),
                }
            )

    return pd.DataFrame(rows)

