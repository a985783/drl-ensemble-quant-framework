"""Data version recording utilities for reproducibility."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Dict, List, Optional

import pandas as pd


@dataclass
class DataVersionRecord:
    recorded_at_utc: str
    source: str
    symbol: str
    interval: str
    requested_start: str
    requested_end: str
    data_start: str
    data_end: str
    rows: int
    cols: int
    columns: str
    hash_sha256: str
    note: str = ""


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    # Stable index ordering
    df_norm = df_norm.sort_index()
    # Stable column ordering
    df_norm = df_norm.reindex(sorted(df_norm.columns), axis=1)
    return df_norm


def _index_to_datestr(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def hash_dataframe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    df_norm = _normalize_df(df)
    hashed = pd.util.hash_pandas_object(df_norm, index=True).values
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return digest


def build_data_version_record(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    requested_start: str,
    requested_end: str,
    source: str,
    note: str = "",
) -> Dict[str, object]:
    if df is None or df.empty:
        data_start = ""
        data_end = ""
        cols = 0
        columns = ""
        rows = 0
    else:
        data_start = _index_to_datestr(df.index.min())
        data_end = _index_to_datestr(df.index.max())
        cols = len(df.columns)
        columns = "|".join([str(c) for c in df.columns])
        rows = int(len(df))

    record = DataVersionRecord(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        source=source,
        symbol=symbol,
        interval=interval,
        requested_start=requested_start,
        requested_end=requested_end,
        data_start=data_start,
        data_end=data_end,
        rows=rows,
        cols=cols,
        columns=columns,
        hash_sha256=hash_dataframe(df),
        note=note,
    )
    return asdict(record)


def append_data_version_record(record: Dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def record_data_version(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    requested_start: str,
    requested_end: str,
    source: str,
    output_path: Path,
    note: str = "",
) -> Dict[str, object]:
    record = build_data_version_record(
        df,
        symbol=symbol,
        interval=interval,
        requested_start=requested_start,
        requested_end=requested_end,
        source=source,
        note=note,
    )
    append_data_version_record(record, output_path)
    return record
