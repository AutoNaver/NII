"""Day-count utilities."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def _to_date(value: pd.Timestamp | datetime | date | str) -> date:
    return pd.Timestamp(value).date()


def days_30_360(start_date: pd.Timestamp | datetime | date | str, end_date: pd.Timestamp | datetime | date | str) -> int:
    """Compute day count using the US/NASD 30/360 convention."""
    start = _to_date(start_date)
    end = _to_date(end_date)

    d1 = start.day
    d2 = end.day

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 in (30, 31):
        d2 = 30

    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
