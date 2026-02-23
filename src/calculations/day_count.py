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


def days_30_360_vectorized(start_dates: pd.Series, end_dates: pd.Series) -> pd.Series:
    """Vectorized US/NASD 30/360 day count for aligned start/end date series."""
    start = pd.to_datetime(start_dates)
    end = pd.to_datetime(end_dates)

    d1 = start.dt.day.astype(int)
    d2 = end.dt.day.astype(int)

    d1 = d1.where(d1 != 31, 30)
    d2 = d2.where(~((d2 == 31) & d1.isin([30, 31])), 30)

    return (
        360 * (end.dt.year - start.dt.year)
        + 30 * (end.dt.month - start.dt.month)
        + (d2 - d1)
    ).astype(int)
