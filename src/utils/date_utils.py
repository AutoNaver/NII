"""Date helpers shared across calculations and dashboard layers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def to_timestamp(value: pd.Timestamp | datetime | date | str) -> pd.Timestamp:
    """Convert an input value to a timezone-naive pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts


def previous_calendar_month_window(as_of_month_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start and end timestamps of the month preceding the provided month-end date."""
    as_of = to_timestamp(as_of_month_end)
    prev_end = (as_of - pd.offsets.MonthEnd(1)).normalize()
    prev_start = prev_end.replace(day=1)
    return prev_start, prev_end


def month_end_sequence(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Generate inclusive month-end dates between start and end."""
    start = to_timestamp(start_date) + pd.offsets.MonthEnd(0)
    end = to_timestamp(end_date) + pd.offsets.MonthEnd(0)
    if start > end:
        return []
    return list(pd.date_range(start=start, end=end, freq='ME'))
