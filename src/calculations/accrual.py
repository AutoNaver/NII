"""Accrual and deal activity logic."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.calculations.day_count import days_30_360
from src.calculations.day_count import days_30_360_vectorized


def _to_ts(value: pd.Timestamp | datetime | date | str) -> pd.Timestamp:
    return pd.Timestamp(value)


def is_active(
    value_date: pd.Timestamp | datetime | date | str,
    maturity_date: pd.Timestamp | datetime | date | str,
    as_of_date: pd.Timestamp | datetime | date | str,
) -> bool:
    """A deal is active at t when value_date <= t < maturity_date."""
    v = _to_ts(value_date)
    m = _to_ts(maturity_date)
    t = _to_ts(as_of_date)
    return v <= t < m


def accrued_interest_eur(
    notional: float,
    annual_coupon: float,
    start_date: pd.Timestamp | datetime | date | str,
    end_date: pd.Timestamp | datetime | date | str,
) -> float:
    """Accrued EUR interest for the given period under 30/360."""
    start = _to_ts(start_date)
    end = _to_ts(end_date)
    if end <= start:
        return 0.0
    days = days_30_360(start, end)
    return float(notional) * float(annual_coupon) * days / 360.0


def accrued_interest_for_overlap(
    notional: float,
    annual_coupon: float,
    deal_value_date: pd.Timestamp,
    deal_maturity_date: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> float:
    """Accrue interest only for the overlap between deal life and a time window.

    The overlap is treated as [start, end) by clipping to deal boundaries.
    """
    start = max(pd.Timestamp(deal_value_date), pd.Timestamp(window_start))
    end = min(pd.Timestamp(deal_maturity_date), pd.Timestamp(window_end))
    if end <= start:
        return 0.0
    return accrued_interest_eur(notional, annual_coupon, start, end)


def accrued_interest_for_overlap_vectorized(
    notional: pd.Series,
    annual_coupon: pd.Series,
    deal_value_date: pd.Series,
    deal_maturity_date: pd.Series,
    window_start: pd.Series | pd.Timestamp,
    window_end: pd.Series | pd.Timestamp,
) -> pd.Series:
    """Vectorized overlap accrual for [window_start, window_end) under 30/360."""
    index = deal_value_date.index
    n = pd.Series(notional, index=index, dtype=float)
    c = pd.Series(annual_coupon, index=index, dtype=float)
    v = pd.to_datetime(pd.Series(deal_value_date, index=index))
    m = pd.to_datetime(pd.Series(deal_maturity_date, index=index))
    ws = pd.to_datetime(pd.Series(window_start, index=index))
    we = pd.to_datetime(pd.Series(window_end, index=index))

    start = pd.concat([v, ws], axis=1).max(axis=1)
    end = pd.concat([m, we], axis=1).min(axis=1)

    out = pd.Series(0.0, index=index, dtype=float)
    active = end > start
    if not bool(active.any()):
        return out

    days = days_30_360_vectorized(start.loc[active], end.loc[active]).astype(float)
    out.loc[active] = n.loc[active] * c.loc[active] * days / 360.0
    return out
