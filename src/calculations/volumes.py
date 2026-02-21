"""Monthly bucket volume and coupon metrics."""

from __future__ import annotations

import pandas as pd

from src.calculations.accrual import is_active
from src.calculations.nii import active_deals_snapshot
from src.calculations.nii import compute_monthly_realized_nii
from src.utils.date_utils import month_end_sequence


def _weighted_coupon(active_df: pd.DataFrame) -> float:
    if active_df.empty:
        return 0.0
    weights = active_df['notional'].abs()
    if float(weights.sum()) == 0.0:
        return 0.0
    return float((active_df['coupon'] * weights).sum() / weights.sum())


def compute_monthly_buckets(deals_df: pd.DataFrame, month_end: pd.Timestamp) -> pd.DataFrame:
    """Build monthly aggregate rows through the selected month-end."""
    max_month_end = pd.Timestamp(month_end) + pd.offsets.MonthEnd(0)
    min_start = deals_df['value_date'].min()
    month_ends = month_end_sequence(min_start, max_month_end)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for me in month_ends:
        active = deals_df[
            deals_df.apply(lambda r: is_active(r['value_date'], r['maturity_date'], me), axis=1)
        ]
        month_start = me.replace(day=1)
        interest_paid = compute_monthly_realized_nii(deals_df, month_start, me)
        rows.append(
            {
                'month_end': me,
                'total_active_notional': float(active['notional'].sum()) if not active.empty else 0.0,
                'weighted_avg_coupon': _weighted_coupon(active),
                'interest_paid_eur': float(interest_paid),
                'active_deal_count': int(len(active)),
            }
        )

    return pd.DataFrame(rows)


def compute_runoff_profile(deals_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Compute a monthly runoff profile for deals active at as-of date.

    Month 0 starts with the full active cohort. Subsequent months roll off as deals mature.
    """
    as_of = pd.Timestamp(as_of_date) + pd.offsets.MonthEnd(0)
    cohort = active_deals_snapshot(deals_df, as_of)
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                'month_offset',
                'month_end',
                'remaining_notional',
                'remaining_abs_notional',
                'matured_abs_notional',
                'remaining_pct_of_initial_abs',
                'active_deal_count',
            ]
        )

    max_me = cohort['maturity_date'].max() + pd.offsets.MonthEnd(0)
    month_ends = month_end_sequence(as_of, max_me)
    initial_abs = float(cohort['notional'].abs().sum())

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    previous_me = as_of - pd.offsets.MonthEnd(1)
    for offset, me in enumerate(month_ends):
        remaining = cohort[cohort['maturity_date'] > me]
        matured_in_month = cohort[
            (cohort['maturity_date'] > previous_me) & (cohort['maturity_date'] <= me)
        ]
        remaining_abs = float(remaining['notional'].abs().sum()) if not remaining.empty else 0.0
        rows.append(
            {
                'month_offset': int(offset),
                'month_end': me,
                'remaining_notional': float(remaining['notional'].sum()) if not remaining.empty else 0.0,
                'remaining_abs_notional': remaining_abs,
                'matured_abs_notional': float(matured_in_month['notional'].abs().sum())
                if not matured_in_month.empty
                else 0.0,
                'remaining_pct_of_initial_abs': (remaining_abs / initial_abs) if initial_abs else 0.0,
                'active_deal_count': int(len(remaining)),
            }
        )
        previous_me = me

    return pd.DataFrame(rows)


def compare_runoff_profiles(profile_d1: pd.DataFrame, profile_d2: pd.DataFrame) -> pd.DataFrame:
    """Align two runoff profiles by month offset and compute deltas (d2 - d1)."""
    left = profile_d1.add_suffix('_d1')
    right = profile_d2.add_suffix('_d2')
    merged = left.merge(
        right,
        left_on='month_offset_d1',
        right_on='month_offset_d2',
        how='outer',
    )

    merged['month_offset'] = merged['month_offset_d1'].fillna(merged['month_offset_d2']).astype(int)
    merged['month_end_d1'] = pd.to_datetime(merged['month_end_d1'])
    merged['month_end_d2'] = pd.to_datetime(merged['month_end_d2'])

    numeric_cols = [
        'remaining_notional',
        'remaining_abs_notional',
        'matured_abs_notional',
        'remaining_pct_of_initial_abs',
        'active_deal_count',
    ]
    for col in numeric_cols:
        merged[f'{col}_d1'] = merged[f'{col}_d1'].fillna(0.0)
        merged[f'{col}_d2'] = merged[f'{col}_d2'].fillna(0.0)
        merged[f'{col}_delta'] = merged[f'{col}_d2'] - merged[f'{col}_d1']

    return merged.sort_values('month_offset').reset_index(drop=True)


def compare_monthly_bucket_series(series_d1: pd.DataFrame, series_d2: pd.DataFrame) -> pd.DataFrame:
    """Compare monthly bucket outputs on calendar month_end and compute deltas (d2 - d1)."""
    cols = ['month_end', 'total_active_notional', 'weighted_avg_coupon', 'interest_paid_eur', 'active_deal_count']
    d1 = series_d1[cols].copy().add_suffix('_d1').rename(columns={'month_end_d1': 'month_end'})
    d2 = series_d2[cols].copy().add_suffix('_d2').rename(columns={'month_end_d2': 'month_end'})
    merged = d1.merge(d2, on='month_end', how='outer').sort_values('month_end').reset_index(drop=True)

    for col in ['total_active_notional', 'weighted_avg_coupon', 'interest_paid_eur', 'active_deal_count']:
        merged[f'{col}_d1'] = merged[f'{col}_d1'].fillna(0.0)
        merged[f'{col}_d2'] = merged[f'{col}_d2'].fillna(0.0)
        merged[f'{col}_delta'] = merged[f'{col}_d2'] - merged[f'{col}_d1']

    return merged


def _remaining_abs_series(cohort_df: pd.DataFrame, as_of_date: pd.Timestamp, max_offset: int) -> dict[int, float]:
    """Return remaining absolute notional by month offset for a fixed cohort."""
    as_of = pd.Timestamp(as_of_date) + pd.offsets.MonthEnd(0)
    values: dict[int, float] = {}
    for offset in range(max_offset + 1):
        month_end = as_of + pd.offsets.MonthEnd(offset)
        remaining = cohort_df[cohort_df['maturity_date'] > month_end]
        values[offset] = float(remaining['notional'].abs().sum()) if not remaining.empty else 0.0
    return values


def compute_runoff_delta_attribution(
    deals_df: pd.DataFrame,
    d1: pd.Timestamp,
    d2: pd.Timestamp,
) -> pd.DataFrame:
    """Compare runoff between two dates with attribution to added vs matured cohorts."""
    t1 = pd.Timestamp(d1) + pd.offsets.MonthEnd(0)
    t2 = pd.Timestamp(d2) + pd.offsets.MonthEnd(0)

    profile_d1 = compute_runoff_profile(deals_df, t1)
    profile_d2 = compute_runoff_profile(deals_df, t2)
    if profile_d1.empty and profile_d2.empty:
        return pd.DataFrame()

    max_offset = int(
        max(
            profile_d1['month_offset'].max() if not profile_d1.empty else 0,
            profile_d2['month_offset'].max() if not profile_d2.empty else 0,
        )
    )

    active1 = active_deals_snapshot(deals_df, t1)
    active2 = active_deals_snapshot(deals_df, t2)
    ids1 = set(active1['deal_id'].tolist())
    ids2 = set(active2['deal_id'].tolist())
    added_cohort = active2[active2['deal_id'].isin(ids2 - ids1)]
    matured_cohort = active1[active1['deal_id'].isin(ids1 - ids2)]

    rem_d1 = _remaining_abs_series(active1, t1, max_offset)
    rem_d2 = _remaining_abs_series(active2, t2, max_offset)
    rem_added = _remaining_abs_series(added_cohort, t2, max_offset)
    rem_matured = _remaining_abs_series(matured_cohort, t1, max_offset)

    rows: list[dict[str, float | int]] = []
    for offset in range(max_offset + 1):
        delta = rem_d2[offset] - rem_d1[offset]
        explained = rem_added[offset] - rem_matured[offset]
        rows.append(
            {
                'month_offset': offset,
                'remaining_abs_notional_d1': rem_d1[offset],
                'remaining_abs_notional_d2': rem_d2[offset],
                'remaining_abs_notional_delta': delta,
                'added_remaining_abs_notional': rem_added[offset],
                'matured_remaining_abs_notional': rem_matured[offset],
                'explained_delta': explained,
                'residual_delta': delta - explained,
            }
        )
    return pd.DataFrame(rows)
