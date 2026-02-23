"""Net interest income aggregation functions."""

from __future__ import annotations

import pandas as pd

from src.calculations.accrual import accrued_interest_for_overlap_vectorized


def compute_monthly_realized_nii(
    deals_df: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> float:
    """Sum realized interest paid in a month in EUR.

    Month window is treated as [month_start, month_end + 1 day).
    """
    start = pd.Timestamp(month_start)
    end_exclusive = pd.Timestamp(month_end) + pd.Timedelta(days=1)

    accrued = accrued_interest_for_overlap_vectorized(
        notional=deals_df['notional'],
        annual_coupon=deals_df['coupon'],
        deal_value_date=deals_df['value_date'],
        deal_maturity_date=deals_df['maturity_date'],
        window_start=start,
        window_end=end_exclusive,
    )
    return float(accrued.sum())


def accrued_interest_to_date(deals_df: pd.DataFrame, as_of_date: pd.Timestamp) -> float:
    """Accrued interest for active deals from value date to as-of date."""
    as_of = pd.Timestamp(as_of_date)
    active = active_deals_snapshot(deals_df, as_of)
    if active.empty:
        return 0.0
    accrued = accrued_interest_for_overlap_vectorized(
        notional=active['notional'],
        annual_coupon=active['coupon'],
        deal_value_date=active['value_date'],
        deal_maturity_date=active['maturity_date'],
        window_start=active['value_date'],
        window_end=as_of,
    )
    return float(accrued.sum())


def active_deals_snapshot(deals_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Return active deals at a specific date."""
    as_of = pd.Timestamp(as_of_date)
    mask = (deals_df['value_date'] <= as_of) & (as_of < deals_df['maturity_date'])
    return deals_df.loc[mask].copy()


def compare_month_ends(deals_df: pd.DataFrame, d1: pd.Timestamp, d2: pd.Timestamp) -> dict[str, pd.DataFrame | float]:
    """Compare two month-end dates and classify deal differences."""
    t1 = pd.Timestamp(d1)
    t2 = pd.Timestamp(d2)

    active1 = active_deals_snapshot(deals_df, t1)
    active2 = active_deals_snapshot(deals_df, t2)

    merged = active1.merge(
        active2,
        on='deal_id',
        how='outer',
        suffixes=('_d1', '_d2'),
        indicator=True,
    )

    added = merged[merged['_merge'] == 'right_only'][['deal_id', 'notional_d2', 'coupon_d2']].copy()
    matured = merged[merged['_merge'] == 'left_only'][['deal_id', 'notional_d1', 'coupon_d1']].copy()

    both = merged[merged['_merge'] == 'both'].copy()
    notional_changed = both[both['notional_d1'] != both['notional_d2']][
        ['deal_id', 'notional_d1', 'notional_d2']
    ].copy()
    coupon_changed = both[both['coupon_d1'] != both['coupon_d2']][
        ['deal_id', 'coupon_d1', 'coupon_d2']
    ].copy()

    change_rows: list[dict[str, float | int | str]] = []
    for _, row in merged.iterrows():
        status_parts: list[str] = []
        if row['_merge'] == 'right_only':
            status_parts.append('new')
        elif row['_merge'] == 'left_only':
            status_parts.append('matured')
        else:
            if row['notional_d1'] != row['notional_d2']:
                status_parts.append('notional_changed')
            if row['coupon_d1'] != row['coupon_d2']:
                status_parts.append('coupon_changed')
        if status_parts:
            change_rows.append(
                {
                    'deal_id': row['deal_id'],
                    'status': ','.join(status_parts),
                    'notional_d1': row['notional_d1'],
                    'notional_d2': row['notional_d2'],
                    'coupon_d1': row['coupon_d1'],
                    'coupon_d2': row['coupon_d2'],
                }
            )

    deal_changes = pd.DataFrame(change_rows)

    return {
        'active_count_d1': int(len(active1)),
        'active_count_d2': int(len(active2)),
        'active_count_delta': int(len(active2) - len(active1)),
        'added': added,
        'matured': matured,
        'notional_changed': notional_changed,
        'coupon_changed': coupon_changed,
        'deal_changes': deal_changes,
    }
