"""Monthly bucket volume and coupon metrics."""

from __future__ import annotations

import pandas as pd

from src.calculations.accrual import accrued_interest_for_overlap, is_active
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


def _remaining_maturity_months(as_of_me: pd.Timestamp, maturity_date: pd.Timestamp) -> int:
    """Remaining maturity in whole months from as_of month-end to maturity month-end."""
    as_of = pd.Timestamp(as_of_me) + pd.offsets.MonthEnd(0)
    maturity_me = pd.Timestamp(maturity_date) + pd.offsets.MonthEnd(0)
    months = (maturity_me.to_period('M') - as_of.to_period('M')).n
    return max(0, int(months))


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


def compute_remaining_maturity_buckets(deals_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate active deals into remaining-maturity buckets with cumulative metrics.

    Buckets are month offsets from the as-of month-end (0,1,2,...horizon),
    where offset k contains deals maturing at as-of month-end + k months.
    """
    as_of = pd.Timestamp(as_of_date) + pd.offsets.MonthEnd(0)
    active = active_deals_snapshot(deals_df, as_of)
    buckets_range = range(0, 241)  # fixed 0..240

    active = active.copy()
    active['bucket'] = active['maturity_date'].apply(lambda d: min(240, _remaining_maturity_months(as_of, d)))

    grouped = active.groupby('bucket').agg(
        deal_count=('deal_id', 'count'),
        abs_notional=('notional', lambda s: float(s.abs().sum())),
        signed_notional=('notional', 'sum'),
    )
    notional_coupon = (
        active.assign(_nc=active['notional'] * active['coupon'])
        .groupby('bucket')['_nc']
        .sum()
        .astype(float)
    )
    grouped['notional_coupon'] = notional_coupon
    grouped = grouped.reset_index().rename(columns={'bucket': 'remaining_maturity_months'}).fillna(0.0)

    full_index = pd.Index(buckets_range, name='remaining_maturity_months')
    grouped = grouped.set_index('remaining_maturity_months').reindex(full_index, fill_value=0.0).reset_index()
    grouped = grouped.sort_values('remaining_maturity_months')
    grouped['cumulative_abs_notional'] = grouped['abs_notional'][::-1].cumsum()[::-1]
    grouped['cumulative_signed_notional'] = grouped['signed_notional'][::-1].cumsum()[::-1]
    grouped['cumulative_notional_coupon'] = grouped['notional_coupon'][::-1].cumsum()[::-1]
    return grouped.reset_index(drop=True)


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


def compute_monthly_activity_metrics(
    deals_df: pd.DataFrame,
    month_ends: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute monthly active/added deal counts and notional*coupon bucket metrics."""
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for me_raw in month_ends:
        me = pd.Timestamp(me_raw) + pd.offsets.MonthEnd(0)
        ms = me.replace(day=1)

        active = deals_df[
            deals_df.apply(lambda r: is_active(r['value_date'], r['maturity_date'], me), axis=1)
        ]
        added = deals_df[(deals_df['value_date'] >= ms) & (deals_df['value_date'] <= me)]

        rows.append(
            {
                'month_end': me,
                'active_deal_count': int(len(active)),
                'added_deal_count': int(len(added)),
                'active_notional_coupon': float((active['notional'] * active['coupon']).sum())
                if not active.empty
                else 0.0,
                'added_notional_coupon': float((added['notional'] * added['coupon']).sum())
                if not added.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values('month_end').reset_index(drop=True)


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
    """Compare runoff between two month-end views using remaining-maturity buckets with attribution."""
    t1 = pd.Timestamp(d1) + pd.offsets.MonthEnd(0)
    t2 = pd.Timestamp(d2) + pd.offsets.MonthEnd(0)

    buckets1 = compute_remaining_maturity_buckets(deals_df, t1)
    buckets2 = compute_remaining_maturity_buckets(deals_df, t2)
    if buckets1.empty and buckets2.empty:
        return pd.DataFrame()

    active1 = active_deals_snapshot(deals_df, t1)
    active2 = active_deals_snapshot(deals_df, t2)
    ids1 = set(active1['deal_id'].tolist())
    ids2 = set(active2['deal_id'].tolist())
    added = active2[active2['deal_id'].isin(ids2 - ids1)].copy()
    matured = active1[active1['deal_id'].isin(ids1 - ids2)].copy()
    for df, as_of in ((added, t2), (matured, t1)):
        if not df.empty:
            df['remaining_maturity_months'] = df['maturity_date'].apply(
                lambda d: _remaining_maturity_months(as_of, d)
            )

    def _agg(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    'remaining_maturity_months',
                    'abs_notional',
                    'notional',
                    'notional_coupon',
                    'effective_interest',
                    'deal_count',
                ]
            )
        work = df.copy()
        if 'remaining_maturity_months' not in work.columns:
            work['remaining_maturity_months'] = work['maturity_date'].apply(
                lambda m: _remaining_maturity_months(as_of, m)
            )
        work['remaining_maturity_months'] = work['remaining_maturity_months'].astype(int).clip(lower=0, upper=240)

        # Effective-interest per bucket is accrued over the bucket month window
        # (as_of + bucket months), so mid-month maturities/additions only
        # contribute for active overlap days under 30/360.
        work['_bucket_month_end'] = work['remaining_maturity_months'].apply(
            lambda k: pd.Timestamp(as_of) + pd.offsets.MonthEnd(int(k))
        )
        work['_effective_interest'] = work.apply(
            lambda r: accrued_interest_for_overlap(
                notional=r['notional'],
                annual_coupon=r['coupon'],
                deal_value_date=r['value_date'],
                deal_maturity_date=r['maturity_date'],
                window_start=pd.Timestamp(r['_bucket_month_end']).replace(day=1),
                window_end=pd.Timestamp(r['_bucket_month_end']) + pd.Timedelta(days=1),
            ),
            axis=1,
        )

        grouped = work.groupby('remaining_maturity_months').agg(
            abs_notional=('notional', lambda s: float(s.abs().sum())),
            notional=('notional', 'sum'),
            deal_count=('deal_id', 'count'),
        )
        notional_coupon = (
            work.assign(_nc=work['notional'] * work['coupon'])
            .groupby('remaining_maturity_months')['_nc']
            .sum()
            .astype(float)
        )
        effective_interest = (
            work.groupby('remaining_maturity_months')['_effective_interest']
            .sum()
            .astype(float)
        )
        grouped['notional_coupon'] = notional_coupon
        grouped['effective_interest'] = effective_interest
        return grouped.reset_index()

    active1_buckets = _agg(active1, t1)
    active2_buckets = _agg(active2, t2)
    added_buckets = _agg(added, t2)
    matured_buckets = _agg(matured, t1)

    buckets1 = buckets1.set_index('remaining_maturity_months')
    buckets2 = buckets2.set_index('remaining_maturity_months')
    merged = (
        buckets1.join(buckets2, how='outer', lsuffix='_d1', rsuffix='_d2')
        .fillna(0.0)
        .reset_index()
    )
    merged = (
        merged.merge(
            active1_buckets[['remaining_maturity_months', 'effective_interest']].rename(
                columns={'effective_interest': 'effective_interest_d1'}
            ),
            on='remaining_maturity_months',
            how='left',
        )
        .merge(
            active2_buckets[['remaining_maturity_months', 'effective_interest']].rename(
                columns={'effective_interest': 'effective_interest_d2'}
            ),
            on='remaining_maturity_months',
            how='left',
        )
    )
    base_fill_cols = [
        'abs_notional_d1',
        'abs_notional_d2',
        'signed_notional_d1',
        'signed_notional_d2',
        'notional_coupon_d1',
        'notional_coupon_d2',
        'effective_interest_d1',
        'effective_interest_d2',
        'cumulative_abs_notional_d1',
        'cumulative_abs_notional_d2',
        'cumulative_signed_notional_d1',
        'cumulative_signed_notional_d2',
        'cumulative_notional_coupon_d1',
        'cumulative_notional_coupon_d2',
        'deal_count_d1',
        'deal_count_d2',
    ]
    for col in base_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(float).fillna(0.0)

    for col in [
        'abs_notional',
        'signed_notional',
        'notional_coupon',
        'cumulative_abs_notional',
        'cumulative_signed_notional',
        'cumulative_notional_coupon',
        'deal_count',
        'effective_interest',
    ]:
        merged[f'{col}_delta'] = merged[f'{col}_d2'] - merged[f'{col}_d1']

    merged = (
        merged.merge(
            added_buckets.rename(
                columns={
                    'abs_notional': 'added_abs_notional',
                    'notional': 'added_notional',
                    'notional_coupon': 'added_notional_coupon',
                    'effective_interest': 'added_effective_interest',
                    'deal_count': 'added_deal_count',
                }
            ),
            on='remaining_maturity_months',
            how='left',
        )
        .merge(
            matured_buckets.rename(
                columns={
                    'abs_notional': 'matured_abs_notional',
                    'notional': 'matured_notional',
                    'notional_coupon': 'matured_notional_coupon',
                    'effective_interest': 'matured_effective_interest',
                    'deal_count': 'matured_deal_count',
                }
            ),
            on='remaining_maturity_months',
            how='left',
        )
    )
    added_fill_cols = [
        'added_abs_notional',
        'added_notional',
        'added_notional_coupon',
        'added_effective_interest',
        'added_deal_count',
        'matured_abs_notional',
        'matured_notional',
        'matured_notional_coupon',
        'matured_effective_interest',
        'matured_deal_count',
    ]
    for col in added_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(float).fillna(0.0)

    numeric_cols = [
        'added_abs_notional',
        'added_notional',
        'added_notional_coupon',
        'added_deal_count',
        'matured_abs_notional',
        'matured_notional',
        'matured_notional_coupon',
        'matured_deal_count',
        'abs_notional_d1',
        'abs_notional_d2',
        'signed_notional_d1',
        'signed_notional_d2',
        'notional_coupon_d1',
        'notional_coupon_d2',
        'effective_interest_d1',
        'effective_interest_d2',
        'cumulative_abs_notional_d1',
        'cumulative_abs_notional_d2',
        'cumulative_signed_notional_d1',
        'cumulative_signed_notional_d2',
        'cumulative_notional_coupon_d1',
        'cumulative_notional_coupon_d2',
        'abs_notional_delta',
        'signed_notional_delta',
        'notional_coupon_delta',
        'effective_interest_delta',
        'cumulative_abs_notional_delta',
        'cumulative_signed_notional_delta',
        'cumulative_notional_coupon_delta',
        'deal_count_delta',
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    merged['explained_delta'] = merged['added_abs_notional'] - merged['matured_abs_notional']
    merged['remaining_abs_notional_delta'] = merged['abs_notional_delta']
    merged['residual_delta'] = merged['remaining_abs_notional_delta'] - merged['explained_delta']
    merged = merged.sort_values('remaining_maturity_months').reset_index(drop=True)
    merged['cumulative_effective_interest_d1'] = merged['effective_interest_d1'][::-1].cumsum()[::-1]
    merged['cumulative_effective_interest_d2'] = merged['effective_interest_d2'][::-1].cumsum()[::-1]
    merged['cumulative_effective_interest_delta'] = (
        merged['cumulative_effective_interest_d2'] - merged['cumulative_effective_interest_d1']
    )

    return merged


def compute_calendar_month_runoff_view(
    runoff_delta_df: pd.DataFrame,
    d1: pd.Timestamp,
    d2: pd.Timestamp,
    deals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map aligned remaining-maturity buckets into actual calendar month-end values.

    If deals_df is provided, effective interest metrics are calculated with 30/360
    month-window overlap so mid-month adds/maturities are reflected.
    """

    def _monthly_effective_interest(cohort: pd.DataFrame, month_ends: list[pd.Timestamp]) -> pd.Series:
        if cohort.empty:
            return pd.Series(0.0, index=month_ends, dtype=float)
        values: dict[pd.Timestamp, float] = {}
        for me_raw in month_ends:
            me = pd.Timestamp(me_raw) + pd.offsets.MonthEnd(0)
            ms = me.replace(day=1)
            end_exclusive = me + pd.Timedelta(days=1)
            total = 0.0
            for row in cohort.itertuples(index=False):
                total += accrued_interest_for_overlap(
                    notional=row.notional,
                    annual_coupon=row.coupon,
                    deal_value_date=row.value_date,
                    deal_maturity_date=row.maturity_date,
                    window_start=ms,
                    window_end=end_exclusive,
                )
            values[me] = float(total)
        return pd.Series(values, dtype=float)

    def _daily_style_matured_sum(month_end: pd.Timestamp) -> float:
        """Monthly sum of daily matured component, matching dashboard daily logic."""
        me = pd.Timestamp(month_end).normalize()
        start = me.replace(day=1)
        end = me
        all_days = pd.date_range(start=start, end=end, freq='D')
        days_in_month = len(all_days)

        if days_in_month == 31:
            days = all_days[:-1]
            weights = pd.Series(1.0, index=days)
        else:
            days = all_days
            weights = pd.Series(1.0, index=days)
            if days_in_month < 30:
                weights.iloc[-1] += float(30 - days_in_month)

        matured_in_month_all = deals_df[
            (deals_df['maturity_date'] >= start)
            & (deals_df['maturity_date'] <= end)
            & (deals_df['value_date'] < deals_df['maturity_date'])
        ].copy()
        if matured_in_month_all.empty:
            return 0.0

        total = 0.0
        for d in days:
            matured_cum = matured_in_month_all[matured_in_month_all['maturity_date'] <= d]
            matured_contrib = (
                float((matured_cum['notional'] * matured_cum['coupon'] / 360.0).sum())
                if not matured_cum.empty
                else 0.0
            )
            total += (-matured_contrib) * float(weights.loc[d])
        return float(total)

    if runoff_delta_df.empty:
        return pd.DataFrame(
            columns=[
                'calendar_month_end',
                'abs_notional_t1',
                'abs_notional_t2',
                'abs_notional_delta',
                'signed_notional_t1',
                'signed_notional_t2',
                'signed_notional_delta',
                'notional_coupon_t1',
                'notional_coupon_t2',
                'notional_coupon_delta',
                'effective_interest_t1',
                'effective_interest_t2',
                'effective_interest_delta',
                'deal_count_t1',
                'deal_count_t2',
                'deal_count_delta',
                'added_abs_notional',
                'matured_abs_notional',
                'added_notional',
                'matured_notional',
                'added_notional_coupon',
                'matured_notional_coupon',
                'added_effective_interest',
                'matured_effective_interest',
                'added_deal_count',
                'matured_deal_count',
                'cumulative_abs_notional_t1',
                'cumulative_abs_notional_t2',
                'cumulative_signed_notional_t1',
                'cumulative_signed_notional_t2',
                'cumulative_notional_coupon_t1',
                'cumulative_notional_coupon_t2',
                'cumulative_effective_interest_t1',
                'cumulative_effective_interest_t2',
            ]
        )

    t1 = pd.Timestamp(d1) + pd.offsets.MonthEnd(0)
    t2 = pd.Timestamp(d2) + pd.offsets.MonthEnd(0)
    base = runoff_delta_df.copy()
    base['remaining_maturity_months'] = base['remaining_maturity_months'].astype(int)

    def _series_or_zero(col: str) -> pd.Series:
        if col in base.columns:
            return base[col].astype(float)
        return pd.Series(0.0, index=base.index, dtype=float)

    t1_rows = pd.DataFrame(
        {
            'calendar_month_end': base['remaining_maturity_months'].apply(lambda k: t1 + pd.offsets.MonthEnd(int(k))),
            'abs_notional_t1': base['abs_notional_d1'].astype(float),
            'signed_notional_t1': _series_or_zero('signed_notional_d1'),
            'notional_coupon_t1': base['notional_coupon_d1'].astype(float),
            'deal_count_t1': base['deal_count_d1'].astype(float),
            'matured_abs_notional': _series_or_zero('matured_abs_notional'),
            'matured_notional': _series_or_zero('matured_notional'),
            'matured_notional_coupon': _series_or_zero('matured_notional_coupon'),
            'matured_deal_count': _series_or_zero('matured_deal_count'),
        }
    )
    t2_rows = pd.DataFrame(
        {
            'calendar_month_end': base['remaining_maturity_months'].apply(lambda k: t2 + pd.offsets.MonthEnd(int(k))),
            'abs_notional_t2': base['abs_notional_d2'].astype(float),
            'signed_notional_t2': _series_or_zero('signed_notional_d2'),
            'notional_coupon_t2': base['notional_coupon_d2'].astype(float),
            'deal_count_t2': base['deal_count_d2'].astype(float),
            'added_abs_notional': _series_or_zero('added_abs_notional'),
            'added_notional': _series_or_zero('added_notional'),
            'added_notional_coupon': _series_or_zero('added_notional_coupon'),
            'added_deal_count': _series_or_zero('added_deal_count'),
        }
    )

    t1_grouped = t1_rows.groupby('calendar_month_end', as_index=False).sum()
    t2_grouped = t2_rows.groupby('calendar_month_end', as_index=False).sum()
    t1_grouped = t1_grouped.sort_values('calendar_month_end').reset_index(drop=True)
    t2_grouped = t2_grouped.sort_values('calendar_month_end').reset_index(drop=True)

    t1_grouped['cumulative_abs_notional_t1'] = t1_grouped['abs_notional_t1'][::-1].cumsum()[::-1]
    t2_grouped['cumulative_abs_notional_t2'] = t2_grouped['abs_notional_t2'][::-1].cumsum()[::-1]
    t1_grouped['cumulative_signed_notional_t1'] = t1_grouped['signed_notional_t1'][::-1].cumsum()[::-1]
    t2_grouped['cumulative_signed_notional_t2'] = t2_grouped['signed_notional_t2'][::-1].cumsum()[::-1]
    t1_grouped['cumulative_notional_coupon_t1'] = t1_grouped['notional_coupon_t1'][::-1].cumsum()[::-1]
    t2_grouped['cumulative_notional_coupon_t2'] = t2_grouped['notional_coupon_t2'][::-1].cumsum()[::-1]

    merged = (
        t1_grouped.merge(t2_grouped, on='calendar_month_end', how='outer')
        .fillna(0.0)
        .sort_values('calendar_month_end')
        .reset_index(drop=True)
    )
    merged['abs_notional_delta'] = merged['abs_notional_t2'] - merged['abs_notional_t1']
    merged['signed_notional_delta'] = merged['signed_notional_t2'] - merged['signed_notional_t1']
    merged['notional_coupon_delta'] = merged['notional_coupon_t2'] - merged['notional_coupon_t1']
    merged['deal_count_delta'] = merged['deal_count_t2'] - merged['deal_count_t1']
    merged['effective_interest_t1'] = merged['notional_coupon_t1']
    merged['effective_interest_t2'] = merged['notional_coupon_t2']
    merged['added_effective_interest'] = merged.get('added_notional_coupon', 0.0)
    merged['matured_effective_interest'] = merged.get('matured_notional_coupon', 0.0)

    if deals_df is not None and not deals_df.empty:
        active1 = active_deals_snapshot(deals_df, t1)
        active2 = active_deals_snapshot(deals_df, t2)
        ids1 = set(active1['deal_id'].tolist())
        ids2 = set(active2['deal_id'].tolist())
        added = active2[active2['deal_id'].isin(ids2 - ids1)].copy()
        matured = active1[active1['deal_id'].isin(ids1 - ids2)].copy()

        month_ends = merged['calendar_month_end'].drop_duplicates().sort_values().tolist()
        eff1 = _monthly_effective_interest(active1, month_ends)
        eff2 = _monthly_effective_interest(active2, month_ends)
        eff_added = _monthly_effective_interest(added, month_ends)
        eff_matured = _monthly_effective_interest(matured, month_ends)

        merged['effective_interest_t1'] = merged['calendar_month_end'].map(eff1.to_dict()).fillna(0.0).astype(float)
        merged['effective_interest_t2'] = merged['calendar_month_end'].map(eff2.to_dict()).fillna(0.0).astype(float)
        merged['added_effective_interest'] = (
            merged['calendar_month_end'].map(eff_added.to_dict()).fillna(0.0).astype(float)
        )
        merged['matured_effective_interest'] = (
            merged['calendar_month_end'].map(eff_matured.to_dict()).fillna(0.0).astype(float)
        )

        # Reconcile T2 anchor-month total with daily decomposition:
        # active2 excludes deals that mature inside the T2 month, while the daily
        # view includes their in-month accrued contribution up to maturity.
        anchor_mask_t2 = merged['calendar_month_end'] == t2
        merged.loc[anchor_mask_t2, 'effective_interest_t2'] = (
            merged.loc[anchor_mask_t2, 'effective_interest_t2']
            + merged.loc[anchor_mask_t2, 'matured_effective_interest']
        )
        daily_matured_sum_t2 = _daily_style_matured_sum(t2)
        # Store magnitude/sign convention used by runoff chart:
        # plotted matured bar is -matured_effective_interest.
        merged.loc[anchor_mask_t2, 'matured_effective_interest'] = -daily_matured_sum_t2

    merged['effective_interest_delta'] = merged['effective_interest_t2'] - merged['effective_interest_t1']
    merged['cumulative_effective_interest_t1'] = merged['effective_interest_t1'][::-1].cumsum()[::-1]
    merged['cumulative_effective_interest_t2'] = merged['effective_interest_t2'][::-1].cumsum()[::-1]

    return merged
