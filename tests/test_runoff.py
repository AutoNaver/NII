import pandas as pd

from src.calculations.accrual import accrued_interest_for_overlap
from src.calculations.nii import active_deals_snapshot
from src.calculations.volumes import (
    compare_monthly_bucket_series,
    compute_calendar_month_runoff_view,
    compute_runoff_delta_attribution,
)


def test_compare_monthly_bucket_series_computes_calendar_deltas() -> None:
    d1 = pd.DataFrame(
        {
            'month_end': pd.to_datetime(['2025-01-31']),
            'total_active_notional': [100.0],
            'weighted_avg_coupon': [0.02],
            'interest_paid_eur': [1.0],
            'active_deal_count': [1],
        }
    )
    d2 = pd.DataFrame(
        {
            'month_end': pd.to_datetime(['2025-01-31']),
            'total_active_notional': [120.0],
            'weighted_avg_coupon': [0.03],
            'interest_paid_eur': [1.2],
            'active_deal_count': [2],
        }
    )
    cmp_df = compare_monthly_bucket_series(d1, d2)
    assert float(cmp_df.iloc[0]['total_active_notional_delta']) == 20.0
    assert round(float(cmp_df.iloc[0]['weighted_avg_coupon_delta']), 10) == 0.01


def test_runoff_delta_attribution_has_added_and_matured_components() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.02,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-02-01'),
                'value_date': pd.Timestamp('2025-02-01'),
                'maturity_date': pd.Timestamp('2025-05-31'),
                'notional': 150.0,
                'coupon': 0.02,
            },
        ]
    )
    out = compute_runoff_delta_attribution(deals, pd.Timestamp('2025-01-31'), pd.Timestamp('2025-02-28'))
    assert {
        'remaining_maturity_months',
        'added_abs_notional',
        'matured_abs_notional',
        'added_deal_count',
        'matured_deal_count',
        'remaining_abs_notional_delta',
    }.issubset(out.columns)
    assert (out['remaining_abs_notional_delta'] == (out['abs_notional_d2'] - out['abs_notional_d1'])).all()


def test_calendar_month_runoff_view_maps_buckets_to_calendar_month_end() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.02,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-02-01'),
                'value_date': pd.Timestamp('2025-02-01'),
                'maturity_date': pd.Timestamp('2025-05-31'),
                'notional': 150.0,
                'coupon': 0.02,
            },
        ]
    )
    t1 = pd.Timestamp('2025-01-31')
    t2 = pd.Timestamp('2025-02-28')
    runoff = compute_runoff_delta_attribution(deals, t1, t2)
    calendar = compute_calendar_month_runoff_view(runoff, t1, t2, deals_df=deals)

    assert 'calendar_month_end' in calendar.columns
    assert 'abs_notional_t1' in calendar.columns
    assert 'abs_notional_t2' in calendar.columns
    assert 'effective_interest_t1' in calendar.columns
    assert 'effective_interest_t2' in calendar.columns
    assert 'added_abs_notional' in calendar.columns
    assert 'matured_abs_notional' in calendar.columns
    assert (calendar['abs_notional_delta'] == (calendar['abs_notional_t2'] - calendar['abs_notional_t1'])).all()
    assert calendar['calendar_month_end'].min() == pd.Timestamp('2025-01-31')


def test_calendar_month_effective_interest_uses_day_count_for_mid_month_adds_and_maturities() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.36,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-02-12'),
                'value_date': pd.Timestamp('2025-02-12'),
                'maturity_date': pd.Timestamp('2025-04-30'),
                'notional': 100.0,
                'coupon': 0.36,
            },
            {
                'deal_id': 3,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-02-12'),
                'notional': 100.0,
                'coupon': 0.36,
            },
        ]
    )
    t1 = pd.Timestamp('2025-01-31')
    t2 = pd.Timestamp('2025-02-28')
    runoff = compute_runoff_delta_attribution(deals, t1, t2)
    calendar = compute_calendar_month_runoff_view(runoff, t1, t2, deals_df=deals)

    feb_row = calendar.loc[calendar['calendar_month_end'] == pd.Timestamp('2025-02-28')].iloc[0]
    assert round(float(feb_row['added_effective_interest']), 6) == 1.9
    assert round(float(feb_row['matured_effective_interest']), 6) == 1.9
    assert round(float(feb_row['effective_interest_t1']), 6) == 4.1
    assert round(float(feb_row['effective_interest_t2']), 6) == 6.0


def test_runoff_signed_notional_fields_handle_negative_notionals() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.02,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-05-31'),
                'notional': -80.0,
                'coupon': 0.02,
            },
        ]
    )
    t1 = pd.Timestamp('2025-01-31')
    t2 = pd.Timestamp('2025-02-28')
    runoff = compute_runoff_delta_attribution(deals, t1, t2)
    calendar = compute_calendar_month_runoff_view(runoff, t1, t2, deals_df=deals)

    assert {'signed_notional_d1', 'signed_notional_d2', 'signed_notional_delta'}.issubset(runoff.columns)
    assert {'signed_notional_t1', 'signed_notional_t2', 'signed_notional_delta'}.issubset(calendar.columns)
    assert (runoff['signed_notional_delta'] == (runoff['signed_notional_d2'] - runoff['signed_notional_d1'])).all()
    assert (calendar['signed_notional_delta'] == (calendar['signed_notional_t2'] - calendar['signed_notional_t1'])).all()


def test_runoff_effective_interest_bucket_uses_month_overlap_days() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-02-12'),
                'notional': 100.0,
                'coupon': 0.36,
            },
        ]
    )
    t1 = pd.Timestamp('2025-01-31')
    t2 = pd.Timestamp('2025-02-28')
    runoff = compute_runoff_delta_attribution(deals, t1, t2)

    # Bucket 1 for T1 corresponds to February; overlap is Feb 1 to Feb 12 -> 11 days (30/360).
    bucket1 = runoff.loc[runoff['remaining_maturity_months'] == 1].iloc[0]
    assert round(float(bucket1['effective_interest_d1']), 6) == 1.1


def test_calendar_month_effective_interest_matches_scalar_for_non_anchor_months() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-06-30'),
                'notional': 100.0,
                'coupon': 0.12,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-02-10'),
                'value_date': pd.Timestamp('2025-02-10'),
                'maturity_date': pd.Timestamp('2025-08-31'),
                'notional': 200.0,
                'coupon': 0.08,
            },
        ]
    )
    t1 = pd.Timestamp('2025-01-31')
    t2 = pd.Timestamp('2025-02-28')
    runoff = compute_runoff_delta_attribution(deals, t1, t2)
    calendar = compute_calendar_month_runoff_view(runoff, t1, t2, deals_df=deals)

    month_ends = sorted(pd.to_datetime(calendar['calendar_month_end']).tolist())
    active1 = active_deals_snapshot(deals, t1)
    active2 = active_deals_snapshot(deals, t2)

    def _scalar_monthly(cohort: pd.DataFrame) -> dict[pd.Timestamp, float]:
        out: dict[pd.Timestamp, float] = {}
        for me in month_ends:
            ms = pd.Timestamp(me).replace(day=1)
            we = pd.Timestamp(me) + pd.Timedelta(days=1)
            total = 0.0
            for row in cohort.itertuples(index=False):
                total += accrued_interest_for_overlap(
                    row.notional,
                    row.coupon,
                    row.value_date,
                    row.maturity_date,
                    ms,
                    we,
                )
            out[pd.Timestamp(me)] = float(total)
        return out

    expected_t1 = _scalar_monthly(active1)
    expected_t2 = _scalar_monthly(active2)

    for row in calendar.itertuples(index=False):
        me = pd.Timestamp(row.calendar_month_end)
        assert abs(float(row.effective_interest_t1) - expected_t1.get(me, 0.0)) < 1e-9
        if me != t2:  # anchor month has explicit reconciliation against daily decomposition
            assert abs(float(row.effective_interest_t2) - expected_t2.get(me, 0.0)) < 1e-9
