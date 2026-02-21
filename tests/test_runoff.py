import pandas as pd

from src.calculations.volumes import (
    compare_monthly_bucket_series,
    compare_runoff_profiles,
    compute_runoff_delta_attribution,
    compute_runoff_profile,
)


def test_runoff_profile_rolls_down_after_maturity() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-01-31'),
                'notional': 100.0,
                'coupon': 0.02,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-02-28'),
                'notional': 200.0,
                'coupon': 0.02,
            },
        ]
    )
    profile = compute_runoff_profile(deals, pd.Timestamp('2025-01-31'))
    assert float(profile.iloc[0]['remaining_abs_notional']) == 200.0
    assert float(profile.iloc[1]['remaining_abs_notional']) == 0.0


def test_compare_runoff_profiles_computes_deltas() -> None:
    p1 = pd.DataFrame(
        {
            'month_offset': [0, 1],
            'month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'remaining_notional': [100.0, 0.0],
            'remaining_abs_notional': [100.0, 0.0],
            'matured_abs_notional': [0.0, 100.0],
            'remaining_pct_of_initial_abs': [1.0, 0.0],
            'active_deal_count': [1, 0],
        }
    )
    p2 = pd.DataFrame(
        {
            'month_offset': [0, 1],
            'month_end': pd.to_datetime(['2025-02-28', '2025-03-31']),
            'remaining_notional': [200.0, 100.0],
            'remaining_abs_notional': [200.0, 100.0],
            'matured_abs_notional': [0.0, 100.0],
            'remaining_pct_of_initial_abs': [1.0, 0.5],
            'active_deal_count': [2, 1],
        }
    )
    cmp_df = compare_runoff_profiles(p1, p2)
    assert float(cmp_df.iloc[0]['remaining_abs_notional_delta']) == 100.0
    assert float(cmp_df.iloc[1]['active_deal_count_delta']) == 1.0


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
    assert {'added_remaining_abs_notional', 'matured_remaining_abs_notional', 'remaining_abs_notional_delta'}.issubset(
        out.columns
    )
    assert (out['remaining_abs_notional_delta'] == (out['remaining_abs_notional_d2'] - out['remaining_abs_notional_d1'])).all()
