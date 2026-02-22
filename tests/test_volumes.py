import pandas as pd

from src.calculations.volumes import compute_monthly_activity_metrics, compute_monthly_buckets
from src.calculations.volumes import compute_remaining_maturity_buckets


def test_monthly_buckets_weighted_coupon_uses_abs_notional() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.10,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': -100.0,
                'coupon': 0.02,
            },
        ]
    )

    result = compute_monthly_buckets(deals, pd.Timestamp('2025-01-31'))
    assert len(result) == 1
    assert round(float(result.iloc[0]['weighted_avg_coupon']), 6) == 0.06


def test_monthly_activity_metrics_counts_added_deals() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-10'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 100.0,
                'coupon': 0.10,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-02-01'),
                'value_date': pd.Timestamp('2025-02-05'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 50.0,
                'coupon': 0.20,
            },
        ]
    )
    out = compute_monthly_activity_metrics(
        deals,
        [pd.Timestamp('2025-01-31'), pd.Timestamp('2025-02-28')],
    )
    assert int(out.iloc[0]['added_deal_count']) == 1
    assert int(out.iloc[1]['added_deal_count']) == 1
    assert round(float(out.iloc[0]['active_notional_coupon']), 6) == 10.0
    assert round(float(out.iloc[1]['active_notional_coupon']), 6) == 20.0


def test_remaining_maturity_buckets_assign_and_cumulate() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-01-15'),
                'notional': 100.0,
                'coupon': 0.10,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-03-31'),
                'notional': 200.0,
                'coupon': 0.05,
            },
        ]
    )
    buckets = compute_remaining_maturity_buckets(deals, pd.Timestamp('2025-01-31'))
    # 241 buckets expected (0..240)
    assert len(buckets) == 241
    bucket2 = buckets.loc[buckets['remaining_maturity_months'] == 2].iloc[0]
    assert round(float(bucket2['abs_notional']), 6) == 200.0
    assert round(float(bucket2['cumulative_abs_notional']), 6) == 200.0
