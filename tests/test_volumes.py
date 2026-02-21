import pandas as pd

from src.calculations.volumes import compute_monthly_buckets


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
