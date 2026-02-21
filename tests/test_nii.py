import pandas as pd

from src.calculations.nii import compute_monthly_realized_nii


def test_monthly_nii_aggregation() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-02-01'),
                'notional': 1000.0,
                'coupon': 0.12,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-10'),
                'value_date': pd.Timestamp('2025-01-10'),
                'maturity_date': pd.Timestamp('2025-02-10'),
                'notional': -500.0,
                'coupon': 0.06,
            },
        ]
    )

    result = compute_monthly_realized_nii(deals, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-31'))

    assert round(result, 6) == round(10.0 - 1.75, 6)
