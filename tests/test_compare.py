import pandas as pd

from src.calculations.nii import compare_month_ends


def test_compare_month_ends_classifies_changes() -> None:
    deals = pd.DataFrame(
        [
            {
                'deal_id': 1,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-04-30'),
                'notional': 1000.0,
                'coupon': 0.03,
            },
            {
                'deal_id': 2,
                'trade_date': pd.Timestamp('2025-01-01'),
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-01-31'),
                'notional': 500.0,
                'coupon': 0.02,
            },
            {
                'deal_id': 3,
                'trade_date': pd.Timestamp('2025-02-01'),
                'value_date': pd.Timestamp('2025-02-01'),
                'maturity_date': pd.Timestamp('2025-06-30'),
                'notional': 300.0,
                'coupon': 0.025,
            },
        ]
    )

    diff = compare_month_ends(deals, pd.Timestamp('2025-01-31'), pd.Timestamp('2025-02-28'))

    assert diff['active_count_d1'] == 1
    assert diff['active_count_d2'] == 2
    assert diff['active_count_delta'] == 1
    assert len(diff['added']) == 1
    assert int(diff['added'].iloc[0]['deal_id']) == 3
    assert len(diff['deal_changes']) == 1
    assert diff['deal_changes'].iloc[0]['status'] == 'new'
