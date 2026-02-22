import pandas as pd

from src.calculations.nii import active_deals_snapshot
from src.dashboard.app import _compute_daily_interest


def test_daily_notional_decomposition_handles_negative_notionals() -> None:
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
                'maturity_date': pd.Timestamp('2025-02-15'),
                'notional': -60.0,
                'coupon': 0.10,
            },
            {
                'deal_id': 3,
                'trade_date': pd.Timestamp('2025-02-10'),
                'value_date': pd.Timestamp('2025-02-10'),
                'maturity_date': pd.Timestamp('2025-04-30'),
                'notional': -40.0,
                'coupon': 0.10,
            },
        ]
    )
    month_end = pd.Timestamp('2025-02-28')
    daily = _compute_daily_interest(deals, month_end)

    # Daily total notional should reconcile to the active signed notional at month end.
    active = active_deals_snapshot(deals, month_end)
    assert round(float(daily['notional_total'].iloc[-1]), 10) == round(float(active['notional'].sum()), 10)

    # Decomposition identity in daily notional chart logic.
    assert (
        round(
            float((daily['notional_existing'] + daily['notional_added'] - daily['notional_total']).abs().sum()),
            10,
        )
        == 0.0
    )

    # A matured negative-notional deal contributes positive matured effect after maturity.
    matured_after_15th = daily.loc[daily['date'] >= pd.Timestamp('2025-02-15'), 'notional_matured']
    assert float(matured_after_15th.max()) > 0.0

    matured_interest_after_15th = daily.loc[daily['date'] >= pd.Timestamp('2025-02-15'), 'interest_matured']
    assert float(matured_interest_after_15th.max()) > 0.0
