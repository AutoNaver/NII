import pandas as pd

from src.calculations.refill_growth import (
    compute_refill_growth_components_anchor_safe,
    growth_outstanding_profile,
    shifted_portfolio_refill_weights,
)


def test_shifted_portfolio_refill_weights_uses_one_month_shift_delta() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 1, 2],
            'abs_notional_d1': [50.0, 40.0, 30.0],
            'abs_notional_d2': [45.0, 35.0, 15.0],
        }
    )
    out = shifted_portfolio_refill_weights(compare)
    assert out is not None
    assert out['tenor'].tolist() == [0, 1, 2]
    assert out['shift_delta'].tolist() == [5.0, 5.0, 15.0]
    assert round(float(out['weight'].sum()), 10) == 1.0


def test_refill_growth_components_anchor_safe_ignores_leading_zero_anchor() -> None:
    cumulative = pd.Series([0.0, 100.0, 90.0, 80.0], dtype=float)
    out = compute_refill_growth_components_anchor_safe(
        cumulative_notional=cumulative,
        growth_mode='constant',
        monthly_growth_amount=0.0,
    )
    assert out['refill_required'].tolist() == [0.0, 0.0, 10.0, 20.0]
    assert out['growth_required'].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_growth_outstanding_profile_preserves_negative_liability_growth() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 2],
            'abs_notional_d1': [50.0, 50.0],
            'abs_notional_d2': [0.0, 0.0],
        }
    )
    flow = pd.Series([0.0, -10.0, -10.0], dtype=float)
    out = growth_outstanding_profile(
        growth_flow=flow,
        runoff_compare_df=compare,
        basis='T1',
    )
    assert out.tolist() == [0.0, -10.0, -15.0]
