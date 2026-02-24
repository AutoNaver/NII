import pandas as pd

from src.dashboard.plots.runoff_plots import _compute_refill_growth_components


def test_growth_mode_constant_has_no_growth_component() -> None:
    cumulative = pd.Series([100.0, 90.0, 80.0], dtype=float)
    out = _compute_refill_growth_components(
        cumulative_notional=cumulative,
        growth_mode='constant',
        monthly_growth_amount=25.0,
    )
    assert out['refill_required'].tolist() == [0.0, 10.0, 20.0]
    assert out['growth_required'].tolist() == [0.0, 0.0, 0.0]
    assert out['total_required'].tolist() == [0.0, 10.0, 20.0]


def test_growth_mode_user_defined_applies_increment_from_second_step() -> None:
    cumulative = pd.Series([100.0, 90.0, 80.0], dtype=float)
    out = _compute_refill_growth_components(
        cumulative_notional=cumulative,
        growth_mode='user_defined',
        monthly_growth_amount=5.0,
    )
    assert out['refill_required'].tolist() == [0.0, 10.0, 20.0]
    assert out['growth_required'].tolist() == [0.0, 5.0, 5.0]
    assert out['total_required'].tolist() == [0.0, 15.0, 25.0]
