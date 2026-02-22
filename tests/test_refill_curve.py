import pandas as pd

from src.dashboard.plots.runoff_plots import _build_refill_series


def test_refill_uses_interest_curve_by_tenor_with_interpolation() -> None:
    base_notional = pd.Series([100.0, 100.0], index=[0, 1], dtype=float)
    base_effective = pd.Series([0.0, 0.0], index=[0, 1], dtype=float)
    tenors = pd.Series([1.0, 5.0], index=[0, 1], dtype=float)

    refill_logic = pd.DataFrame(
        {
            'tenor': [1, 5],
            't_0_notional': [100.0, 100.0],
            'Existing_Deals': [0.0, 0.0],
            'Delta_Deals': [100.0, 100.0],
            't_1_notional': [100.0, 100.0],
        }
    )
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 3, 6],
            'rate': [0.01, 0.03, 0.06],
        }
    )

    out = _build_refill_series(
        base_notional_total=base_notional,
        base_effective_total=base_effective,
        tenor_points=tenors,
        refill_logic_df=refill_logic,
        curve_df=curve,
        basis_date=pd.Timestamp('2025-01-31'),
    )
    assert out is not None
    # 1M uses 1M rate directly, 5M interpolates between 3M and 6M.
    assert round(float(out['delta_effective'].iloc[0]), 6) == round(100.0 * 0.01 * (30.0 / 360.0), 6)
    assert round(float(out['delta_effective'].iloc[1]), 6) == round(100.0 * 0.05 * (30.0 / 360.0), 6)
