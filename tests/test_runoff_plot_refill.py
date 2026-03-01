import numpy as np
import pandas as pd

from src.dashboard.plots.runoff_plots import _compute_refill_growth_components_anchor_safe
from src.dashboard.plots.runoff_plots import _growth_outstanding_profile
from src.dashboard.plots.runoff_plots import _refill_allocation_heatmap
from src.dashboard.plots.runoff_plots import _refill_volume_interest_chart
from src.dashboard.plots.runoff_plots import _shifted_portfolio_refill_weights


def test_refill_growth_components_anchor_safe_ignores_leading_zero_anchor() -> None:
    cumulative = pd.Series([0.0, 100.0, 90.0, 80.0], dtype=float)
    out = _compute_refill_growth_components_anchor_safe(
        cumulative_notional=cumulative,
        growth_mode='constant',
        monthly_growth_amount=0.0,
    )
    refill = out['refill_required']
    growth = out['growth_required']

    assert refill.tolist() == [0.0, 0.0, 10.0, 20.0]
    assert growth.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_refill_growth_components_anchor_safe_handles_all_zero_series() -> None:
    cumulative = pd.Series([0.0, 0.0, 0.0], dtype=float)
    out = _compute_refill_growth_components_anchor_safe(
        cumulative_notional=cumulative,
        growth_mode='constant',
        monthly_growth_amount=0.0,
    )
    assert out['refill_required'].sum() == 0.0
    assert out['growth_required'].sum() == 0.0


def test_shifted_portfolio_refill_weights_uses_one_month_shift_delta() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 1, 2],
            'abs_notional_d1': [50.0, 40.0, 30.0],
            'abs_notional_d2': [45.0, 35.0, 15.0],
        }
    )
    # d1 shifted = [40, 30, 0]; delta = [5, 5, 15]
    out = _shifted_portfolio_refill_weights(compare)
    assert out is not None
    assert out['tenor'].tolist() == [0, 1, 2]
    assert out['shift_delta'].tolist() == [5.0, 5.0, 15.0]
    assert round(float(out['weight'].sum()), 10) == 1.0


def test_refill_volume_interest_chart_has_dual_axes_traces() -> None:
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31'])
    refill_required = pd.Series([10.0, 20.0, 30.0], dtype=float)
    growth_required = pd.Series([1.0, 2.0, 3.0], dtype=float)
    refill_rate = pd.Series([0.1, 0.2, 0.3], dtype=float)
    fig = _refill_volume_interest_chart(
        month_ends=month_ends,
        refill_required=refill_required,
        growth_volume=growth_required,
        refill_rate=refill_rate,
        title='t',
        x_label='x',
    )
    assert len(fig.data) == 6
    assert fig.data[0].name == 'Refill Volume'
    assert fig.data[1].name == 'Growth Volume'
    assert fig.data[2].name == 'Total Volume'
    assert fig.data[2].yaxis in {None, 'y'}
    assert fig.data[3].name == 'Refill Interest (Annualized)'
    assert fig.data[3].yaxis == 'y2'
    assert fig.data[4].name == 'Growth Interest (Annualized)'
    assert fig.data[4].yaxis == 'y2'
    assert fig.data[5].name == 'Total Interest (Annualized)'
    assert fig.data[5].yaxis == 'y2'


def test_refill_volume_interest_chart_preserves_negative_signed_flows() -> None:
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31'])
    refill_required = pd.Series([0.0, -20.0, -30.0], dtype=float)
    growth_required = pd.Series([0.0, -5.0, -5.0], dtype=float)
    refill_rate = pd.Series([0.1, 0.2, 0.3], dtype=float)
    fig = _refill_volume_interest_chart(
        month_ends=month_ends,
        refill_required=refill_required,
        growth_volume=growth_required,
        refill_rate=refill_rate,
        title='t',
        x_label='x',
    )
    assert min(fig.data[0].y) < 0.0
    assert min(fig.data[1].y) < 0.0
    assert min(fig.data[2].y) < 0.0
    assert min(fig.data[3].y) < 0.0


def test_refill_allocation_heatmap_user_defined_includes_growth_with_t0_distribution() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 1],
            'abs_notional_d1': [100.0, 0.0],  # T0 portfolio for T1 basis -> all growth to tenor 0
            'abs_notional_d2': [0.0, 100.0],  # shifted refill weight -> all refill to tenor 1
        }
    )
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28'])
    refill_required = pd.Series([10.0, 10.0], dtype=float)
    growth_required = pd.Series([0.0, 20.0], dtype=float)
    fig = _refill_allocation_heatmap(
        month_ends=month_ends,
        refill_required=refill_required,
        growth_required=growth_required,
        growth_mode='user_defined',
        basis='T1',
        runoff_compare_df=compare,
        title='t',
        x_label='x',
    )
    assert fig is not None
    tenors = [int(v) for v in fig.data[0].y]
    z = np.asarray(fig.data[0].z, dtype=float)
    by_tenor = {tenor: z[idx] for idx, tenor in enumerate(tenors)}
    assert np.isnan(by_tenor[0][0])
    assert by_tenor[0][1] == 20.0
    assert by_tenor[1][0] == 10.0
    assert by_tenor[1][1] == 10.0


def test_refill_allocation_heatmap_constant_mode_excludes_growth_component() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 1],
            'abs_notional_d1': [100.0, 0.0],
            'abs_notional_d2': [0.0, 100.0],
        }
    )
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28'])
    refill_required = pd.Series([10.0, 10.0], dtype=float)
    growth_required = pd.Series([0.0, 20.0], dtype=float)
    fig = _refill_allocation_heatmap(
        month_ends=month_ends,
        refill_required=refill_required,
        growth_required=growth_required,
        growth_mode='constant',
        basis='T1',
        runoff_compare_df=compare,
        title='t',
        x_label='x',
    )
    assert fig is not None
    tenors = [int(v) for v in fig.data[0].y]
    z = np.asarray(fig.data[0].z, dtype=float)
    by_tenor = {tenor: z[idx] for idx, tenor in enumerate(tenors)}
    assert 0 not in by_tenor
    assert by_tenor[1][0] == 10.0
    assert by_tenor[1][1] == 10.0


def test_growth_outstanding_profile_uses_t0_survival_weights() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 2],
            'abs_notional_d1': [50.0, 50.0],  # T0: 50% immediate, 50% at tenor 2
            'abs_notional_d2': [0.0, 0.0],
        }
    )
    flow = pd.Series([0.0, 10.0, 10.0], dtype=float)
    out = _growth_outstanding_profile(
        growth_flow=flow,
        runoff_compare_df=compare,
        basis='T1',
    )
    # survival lags: [1.0, 0.5, 0.5]
    # outstanding: [0, 10, 15]
    assert out.tolist() == [0.0, 10.0, 15.0]


def test_growth_outstanding_profile_preserves_negative_liability_growth() -> None:
    compare = pd.DataFrame(
        {
            'remaining_maturity_months': [0, 2],
            'abs_notional_d1': [50.0, 50.0],
            'abs_notional_d2': [0.0, 0.0],
        }
    )
    flow = pd.Series([0.0, -10.0, -10.0], dtype=float)
    out = _growth_outstanding_profile(
        growth_flow=flow,
        runoff_compare_df=compare,
        basis='T1',
    )
    assert out.tolist() == [0.0, -10.0, -15.0]
