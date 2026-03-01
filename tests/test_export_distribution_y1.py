from __future__ import annotations

import pandas as pd

from src.dashboard.reporting.export_pack import build_export_context


def test_distribution_y1_shape_and_sums() -> None:
    t1 = pd.Timestamp('2024-12-31')
    t2 = pd.Timestamp('2025-01-31')
    months = pd.date_range(t2, periods=12, freq='ME')
    calendar_runoff = pd.DataFrame(
        {
            'calendar_month_end': months,
            'cumulative_signed_notional_t2': [1_200_000.0 - (i * 50_000.0) for i in range(12)],
            'abs_notional_t2': [100_000.0] * 12,
            'effective_interest_t2': [2_000.0] * 12,
        }
    )
    runoff_delta = pd.DataFrame(
        {
            'remaining_maturity_months': [1, 3, 6, 12],
            'abs_notional_d1': [400_000.0, 300_000.0, 200_000.0, 100_000.0],
            'abs_notional_d2': [380_000.0, 320_000.0, 210_000.0, 90_000.0],
        }
    )
    curve_df = pd.DataFrame(
        {
            'ir_date': [t2, t2, t2],
            'ir_tenor': [1, 6, 12],
            'rate': [0.01, 0.015, 0.02],
        }
    )
    ctx = build_export_context(
        path='Input.xlsx',
        product='Mortgages',
        t1=t1,
        t2=t2,
        growth_mode='user_defined',
        growth_monthly_value=100_000.0,
        scenario_payload_json='[]',
        active_ids_json='[]',
        overview_metrics=pd.DataFrame(),
        overview_delta_kpis={},
        yearly_summary=pd.DataFrame(),
        monthly_base=pd.DataFrame(),
        monthly_scenarios=pd.DataFrame(),
        calendar_runoff=calendar_runoff,
        runoff_delta=runoff_delta,
        curve_df=curve_df,
    )
    grid = ctx['distribution_month_tenor']
    monthly = ctx['distribution_monthly_summary']

    month_cols = [c for c in grid.columns if c != 'Refill Tenor Bucket (Months)']
    assert len(month_cols) == 12
    assert (grid[month_cols].fillna(0.0) >= 0.0).all().all()

    assert len(monthly) == 12
    assert ((monthly['Total Volume'] - (monthly['Refill Volume'] + monthly['Growth Volume'])).abs() < 1e-9).all()
    assert (
        (monthly['Total Interest (Annualized)'] - (monthly['Refill Interest (Annualized)'] + monthly['Growth Interest (Annualized)'])).abs()
        < 1e-9
    ).all()

