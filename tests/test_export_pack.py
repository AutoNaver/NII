from __future__ import annotations

from io import BytesIO

import pandas as pd

from src.dashboard.reporting.export_pack import (
    build_export_context,
    build_export_workbook_bytes,
    default_export_filename,
)


def _fixtures() -> dict[str, object]:
    t1 = pd.Timestamp('2024-12-31')
    t2 = pd.Timestamp('2025-01-31')
    months = pd.date_range(t2, periods=12, freq='ME')
    overview_metrics = pd.DataFrame(
        {
            f'T1 {t1.date()}': [1_000_000.0, 0.02, 1_500.0, 10.0],
            f'T2 {t2.date()}': [1_050_000.0, 0.021, 1_650.0, 11.0],
            'Delta (T2-T1)': [50_000.0, 0.001, 150.0, 1.0],
        },
        index=[
            'Total Active Notional (EUR)',
            'Weighted Avg Coupon (pp)',
            'Interest Paid EUR (30/360)',
            'Active Deal Count',
        ],
    )
    yearly_summary = pd.DataFrame(
        [
            {
                'scenario_id': 'inst_up_50',
                'scenario_label': 'Instant +50 bps',
                'Y1 Delta': 10.0,
                'Y2 Delta': 11.0,
                'Y3 Delta': 12.0,
                'Y4 Delta': 13.0,
                'Y5 Delta': 14.0,
                '5Y Cumulative Delta': 60.0,
            }
        ]
    )
    monthly_base = pd.DataFrame(
        {
            'month_idx': list(range(12)),
            'calendar_month_end': months,
            'base_total_interest': [100.0] * 12,
        }
    )
    monthly_scenarios = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50'] * 12,
            'scenario_label': ['Instant +50 bps'] * 12,
            'month_idx': list(range(12)),
            'calendar_month_end': months,
            'shocked_total_interest': [110.0] * 12,
            'delta_vs_base': [10.0] * 12,
            'cumulative_delta': [10.0 * (i + 1) for i in range(12)],
        }
    )
    calendar_runoff = pd.DataFrame(
        {
            'calendar_month_end': months,
            'cumulative_signed_notional_t2': [1_000_000.0 - (i * 25_000.0) for i in range(12)],
            'abs_notional_t2': [80_000.0] * 12,
            'effective_interest_t2': [1_600.0] * 12,
        }
    )
    runoff_delta = pd.DataFrame(
        {
            'remaining_maturity_months': list(range(1, 13)),
            'abs_notional_d1': [80_000.0] * 12,
            'abs_notional_d2': [80_000.0] * 12,
        }
    )
    curve_df = pd.DataFrame(
        {
            'ir_date': [t2, t2],
            'ir_tenor': [1, 12],
            'rate': [0.01, 0.02],
        }
    )
    return {
        't1': t1,
        't2': t2,
        'overview_metrics': overview_metrics,
        'yearly_summary': yearly_summary,
        'monthly_base': monthly_base,
        'monthly_scenarios': monthly_scenarios,
        'calendar_runoff': calendar_runoff,
        'runoff_delta': runoff_delta,
        'curve_df': curve_df,
    }


def test_default_export_filename() -> None:
    name = default_export_filename('Mortgage Loans', pd.Timestamp('2024-12-31'), pd.Timestamp('2025-01-31'))
    assert name.startswith('nii_executive_pack_Mortgage_Loans_T1_2024-12-31_T2_2025-01-31')
    assert name.endswith('.xlsx')


def test_build_export_workbook_bytes_contains_expected_sheets() -> None:
    fx = _fixtures()
    ctx = build_export_context(
        path='Input.xlsx',
        product='Mortgages',
        t1=fx['t1'],
        t2=fx['t2'],
        growth_mode='user_defined',
        growth_monthly_value=100_000.0,
        scenario_payload_json='[{"scenario_id":"inst_up_50","scenario_label":"Instant +50 bps"}]',
        active_ids_json='["inst_up_50"]',
        overview_metrics=fx['overview_metrics'],
        overview_delta_kpis={'Realized NII Delta (EUR)': 123.0},
        yearly_summary=fx['yearly_summary'],
        monthly_base=fx['monthly_base'],
        monthly_scenarios=fx['monthly_scenarios'],
        calendar_runoff=fx['calendar_runoff'],
        runoff_delta=fx['runoff_delta'],
        curve_df=fx['curve_df'],
    )
    data = build_export_workbook_bytes(ctx, workbook_title='Test Export')
    assert isinstance(data, bytes)
    assert len(data) > 0

    xls = pd.ExcelFile(BytesIO(data))
    assert set(xls.sheet_names) == {
        'Summary_Metadata',
        'Overview_Metrics',
        'Scenario_Matrix_Delta',
        'Scenario_Matrix_Absolute',
        'Distribution_Month_Tenor_Y1',
        'Distribution_Monthly_Summary_Y1',
    }

    delta = pd.read_excel(BytesIO(data), sheet_name='Scenario_Matrix_Delta')
    absolute = pd.read_excel(BytesIO(data), sheet_name='Scenario_Matrix_Absolute')
    assert delta.columns.tolist() == ['Scenario', 'Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta', '5Y Cumulative Delta']
    assert absolute.columns.tolist() == ['Scenario', 'Y1 Absolute', 'Y2 Absolute', 'Y3 Absolute', 'Y4 Absolute', 'Y5 Absolute', '5Y Cumulative Absolute']
    assert 'BaseCase' in absolute['Scenario'].astype(str).tolist()


def test_build_export_context_accepts_lowercase_metric_index_name() -> None:
    fx = _fixtures()
    metrics = fx['overview_metrics'].copy()
    metrics.index.name = 'metric'
    ctx = build_export_context(
        path='Input.xlsx',
        product='Mortgages',
        t1=fx['t1'],
        t2=fx['t2'],
        growth_mode='constant',
        growth_monthly_value=0.0,
        scenario_payload_json='[]',
        active_ids_json='[]',
        overview_metrics=metrics,
        overview_delta_kpis={},
        yearly_summary=pd.DataFrame(),
        monthly_base=pd.DataFrame(),
        monthly_scenarios=pd.DataFrame(),
        calendar_runoff=fx['calendar_runoff'],
        runoff_delta=fx['runoff_delta'],
        curve_df=fx['curve_df'],
    )
    out = pd.DataFrame(ctx['overview_metrics'])
    assert 'Metric' in out.columns
