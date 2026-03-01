from __future__ import annotations

import pandas as pd

from src.dashboard.plots.rate_scenario_plots import build_scenario_matrix_table
from src.dashboard.reporting.export_pack import build_export_context


def test_export_scenario_matrices_match_source_logic() -> None:
    t1 = pd.Timestamp('2024-12-31')
    t2 = pd.Timestamp('2025-01-31')
    months = pd.date_range(t2, periods=24, freq='ME')
    yearly_summary = pd.DataFrame(
        [
            {
                'scenario_id': 'inst_up_50',
                'scenario_label': 'Instant +50 bps',
                'Y1 Delta': 10.0,
                'Y2 Delta': 20.0,
                'Y3 Delta': 0.0,
                'Y4 Delta': 0.0,
                'Y5 Delta': 0.0,
                '5Y Cumulative Delta': 30.0,
            }
        ]
    )
    monthly_base = pd.DataFrame(
        {
            'month_idx': list(range(24)),
            'calendar_month_end': months,
            'base_total_interest': [100.0] * 24,
        }
    )
    monthly_scenarios = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50'] * 24,
            'scenario_label': ['Instant +50 bps'] * 24,
            'month_idx': list(range(24)),
            'calendar_month_end': months,
            'shocked_total_interest': [110.0] * 24,
            'delta_vs_base': [10.0] * 24,
            'cumulative_delta': [10.0 * (i + 1) for i in range(24)],
        }
    )

    ctx = build_export_context(
        path='Input.xlsx',
        product='Mortgages',
        t1=t1,
        t2=t2,
        growth_mode='constant',
        growth_monthly_value=0.0,
        scenario_payload_json='[{"scenario_id":"inst_up_50","scenario_label":"Instant +50 bps"}]',
        active_ids_json='["inst_up_50"]',
        overview_metrics=pd.DataFrame(),
        overview_delta_kpis={},
        yearly_summary=yearly_summary,
        monthly_base=monthly_base,
        monthly_scenarios=monthly_scenarios,
        calendar_runoff=pd.DataFrame({'calendar_month_end': months}),
        runoff_delta=pd.DataFrame(),
        curve_df=pd.DataFrame(),
    )
    expected_delta = build_scenario_matrix_table(yearly_summary, view_mode='delta').reset_index(drop=True)
    expected_abs = build_scenario_matrix_table(
        yearly_summary,
        view_mode='absolute',
        monthly_base=monthly_base,
        monthly_scenarios=monthly_scenarios,
    ).reset_index(drop=True)
    got_delta = pd.DataFrame(ctx['scenario_matrix_delta']).reset_index(drop=True)
    got_abs = pd.DataFrame(ctx['scenario_matrix_absolute']).reset_index(drop=True)
    pd.testing.assert_frame_equal(got_delta, expected_delta)
    pd.testing.assert_frame_equal(got_abs, expected_abs)

