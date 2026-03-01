import pandas as pd

from src.dashboard.plots.rate_scenario_plots import (
    build_curve_comparison_figure,
    build_scenario_matrix_table,
    build_selected_scenario_impact_figure,
)


def test_build_scenario_matrix_table_structure() -> None:
    yearly = pd.DataFrame(
        [
            {
                'scenario_id': 'inst_up_50',
                'scenario_label': 'Instant +50 bps',
                'Y1 Delta': 1.0,
                'Y2 Delta': 2.0,
                'Y3 Delta': 3.0,
                'Y4 Delta': 4.0,
                'Y5 Delta': 5.0,
                '5Y Cumulative Delta': 15.0,
            }
        ]
    )
    out = build_scenario_matrix_table(yearly)
    assert out.columns.tolist() == ['Scenario', 'Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta', '5Y Cumulative Delta']
    assert out.iloc[0]['Scenario'] == 'Instant +50 bps'


def test_build_selected_scenario_impact_figure_traces() -> None:
    monthly_base = pd.DataFrame(
        {
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'base_total_interest': [10.0, 11.0],
        }
    )
    monthly_scenarios = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50', 'inst_up_50'],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'delta_vs_base': [1.0, 1.5],
            'shocked_total_interest': [11.0, 12.5],
            'cumulative_delta': [1.0, 2.5],
        }
    )
    fig = build_selected_scenario_impact_figure(
        monthly_base=monthly_base,
        monthly_scenarios=monthly_scenarios,
        scenario_id='inst_up_50',
        scenario_label='Instant +50 bps',
        show_totals=True,
    )
    assert len(fig.data) == 4
    assert fig.data[0].name == 'Delta vs Base'
    assert fig.data[-1].name == 'Cumulative Delta'


def test_build_curve_comparison_figure_traces() -> None:
    curve_points = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50'] * 2,
            'scenario_label': ['Instant +50 bps'] * 2,
            'state': ['anchor', 'anchor'],
            'tenor_months': [1, 12],
            'base_rate': [0.01, 0.02],
            'shocked_rate': [0.015, 0.025],
        }
    )
    tenor_paths = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50'] * 10,
            'scenario_label': ['Instant +50 bps'] * 10,
            'calendar_month_end': pd.to_datetime(
                [
                    '2025-01-31', '2025-02-28',
                    '2025-01-31', '2025-02-28',
                    '2025-01-31', '2025-02-28',
                    '2025-01-31', '2025-02-28',
                    '2025-01-31', '2025-02-28',
                ]
            ),
            'tenor_label': ['1M', '1M', '6M', '6M', '1Y', '1Y', '5Y', '5Y', '10Y', '10Y'],
            'base_rate': [0.01, 0.011, 0.015, 0.016, 0.02, 0.021, 0.025, 0.026, 0.03, 0.031],
            'shocked_rate': [0.015, 0.016, 0.02, 0.021, 0.025, 0.026, 0.03, 0.031, 0.035, 0.036],
            'shock_bps': [50.0] * 10,
        }
    )
    fig = build_curve_comparison_figure(
        curve_points=curve_points,
        tenor_paths=tenor_paths,
        scenario_id='inst_up_50',
        scenario_label='Instant +50 bps',
    )
    assert len(fig.data) == 12
    names = [str(t.name) for t in fig.data]
    assert 'Shocked 1M' in names
    assert 'Shocked 6M' in names
    assert 'Shocked 10Y' in names
    assert 'Base 1M' in names
    assert 'Base 6M' in names
    assert 'Base 10Y' in names


def test_build_curve_comparison_figure_ramp_shows_6m_and_12m_anchor_curves() -> None:
    curve_points = pd.DataFrame(
        {
            'scenario_id': ['ramp_up_100'] * 6,
            'scenario_label': ['Linear 12M +100 bps'] * 6,
            'state': ['anchor', 'anchor', 'month6', 'month6', 'month12', 'month12'],
            'tenor_months': [1, 12, 1, 12, 1, 12],
            'base_rate': [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            'shocked_rate': [0.01, 0.02, 0.015, 0.025, 0.02, 0.03],
        }
    )
    tenor_paths = pd.DataFrame(
        {
            'scenario_id': ['ramp_up_100'] * 8,
            'scenario_label': ['Linear 12M +100 bps'] * 8,
            'calendar_month_end': pd.to_datetime(
                ['2025-01-31', '2025-02-28', '2025-01-31', '2025-02-28', '2025-01-31', '2025-02-28', '2025-01-31', '2025-02-28']
            ),
            'tenor_label': ['1M', '1M', '1Y', '1Y', '5Y', '5Y', '10Y', '10Y'],
            'base_rate': [0.01] * 8,
            'shocked_rate': [0.01, 0.0108333, 0.02, 0.0208333, 0.03, 0.0308333, 0.04, 0.0408333],
            'shock_bps': [0.0, 8.333] * 4,
        }
    )
    fig = build_curve_comparison_figure(
        curve_points=curve_points,
        tenor_paths=tenor_paths,
        scenario_id='ramp_up_100',
        scenario_label='Linear 12M +100 bps',
    )
    names = [str(t.name) for t in fig.data]
    assert 'Shocked Curve (6M)' in names
    assert 'Shocked Curve (12M)' in names
