from __future__ import annotations

import pandas as pd

from src.dashboard.app import _combine_metric_rows, _combine_monthly_base_for_net, _combine_monthly_scenarios_for_net


def test_combine_metric_rows_builds_net_layer() -> None:
    internal = pd.Series(
        {
            'total_active_notional': 1_000_000.0,
            'weighted_avg_coupon': 0.02,
            'interest_paid_eur': 2_000.0,
            'active_deal_count': 10,
        }
    )
    external = pd.Series(
        {
            'total_active_notional': -300_000.0,
            'weighted_avg_coupon': 0.01,
            'interest_paid_eur': -250.0,
            'active_deal_count': 1,
        }
    )
    net = _combine_metric_rows(internal, external)
    assert round(float(net['total_active_notional']), 6) == 700000.0
    assert round(float(net['interest_paid_eur']), 6) == 1750.0
    assert int(net['active_deal_count']) == 11


def test_combine_monthly_scenario_outputs_reconciles_net_delta() -> None:
    base_internal = pd.DataFrame(
        {
            'month_idx': [0, 1],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'base_total_interest': [100.0, 90.0],
        }
    )
    base_external = pd.DataFrame(
        {
            'month_idx': [0, 1],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'base_total_interest': [20.0, 25.0],
        }
    )
    scen_internal = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50', 'inst_up_50'],
            'scenario_label': ['Instant +50 bps', 'Instant +50 bps'],
            'month_idx': [0, 1],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'shock_bps': [50.0, 50.0],
            'base_total_interest': [100.0, 90.0],
            'shocked_total_interest': [110.0, 99.0],
            'delta_vs_base': [10.0, 9.0],
        }
    )
    scen_external = pd.DataFrame(
        {
            'scenario_id': ['inst_up_50', 'inst_up_50'],
            'scenario_label': ['Instant +50 bps', 'Instant +50 bps'],
            'month_idx': [0, 1],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'base_total_interest': [20.0, 25.0],
            'shocked_total_interest': [22.0, 27.5],
            'delta_vs_base': [2.0, 2.5],
        }
    )

    net_base = _combine_monthly_base_for_net(base_internal, base_external)
    net_scenarios = _combine_monthly_scenarios_for_net(scen_internal, scen_external)

    assert net_base['base_total_interest'].tolist() == [120.0, 115.0]
    assert net_scenarios['delta_vs_base'].tolist() == [12.0, 11.5]
    assert net_scenarios['cumulative_delta'].tolist() == [12.0, 23.5]
