from __future__ import annotations

import pandas as pd

from src.calculations.external_models import (
    DEFAULT_DAILY_DUE_SAVINGS_SETTINGS,
    build_external_model,
    compute_external_monthly_snapshot,
    filter_external_profile_by_product,
    normalize_external_settings,
    simulate_external_portfolio,
)


def _manual_profile_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'product': ['Mortgages', 'Mortgages'],
            'external_product_type': ['manual_profile', 'manual_profile'],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-02-28']),
            'external_notional': [1_000_000.0, 900_000.0],
            'repricing_tenor_months': [12, 12],
            'manual_rate': [0.03, 0.03],
        }
    )


def _savings_profile_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'product': ['Savings'],
            'external_product_type': ['daily_due_savings'],
            'calendar_month_end': [pd.NaT],
            'external_notional': [-1.0],
            'repricing_tenor_months': [pd.NA],
            'manual_rate': [pd.NA],
        }
    )


def _curve_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 12, 120],
            'rate': [0.01, 0.02, 0.03],
        }
    )


def test_compute_external_monthly_snapshot_mirrors_internal_volume() -> None:
    profile = _manual_profile_fixture()
    mirrored = pd.Series([2_000_000.0, 1_800_000.0], dtype=float)
    monthly = compute_external_monthly_snapshot(
        profile,
        month_ends=pd.to_datetime(['2025-01-31', '2025-02-28']),
        mirrored_notional=mirrored,
        curve_df=_curve_fixture(),
        anchor_date=pd.Timestamp('2025-01-31'),
        settings_by_model={},
    )
    assert monthly['total_active_notional'].tolist() == [2_000_000.0, 1_800_000.0]
    assert round(float(monthly.iloc[0]['interest_paid_eur']), 6) == round(2_000_000.0 * 0.03 / 12.0, 6)


def test_filter_external_profile_by_product() -> None:
    savings_row = pd.DataFrame(
        {
            'product': ['Savings'],
            'external_product_type': ['daily_due_savings'],
            'calendar_month_end': [pd.NaT],
            'external_notional': [-1.0],
            'repricing_tenor_months': [float('nan')],
            'manual_rate': [float('nan')],
        }
    )
    profile = pd.concat([_manual_profile_fixture(), savings_row], ignore_index=True)
    filtered = filter_external_profile_by_product(profile, 'Savings')
    assert len(filtered) == 1
    assert filtered['product'].tolist() == ['Savings']


def test_manual_external_model_shocks_using_repricing_tenor_with_mirrored_volume() -> None:
    profile = _manual_profile_fixture()
    model = build_external_model('manual_profile', profile)
    scenarios = pd.DataFrame(
        [
            {'scenario_id': 'inst_up_50', 'scenario_label': 'Instant +50 bps'},
        ]
    )
    out = model.simulate(
        month_ends=pd.to_datetime(['2025-01-31', '2025-02-28']),
        mirrored_notional=pd.Series([1_000_000.0, 900_000.0], dtype=float),
        curve_df=_curve_fixture(),
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
    )
    monthly_base = out['monthly_base']
    monthly_scenarios = out['monthly_scenarios']

    assert round(float(monthly_base.iloc[0]['base_total_interest']), 6) == round(1_000_000.0 * 0.03 / 12.0, 6)
    jan = monthly_scenarios[monthly_scenarios['calendar_month_end'] == pd.Timestamp('2025-01-31')].iloc[0]
    expected_delta = 1_000_000.0 * 0.005 / 12.0
    assert round(float(jan['delta_vs_base']), 6) == round(expected_delta, 6)


def test_daily_due_savings_base_path_mean_reverts_to_equilibrium() -> None:
    model = build_external_model('daily_due_savings', _savings_profile_fixture())
    settings = normalize_external_settings('daily_due_savings', DEFAULT_DAILY_DUE_SAVINGS_SETTINGS)
    out = model.simulate(
        month_ends=pd.date_range('2025-01-31', periods=3, freq='ME'),
        mirrored_notional=pd.Series([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
        curve_df=_curve_fixture(),
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=None,
        settings=settings,
    )
    base = out['monthly_base']
    equilibrium = 0.0 + 0.5 * 0.01 + 0.1 * (0.03 - 0.01)
    assert round(float(base.iloc[0]['equilibrium_rate']), 8) == round(equilibrium, 8)
    assert round(float(base.iloc[0]['client_rate']), 8) == 0.01
    assert round(float(base.iloc[1]['client_rate']), 8) == round(0.01 + 0.25 * (equilibrium - 0.01), 8)


def test_daily_due_savings_scenario_paths_differ_for_instant_and_ramp() -> None:
    model = build_external_model('daily_due_savings', _savings_profile_fixture())
    scenarios = pd.DataFrame(
        [
            {'scenario_id': 'inst_up_100', 'scenario_label': 'Instant +100 bps'},
            {'scenario_id': 'ramp_up_100', 'scenario_label': 'Ramp +100 bps'},
        ]
    )
    out = model.simulate(
        month_ends=pd.date_range('2025-01-31', periods=3, freq='ME'),
        mirrored_notional=pd.Series([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
        curve_df=_curve_fixture(),
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
        settings=DEFAULT_DAILY_DUE_SAVINGS_SETTINGS,
    )
    scen = out['monthly_scenarios']
    inst = scen[(scen['scenario_id'] == 'inst_up_100') & (scen['month_idx'] == 1)].iloc[0]
    ramp = scen[(scen['scenario_id'] == 'ramp_up_100') & (scen['month_idx'] == 1)].iloc[0]
    assert float(inst['client_rate_shocked']) != float(ramp['client_rate_shocked'])
    assert float(inst['equilibrium_rate_shocked']) != float(ramp['equilibrium_rate_shocked'])


def test_daily_due_savings_negative_liability_sign_is_preserved() -> None:
    out = simulate_external_portfolio(
        profile_df=_savings_profile_fixture(),
        month_ends=pd.date_range('2025-01-31', periods=2, freq='ME'),
        mirrored_notional=pd.Series([1_200_000.0, 1_100_000.0], dtype=float),
        curve_df=_curve_fixture(),
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=None,
        settings_by_model={'daily_due_savings': DEFAULT_DAILY_DUE_SAVINGS_SETTINGS},
    )
    base = out['monthly_base']
    assert base['total_active_notional'].tolist() == [-1_200_000.0, -1_100_000.0]
    assert (base['base_total_interest'] < 0.0).all()
