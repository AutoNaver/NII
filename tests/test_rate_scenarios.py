import pandas as pd

from src.calculations.rate_scenarios import (
    build_parallel_scenarios,
    interpolate_curve_rate,
    shock_path_bps,
    simulate_rate_scenarios,
)


def _curve_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31', '2025-02-28', '2025-02-28', '2025-02-28']),
            'ir_tenor': [1, 12, 24, 1, 12, 24],
            'rate': [0.01, 0.02, 0.03, 0.011, 0.021, 0.031],
        }
    )


def test_build_parallel_scenarios_has_expected_ids() -> None:
    out = build_parallel_scenarios()
    assert len(out) == 20
    assert out['scenario_id'].tolist() == [
        'inst_up_50',
        'inst_up_100',
        'inst_up_200',
        'inst_dn_50',
        'inst_dn_100',
        'inst_dn_200',
        'ramp_up_50',
        'ramp_up_100',
        'ramp_up_200',
        'ramp_dn_50',
        'ramp_dn_100',
        'ramp_dn_200',
        'inst_twist_up_5',
        'inst_twist_up_10',
        'inst_twist_dn_5',
        'inst_twist_dn_10',
        'ramp_twist_up_5',
        'ramp_twist_up_10',
        'ramp_twist_dn_5',
        'ramp_twist_dn_10',
    ]


def test_shock_path_instant_and_ramp() -> None:
    idx = pd.Series([0, 1, 6, 12, 18], dtype=float)
    inst = shock_path_bps('inst_up_100', idx)
    ramp = shock_path_bps('ramp_dn_120', idx)
    assert inst.tolist() == [100.0, 100.0, 100.0, 100.0, 100.0]
    assert ramp.tolist() == [0.0, -10.0, -60.0, -120.0, -120.0]


def test_shock_path_twist_signs_and_pivot() -> None:
    idx = pd.Series([0, 12], dtype=float)
    tenor_left = pd.Series([1, 1], dtype=float)
    tenor_pivot = pd.Series([6, 6], dtype=float)
    tenor_right = pd.Series([12, 12], dtype=float)

    up_left = shock_path_bps('inst_twist_up_10', idx, tenor_months=tenor_left)
    up_pivot = shock_path_bps('inst_twist_up_10', idx, tenor_months=tenor_pivot)
    up_right = shock_path_bps('inst_twist_up_10', idx, tenor_months=tenor_right)
    assert up_left.tolist() == [-10.0, -10.0]
    assert up_pivot.tolist() == [0.0, 0.0]
    assert up_right.tolist() == [10.0, 10.0]

    ramp_right = shock_path_bps('ramp_twist_dn_5', idx, tenor_months=tenor_right)
    # dn twist flips sign on right side; ramp reaches full at month 12
    assert ramp_right.tolist() == [-0.0, -5.0]


def test_interpolate_curve_rate_handles_date_fallback_and_tenor_clamp() -> None:
    curve = _curve_fixture()
    # before first curve date -> earliest date
    r_early = interpolate_curve_rate(curve, pd.Timestamp('2024-12-31'), tenor_months=6)
    # after available date -> latest <= as_of
    r_late = interpolate_curve_rate(curve, pd.Timestamp('2025-02-28'), tenor_months=6)
    # tenor clamp above max
    r_clamp = interpolate_curve_rate(curve, pd.Timestamp('2025-02-28'), tenor_months=600)
    assert round(r_early, 6) == 0.014545
    assert round(r_late, 6) == 0.015545
    assert round(r_clamp, 6) == 0.031


def test_simulate_rate_scenarios_allows_negative_downshock_rates() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 60],
            'rate': [0.001, 0.002],
        }
    )
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31'])
    existing = pd.Series([10.0, 10.0, 10.0], dtype=float)
    refill = pd.Series([1000.0, 1000.0, 1000.0], dtype=float)
    growth = pd.Series([0.0, 0.0, 0.0], dtype=float)
    tenor = pd.Series([1, 2, 3], dtype=int)
    scenarios = pd.DataFrame(
        [
            {'scenario_id': 'inst_dn_200', 'scenario_label': 'Instant -200 bps'},
        ]
    )
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=existing,
        refill_notional=refill,
        growth_notional=growth,
        tenor_months=tenor,
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
    )
    scen = out['monthly_scenarios']
    assert float(scen['shocked_rate'].min()) < 0.0


def test_simulate_rate_scenarios_zero_shock_matches_basecase_delta_zero() -> None:
    curve = _curve_fixture()
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31'])
    existing = pd.Series([5.0, 6.0, 7.0], dtype=float)
    refill = pd.Series([100.0, 80.0, 60.0], dtype=float)
    growth = pd.Series([10.0, 10.0, 10.0], dtype=float)
    tenor = pd.Series([1, 2, 3], dtype=int)
    scenarios = pd.DataFrame(
        [
            {'scenario_id': 'inst_up_0', 'scenario_label': 'Instant +0 bps'},
        ]
    )
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=existing,
        refill_notional=refill,
        growth_notional=growth,
        tenor_months=tenor,
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
    )
    scen = out['monthly_scenarios']
    assert round(float(scen['delta_vs_base'].abs().sum()), 10) == 0.0
    summary = out['yearly_summary']
    assert round(float(summary['5Y Cumulative Delta'].abs().sum()), 10) == 0.0


def test_simulate_rate_scenarios_includes_first_year_tenor_paths() -> None:
    curve = _curve_fixture()
    month_ends = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30'])
    existing = pd.Series([1.0, 1.0, 1.0, 1.0], dtype=float)
    refill = pd.Series([10.0, 10.0, 10.0, 10.0], dtype=float)
    growth = pd.Series([0.0, 0.0, 0.0, 0.0], dtype=float)
    tenor = pd.Series([1, 2, 3, 4], dtype=int)
    scenarios = pd.DataFrame([{'scenario_id': 'ramp_up_100', 'scenario_label': 'Linear 12M +100 bps'}])
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=existing,
        refill_notional=refill,
        growth_notional=growth,
        tenor_months=tenor,
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
    )
    tp = out['tenor_paths']
    assert not tp.empty
    assert set(['1M', '6M', '1Y', '5Y', '10Y']).issubset(set(tp['tenor_label'].tolist()))


def test_simulate_rate_scenarios_zero_refill_growth_has_zero_delta_for_all_scenarios() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 60],
            'rate': [0.02, 0.02],
        }
    )
    month_ends = pd.date_range('2025-01-31', periods=24, freq='ME')
    existing = pd.Series([1000.0] * 24, dtype=float)
    refill = pd.Series([0.0] * 24, dtype=float)
    growth = pd.Series([0.0] * 24, dtype=float)
    tenor = pd.Series([12] * 24, dtype=int)
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=existing,
        refill_notional=refill,
        growth_notional=growth,
        tenor_months=tenor,
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
    )
    scen = out['monthly_scenarios']
    assert round(float(scen['delta_vs_base'].abs().sum()), 10) == 0.0
    yearly = out['yearly_summary']
    assert round(float(yearly['5Y Cumulative Delta'].abs().sum()), 10) == 0.0


def test_tenor_paths_instant_and_ramp_jump_sizes_are_correct() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 12, 60, 120],
            'rate': [0.01, 0.015, 0.02, 0.025],
        }
    )
    month_ends = pd.date_range('2025-01-31', periods=13, freq='ME')
    existing = pd.Series([0.0] * 13, dtype=float)
    refill = pd.Series([1_000_000.0] * 13, dtype=float)
    growth = pd.Series([0.0] * 13, dtype=float)
    tenor = pd.Series([12] * 13, dtype=int)
    scenarios = pd.DataFrame(
        [
            {'scenario_id': 'inst_up_100', 'scenario_label': 'Instant +100 bps'},
            {'scenario_id': 'ramp_up_100', 'scenario_label': 'Linear 12M +100 bps'},
        ]
    )
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=existing,
        refill_notional=refill,
        growth_notional=growth,
        tenor_months=tenor,
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=scenarios,
    )
    tp = out['tenor_paths']
    # Check all requested tenor labels at month 0 and month 12.
    for tenor_label in ['1M', '6M', '1Y', '5Y', '10Y']:
        inst_m0 = tp[(tp['scenario_id'] == 'inst_up_100') & (tp['tenor_label'] == tenor_label) & (tp['month_idx'] == 0)]
        ramp_m0 = tp[(tp['scenario_id'] == 'ramp_up_100') & (tp['tenor_label'] == tenor_label) & (tp['month_idx'] == 0)]
        ramp_m12 = tp[(tp['scenario_id'] == 'ramp_up_100') & (tp['tenor_label'] == tenor_label) & (tp['month_idx'] == 12)]

        inst_delta_bps = float(((inst_m0['shocked_rate'] - inst_m0['base_rate']) * 10000.0).iloc[0])
        ramp_m0_delta_bps = float(((ramp_m0['shocked_rate'] - ramp_m0['base_rate']) * 10000.0).iloc[0])
        ramp_m12_delta_bps = float(((ramp_m12['shocked_rate'] - ramp_m12['base_rate']) * 10000.0).iloc[0])

        assert round(inst_delta_bps, 8) == 100.0
        assert round(ramp_m0_delta_bps, 8) == 0.0
        assert round(ramp_m12_delta_bps, 8) == 100.0


def test_tenor_paths_twist_up_has_opposite_signs_around_6m_pivot() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31'] * 4),
            'ir_tenor': [1, 6, 12, 120],
            'rate': [0.01, 0.012, 0.015, 0.025],
        }
    )
    out = simulate_rate_scenarios(
        month_ends=pd.date_range('2025-01-31', periods=13, freq='ME'),
        existing_contractual_interest=pd.Series([0.0] * 13, dtype=float),
        refill_notional=pd.Series([1.0] * 13, dtype=float),
        growth_notional=pd.Series([0.0] * 13, dtype=float),
        tenor_months=pd.Series([12] * 13, dtype=int),
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=pd.DataFrame([{'scenario_id': 'inst_twist_up_10', 'scenario_label': 'Instant Twist +10 bps (pivot 6M)'}]),
    )
    cp = out['curve_points']
    anchor = cp[(cp['state'] == 'anchor')]
    c1 = anchor[anchor['tenor_months'] == 1].iloc[0]
    c6 = anchor[anchor['tenor_months'] == 6].iloc[0]
    c12 = anchor[anchor['tenor_months'] == 12].iloc[0]
    d1 = float((c1['shocked_rate'] - c1['base_rate']) * 10000.0)
    d6 = float((c6['shocked_rate'] - c6['base_rate']) * 10000.0)
    d12 = float((c12['shocked_rate'] - c12['base_rate']) * 10000.0)
    assert round(d1, 8) == -10.0
    assert round(d6, 8) == 0.0
    assert round(d12, 8) == 10.0


def test_tenor_paths_base_is_anchor_fixed_over_first_year() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31']),
            'ir_tenor': [1, 1, 1],
            'rate': [0.01, 0.015, 0.02],
        }
    )
    month_ends = pd.date_range('2025-01-31', periods=13, freq='ME')
    out = simulate_rate_scenarios(
        month_ends=month_ends,
        existing_contractual_interest=pd.Series([0.0] * 13, dtype=float),
        refill_notional=pd.Series([1.0] * 13, dtype=float),
        growth_notional=pd.Series([0.0] * 13, dtype=float),
        tenor_months=pd.Series([1] * 13, dtype=int),
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=pd.DataFrame([{'scenario_id': 'inst_up_50', 'scenario_label': 'Instant +50 bps'}]),
    )
    tp = out['tenor_paths']
    one_m = tp[(tp['scenario_id'] == 'inst_up_50') & (tp['tenor_label'] == '1M')].copy()
    assert not one_m.empty
    # Base line is pinned to anchor date (1.00%) for the full first-year movement chart.
    assert round(float(one_m['base_rate'].iloc[0]), 8) == 0.01
    assert round(float(one_m['base_rate'].iloc[-1]), 8) == 0.01


def test_curve_points_include_month6_and_month12_for_ramp() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 12, 60, 120],
            'rate': [0.01, 0.015, 0.02, 0.025],
        }
    )
    out = simulate_rate_scenarios(
        month_ends=pd.date_range('2025-01-31', periods=13, freq='ME'),
        existing_contractual_interest=pd.Series([0.0] * 13, dtype=float),
        refill_notional=pd.Series([1.0] * 13, dtype=float),
        growth_notional=pd.Series([0.0] * 13, dtype=float),
        tenor_months=pd.Series([12] * 13, dtype=int),
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=pd.DataFrame([{'scenario_id': 'ramp_up_100', 'scenario_label': 'Linear 12M +100 bps'}]),
    )
    cp = out['curve_points']
    assert set(['anchor', 'month6', 'month12']).issubset(set(cp['state'].astype(str).unique().tolist()))
    c1_m6 = cp[(cp['state'] == 'month6') & (cp['tenor_months'] == 1)].iloc[0]
    c1_m12 = cp[(cp['state'] == 'month12') & (cp['tenor_months'] == 1)].iloc[0]
    m6_delta_bps = float((c1_m6['shocked_rate'] - c1_m6['base_rate']) * 10000.0)
    m12_delta_bps = float((c1_m12['shocked_rate'] - c1_m12['base_rate']) * 10000.0)
    assert round(m6_delta_bps, 8) == 50.0
    assert round(m12_delta_bps, 8) == 100.0


def test_tenor_paths_ramp_plateaus_after_month12() -> None:
    curve = pd.DataFrame(
        {
            'ir_date': pd.to_datetime(['2025-01-31', '2025-01-31', '2025-01-31', '2025-01-31']),
            'ir_tenor': [1, 12, 60, 120],
            'rate': [0.01, 0.015, 0.02, 0.025],
        }
    )
    out = simulate_rate_scenarios(
        month_ends=pd.date_range('2025-01-31', periods=25, freq='ME'),
        existing_contractual_interest=pd.Series([0.0] * 25, dtype=float),
        refill_notional=pd.Series([1.0] * 25, dtype=float),
        growth_notional=pd.Series([0.0] * 25, dtype=float),
        tenor_months=pd.Series([12] * 25, dtype=int),
        curve_df=curve,
        anchor_date=pd.Timestamp('2025-01-31'),
        scenarios=pd.DataFrame([{'scenario_id': 'ramp_up_100', 'scenario_label': 'Linear 12M +100 bps'}]),
    )
    tp = out['tenor_paths']
    one_y = tp[(tp['scenario_id'] == 'ramp_up_100') & (tp['tenor_label'] == '1Y')].copy()
    assert not one_y.empty
    m12 = one_y[one_y['month_idx'] == 12].iloc[0]
    m24 = one_y[one_y['month_idx'] == 24].iloc[0]
    d12 = float((m12['shocked_rate'] - m12['base_rate']) * 10000.0)
    d24 = float((m24['shocked_rate'] - m24['base_rate']) * 10000.0)
    assert round(d12, 8) == 100.0
    assert round(d24, 8) == 100.0
