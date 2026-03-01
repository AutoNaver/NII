"""Parallel rate-shock scenario calculations for Overview analytics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


_PARALLEL_RE = re.compile(r'^(inst|ramp)_(up|dn)_(\d+)$')
_TWIST_RE = re.compile(r'^(inst|ramp)_twist_(up|dn)_(\d+)$')


def build_parallel_scenarios() -> pd.DataFrame:
    """Return canonical scenario set (parallel + twist)."""
    rows: list[dict[str, object]] = []
    for materialization in ('inst', 'ramp'):
        for direction, sign in (('up', 1.0), ('dn', -1.0)):
            for magnitude in (50, 100, 200):
                sid = f'{materialization}_{direction}_{magnitude}'
                mtxt = 'Instant' if materialization == 'inst' else 'Linear 12M'
                stxt = '+' if sign > 0 else '-'
                rows.append(
                    {
                        'scenario_id': sid,
                        'scenario_label': f'{mtxt} {stxt}{magnitude} bps',
                        'shock_type': 'parallel',
                        'materialization': materialization,
                        'direction': direction,
                        'shock_bps': sign * float(magnitude),
                    }
                )
    for materialization in ('inst', 'ramp'):
        for direction, sign in (('up', 1.0), ('dn', -1.0)):
            for magnitude in (5, 10):
                sid = f'{materialization}_twist_{direction}_{magnitude}'
                mtxt = 'Instant' if materialization == 'inst' else 'Linear 12M'
                stxt = '+' if sign > 0 else '-'
                rows.append(
                    {
                        'scenario_id': sid,
                        'scenario_label': f'{mtxt} Twist {stxt}{magnitude} bps (pivot 6M)',
                        'shock_type': 'twist',
                        'materialization': materialization,
                        'direction': direction,
                        'shock_bps': sign * float(magnitude),
                    }
                )
    order = [
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
    out = pd.DataFrame(rows)
    out['sort_key'] = out['scenario_id'].apply(lambda s: order.index(str(s)) if str(s) in order else 999)
    out = out.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
    return out


def _parse_scenario_id(scenario_id: str) -> tuple[str, str, float]:
    sid = str(scenario_id).strip().lower()
    m_twist = _TWIST_RE.match(sid)
    if m_twist:
        materialization = str(m_twist.group(1))
        direction = str(m_twist.group(2))
        magnitude = float(m_twist.group(3))
        sign = 1.0 if direction == 'up' else -1.0
        return materialization, 'twist', sign * magnitude
    m_parallel = _PARALLEL_RE.match(sid)
    if m_parallel:
        materialization = str(m_parallel.group(1))
        direction = str(m_parallel.group(2))
        magnitude = float(m_parallel.group(3))
        sign = 1.0 if direction == 'up' else -1.0
        return materialization, 'parallel', sign * magnitude
    raise ValueError(f'Invalid scenario_id `{scenario_id}`.')


def shock_path_bps(
    scenario_id: str,
    month_idx: pd.Series,
    tenor_months: pd.Series | float | int | None = None,
    pivot_tenor_months: float = 6.0,
) -> pd.Series:
    """Return shock path in bps for month indices anchored at 0.

    - Parallel shocks: same sign/magnitude across all tenors.
    - Twist shocks: pivot tenor is unchanged; left/right tenors get opposite signs.
      `up` => right side up, left side down; `dn` => right side down, left side up.
    """
    materialization, shock_type, shock_bps = _parse_scenario_id(scenario_id)
    idx = pd.Series(month_idx, dtype=float)
    if materialization == 'inst':
        scale = pd.Series(1.0, index=idx.index, dtype=float)
    else:
        scale = (idx / 12.0).clip(lower=0.0, upper=1.0)
    if shock_type == 'parallel':
        tenor_sign = pd.Series(1.0, index=idx.index, dtype=float)
    else:
        if tenor_months is None:
            raise ValueError(f'Twist scenario `{scenario_id}` requires tenor_months.')
        if isinstance(tenor_months, pd.Series):
            tenor = pd.Series(tenor_months, dtype=float)
            if len(tenor) == len(idx):
                tenor = tenor.reset_index(drop=True)
                tenor.index = idx.index
            elif len(tenor) == 1:
                tenor = pd.Series(float(tenor.iloc[0]), index=idx.index, dtype=float)
            else:
                raise ValueError('tenor_months length must be 1 or match month_idx length.')
        else:
            tenor = pd.Series(float(tenor_months), index=idx.index, dtype=float)
        tenor_sign = pd.Series(np.sign(tenor - float(pivot_tenor_months)), index=idx.index, dtype=float)
    return scale * float(shock_bps) * tenor_sign


def interpolate_curve_rate(curve_df: pd.DataFrame, as_of_date: pd.Timestamp, tenor_months: int) -> float:
    """Interpolate curve rate using latest curve date <= as_of_date."""
    if curve_df is None or curve_df.empty:
        raise ValueError('Curve dataframe is empty.')

    c = curve_df.copy()
    c.columns = [str(col).strip().lower() for col in c.columns]
    needed = {'ir_date', 'ir_tenor', 'rate'}
    if not needed.issubset(c.columns):
        raise ValueError('Curve dataframe missing required columns ir_date/ir_tenor/rate.')

    c['ir_date'] = pd.to_datetime(c['ir_date'])
    c['ir_tenor'] = pd.to_numeric(c['ir_tenor'], errors='coerce')
    c['rate'] = pd.to_numeric(c['rate'], errors='coerce')
    c = c.dropna(subset=['ir_date', 'ir_tenor', 'rate']).copy()
    if c.empty:
        raise ValueError('Curve dataframe has no valid rows after normalization.')

    as_of = pd.Timestamp(as_of_date) + pd.offsets.MonthEnd(0)
    prior = c[c['ir_date'] <= as_of]
    curve_date = c['ir_date'].min() if prior.empty else prior['ir_date'].max()
    cs = c[c['ir_date'] == curve_date].copy()
    cs = cs.sort_values('ir_tenor').drop_duplicates('ir_tenor', keep='last')
    if cs.empty:
        raise ValueError(f'No curve rows found for as-of {as_of.date().isoformat()}.')

    x = cs['ir_tenor'].astype(float).to_numpy()
    y = cs['rate'].astype(float).to_numpy()
    q = float(max(int(tenor_months), 1))
    return float(np.interp(q, x, y))


def _tenor_grid(curve_df: pd.DataFrame) -> list[int]:
    c = curve_df.copy()
    c.columns = [str(col).strip().lower() for col in c.columns]
    if 'ir_tenor' not in c.columns:
        return list(range(1, 61))
    t = pd.to_numeric(c['ir_tenor'], errors='coerce').dropna().astype(int)
    t = t[t > 0]
    if t.empty:
        return list(range(1, 61))
    return sorted(set(t.tolist()))


def simulate_rate_scenarios(
    *,
    month_ends: pd.Series,
    existing_contractual_interest: pd.Series,
    refill_notional: pd.Series,
    growth_notional: pd.Series,
    tenor_months: pd.Series,
    curve_df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    scenarios: pd.DataFrame | None = None,
    reprice_existing_with_shock: bool = False,
) -> dict[str, pd.DataFrame]:
    """Simulate shocked interest paths for refill/growth vs base case."""
    idx = pd.RangeIndex(len(month_ends))
    me = pd.to_datetime(pd.Series(month_ends)).reset_index(drop=True)
    existing = pd.Series(existing_contractual_interest, dtype=float).reset_index(drop=True)
    refill = pd.Series(refill_notional, dtype=float).reset_index(drop=True)
    growth = pd.Series(growth_notional, dtype=float).reset_index(drop=True)
    tenor = pd.Series(tenor_months, dtype=float).round().astype(int).clip(lower=1).reset_index(drop=True)
    month_idx = pd.Series(np.arange(len(idx), dtype=float), index=idx)

    if not (len(me) == len(existing) == len(refill) == len(growth) == len(tenor)):
        raise ValueError('Scenario inputs must have matching lengths.')

    if scenarios is None:
        scenarios = build_parallel_scenarios()
    sdef = scenarios.copy()
    for col in ['scenario_id', 'scenario_label']:
        if col not in sdef.columns:
            raise ValueError(f'Scenarios dataframe missing `{col}`.')

    base_rate = pd.Series(
        [
            interpolate_curve_rate(curve_df, as_of_date=me.iloc[i], tenor_months=int(tenor.iloc[i]))
            for i in range(len(me))
        ],
        index=idx,
        dtype=float,
    )

    month_fraction = 30.0 / 360.0
    refill_interest_base = refill * base_rate * month_fraction
    growth_interest_base = growth * base_rate * month_fraction
    base_total = existing + refill_interest_base + growth_interest_base

    monthly_base = pd.DataFrame(
        {
            'month_idx': month_idx.astype(int),
            'calendar_month_end': me,
            'tenor_months': tenor.astype(int),
            'existing_contractual_interest': existing,
            'refill_notional': refill,
            'growth_notional': growth,
            'base_rate': base_rate,
            'refill_interest_base': refill_interest_base,
            'growth_interest_base': growth_interest_base,
            'base_total_interest': base_total,
        }
    )

    scenario_rows: list[pd.DataFrame] = []
    for row in sdef.itertuples(index=False):
        sid = str(row.scenario_id)
        label = str(row.scenario_label)
        shock_bps = shock_path_bps(
            sid,
            month_idx=month_idx,
            tenor_months=tenor,
            pivot_tenor_months=6.0,
        ).astype(float)
        shock_rate = shock_bps / 10000.0
        shocked_rate = base_rate + shock_rate
        if reprice_existing_with_shock:
            existing_shocked = existing.copy()
            nz = base_rate.abs() > 1e-12
            existing_shocked.loc[nz] = (
                existing.loc[nz] * (shocked_rate.loc[nz] / base_rate.loc[nz])
            ).astype(float)
        else:
            existing_shocked = existing.copy()
        refill_interest_shocked = refill * shocked_rate * month_fraction
        growth_interest_shocked = growth * shocked_rate * month_fraction
        shocked_total = existing_shocked + refill_interest_shocked + growth_interest_shocked
        delta = shocked_total - base_total
        scenario_rows.append(
            pd.DataFrame(
                {
                    'scenario_id': sid,
                    'scenario_label': label,
                    'month_idx': month_idx.astype(int),
                    'calendar_month_end': me,
                    'shock_bps': shock_bps,
                    'existing_interest_shocked': existing_shocked,
                    'shocked_rate': shocked_rate,
                    'refill_interest_shocked': refill_interest_shocked,
                    'growth_interest_shocked': growth_interest_shocked,
                    'shocked_total_interest': shocked_total,
                    'delta_vs_base': delta,
                    'cumulative_delta': delta.cumsum(),
                }
            )
        )

    monthly_scenarios = (
        pd.concat(scenario_rows, ignore_index=True)
        if scenario_rows
        else pd.DataFrame(
            columns=[
                'scenario_id',
                'scenario_label',
                'month_idx',
                'calendar_month_end',
                'shock_bps',
                'shocked_rate',
                'refill_interest_shocked',
                'growth_interest_shocked',
                'shocked_total_interest',
                'delta_vs_base',
                'cumulative_delta',
            ]
        )
    )

    yearly = monthly_scenarios.copy()
    if not yearly.empty:
        yearly['year_bucket'] = (yearly['month_idx'] // 12) + 1
        yearly = yearly[yearly['year_bucket'].between(1, 5)].copy()
        grouped = (
            yearly.groupby(['scenario_id', 'scenario_label', 'year_bucket'])['delta_vs_base']
            .sum()
            .reset_index()
        )
        pivot = grouped.pivot_table(
            index=['scenario_id', 'scenario_label'],
            columns='year_bucket',
            values='delta_vs_base',
            aggfunc='sum',
            fill_value=0.0,
        )
        pivot = pivot.rename(columns={1: 'Y1 Delta', 2: 'Y2 Delta', 3: 'Y3 Delta', 4: 'Y4 Delta', 5: 'Y5 Delta'})
        for col in ['Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta']:
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot['5Y Cumulative Delta'] = pivot[['Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta']].sum(axis=1)
        yearly_summary = pivot.reset_index()
        yearly_summary = (
            sdef[['scenario_id', 'scenario_label']]
            .merge(yearly_summary, on=['scenario_id', 'scenario_label'], how='left')
            .fillna(0.0)
        )
    else:
        yearly_summary = sdef[['scenario_id', 'scenario_label']].copy()
        for col in ['Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta', '5Y Cumulative Delta']:
            yearly_summary[col] = 0.0

    tenor_grid = _tenor_grid(curve_df)
    anchor = pd.Timestamp(anchor_date) + pd.offsets.MonthEnd(0)
    month6 = anchor + pd.offsets.MonthEnd(6)
    month12 = anchor + pd.offsets.MonthEnd(12)
    curve_rows: list[dict[str, object]] = []
    for row in sdef.itertuples(index=False):
        sid = str(row.scenario_id)
        label = str(row.scenario_label)
        for state, as_of, midx in (
            ('anchor', anchor, 0.0),
            ('month6', month6, 6.0),
            ('month12', month12, 12.0),
        ):
            for tenor_m in tenor_grid:
                shock_rate = float(
                    shock_path_bps(
                        sid,
                        pd.Series([midx], dtype=float),
                        tenor_months=pd.Series([tenor_m], dtype=float),
                        pivot_tenor_months=6.0,
                    ).iloc[0]
                ) / 10000.0
                base_r = interpolate_curve_rate(curve_df, as_of_date=as_of, tenor_months=int(tenor_m))
                curve_rows.append(
                    {
                        'scenario_id': sid,
                        'scenario_label': label,
                        'state': state,
                        'as_of_date': pd.Timestamp(as_of),
                        'tenor_months': int(tenor_m),
                        'base_rate': float(base_r),
                        'shocked_rate': float(base_r + shock_rate),
                        'shock_bps_state': float(shock_rate * 10000.0),
                    }
                )
    curve_points = pd.DataFrame(curve_rows)

    # Selected tenor movement path for visualization (month 0..24).
    # Ramp shocks build linearly to month 12 and then stay at the shocked level.
    # Tenors requested by product spec: 1M, 6M, 1Y, 5Y, 10Y.
    # For visualization clarity this path is anchored to the T2 base curve:
    # base tenor levels stay fixed; only the scenario shock path drives movement.
    tenor_targets: list[tuple[int, str]] = [(1, '1M'), (6, '6M'), (12, '1Y'), (60, '5Y'), (120, '10Y')]
    max_idx = int(min(len(me) - 1, 24))
    base_anchor_by_tenor: dict[int, float] = {
        tenor_m: float(interpolate_curve_rate(curve_df, as_of_date=anchor, tenor_months=tenor_m))
        for tenor_m, _ in tenor_targets
    }
    tenor_path_rows: list[dict[str, object]] = []
    for row in sdef.itertuples(index=False):
        sid = str(row.scenario_id)
        label = str(row.scenario_label)
        idx_series = pd.Series(range(max_idx + 1), dtype=float)
        for i in range(max_idx + 1):
            as_of = pd.Timestamp(me.iloc[i])
            for tenor_m, tenor_label in tenor_targets:
                base_r = float(base_anchor_by_tenor.get(tenor_m, 0.0))
                shock_bps_val = float(
                    shock_path_bps(
                        sid,
                        month_idx=pd.Series([float(i)], dtype=float),
                        tenor_months=pd.Series([float(tenor_m)], dtype=float),
                        pivot_tenor_months=6.0,
                    ).iloc[0]
                )
                shock_rate = shock_bps_val / 10000.0
                tenor_path_rows.append(
                    {
                        'scenario_id': sid,
                        'scenario_label': label,
                        'month_idx': int(i),
                        'calendar_month_end': as_of,
                        'tenor_months': int(tenor_m),
                        'tenor_label': tenor_label,
                        'base_rate': float(base_r),
                        'shocked_rate': float(base_r + shock_rate),
                        'shock_bps': shock_bps_val,
                    }
                )
    tenor_paths = pd.DataFrame(tenor_path_rows)

    return {
        'scenarios': sdef.reset_index(drop=True),
        'monthly_base': monthly_base.reset_index(drop=True),
        'monthly_scenarios': monthly_scenarios.reset_index(drop=True),
        'yearly_summary': yearly_summary.reset_index(drop=True),
        'curve_points': curve_points.reset_index(drop=True),
        'tenor_paths': tenor_paths.reset_index(drop=True),
    }
