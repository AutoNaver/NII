"""Runoff charts for aligned buckets and calendar-month views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.calculations.accrual import accrued_interest_eur, accrued_interest_for_overlap
from src.calculations.refill_growth import (
    compute_refill_growth_components as calc_compute_refill_growth_components,
    compute_refill_growth_components_anchor_safe as calc_compute_refill_growth_components_anchor_safe,
    growth_outstanding_profile as calc_growth_outstanding_profile,
    shifted_portfolio_refill_weights as calc_shifted_portfolio_refill_weights,
    t0_portfolio_weights as calc_t0_portfolio_weights,
)
from src.dashboard.components.controls import coerce_option
from src.dashboard.components.formatting import plot_axis_number_format, style_numeric_table


def _series_range(series_list: list[pd.Series]) -> tuple[float, float]:
    valid = [s.astype(float).dropna() for s in series_list if s is not None]
    if not valid:
        return (0.0, 1.0)
    combined = pd.concat(valid, ignore_index=True)
    if combined.empty:
        return (0.0, 1.0)
    lo = float(combined.min())
    hi = float(combined.max())
    if lo == hi:
        if lo == 0.0:
            return (-1.0, 1.0)
        pad = abs(lo) * 0.1
        return (lo - pad, hi + pad)
    return (lo, hi)


def _aligned_secondary_range(
    primary_min: float,
    primary_max: float,
    secondary_min: float,
    secondary_max: float,
) -> tuple[float, float]:
    primary_min = min(primary_min, 0.0)
    primary_max = max(primary_max, 0.0)
    if primary_max == primary_min:
        primary_min, primary_max = -1.0, 1.0

    p = -primary_min / (primary_max - primary_min)
    if p <= 0.0:
        return (0.0, max(secondary_max, 0.0))
    if p >= 1.0:
        return (min(secondary_min, 0.0), 0.0)

    secondary_min = min(secondary_min, 0.0)
    secondary_max = max(secondary_max, 0.0)
    k = (1.0 - p) / p
    if k <= 0.0:
        return (secondary_min, secondary_max if secondary_max > secondary_min else secondary_min + 1.0)

    a = max(0.0, -secondary_min, secondary_max / k)
    if a == 0.0:
        a = 1.0
    return (-a, k * a)


def _maybe_flip_range(lo: float, hi: float, flip_y_axis: bool) -> list[float]:
    if flip_y_axis:
        return [float(hi), float(lo)]
    return [float(lo), float(hi)]


def _available_runoff_chart_options(include_refill_views: bool) -> list[str]:
    options = [
        'Notional Decomposition',
        'Effective Interest Decomposition',
        'Effective Interest Contribution',
        'Deal Count Decomposition',
        'Cumulative Notional',
    ]
    if include_refill_views:
        options.extend(
            [
                'Effective Interest Decomposition (Refill/Growth)',
                'Cumulative Notional (Refill/Growth)',
                'Refill Allocation Heatmap',
            ]
        )
    return options


def _refill_allocation_heatmap(
    *,
    month_ends: pd.Series,
    refill_required: pd.Series,
    growth_required: pd.Series,
    growth_mode: str,
    basis: str,
    runoff_compare_df: pd.DataFrame | None,
    title: str,
    x_label: str,
) -> go.Figure | None:
    month_idx = pd.Series(month_ends, dtype='datetime64[ns]')
    shifted = _shifted_portfolio_refill_weights(runoff_compare_df)
    t0 = _t0_portfolio_weights(runoff_compare_df, basis=basis)
    if shifted is None and t0 is None:
        return None

    months = pd.to_datetime(month_idx, errors='coerce').dt.strftime('%Y-%m').fillna('n/a')
    # Heatmap encodes allocation magnitude; signed direction is shown in line/decomposition charts.
    refill_total = pd.Series(refill_required, index=month_idx.index, dtype=float).abs()
    growth_total = pd.Series(growth_required, index=month_idx.index, dtype=float).abs()
    include_growth = str(growth_mode).strip().lower() == 'user_defined' and float(growth_total.sum()) > 1e-12

    refill_w = None
    if shifted is not None:
        refill_w = shifted.set_index('tenor')['weight'].astype(float)
    elif t0 is not None:
        refill_w = t0.astype(float)
    if refill_w is None:
        return None
    refill_w = refill_w.clip(lower=0.0)
    if float(refill_w.sum()) <= 1e-12:
        return None
    refill_w = refill_w / float(refill_w.sum())

    growth_w = pd.Series(dtype=float)
    if include_growth:
        if t0 is not None and float(t0.sum()) > 1e-12:
            growth_w = t0.astype(float).clip(lower=0.0)
        else:
            growth_w = refill_w.copy()
        if float(growth_w.sum()) > 1e-12:
            growth_w = growth_w / float(growth_w.sum())
        else:
            growth_w = pd.Series(dtype=float)

    tenor_index = pd.Index(sorted(set(refill_w.index.astype(int)).union(set(growth_w.index.astype(int)))), dtype=int)
    refill_vec = refill_w.reindex(tenor_index, fill_value=0.0).to_numpy(dtype=float)
    growth_vec = growth_w.reindex(tenor_index, fill_value=0.0).to_numpy(dtype=float)
    z = np.outer(refill_vec, refill_total.to_numpy(dtype=float))
    if include_growth and growth_vec.size:
        z = z + np.outer(growth_vec, growth_total.to_numpy(dtype=float))

    row_mask = np.nansum(z, axis=1) > 1e-12
    if bool(row_mask.any()):
        z = z[row_mask, :]
        tenor_index = tenor_index[row_mask]

    z_plot = np.where(z > 0.0, z, np.nan)
    if np.isnan(z_plot).all():
        z_plot = z

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=months,
                y=tenor_index.to_numpy(),
                z=z_plot,
                colorscale='YlGnBu',
                colorbar=dict(title='Refill + Growth (EUR)' if include_growth else 'Refill (EUR)'),
                hovertemplate='Month: %{x}<br>Tenor: %{y}M<br>Allocation: %{z:,.2f}<extra></extra>',
            )
        ]
    )
    fig.update_layout(
        title=title,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(title=x_label, type='category')
    fig.update_yaxes(title='Refill Tenor Bucket (Months)')
    return fig


def _refill_volume_interest_chart(
    *,
    month_ends: pd.Series,
    refill_required: pd.Series,
    growth_volume: pd.Series,
    refill_rate: pd.Series,
    title: str,
    x_label: str,
    flip_y_axis: bool = False,
) -> go.Figure:
    month_idx = pd.Series(month_ends, dtype='datetime64[ns]')
    months = pd.to_datetime(month_idx, errors='coerce').dt.strftime('%Y-%m').fillna('n/a')
    # Keep signed flows so liability products (negative notionals) remain visible.
    refill_volume = pd.Series(refill_required, index=month_idx.index, dtype=float)
    growth_volume = pd.Series(growth_volume, index=month_idx.index, dtype=float)
    total_volume = refill_volume + growth_volume
    annual_refill_interest = refill_volume * pd.Series(refill_rate, index=month_idx.index, dtype=float)
    annual_growth_interest = growth_volume * pd.Series(refill_rate, index=month_idx.index, dtype=float)
    annual_total_interest = annual_refill_interest + annual_growth_interest

    fig = go.Figure()
    fig.add_scatter(
        x=months,
        y=refill_volume,
        mode='lines+markers',
        name='Refill Volume',
        line=dict(color='#22c55e', width=2),
    )
    fig.add_scatter(
        x=months,
        y=growth_volume,
        mode='lines+markers',
        name='Growth Volume',
        line=dict(color='#f59e0b', width=2),
    )
    fig.add_scatter(
        x=months,
        y=total_volume,
        mode='lines+markers',
        name='Total Volume',
        line=dict(color='#e2e8f0', width=2, dash='dash'),
    )
    fig.add_scatter(
        x=months,
        y=annual_refill_interest,
        mode='lines+markers',
        name='Refill Interest (Annualized)',
        line=dict(color='#60a5fa', width=2, dash='dot'),
        yaxis='y2',
    )
    fig.add_scatter(
        x=months,
        y=annual_growth_interest,
        mode='lines+markers',
        name='Growth Interest (Annualized)',
        line=dict(color='#c084fc', width=2, dash='dot'),
        yaxis='y2',
    )
    fig.add_scatter(
        x=months,
        y=annual_total_interest,
        mode='lines+markers',
        name='Total Interest (Annualized)',
        line=dict(color='#22d3ee', width=2, dash='dash'),
        yaxis='y2',
    )
    fig.update_layout(
        title=title,
        hovermode='x unified',
        legend=dict(orientation='h'),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title='Refill/Growth Volume (EUR)',
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
            automargin=True,
        ),
        yaxis2=dict(
            title='Annual Refill/Growth Interest (EUR)',
            overlaying='y',
            side='right',
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
            automargin=True,
            showgrid=False,
        ),
    )
    if flip_y_axis:
        fig.update_layout(yaxis_autorange='reversed', yaxis2_autorange='reversed')
    fig.update_xaxes(title=x_label, type='category')
    return plot_axis_number_format(fig, y_axes=['yaxis', 'yaxis2'])


def _component_chart(
    *,
    x: pd.Series,
    y_existing: pd.Series,
    y_added: pd.Series,
    y_matured: pd.Series,
    y_total: pd.Series,
    y_cumulative: pd.Series | None,
    title: str,
    x_label: str,
    y_label: str,
    cumulative_label: str,
    y_refilled: pd.Series | None = None,
    y_growth: pd.Series | None = None,
    flip_y_axis: bool = False,
) -> go.Figure:
    matured_effect = -y_matured
    primary_min, primary_max = _series_range([y_existing, y_added, y_refilled, y_growth, matured_effect, y_total])
    primary_min = min(primary_min, 0.0)
    primary_max = max(primary_max, 0.0)

    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=y_existing,
        name='Existing',
        marker=dict(color='#1f77b4'),
    )
    fig.add_bar(
        x=x,
        y=y_added,
        name='Added',
        marker=dict(color='#2ca02c'),
    )
    if y_refilled is not None:
        fig.add_bar(
            x=x,
            y=y_refilled,
            name='Refilled',
            marker=dict(color='#8c564b'),
        )
    if y_growth is not None:
        fig.add_bar(
            x=x,
            y=y_growth,
            name='Growth',
            marker=dict(color='#ff7f0e'),
        )
    fig.add_bar(
        x=x,
        y=matured_effect,
        name='Matured',
        marker=dict(color='#ef4444'),
    )
    fig.add_scatter(
        x=x,
        y=y_total,
        mode='lines+markers',
        name='Total',
        line=dict(color='#00e5ff', width=2),
    )
    if y_cumulative is not None:
        fig.add_scatter(
            x=x,
            y=y_cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#ffd166', width=2, dash='dot'),
            yaxis='y2',
        )

    layout_kwargs = {
        'title': title,
        'barmode': 'relative',
        'legend': dict(orientation='h'),
        'hovermode': 'x unified',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'yaxis': dict(
            title=y_label,
            range=_maybe_flip_range(primary_min, primary_max, flip_y_axis),
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        ),
    }
    if y_cumulative is not None:
        secondary_min, secondary_max = _series_range([y_cumulative])
        sec_lo, sec_hi = _aligned_secondary_range(primary_min, primary_max, secondary_min, secondary_max)
        layout_kwargs['yaxis2'] = dict(
            title=cumulative_label,
            overlaying='y',
            side='right',
            range=_maybe_flip_range(sec_lo, sec_hi, flip_y_axis),
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title=x_label)
    y_axes = ['yaxis']
    if y_cumulative is not None:
        y_axes.append('yaxis2')
    return plot_axis_number_format(fig, y_axes=y_axes)


def _effective_contribution_chart(
    *,
    x: pd.Series,
    y_existing: pd.Series,
    y_added: pd.Series,
    y_matured: pd.Series,
    y_total: pd.Series,
    y_cumulative: pd.Series | None,
    title: str,
    x_label: str,
    cumulative_label: str,
    flip_y_axis: bool = False,
) -> go.Figure:
    matured_effect = -y_matured
    primary_min, primary_max = _series_range([y_existing, y_added, matured_effect, y_total])
    primary_min = min(primary_min, 0.0)
    primary_max = max(primary_max, 0.0)

    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=y_existing,
        name='Existing',
        marker=dict(color='#1f77b4'),
    )
    fig.add_bar(
        x=x,
        y=y_added,
        name='Added',
        marker=dict(color='#2ca02c'),
    )
    fig.add_bar(
        x=x,
        y=matured_effect,
        name='Matured',
        marker=dict(color='#ef4444'),
    )
    fig.add_scatter(
        x=x,
        y=y_total,
        mode='lines+markers',
        name='Total',
        line=dict(color='#00e5ff', width=2),
    )
    if y_cumulative is not None:
        fig.add_scatter(
            x=x,
            y=y_cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#ffd166', width=2, dash='dot'),
            yaxis='y2',
        )

    fig.update_layout(
        title=title,
        barmode='relative',
        legend=dict(orientation='h'),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title='Effective Interest',
            range=_maybe_flip_range(primary_min, primary_max, flip_y_axis),
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        ),
    )
    if y_cumulative is not None:
        secondary_min, secondary_max = _series_range([y_cumulative])
        sec_lo, sec_hi = _aligned_secondary_range(primary_min, primary_max, secondary_min, secondary_max)
        fig.update_layout(
            yaxis2=dict(
                title=cumulative_label,
                overlaying='y',
                side='right',
                range=_maybe_flip_range(sec_lo, sec_hi, flip_y_axis),
                showgrid=False,
                zeroline=True,
                zerolinewidth=1,
                separatethousands=True,
            )
        )
    fig.update_xaxes(title=x_label)
    y_axes = ['yaxis']
    if y_cumulative is not None:
        y_axes.append('yaxis2')
    return plot_axis_number_format(fig, y_axes=y_axes)


def _remaining_maturity_months(basis_date: pd.Timestamp, maturity_date: pd.Timestamp) -> int:
    b = pd.Timestamp(basis_date) + pd.offsets.MonthEnd(0)
    m = pd.Timestamp(maturity_date) + pd.offsets.MonthEnd(0)
    return max(0, min(240, int((m.to_period('M') - b.to_period('M')).n)))


def _normalize_refill_logic(refill_logic_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if refill_logic_df is None or refill_logic_df.empty:
        return None
    raw = refill_logic_df.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    col_map = {}
    for c in raw.columns:
        key = c.replace(' ', '_')
        if key == 'tenor':
            col_map[c] = 'tenor'
        elif key in {'t_0_notional', 't0_notional'}:
            col_map[c] = 't_0_notional'
        elif key in {'existing_deals', 'existing'}:
            col_map[c] = 'existing_deals'
        elif key in {'delta_deals', 'delta'}:
            col_map[c] = 'delta_deals'
        elif key in {'t_1_notional', 't1_notional'}:
            col_map[c] = 't_1_notional'
    raw = raw.rename(columns=col_map)
    needed = {'tenor', 't_0_notional', 'existing_deals', 'delta_deals', 't_1_notional'}
    if not needed.issubset(raw.columns):
        return None
    ordered_cols = ['tenor', 't_0_notional', 'existing_deals', 'delta_deals', 't_1_notional']
    out = raw[ordered_cols].copy()
    out['tenor'] = pd.to_numeric(out['tenor'], errors='coerce').astype('Int64')
    out['t_0_notional'] = pd.to_numeric(out['t_0_notional'], errors='coerce')
    out['existing_deals'] = pd.to_numeric(out['existing_deals'], errors='coerce')
    out['delta_deals'] = pd.to_numeric(out['delta_deals'], errors='coerce')
    out['t_1_notional'] = pd.to_numeric(out['t_1_notional'], errors='coerce')
    out = out.dropna().copy()
    if out.empty:
        return None
    out['tenor'] = out['tenor'].astype(int)
    out = out.sort_values('tenor').drop_duplicates('tenor', keep='last').reset_index(drop=True)
    return out


def _shifted_portfolio_refill_weights(runoff_compare_df: pd.DataFrame | None) -> pd.DataFrame | None:
    return calc_shifted_portfolio_refill_weights(runoff_compare_df)


def _t0_portfolio_weights(runoff_compare_df: pd.DataFrame | None, *, basis: str) -> pd.Series | None:
    return calc_t0_portfolio_weights(runoff_compare_df, basis=basis)


def _growth_outstanding_profile(
    *,
    growth_flow: pd.Series,
    runoff_compare_df: pd.DataFrame | None,
    basis: str,
) -> pd.Series:
    return calc_growth_outstanding_profile(
        growth_flow=growth_flow,
        runoff_compare_df=runoff_compare_df,
        basis=basis,
    )


def _curve_rate_by_tenor(
    *,
    tenor_points: pd.Series,
    base_notional_total: pd.Series,
    base_effective_total: pd.Series,
    curve_df: pd.DataFrame | None = None,
    basis_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Return annualized curve rate by tenor; fallback to observed ratio."""
    idx = base_notional_total.index
    eff_rate = pd.Series(0.0, index=idx, dtype=float)
    tenor = pd.Series(tenor_points, index=idx).astype(float).round().astype(int)
    curve_rate = None

    if curve_df is not None and basis_date is not None and not curve_df.empty:
        c = curve_df.copy()
        c.columns = [str(col).strip().lower() for col in c.columns]
        needed_cols = {'ir_date', 'ir_tenor', 'rate'}
        if needed_cols.issubset(c.columns):
            c['ir_date'] = pd.to_datetime(c['ir_date'])
            c['ir_tenor'] = pd.to_numeric(c['ir_tenor'], errors='coerce')
            c['rate'] = pd.to_numeric(c['rate'], errors='coerce')
            c = c.dropna(subset=['ir_date', 'ir_tenor', 'rate']).copy()
            if not c.empty:
                as_of = pd.Timestamp(basis_date) + pd.offsets.MonthEnd(0)
                prior = c[c['ir_date'] <= as_of]
                curve_date = c['ir_date'].min() if prior.empty else prior['ir_date'].max()
                cs = c[c['ir_date'] == curve_date].copy()
                cs = cs.sort_values('ir_tenor').drop_duplicates('ir_tenor', keep='last')
                if len(cs) >= 2:
                    x = cs['ir_tenor'].astype(float).to_numpy()
                    y = cs['rate'].astype(float).to_numpy()
                    q = tenor.astype(float).to_numpy()
                    curve_rate = pd.Series(np.interp(q, x, y), index=idx, dtype=float)
                elif len(cs) == 1:
                    curve_rate = pd.Series(float(cs['rate'].iloc[0]), index=idx, dtype=float)

    if curve_rate is None:
        non_zero = base_notional_total.abs() > 1e-12
        eff_rate.loc[non_zero] = (
            base_effective_total.loc[non_zero].abs() / base_notional_total.loc[non_zero].abs()
        ).astype(float)
    else:
        eff_rate = curve_rate.astype(float)
    return eff_rate


def _build_refill_series(
    *,
    base_notional_total: pd.Series,
    base_effective_total: pd.Series,
    tenor_points: pd.Series,
    refill_logic_df: pd.DataFrame | None,
    curve_df: pd.DataFrame | None = None,
    basis_date: pd.Timestamp | None = None,
) -> dict[str, pd.Series] | None:
    refill = _normalize_refill_logic(refill_logic_df)
    if refill is None:
        return None

    tenor_min = int(refill['tenor'].min())
    tenor_max = int(refill['tenor'].max())
    tenor_clipped = pd.Series(tenor_points, index=base_notional_total.index).astype(float).round().astype(int)
    tenor_clipped = tenor_clipped.clip(lower=tenor_min, upper=tenor_max)

    x_template = refill['tenor'].astype(float).to_numpy()
    t0_template = refill['t_0_notional'].astype(float).to_numpy()
    existing_template = refill['existing_deals'].astype(float).to_numpy()
    delta_template = refill['delta_deals'].astype(float).to_numpy()

    x_query = tenor_clipped.astype(float).to_numpy()
    t0_interp = pd.Series(np.interp(x_query, x_template, t0_template), index=base_notional_total.index, dtype=float)
    existing_interp = pd.Series(np.interp(x_query, x_template, existing_template), index=base_notional_total.index, dtype=float)
    delta_interp = pd.Series(np.interp(x_query, x_template, delta_template), index=base_notional_total.index, dtype=float)

    # Scale template notionals once to the current portfolio magnitude.
    template_total = float(t0_interp.sum())
    current_total = float(base_notional_total.abs().sum())
    scale = (current_total / template_total) if abs(template_total) > 1e-12 else 1.0

    refill_existing_notional = existing_interp * scale
    refill_delta_notional = delta_interp * scale
    refill_total_notional = refill_existing_notional + refill_delta_notional

    # Remunerate refill delta deals from Interest_Curve by tenor (interpolated).
    # Fallback to observed effective/notional ratio if curve is unavailable.
    eff_rate = pd.Series(0.0, index=base_notional_total.index, dtype=float)
    curve_rate = None
    if curve_df is not None and basis_date is not None and not curve_df.empty:
        c = curve_df.copy()
        c.columns = [str(col).strip().lower() for col in c.columns]
        needed_cols = {'ir_date', 'ir_tenor', 'rate'}
        if needed_cols.issubset(c.columns):
            c['ir_date'] = pd.to_datetime(c['ir_date'])
            c['ir_tenor'] = pd.to_numeric(c['ir_tenor'], errors='coerce')
            c['rate'] = pd.to_numeric(c['rate'], errors='coerce')
            c = c.dropna(subset=['ir_date', 'ir_tenor', 'rate']).copy()
            if not c.empty:
                as_of = pd.Timestamp(basis_date) + pd.offsets.MonthEnd(0)
                prior = c[c['ir_date'] <= as_of]
                if prior.empty:
                    curve_date = c['ir_date'].min()
                else:
                    curve_date = prior['ir_date'].max()
                cs = c[c['ir_date'] == curve_date].copy()
                cs = cs.sort_values('ir_tenor').drop_duplicates('ir_tenor', keep='last')
                if len(cs) >= 2:
                    x = cs['ir_tenor'].astype(float).to_numpy()
                    y = cs['rate'].astype(float).to_numpy()
                    q = tenor_clipped.astype(float).to_numpy()
                    curve_rate = pd.Series(np.interp(q, x, y), index=base_notional_total.index, dtype=float)
                elif len(cs) == 1:
                    curve_rate = pd.Series(float(cs['rate'].iloc[0]), index=base_notional_total.index, dtype=float)
    if curve_rate is None:
        non_zero = base_notional_total.abs() > 1e-12
        eff_rate.loc[non_zero] = (
            base_effective_total.loc[non_zero].abs() / base_notional_total.loc[non_zero].abs()
        ).astype(float)
    else:
        eff_rate = curve_rate.astype(float)

    month_fraction = 30.0 / 360.0

    refill_existing_effective = refill_existing_notional * eff_rate * month_fraction
    refill_delta_effective = refill_delta_notional * eff_rate * month_fraction
    refill_total_effective = refill_total_notional * eff_rate * month_fraction

    return {
        'curve_rate': eff_rate,
        'existing_notional': refill_existing_notional,
        'delta_notional': refill_delta_notional,
        'total_notional': refill_total_notional,
        'existing_effective': refill_existing_effective,
        'delta_effective': refill_delta_effective,
        'total_effective': refill_total_effective,
    }


def _compute_refill_growth_components(
    *,
    cumulative_notional: pd.Series,
    growth_mode: str,
    monthly_growth_amount: float,
) -> dict[str, pd.Series]:
    return calc_compute_refill_growth_components(
        cumulative_notional=cumulative_notional,
        growth_mode=growth_mode,
        monthly_growth_amount=monthly_growth_amount,
    )


def _compute_refill_growth_components_anchor_safe(
    *,
    cumulative_notional: pd.Series,
    growth_mode: str,
    monthly_growth_amount: float,
) -> dict[str, pd.Series]:
    return calc_compute_refill_growth_components_anchor_safe(
        cumulative_notional=cumulative_notional,
        growth_mode=growth_mode,
        monthly_growth_amount=monthly_growth_amount,
    )


def _build_aggregation_windows(month_ends: pd.Series, window_mode: str) -> list[tuple[str, pd.Series]]:
    if month_ends.empty:
        return []

    mode = str(window_mode or '').strip().lower()
    positions = pd.Series(np.arange(len(month_ends)), index=month_ends.index)
    windows: list[tuple[str, pd.Series]] = []

    if mode == 'next 5 years':
        all_mask = (positions >= 60) & (positions < 120)
        windows.append(('All (Y1-Y5)', all_mask))
        for y in range(1, 6):
            start = 60 + (y - 1) * 12
            end = start + 12
            y_mask = (positions >= start) & (positions < end)
            if bool(y_mask.any()):
                windows.append((f'Y{y}', y_mask))
        return windows

    years = month_ends.dt.year.drop_duplicates().tolist()[:5]
    if not years:
        return []
    all_mask = month_ends.dt.year.isin(set(years))
    windows.append(('All (5 calendar years)', all_mask))
    for year in years:
        windows.append((str(year), month_ends.dt.year == int(year)))
    return windows


def _render_aggregation_table(
    *,
    month_ends: pd.Series,
    notional_existing: pd.Series,
    notional_added: pd.Series,
    notional_matured_effect: pd.Series,
    notional_refilled: pd.Series,
    notional_growth: pd.Series,
    notional_total: pd.Series,
    effective_existing: pd.Series,
    effective_added: pd.Series,
    effective_matured_effect: pd.Series,
    effective_refilled: pd.Series,
    effective_growth: pd.Series,
    effective_total: pd.Series,
    key_prefix: str,
) -> None:
    horizon_options = ['Next 5 Years', '5 Calendar Years']
    horizon_key = f'{key_prefix}_aggregation_horizon'
    horizon_default = 'Next 5 Years'
    horizon_current = coerce_option(st.session_state.get(horizon_key, horizon_default), horizon_options, horizon_default)
    st.session_state[horizon_key] = horizon_current
    window_mode = st.radio(
        label='Aggregation horizon',
        options=horizon_options,
        index=horizon_options.index(horizon_current),
        horizontal=True,
        key=horizon_key,
    )

    windows = _build_aggregation_windows(month_ends, window_mode)
    if not windows:
        st.info('No runoff points available for the selected aggregation horizon.')
        return
    split_options = [label for label, _ in windows]
    split_key = f'{key_prefix}_aggregation_split'
    split_default = split_options[0]
    split_current = coerce_option(st.session_state.get(split_key, split_default), split_options, split_default)
    st.session_state[split_key] = split_current
    split = st.radio(
        label='Aggregation split',
        options=split_options,
        index=split_options.index(split_current),
        horizontal=True,
        key=split_key,
    )
    masks = {label: mask for label, mask in windows}
    mask = masks[split]
    if not bool(mask.any()):
        st.info('No runoff points available for the selected aggregation horizon.')
        return

    def _sum(s: pd.Series) -> float:
        return float(s.loc[mask].astype(float).sum())

    rows = [
        {
            'Metric': 'Notional',
            'Existing': _sum(notional_existing),
            'Added': _sum(notional_added),
            'Refilled': _sum(notional_refilled),
            'Growth': _sum(notional_growth),
            'Matured': _sum(notional_matured_effect),
            'Total': _sum(notional_total),
        },
        {
            'Metric': 'Effective Interest',
            'Existing': _sum(effective_existing),
            'Added': _sum(effective_added),
            'Refilled': _sum(effective_refilled),
            'Growth': _sum(effective_growth),
            'Matured': _sum(effective_matured_effect),
            'Total': _sum(effective_total),
        },
    ]
    table = pd.DataFrame(rows).set_index('Metric')
    st.dataframe(style_numeric_table(table), use_container_width=True)


def _bucketed_month_effective_contribution(
    deals_df: pd.DataFrame,
    basis_date: pd.Timestamp,
    target_month_end: pd.Timestamp,
) -> pd.DataFrame:
    """Decompose one target month's effective interest into maturity buckets.

    Existing is the full-month baseline of start-of-month deals.
    Matured is the signed in-month run-off adjustment (baseline - actual overlap).
    Added is interest from deals valued within the target month.
    """
    basis = pd.Timestamp(basis_date) + pd.offsets.MonthEnd(0)
    me = pd.Timestamp(target_month_end) + pd.offsets.MonthEnd(0)
    ms = me.replace(day=1)
    we = me + pd.Timedelta(days=1)

    prior = deals_df[(deals_df['value_date'] < ms) & (deals_df['maturity_date'] > ms)].copy()
    added = deals_df[(deals_df['value_date'] >= ms) & (deals_df['value_date'] < we)].copy()

    rows: list[dict[str, float | int]] = []

    for row in prior.itertuples(index=False):
        bucket = _remaining_maturity_months(basis, row.maturity_date)
        full_month = accrued_interest_eur(row.notional, row.coupon, ms, we)
        actual = accrued_interest_for_overlap(
            notional=row.notional,
            annual_coupon=row.coupon,
            deal_value_date=row.value_date,
            deal_maturity_date=row.maturity_date,
            window_start=ms,
            window_end=we,
        )
        matured_adj = full_month - actual
        rows.append(
            {
                'remaining_maturity_months': bucket,
                'existing': float(full_month),
                'added': 0.0,
                'matured': float(matured_adj),
            }
        )

    for row in added.itertuples(index=False):
        bucket = _remaining_maturity_months(basis, row.maturity_date)
        actual = accrued_interest_for_overlap(
            notional=row.notional,
            annual_coupon=row.coupon,
            deal_value_date=row.value_date,
            deal_maturity_date=row.maturity_date,
            window_start=ms,
            window_end=we,
        )
        if abs(float(actual)) < 1e-12:
            continue
        rows.append(
            {
                'remaining_maturity_months': bucket,
                'existing': 0.0,
                'added': float(actual),
                'matured': 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(
            {
                'remaining_maturity_months': [1],
                'existing': [0.0],
                'added': [0.0],
                'matured': [0.0],
                'total': [0.0],
            }
        )

    out = pd.DataFrame(rows).groupby('remaining_maturity_months', as_index=False).sum()
    out = out.sort_values('remaining_maturity_months').reset_index(drop=True)
    out['total'] = out['existing'] + out['added'] - out['matured']
    return out


def render_runoff_delta_charts(
    compare_df: pd.DataFrame,
    key_prefix: str = 'runoff_buckets_mode',
    deals_df: pd.DataFrame | None = None,
    basis_t1: pd.Timestamp | None = None,
    basis_t2: pd.Timestamp | None = None,
    refill_logic_df: pd.DataFrame | None = None,
    curve_df: pd.DataFrame | None = None,
    ui_state: dict[str, Any] | None = None,
) -> str:
    if compare_df.empty:
        st.info('No runoff data to display.')
        return 'Notional Decomposition'

    # Focus range where there is activity
    mask = (
        (compare_df[['abs_notional_d1', 'abs_notional_d2', 'cumulative_abs_notional_d1', 'cumulative_abs_notional_d2']].sum(axis=1) > 0)
        | ((compare_df['deal_count_d1'] + compare_df['deal_count_d2']) > 0)
    )
    active_df = compare_df.loc[mask].copy()
    if active_df.empty:
        active_df = compare_df.head(24)  # fallback first 24 buckets

    def _col_or_zero(col: str) -> pd.Series:
        if col in active_df.columns:
            return active_df[col].astype(float)
        return pd.Series(0.0, index=active_df.index, dtype=float)

    def _first_available(cols: list[str]) -> pd.Series:
        for col in cols:
            if col in active_df.columns:
                return active_df[col].astype(float)
        return pd.Series(0.0, index=active_df.index, dtype=float)

    ui_state = ui_state or {}
    basis = str(ui_state.get('runoff_decomposition_basis', 'T2'))
    growth_mode = str(ui_state.get('growth_mode', 'constant'))
    monthly_growth_amount = float(ui_state.get('growth_monthly_value', 0.0))
    chart_view = str(ui_state.get('runoff_chart_view', 'Notional Decomposition'))
    flip_y_axis = bool(ui_state.get('flip_y_axis', False))

    if basis == 'T1':
        notional_existing = _first_available(['signed_notional_d1', 'abs_notional_d1'])
        notional_added = pd.Series(0.0, index=active_df.index, dtype=float)
        notional_matured = pd.Series(0.0, index=active_df.index, dtype=float)
        notional_total = _first_available(['signed_notional_d1', 'abs_notional_d1'])
        abs_notional_existing = _first_available(['abs_notional_d1'])
        abs_notional_added = pd.Series(0.0, index=active_df.index, dtype=float)
        abs_notional_matured = pd.Series(0.0, index=active_df.index, dtype=float)
        abs_notional_total = _first_available(['abs_notional_d1'])

        nc_existing = _first_available(['effective_interest_d1', 'notional_coupon_d1'])
        nc_added = pd.Series(0.0, index=active_df.index, dtype=float)
        nc_matured = pd.Series(0.0, index=active_df.index, dtype=float)
        nc_total = _first_available(['effective_interest_d1', 'notional_coupon_d1'])

        deals_existing = active_df['deal_count_d1']
        deals_added = pd.Series(0.0, index=active_df.index, dtype=float)
        deals_matured = pd.Series(0.0, index=active_df.index, dtype=float)
        deals_total = active_df['deal_count_d1']
    else:
        notional_existing = (
            _first_available(['signed_notional_d2', 'abs_notional_d2'])
            - _first_available(['added_notional', 'added_abs_notional'])
            + _first_available(['matured_notional', 'matured_abs_notional'])
        )
        notional_added = _first_available(['added_notional', 'added_abs_notional'])
        notional_matured = _first_available(['matured_notional', 'matured_abs_notional'])
        notional_total = _first_available(['signed_notional_d2', 'abs_notional_d2'])
        abs_notional_existing = (
            _first_available(['abs_notional_d2'])
            - _first_available(['added_abs_notional'])
            + _first_available(['matured_abs_notional'])
        )
        abs_notional_added = _first_available(['added_abs_notional'])
        abs_notional_matured = _first_available(['matured_abs_notional'])
        abs_notional_total = _first_available(['abs_notional_d2'])

        nc_existing = (
            _first_available(['effective_interest_d2', 'notional_coupon_d2'])
            - _first_available(['added_effective_interest', 'added_notional_coupon'])
            + _first_available(['matured_effective_interest', 'matured_notional_coupon'])
        )
        nc_added = _first_available(['added_effective_interest', 'added_notional_coupon'])
        nc_matured = _first_available(['matured_effective_interest', 'matured_notional_coupon'])
        nc_total = _first_available(['effective_interest_d2', 'notional_coupon_d2'])

        deals_existing = (
            active_df['deal_count_d2']
            - _col_or_zero('added_deal_count')
            + _col_or_zero('matured_deal_count')
        )
        deals_added = _col_or_zero('added_deal_count')
        deals_matured = _col_or_zero('matured_deal_count')
        deals_total = active_df['deal_count_d2']

    # Cumulative notional as runoff profile (decreasing over time as maturities roll off)
    cum_notional_existing = abs_notional_existing[::-1].cumsum()[::-1]
    cum_notional_added = abs_notional_added[::-1].cumsum()[::-1]
    cum_notional_matured = abs_notional_matured[::-1].cumsum()[::-1]
    cum_notional_total = abs_notional_total[::-1].cumsum()[::-1]
    cum_notional_existing_chart = notional_existing[::-1].cumsum()[::-1]
    cum_notional_added_chart = notional_added[::-1].cumsum()[::-1]
    cum_notional_matured_chart = notional_matured[::-1].cumsum()[::-1]
    if basis == 'T1':
        cum_notional_total_chart = _first_available(
            ['cumulative_signed_notional_d1', 'cumulative_abs_notional_d1']
        )
    else:
        cum_notional_total_chart = _first_available(
            ['cumulative_signed_notional_d2', 'cumulative_abs_notional_d2']
        )
    growth_components = _compute_refill_growth_components_anchor_safe(
        cumulative_notional=cum_notional_total_chart.astype(float),
        growth_mode=growth_mode,
        monthly_growth_amount=monthly_growth_amount,
    )
    refill_required = growth_components['refill_required']
    growth_required = growth_components['growth_required']
    growth_outstanding = _growth_outstanding_profile(
        growth_flow=growth_required,
        runoff_compare_df=compare_df,
        basis=basis,
    )
    refill_rate = _curve_rate_by_tenor(
        tenor_points=active_df['remaining_maturity_months'].astype(float),
        base_notional_total=abs_notional_total.astype(float),
        base_effective_total=nc_total.astype(float),
        curve_df=curve_df,
        basis_date=(pd.Timestamp(basis_t1) if basis == 'T1' and basis_t1 is not None else pd.Timestamp(basis_t2) if basis_t2 is not None else None),
    )
    refill_effective = refill_required * refill_rate * (30.0 / 360.0)
    growth_effective = growth_outstanding * refill_rate * (30.0 / 360.0)
    refill_title_suffix = 'Refill'
    if str(growth_mode).strip().lower() == 'user_defined' and float(monthly_growth_amount) > 0.0:
        refill_title_suffix = f'Refill + Growth ({monthly_growth_amount:,.2f}/month)'
    include_refill_views = True
    selected_view = coerce_option(
        chart_view,
        _available_runoff_chart_options(include_refill_views),
        'Notional Decomposition',
    )
    if basis == 'T1' and basis_t1 is not None:
        basis_date = pd.Timestamp(basis_t1)
    elif basis_t2 is not None:
        basis_date = pd.Timestamp(basis_t2)
    elif basis_t1 is not None:
        basis_date = pd.Timestamp(basis_t1)
    else:
        basis_date = pd.Timestamp.today().normalize()
    month_ends = active_df['remaining_maturity_months'].astype(int).apply(
        lambda m: (basis_date + pd.offsets.MonthEnd(int(m))).normalize()
    )

    if selected_view == 'Notional Decomposition':
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=notional_existing,
            y_added=notional_added,
            y_matured=notional_matured,
            y_total=notional_total,
            y_cumulative=notional_total.cumsum(),
            title=f'Runoff Buckets: Notional Decomposition ({basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Signed Notional',
            cumulative_label='Cumulative Signed Notional (Running)',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_notional')
    elif selected_view == 'Effective Interest Decomposition':
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=nc_existing,
            y_added=nc_added,
            y_matured=nc_matured,
            y_total=nc_total,
            y_cumulative=nc_total.cumsum(),
            title=f'Runoff Buckets: Effective Interest Decomposition ({basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Effective Interest',
            cumulative_label='Cumulative Effective Interest',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_nc')
    elif selected_view == 'Effective Interest Contribution':
        if deals_df is not None and basis_t1 is not None and basis_t2 is not None:
            basis_date = pd.Timestamp(basis_t1) if basis == 'T1' else pd.Timestamp(basis_t2)
            target_month_end = basis_date + pd.offsets.MonthEnd(1)
            bdf = _bucketed_month_effective_contribution(
                deals_df=deals_df,
                basis_date=basis_date,
                target_month_end=target_month_end,
            )
            fig = _effective_contribution_chart(
                x=bdf['remaining_maturity_months'],
                y_existing=bdf['existing'],
                y_added=bdf['added'],
                y_matured=bdf['matured'],
                y_total=bdf['total'],
                y_cumulative=bdf['total'].cumsum(),
                title=(
                    f'Effective Interest Contribution by Remaining Maturity Bucket '
                    f'for {pd.Timestamp(target_month_end).date().isoformat()} ({basis})'
                ),
                x_label='Remaining Maturity (Months)',
                cumulative_label='Cumulative Effective Interest (Running)',
                flip_y_axis=flip_y_axis,
            )
        else:
            fig = _effective_contribution_chart(
                x=active_df['remaining_maturity_months'],
                y_existing=nc_existing,
                y_added=nc_added,
                y_matured=nc_matured,
                y_total=nc_total,
                y_cumulative=nc_total.cumsum(),
                title=f'Runoff Buckets: Effective Interest Contribution ({basis})',
                x_label='Remaining Maturity (Months)',
                cumulative_label='Cumulative Effective Interest (Running)',
                flip_y_axis=flip_y_axis,
            )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_effective_contribution')
    elif selected_view == 'Deal Count Decomposition':
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=deals_existing,
            y_added=deals_added,
            y_matured=deals_matured,
            y_total=deals_total,
            y_cumulative=deals_total.cumsum(),
            title=f'Runoff Buckets: Deal Count Decomposition ({basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Deal Count',
            cumulative_label='Cumulative Deal Count',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_deals')
    elif selected_view == 'Effective Interest Decomposition (Refill/Growth)' and include_refill_views:
        refill_effective_total = nc_total + refill_effective + growth_effective
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=nc_existing,
            y_added=nc_added,
            y_refilled=refill_effective,
            y_growth=growth_effective,
            y_matured=nc_matured,
            y_total=refill_effective_total,
            y_cumulative=refill_effective_total.cumsum(),
            title=f'Runoff Buckets: Effective Interest Decomposition ({refill_title_suffix}, {basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Effective Interest',
            cumulative_label='Cumulative Effective Interest',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_effective')
    elif selected_view == 'Cumulative Notional (Refill/Growth)' and include_refill_views:
        refill_notional_total = cum_notional_total_chart + refill_required + growth_outstanding
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=cum_notional_existing_chart,
            y_added=cum_notional_added_chart,
            y_refilled=refill_required,
            y_growth=growth_outstanding,
            y_matured=cum_notional_matured_chart,
            y_total=refill_notional_total,
            y_cumulative=None,
            title=f'Runoff Buckets: Cumulative Notional Decomposition ({refill_title_suffix}, {basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Cumulative Notional',
            cumulative_label='Cumulative Notional',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_cumulative')
    elif selected_view == 'Refill Allocation Heatmap' and include_refill_views:
        fig = _refill_allocation_heatmap(
            month_ends=month_ends,
            refill_required=refill_required,
            growth_required=growth_required,
            growth_mode=growth_mode,
            basis=basis,
            runoff_compare_df=compare_df,
            title=f'Refill Allocation by Month and Tenor (Shifted Delta, {refill_title_suffix}, {basis})',
            x_label='Calendar Month End',
        )
        if fig is None:
            st.info('Refill allocation heatmap unavailable: could not derive shifted one-month delta weights.')
        else:
            st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_allocation_heatmap')
            refill_line = _refill_volume_interest_chart(
                month_ends=month_ends,
                refill_required=refill_required,
                growth_volume=growth_outstanding,
                refill_rate=refill_rate,
                title=f'Refill Volume and Annual Interest ({refill_title_suffix}, {basis})',
                x_label='Calendar Month End',
                flip_y_axis=flip_y_axis,
            )
            st.plotly_chart(refill_line, use_container_width=True, key=f'{key_prefix}_refill_volume_interest')
    else:
        fig = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=cum_notional_existing_chart,
            y_added=cum_notional_added_chart,
            y_matured=cum_notional_matured_chart,
            y_total=cum_notional_total_chart,
            y_cumulative=None,
            title=f'Runoff Buckets: Cumulative Notional Decomposition ({basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Cumulative Notional',
            cumulative_label='Cumulative Notional',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_cumulative')

    notional_total_for_table = cum_notional_total_chart + refill_required + growth_outstanding
    effective_total_for_table = nc_total + refill_effective + growth_effective
    with st.expander('Runoff 5Y Aggregation', expanded=False):
        _render_aggregation_table(
            month_ends=month_ends,
            notional_existing=cum_notional_existing_chart,
            notional_added=cum_notional_added_chart,
            notional_matured_effect=-cum_notional_matured_chart,
            notional_refilled=refill_required,
            notional_growth=growth_outstanding,
            notional_total=notional_total_for_table,
            effective_existing=nc_existing,
            effective_added=nc_added,
            effective_matured_effect=-nc_matured,
            effective_refilled=refill_effective,
            effective_growth=growth_effective,
            effective_total=effective_total_for_table,
            key_prefix=f'{key_prefix}_summary',
        )
    return selected_view


def render_calendar_runoff_charts(
    calendar_df: pd.DataFrame,
    label_t1: str,
    label_t2: str,
    key_prefix: str = 'runoff_calendar_mode',
    runoff_compare_df: pd.DataFrame | None = None,
    deals_df: pd.DataFrame | None = None,
    basis_t1: pd.Timestamp | None = None,
    basis_t2: pd.Timestamp | None = None,
    refill_logic_df: pd.DataFrame | None = None,
    curve_df: pd.DataFrame | None = None,
    ui_state: dict[str, Any] | None = None,
) -> str:
    """Render runoff charts using actual calendar month-end x-axis."""
    if calendar_df.empty:
        st.info('No calendar-month runoff data to display.')
        return 'Notional Decomposition'

    df = calendar_df.sort_values('calendar_month_end').copy()
    x_vals = df['calendar_month_end'].dt.strftime('%Y-%m')

    def _col_or_zero(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype(float)
        return pd.Series(0.0, index=df.index, dtype=float)

    def _first_available(cols: list[str]) -> pd.Series:
        for col in cols:
            if col in df.columns:
                return df[col].astype(float)
        return pd.Series(0.0, index=df.index, dtype=float)

    ui_state = ui_state or {}
    basis = str(ui_state.get('runoff_decomposition_basis', 'T2'))
    growth_mode = str(ui_state.get('growth_mode', 'constant'))
    monthly_growth_amount = float(ui_state.get('growth_monthly_value', 0.0))
    chart_view = str(ui_state.get('runoff_chart_view', 'Notional Decomposition'))
    flip_y_axis = bool(ui_state.get('flip_y_axis', False))
    is_t1_basis = basis == 'T1'

    if is_t1_basis:
        notional_existing = _first_available(['signed_notional_t1', 'abs_notional_t1'])
        notional_added = pd.Series(0.0, index=df.index, dtype=float)
        notional_matured = pd.Series(0.0, index=df.index, dtype=float)
        notional_total = _first_available(['signed_notional_t1', 'abs_notional_t1'])
        abs_notional_existing = _first_available(['abs_notional_t1'])
        abs_notional_added = pd.Series(0.0, index=df.index, dtype=float)
        abs_notional_matured = pd.Series(0.0, index=df.index, dtype=float)
        abs_notional_total = _first_available(['abs_notional_t1'])

        nc_existing = _first_available(['effective_interest_t1', 'notional_coupon_t1'])
        nc_added = pd.Series(0.0, index=df.index, dtype=float)
        nc_matured = pd.Series(0.0, index=df.index, dtype=float)
        nc_total = _first_available(['effective_interest_t1', 'notional_coupon_t1'])

        deals_existing = df['deal_count_t1']
        deals_added = pd.Series(0.0, index=df.index, dtype=float)
        deals_matured = pd.Series(0.0, index=df.index, dtype=float)
        deals_total = df['deal_count_t1']
    else:
        notional_existing = (
            _first_available(['signed_notional_t2', 'abs_notional_t2'])
            - _first_available(['added_notional', 'added_abs_notional'])
            + _first_available(['matured_notional', 'matured_abs_notional'])
        )
        notional_added = _first_available(['added_notional', 'added_abs_notional'])
        notional_matured = _first_available(['matured_notional', 'matured_abs_notional'])
        notional_total = _first_available(['signed_notional_t2', 'abs_notional_t2'])
        abs_notional_existing = (
            _first_available(['abs_notional_t2'])
            - _first_available(['added_abs_notional'])
            + _first_available(['matured_abs_notional'])
        )
        abs_notional_added = _first_available(['added_abs_notional'])
        abs_notional_matured = _first_available(['matured_abs_notional'])
        abs_notional_total = _first_available(['abs_notional_t2'])

        nc_existing = (
            _first_available(['effective_interest_t2', 'notional_coupon_t2'])
            - _first_available(['added_effective_interest', 'added_notional_coupon'])
        )
        nc_added = _first_available(['added_effective_interest', 'added_notional_coupon'])
        nc_matured = _first_available(['matured_effective_interest', 'matured_notional_coupon'])
        nc_total = _first_available(['effective_interest_t2', 'notional_coupon_t2'])

        deals_existing = (
            df['deal_count_t2']
            - _col_or_zero('added_deal_count')
            + _col_or_zero('matured_deal_count')
        )
        deals_added = _col_or_zero('added_deal_count')
        deals_matured = _col_or_zero('matured_deal_count')
        deals_total = df['deal_count_t2']

    cum_notional_existing = abs_notional_existing[::-1].cumsum()[::-1]
    cum_notional_added = abs_notional_added[::-1].cumsum()[::-1]
    cum_notional_matured = abs_notional_matured[::-1].cumsum()[::-1]
    cum_notional_total = abs_notional_total[::-1].cumsum()[::-1]
    cum_notional_existing_chart = notional_existing[::-1].cumsum()[::-1]
    cum_notional_added_chart = notional_added[::-1].cumsum()[::-1]
    cum_notional_matured_chart = notional_matured[::-1].cumsum()[::-1]
    if is_t1_basis:
        cum_notional_total_chart = _first_available(
            ['cumulative_signed_notional_t1', 'cumulative_abs_notional_t1']
        )
    else:
        cum_notional_total_chart = _first_available(
            ['cumulative_signed_notional_t2', 'cumulative_abs_notional_t2']
        )
    growth_components = _compute_refill_growth_components_anchor_safe(
        cumulative_notional=cum_notional_total_chart.astype(float),
        growth_mode=growth_mode,
        monthly_growth_amount=monthly_growth_amount,
    )
    refill_required = growth_components['refill_required']
    growth_required = growth_components['growth_required']
    growth_outstanding = _growth_outstanding_profile(
        growth_flow=growth_required,
        runoff_compare_df=runoff_compare_df,
        basis=basis,
    )
    refill_rate = _curve_rate_by_tenor(
        tenor_points=pd.Series(range(1, len(df) + 1), index=df.index, dtype=float),
        base_notional_total=abs_notional_total.astype(float),
        base_effective_total=nc_total.astype(float),
        curve_df=curve_df,
        basis_date=(pd.Timestamp(basis_t1) if is_t1_basis and basis_t1 is not None else pd.Timestamp(basis_t2) if basis_t2 is not None else None),
    )
    refill_effective = refill_required * refill_rate * (30.0 / 360.0)
    growth_effective = growth_outstanding * refill_rate * (30.0 / 360.0)
    refill_title_suffix = 'Refill'
    if str(growth_mode).strip().lower() == 'user_defined' and float(monthly_growth_amount) > 0.0:
        refill_title_suffix = f'Refill + Growth ({monthly_growth_amount:,.2f}/month)'
    include_refill_views = True
    selected_view = coerce_option(
        chart_view,
        _available_runoff_chart_options(include_refill_views),
        'Notional Decomposition',
    )

    if selected_view == 'Notional Decomposition':
        fig = _component_chart(
            x=x_vals,
            y_existing=notional_existing,
            y_added=notional_added,
            y_matured=notional_matured,
            y_total=notional_total,
            y_cumulative=notional_total.cumsum(),
            title=f'Runoff Buckets by Calendar Month: Notional Decomposition ({basis})',
            x_label='Calendar Month End',
            y_label='Signed Notional',
            cumulative_label='Cumulative Signed Notional (Running)',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_notional')
    elif selected_view == 'Effective Interest Decomposition':
        fig = _component_chart(
            x=x_vals,
            y_existing=nc_existing,
            y_added=nc_added,
            y_matured=nc_matured,
            y_total=nc_total,
            y_cumulative=nc_total.cumsum(),
            title=f'Runoff Buckets by Calendar Month: Effective Interest Decomposition ({basis})',
            x_label='Calendar Month End',
            y_label='Effective Interest',
            cumulative_label='Cumulative Effective Interest',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_nc')
    elif selected_view == 'Effective Interest Contribution':
        # In calendar display mode, show bucketed contribution for the first
        # calendar step (the selected basis month) to reconcile with daily totals.
        if deals_df is not None and basis_t1 is not None and basis_t2 is not None:
            basis_date = pd.Timestamp(basis_t1) if is_t1_basis else pd.Timestamp(basis_t2)
            target_month_end = basis_date + pd.offsets.MonthEnd(0)
            bdf = _bucketed_month_effective_contribution(
                deals_df=deals_df,
                basis_date=basis_date,
                target_month_end=target_month_end,
            )
            fig = _effective_contribution_chart(
                x=bdf['remaining_maturity_months'],
                y_existing=bdf['existing'],
                y_added=bdf['added'],
                y_matured=bdf['matured'],
                y_total=bdf['total'],
                y_cumulative=bdf['total'].cumsum(),
                title=(
                    f'Effective Interest Contribution by Remaining Maturity Bucket '
                    f'for {pd.Timestamp(target_month_end).date().isoformat()} ({basis})'
                ),
                x_label='Remaining Maturity (Months)',
                cumulative_label='Cumulative Effective Interest (Running)',
                flip_y_axis=flip_y_axis,
            )
        elif runoff_compare_df is not None and not runoff_compare_df.empty:
            bdf = runoff_compare_df.copy()
            bmask = (
                (bdf[['abs_notional_d1', 'abs_notional_d2', 'cumulative_abs_notional_d1', 'cumulative_abs_notional_d2']].sum(axis=1) > 0)
                | ((bdf['deal_count_d1'] + bdf['deal_count_d2']) > 0)
            )
            bdf = bdf.loc[bmask].copy()
            if bdf.empty:
                bdf = runoff_compare_df.head(24).copy()

            def _bfirst_available(cols: list[str]) -> pd.Series:
                for col in cols:
                    if col in bdf.columns:
                        return bdf[col].astype(float)
                return pd.Series(0.0, index=bdf.index, dtype=float)

            if is_t1_basis:
                bucket_nc_existing = _bfirst_available(['effective_interest_d1', 'notional_coupon_d1'])
                bucket_nc_added = pd.Series(0.0, index=bdf.index, dtype=float)
                bucket_nc_matured = pd.Series(0.0, index=bdf.index, dtype=float)
                bucket_nc_total = _bfirst_available(['effective_interest_d1', 'notional_coupon_d1'])
            else:
                bucket_nc_existing = (
                    _bfirst_available(['effective_interest_d2', 'notional_coupon_d2'])
                    - _bfirst_available(['added_effective_interest', 'added_notional_coupon'])
                    + _bfirst_available(['matured_effective_interest', 'matured_notional_coupon'])
                )
                bucket_nc_added = _bfirst_available(['added_effective_interest', 'added_notional_coupon'])
                bucket_nc_matured = _bfirst_available(['matured_effective_interest', 'matured_notional_coupon'])
                bucket_nc_total = _bfirst_available(['effective_interest_d2', 'notional_coupon_d2'])

            fig = _effective_contribution_chart(
                x=bdf['remaining_maturity_months'],
                y_existing=bucket_nc_existing,
                y_added=bucket_nc_added,
                y_matured=bucket_nc_matured,
                y_total=bucket_nc_total,
                y_cumulative=bucket_nc_total.cumsum(),
                title=f'Runoff Buckets: Effective Interest Contribution ({basis})',
                x_label='Remaining Maturity (Months)',
                cumulative_label='Cumulative Effective Interest (Running)',
                flip_y_axis=flip_y_axis,
            )
        else:
            fig = _effective_contribution_chart(
                x=x_vals,
                y_existing=nc_existing,
                y_added=nc_added,
                y_matured=nc_matured,
                y_total=nc_total,
                y_cumulative=nc_total.cumsum(),
                title=f'Runoff Buckets by Calendar Month: Effective Interest Contribution ({basis})',
                x_label='Calendar Month End',
                cumulative_label='Cumulative Effective Interest (Running)',
                flip_y_axis=flip_y_axis,
            )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_effective_contribution')
    elif selected_view == 'Deal Count Decomposition':
        fig = _component_chart(
            x=x_vals,
            y_existing=deals_existing,
            y_added=deals_added,
            y_matured=deals_matured,
            y_total=deals_total,
            y_cumulative=deals_total.cumsum(),
            title=f'Runoff Buckets by Calendar Month: Deal Count Decomposition ({basis})',
            x_label='Calendar Month End',
            y_label='Deal Count',
            cumulative_label='Cumulative Deal Count',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_deals')
    elif selected_view == 'Effective Interest Decomposition (Refill/Growth)' and include_refill_views:
        refill_effective_total = nc_total + refill_effective + growth_effective
        fig = _component_chart(
            x=x_vals,
            y_existing=nc_existing,
            y_added=nc_added,
            y_refilled=refill_effective,
            y_growth=growth_effective,
            y_matured=nc_matured,
            y_total=refill_effective_total,
            y_cumulative=refill_effective_total.cumsum(),
            title=f'Runoff by Calendar Month: Effective Interest Decomposition ({refill_title_suffix}, {basis})',
            x_label='Calendar Month End',
            y_label='Effective Interest',
            cumulative_label='Cumulative Effective Interest',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_effective')
    elif selected_view == 'Cumulative Notional (Refill/Growth)' and include_refill_views:
        refill_notional_total = cum_notional_total_chart + refill_required + growth_outstanding
        fig = _component_chart(
            x=x_vals,
            y_existing=cum_notional_existing_chart,
            y_added=cum_notional_added_chart,
            y_refilled=refill_required,
            y_growth=growth_outstanding,
            y_matured=cum_notional_matured_chart,
            y_total=refill_notional_total,
            y_cumulative=None,
            title=f'Runoff by Calendar Month: Cumulative Notional Decomposition ({refill_title_suffix}, {basis})',
            x_label='Calendar Month End',
            y_label='Cumulative Notional',
            cumulative_label='Cumulative Notional',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_cumulative')
    elif selected_view == 'Refill Allocation Heatmap' and include_refill_views:
        fig = _refill_allocation_heatmap(
            month_ends=df['calendar_month_end'],
            refill_required=refill_required,
            growth_required=growth_required,
            growth_mode=growth_mode,
            basis=basis,
            runoff_compare_df=runoff_compare_df,
            title=f'Refill Allocation by Calendar Month and Tenor (Shifted Delta, {refill_title_suffix}, {basis})',
            x_label='Calendar Month End',
        )
        if fig is None:
            st.info('Refill allocation heatmap unavailable: could not derive shifted one-month delta weights.')
        else:
            st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_refill_allocation_heatmap')
            refill_line = _refill_volume_interest_chart(
                month_ends=df['calendar_month_end'],
                refill_required=refill_required,
                growth_volume=growth_outstanding,
                refill_rate=refill_rate,
                title=f'Refill Volume and Annual Interest ({refill_title_suffix}, {basis})',
                x_label='Calendar Month End',
                flip_y_axis=flip_y_axis,
            )
            st.plotly_chart(refill_line, use_container_width=True, key=f'{key_prefix}_refill_volume_interest')
    else:
        fig = _component_chart(
            x=x_vals,
            y_existing=cum_notional_existing_chart,
            y_added=cum_notional_added_chart,
            y_matured=cum_notional_matured_chart,
            y_total=cum_notional_total_chart,
            y_cumulative=None,
            title=f'Runoff Buckets by Calendar Month: Cumulative Notional Decomposition ({basis})',
            x_label='Calendar Month End',
            y_label='Cumulative Notional',
            cumulative_label='Cumulative Notional',
            flip_y_axis=flip_y_axis,
        )
        st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_cumulative')

    notional_total_for_table = cum_notional_total_chart + refill_required + growth_outstanding
    effective_total_for_table = nc_total + refill_effective + growth_effective
    with st.expander('Runoff 5Y Aggregation', expanded=False):
        _render_aggregation_table(
            month_ends=df['calendar_month_end'],
            notional_existing=cum_notional_existing_chart,
            notional_added=cum_notional_added_chart,
            notional_matured_effect=-cum_notional_matured_chart,
            notional_refilled=refill_required,
            notional_growth=growth_outstanding,
            notional_total=notional_total_for_table,
            effective_existing=nc_existing,
            effective_added=nc_added,
            effective_matured_effect=-nc_matured,
            effective_refilled=refill_effective,
            effective_growth=growth_effective,
            effective_total=effective_total_for_table,
            key_prefix=f'{key_prefix}_summary',
        )
    return selected_view
