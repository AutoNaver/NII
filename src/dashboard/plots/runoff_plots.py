"""Runoff charts for aligned buckets and calendar-month views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.calculations.accrual import accrued_interest_eur, accrued_interest_for_overlap


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


def _styled_numeric_table(df: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if df.empty:
        return df
    fmt: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            name = str(col).lower()
            if 'count' in name:
                fmt[col] = '{:,.0f}'
            else:
                fmt[col] = '{:,.2f}'
    if not fmt:
        return df
    return df.style.format(fmt, na_rep='-')


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


def _stable_radio(
    *,
    label: str,
    options: list[str],
    key: str,
    default: str,
    horizontal: bool = True,
) -> str:
    current = st.session_state.get(key, default)
    if current not in options:
        current = default if default in options else options[0]
        st.session_state[key] = current
    idx = options.index(current)
    return st.radio(label, options=options, index=idx, horizontal=horizontal, key=key)


def _render_with_toggle(
    *,
    fig_notional: go.Figure,
    fig_cumulative: go.Figure,
    fig_notional_coupon: go.Figure,
    fig_effective_contribution: go.Figure,
    fig_deal_count: go.Figure,
    fig_refill_effective: go.Figure | None,
    fig_refill_cumulative: go.Figure | None,
    key_prefix: str,
) -> None:
    options = [
            'Notional Decomposition',
            'Effective Interest Decomposition',
            'Effective Interest Contribution',
            'Deal Count Decomposition',
            'Cumulative Notional',
    ]
    if fig_refill_effective is not None:
        options.append('Effective Interest Decomposition (Refill/Growth)')
    if fig_refill_cumulative is not None:
        options.append('Cumulative Notional (Refill/Growth)')
    option = _stable_radio(
        label='Runoff chart view',
        options=options,
        key='runoff_chart_view',
        default=options[0],
        horizontal=True,
    )
    if option == 'Notional Decomposition':
        st.plotly_chart(fig_notional, use_container_width=True, key=f'{key_prefix}_notional')
    elif option == 'Effective Interest Decomposition':
        st.plotly_chart(fig_notional_coupon, use_container_width=True, key=f'{key_prefix}_nc')
    elif option == 'Effective Interest Contribution':
        st.plotly_chart(fig_effective_contribution, use_container_width=True, key=f'{key_prefix}_effective_contribution')
    elif option == 'Deal Count Decomposition':
        st.plotly_chart(fig_deal_count, use_container_width=True, key=f'{key_prefix}_deals')
    elif option == 'Effective Interest Decomposition (Refill/Growth)' and fig_refill_effective is not None:
        st.plotly_chart(fig_refill_effective, use_container_width=True, key=f'{key_prefix}_refill_effective')
    elif option == 'Cumulative Notional (Refill/Growth)' and fig_refill_cumulative is not None:
        st.plotly_chart(fig_refill_cumulative, use_container_width=True, key=f'{key_prefix}_refill_cumulative')
    else:
        st.plotly_chart(fig_cumulative, use_container_width=True, key=f'{key_prefix}_cumulative')


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
            range=[primary_min, primary_max],
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
            range=[sec_lo, sec_hi],
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title=x_label)
    return fig


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
            range=[primary_min, primary_max],
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
                range=[sec_lo, sec_hi],
                showgrid=False,
                zeroline=True,
                zerolinewidth=1,
                separatethousands=True,
            )
        )
    fig.update_xaxes(title=x_label)
    return fig


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
        eff_rate.loc[non_zero] = (base_effective_total.loc[non_zero] / base_notional_total.loc[non_zero]).astype(float)
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
    if cumulative_notional.empty:
        empty = pd.Series(dtype=float, index=cumulative_notional.index)
        return {
            'target_total': empty,
            'total_required': empty,
            'refill_required': empty,
            'growth_required': empty,
        }

    base_level = float(cumulative_notional.iloc[0])
    idx = cumulative_notional.index
    steps = pd.Series(np.arange(len(cumulative_notional), dtype=float), index=idx)

    mode = str(growth_mode or 'constant').strip().lower()
    growth_per_step = float(monthly_growth_amount) if mode == 'user_defined' else 0.0
    growth_per_step = max(growth_per_step, 0.0)

    target_total = pd.Series(base_level, index=idx, dtype=float) + steps * growth_per_step
    total_required = (target_total - cumulative_notional).clip(lower=0.0)
    refill_required = (pd.Series(base_level, index=idx, dtype=float) - cumulative_notional).clip(lower=0.0)
    growth_required = (total_required - refill_required).clip(lower=0.0)

    first = idx[0]
    total_required.loc[first] = 0.0
    refill_required.loc[first] = 0.0
    growth_required.loc[first] = 0.0

    return {
        'target_total': target_total,
        'total_required': total_required,
        'refill_required': refill_required,
        'growth_required': growth_required,
    }


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
    st.markdown('**Runoff 5Y Aggregation**')
    window_mode = _stable_radio(
        label='Aggregation horizon',
        options=['Next 5 Years', '5 Calendar Years'],
        key=f'{key_prefix}_aggregation_horizon',
        default='Next 5 Years',
        horizontal=True,
    )

    windows = _build_aggregation_windows(month_ends, window_mode)
    if not windows:
        st.info('No runoff points available for the selected aggregation horizon.')
        return
    split_options = [label for label, _ in windows]
    split = _stable_radio(
        label='Aggregation split',
        options=split_options,
        key=f'{key_prefix}_aggregation_split',
        default=split_options[0],
        horizontal=True,
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
    st.dataframe(_styled_numeric_table(table), use_container_width=True)


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
    growth_mode: str = 'constant',
    monthly_growth_amount: float = 0.0,
) -> None:
    if compare_df.empty:
        st.info('No runoff data to display.')
        return

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

    basis = _stable_radio(
        label='Runoff decomposition basis',
        options=['T1', 'T2'],
        key='runoff_decomposition_basis',
        default='T2',
        horizontal=True,
    )

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

    # Bucket notional decomposition (daily-interest style)
    fig_buckets = _component_chart(
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
    )

    # Cumulative notional as runoff profile (decreasing over time as maturities roll off)
    cum_notional_existing = abs_notional_existing[::-1].cumsum()[::-1]
    cum_notional_added = abs_notional_added[::-1].cumsum()[::-1]
    cum_notional_matured = abs_notional_matured[::-1].cumsum()[::-1]
    cum_notional_total = abs_notional_total[::-1].cumsum()[::-1]
    fig_totals = _component_chart(
        x=active_df['remaining_maturity_months'],
        y_existing=cum_notional_existing,
        y_added=cum_notional_added,
        y_matured=cum_notional_matured,
        y_total=cum_notional_total,
        y_cumulative=None,
        title=f'Runoff Buckets: Cumulative Notional Decomposition ({basis})',
        x_label='Remaining Maturity (Months)',
        y_label='Cumulative Abs Notional',
        cumulative_label='Cumulative Abs Notional',
    )

    # Notional*coupon decomposition
    nc_cumulative = nc_total.cumsum()
    fig_nc = _component_chart(
        x=active_df['remaining_maturity_months'],
        y_existing=nc_existing,
        y_added=nc_added,
        y_matured=nc_matured,
        y_total=nc_total,
        y_cumulative=nc_cumulative,
        title=f'Runoff Buckets: Effective Interest Decomposition ({basis})',
        x_label='Remaining Maturity (Months)',
        y_label='Effective Interest',
        cumulative_label='Cumulative Effective Interest',
    )
    fig_effective_contribution = _effective_contribution_chart(
        x=active_df['remaining_maturity_months'],
        y_existing=nc_existing,
        y_added=nc_added,
        y_matured=nc_matured,
        y_total=nc_total,
        y_cumulative=nc_total.cumsum(),
        title=f'Runoff Buckets: Effective Interest Contribution ({basis})',
        x_label='Remaining Maturity (Months)',
        cumulative_label='Cumulative Effective Interest (Running)',
    )
    if deals_df is not None and basis_t1 is not None and basis_t2 is not None:
        basis_date = pd.Timestamp(basis_t1) if basis == 'T1' else pd.Timestamp(basis_t2)
        target_month_end = basis_date + pd.offsets.MonthEnd(1)
        bdf = _bucketed_month_effective_contribution(
            deals_df=deals_df,
            basis_date=basis_date,
            target_month_end=target_month_end,
        )
        fig_effective_contribution = _effective_contribution_chart(
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
        )

    # Deal count decomposition
    fig_deals = _component_chart(
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
    )

    fig_refill_effective = None
    fig_refill_cumulative = None
    refill_required = pd.Series(0.0, index=active_df.index, dtype=float)
    growth_required = pd.Series(0.0, index=active_df.index, dtype=float)
    refill_effective = pd.Series(0.0, index=active_df.index, dtype=float)
    growth_effective = pd.Series(0.0, index=active_df.index, dtype=float)
    refill_series = _build_refill_series(
        base_notional_total=abs_notional_total.astype(float),
        base_effective_total=nc_total.astype(float),
        tenor_points=active_df['remaining_maturity_months'].astype(float),
        refill_logic_df=refill_logic_df,
        curve_df=curve_df,
        basis_date=(pd.Timestamp(basis_t1) if basis == 'T1' and basis_t1 is not None else pd.Timestamp(basis_t2) if basis_t2 is not None else None),
    )
    if refill_series is not None:
        growth_components = _compute_refill_growth_components(
            cumulative_notional=cum_notional_total.astype(float),
            growth_mode=growth_mode,
            monthly_growth_amount=monthly_growth_amount,
        )
        refill_required = growth_components['refill_required']
        growth_required = growth_components['growth_required']
        refill_rate = refill_series['curve_rate'].astype(float)
        refill_effective = refill_required * refill_rate * (30.0 / 360.0)
        growth_effective = growth_required * refill_rate * (30.0 / 360.0)
        refill_effective_total = nc_total + refill_effective + growth_effective
        refill_title_suffix = 'Refill'
        if str(growth_mode).strip().lower() == 'user_defined' and float(monthly_growth_amount) > 0.0:
            refill_title_suffix = f'Refill + Growth ({monthly_growth_amount:,.2f}/month)'
        fig_refill_effective = _component_chart(
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
        )

        refill_notional_total = cum_notional_total + refill_required + growth_required
        fig_refill_cumulative = _component_chart(
            x=active_df['remaining_maturity_months'],
            y_existing=cum_notional_existing,
            y_added=cum_notional_added,
            y_refilled=refill_required,
            y_growth=growth_required,
            y_matured=cum_notional_matured,
            y_total=refill_notional_total,
            y_cumulative=None,
            title=f'Runoff Buckets: Cumulative Notional ({refill_title_suffix}, {basis})',
            x_label='Remaining Maturity (Months)',
            y_label='Cumulative Abs Notional',
            cumulative_label='Cumulative Abs Notional',
        )

    _render_with_toggle(
        fig_notional=fig_buckets,
        fig_cumulative=fig_totals,
        fig_notional_coupon=fig_nc,
        fig_effective_contribution=fig_effective_contribution,
        fig_deal_count=fig_deals,
        fig_refill_effective=fig_refill_effective,
        fig_refill_cumulative=fig_refill_cumulative,
        key_prefix=key_prefix,
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
    notional_total_for_table = cum_notional_total + refill_required + growth_required
    effective_total_for_table = nc_total + refill_effective + growth_effective
    _render_aggregation_table(
        month_ends=month_ends,
        notional_existing=cum_notional_existing,
        notional_added=cum_notional_added,
        notional_matured_effect=-cum_notional_matured,
        notional_refilled=refill_required,
        notional_growth=growth_required,
        notional_total=notional_total_for_table,
        effective_existing=nc_existing,
        effective_added=nc_added,
        effective_matured_effect=-nc_matured,
        effective_refilled=refill_effective,
        effective_growth=growth_effective,
        effective_total=effective_total_for_table,
        key_prefix=f'{key_prefix}_summary',
    )


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
    growth_mode: str = 'constant',
    monthly_growth_amount: float = 0.0,
) -> None:
    """Render runoff charts using actual calendar month-end x-axis."""
    if calendar_df.empty:
        st.info('No calendar-month runoff data to display.')
        return

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

    basis_options = ['T1', 'T2']
    basis = _stable_radio(
        label='Runoff decomposition basis',
        options=basis_options,
        key='runoff_decomposition_basis',
        default='T2',
        horizontal=True,
    )
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

    fig_buckets = _component_chart(
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
    )

    cum_notional_existing = abs_notional_existing[::-1].cumsum()[::-1]
    cum_notional_added = abs_notional_added[::-1].cumsum()[::-1]
    cum_notional_matured = abs_notional_matured[::-1].cumsum()[::-1]
    cum_notional_total = abs_notional_total[::-1].cumsum()[::-1]
    fig_totals = _component_chart(
        x=x_vals,
        y_existing=cum_notional_existing,
        y_added=cum_notional_added,
        y_matured=cum_notional_matured,
        y_total=cum_notional_total,
        y_cumulative=None,
        title=f'Runoff Buckets by Calendar Month: Cumulative Notional Decomposition ({basis})',
        x_label='Calendar Month End',
        y_label='Cumulative Abs Notional',
        cumulative_label='Cumulative Abs Notional',
    )

    fig_nc = _component_chart(
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
    )
    fig_effective_contribution = _effective_contribution_chart(
        x=x_vals,
        y_existing=nc_existing,
        y_added=nc_added,
        y_matured=nc_matured,
        y_total=nc_total,
        y_cumulative=nc_total.cumsum(),
        title=f'Runoff Buckets by Calendar Month: Effective Interest Contribution ({basis})',
        x_label='Calendar Month End',
        cumulative_label='Cumulative Effective Interest (Running)',
    )
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
        fig_effective_contribution = _effective_contribution_chart(
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

        fig_effective_contribution = _effective_contribution_chart(
            x=bdf['remaining_maturity_months'],
            y_existing=bucket_nc_existing,
            y_added=bucket_nc_added,
            y_matured=bucket_nc_matured,
            y_total=bucket_nc_total,
            y_cumulative=bucket_nc_total.cumsum(),
            title=f'Runoff Buckets: Effective Interest Contribution ({basis})',
            x_label='Remaining Maturity (Months)',
            cumulative_label='Cumulative Effective Interest (Running)',
        )

    fig_deals = _component_chart(
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
    )

    fig_refill_effective = None
    fig_refill_cumulative = None
    refill_required = pd.Series(0.0, index=df.index, dtype=float)
    growth_required = pd.Series(0.0, index=df.index, dtype=float)
    refill_effective = pd.Series(0.0, index=df.index, dtype=float)
    growth_effective = pd.Series(0.0, index=df.index, dtype=float)
    tenor_points = pd.Series(range(1, len(df) + 1), index=df.index, dtype=float)
    refill_series = _build_refill_series(
        base_notional_total=abs_notional_total.astype(float),
        base_effective_total=nc_total.astype(float),
        tenor_points=tenor_points,
        refill_logic_df=refill_logic_df,
        curve_df=curve_df,
        basis_date=(pd.Timestamp(basis_t1) if is_t1_basis and basis_t1 is not None else pd.Timestamp(basis_t2) if basis_t2 is not None else None),
    )
    if refill_series is not None:
        growth_components = _compute_refill_growth_components(
            cumulative_notional=cum_notional_total.astype(float),
            growth_mode=growth_mode,
            monthly_growth_amount=monthly_growth_amount,
        )
        refill_required = growth_components['refill_required']
        growth_required = growth_components['growth_required']
        refill_rate = refill_series['curve_rate'].astype(float)
        refill_effective = refill_required * refill_rate * (30.0 / 360.0)
        growth_effective = growth_required * refill_rate * (30.0 / 360.0)
        refill_effective_total = nc_total + refill_effective + growth_effective
        refill_title_suffix = 'Refill'
        if str(growth_mode).strip().lower() == 'user_defined' and float(monthly_growth_amount) > 0.0:
            refill_title_suffix = f'Refill + Growth ({monthly_growth_amount:,.2f}/month)'
        fig_refill_effective = _component_chart(
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
        )

        refill_notional_total = cum_notional_total + refill_required + growth_required
        fig_refill_cumulative = _component_chart(
            x=x_vals,
            y_existing=cum_notional_existing,
            y_added=cum_notional_added,
            y_refilled=refill_required,
            y_growth=growth_required,
            y_matured=cum_notional_matured,
            y_total=refill_notional_total,
            y_cumulative=None,
            title=f'Runoff by Calendar Month: Cumulative Notional ({refill_title_suffix}, {basis})',
            x_label='Calendar Month End',
            y_label='Cumulative Abs Notional',
            cumulative_label='Cumulative Abs Notional',
        )

    _render_with_toggle(
        fig_notional=fig_buckets,
        fig_cumulative=fig_totals,
        fig_notional_coupon=fig_nc,
        fig_effective_contribution=fig_effective_contribution,
        fig_deal_count=fig_deals,
        fig_refill_effective=fig_refill_effective,
        fig_refill_cumulative=fig_refill_cumulative,
        key_prefix=key_prefix,
    )

    notional_total_for_table = cum_notional_total + refill_required + growth_required
    effective_total_for_table = nc_total + refill_effective + growth_effective
    _render_aggregation_table(
        month_ends=df['calendar_month_end'],
        notional_existing=cum_notional_existing,
        notional_added=cum_notional_added,
        notional_matured_effect=-cum_notional_matured,
        notional_refilled=refill_required,
        notional_growth=growth_required,
        notional_total=notional_total_for_table,
        effective_existing=nc_existing,
        effective_added=nc_added,
        effective_matured_effect=-nc_matured,
        effective_refilled=refill_effective,
        effective_growth=growth_effective,
        effective_total=effective_total_for_table,
        key_prefix=f'{key_prefix}_summary',
    )
