"""Daily decomposition charts for interest and notional with T1/T2 toggle."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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


def _padded_range(
    series_list: list[pd.Series],
    *,
    include_zero: bool,
    pad_ratio: float = 0.08,
    min_span_ratio: float = 0.01,
) -> tuple[float, float]:
    lo, hi = _series_range(series_list)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)

    span = hi - lo
    if span <= 0.0:
        base = max(abs(lo), abs(hi), 1.0)
        pad = base * max(pad_ratio, 0.1)
        return (lo - pad, hi + pad)

    mid = (hi + lo) / 2.0
    min_span = max(abs(mid) * min_span_ratio, 1e-9)
    if span < min_span:
        widen = (min_span - span) / 2.0
        lo -= widen
        hi += widen
        span = hi - lo

    pad = span * pad_ratio
    return (lo - pad, hi + pad)


def _maybe_flip_range(lo: float, hi: float, flip_y_axis: bool) -> list[float]:
    if flip_y_axis:
        return [float(hi), float(lo)]
    return [float(lo), float(hi)]


def _delta_colors(values: pd.Series) -> list[str]:
    colors: list[str] = []
    for value in values.astype(float):
        if value > 0.0:
            colors.append('rgba(34, 197, 94, 0.72)')
        elif value < 0.0:
            colors.append('rgba(239, 68, 68, 0.72)')
        else:
            colors.append('rgba(148, 163, 184, 0.62)')
    return colors


def _delta_line_colors(values: pd.Series) -> list[str]:
    colors: list[str] = []
    for value in values.astype(float):
        if value > 0.0:
            colors.append('rgba(34, 197, 94, 1.0)')
        elif value < 0.0:
            colors.append('rgba(239, 68, 68, 1.0)')
        else:
            colors.append('rgba(148, 163, 184, 0.95)')
    return colors


def _build_interest_cumulative_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.sort_values('date').copy()
    if table.empty:
        return pd.DataFrame(
            columns=[
                'Month End',
                'Cumulative Existing Interest (EUR)',
                'Cumulative Added Interest (EUR)',
                'Cumulative Matured Interest (EUR)',
                'Cumulative Total Interest (EUR)',
            ]
        )
    last_date = pd.to_datetime(table['date'].iloc[-1]).date()
    return pd.DataFrame(
        [
            {
                'Month End': last_date,
                'Cumulative Existing Interest (EUR)': float(table['interest_existing'].astype(float).sum()),
                'Cumulative Added Interest (EUR)': float(table['interest_added'].astype(float).sum()),
                'Cumulative Matured Interest (EUR)': float(table['interest_matured'].astype(float).sum()),
                'Cumulative Total Interest (EUR)': float(table['interest_total'].astype(float).sum()),
            }
        ]
    )


def _plot_daily_metric(
    df: pd.DataFrame,
    title: str,
    metric_prefix: str,
    y_label: str,
    cumulative_label: str,
    include_bottom_cumulative: bool = False,
    include_top_cumulative_total: bool = False,
    flip_y_axis: bool = False,
) -> None:
    df = df.sort_values('date').copy()
    total_col = f'{metric_prefix}_total'
    added_col = f'{metric_prefix}_added'
    matured_col = f'{metric_prefix}_matured'
    month_start_total = float(df[total_col].iloc[0]) if not df.empty else 0.0
    baseline_series = pd.Series(month_start_total, index=df.index, dtype=float)
    df['total_delta_from_start'] = df[total_col] - month_start_total
    delta_colors = _delta_colors(df['total_delta_from_start'])
    delta_line_colors = _delta_line_colors(df['total_delta_from_start'])
    df['cumulative_total'] = df[total_col].cumsum()
    df['cumulative_added'] = df[added_col].cumsum()
    df['cumulative_matured'] = df[matured_col].cumsum()
    df['cumulative_added_matured_aggregate'] = (df[added_col] + df[matured_col]).cumsum()

    top_lo, top_hi = _padded_range(
        [df[total_col], baseline_series],
        include_zero=True,
        pad_ratio=0.12,
    )
    if top_hi <= 0.0:
        primary_top_min = min(top_lo, -1.0)
        primary_top_max = 0.0
    elif top_lo >= 0.0:
        primary_top_min = 0.0
        primary_top_max = max(top_hi, 1.0)
    else:
        primary_top_min = top_lo
        primary_top_max = top_hi
    primary_bottom_min, primary_bottom_max = _padded_range(
        [df[added_col], df[matured_col]],
        include_zero=True,
        pad_ratio=0.1,
    )

    sec_top_lo = sec_top_hi = None
    if include_top_cumulative_total:
        sec_top_min, sec_top_max = _padded_range(
            [df['cumulative_total']],
            include_zero=True,
            pad_ratio=0.08,
        )
        sec_top_lo, sec_top_hi = _aligned_secondary_range(
            primary_top_min, primary_top_max, sec_top_min, sec_top_max
        )

    sec_bottom_lo = sec_bottom_hi = None
    if include_bottom_cumulative:
        sec_bottom_min, sec_bottom_max = _padded_range(
            [df['cumulative_added'], df['cumulative_matured'], df['cumulative_added_matured_aggregate']],
            include_zero=True,
            pad_ratio=0.08,
        )
        sec_bottom_lo, sec_bottom_hi = _aligned_secondary_range(
            primary_bottom_min, primary_bottom_max, sec_bottom_min, sec_bottom_max
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.26,
        specs=[[{"secondary_y": include_top_cumulative_total}], [{"secondary_y": include_bottom_cumulative}]],
    )
    fig.add_bar(
        x=df['date'],
        y=baseline_series,
        name='Month Start Base',
        marker=dict(color='rgba(31, 119, 180, 0.86)'),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Month-start baseline: %{y:,.2f}<extra></extra>',
        row=1,
        col=1,
    )
    fig.add_bar(
        x=df['date'],
        y=df['total_delta_from_start'],
        base=baseline_series,
        name='Delta vs Month Start (Shaded)',
        marker=dict(
            color=delta_colors,
            line=dict(color=delta_line_colors, width=1.2),
            pattern=dict(shape='x', solidity=0.5, fgcolor='rgba(255, 255, 255, 0.78)'),
        ),
        customdata=df[total_col],
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Delta vs month start: %{y:,.2f}<br>Total: %{customdata:,.2f}<extra></extra>',
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=df['date'],
        y=df[total_col],
        mode='lines+markers',
        name='Total',
        line=dict(color='#f8fafc', width=3),
        marker=dict(size=7, color='#f8fafc', line=dict(color='#0b1220', width=1)),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Total: %{y:,.2f}<extra></extra>',
        row=1,
        col=1,
        secondary_y=False,
    )
    if include_top_cumulative_total:
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_total'],
            mode='lines+markers',
            name='Cumulative Total',
            line=dict(color='#a78bfa', width=3, dash='dot'),
            marker=dict(size=6, color='#a78bfa', line=dict(color='#1f2937', width=1)),
            row=1,
            col=1,
            secondary_y=True,
        )
    fig.add_bar(
        x=df['date'],
        y=df[added_col],
        name='Added',
        marker=dict(color='rgba(34, 197, 94, 0.86)'),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_bar(
        x=df['date'],
        y=df[matured_col],
        name='Matured',
        marker=dict(color='rgba(239, 68, 68, 0.86)'),
        row=2,
        col=1,
        secondary_y=False,
    )
    if include_bottom_cumulative:
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_added_matured_aggregate'],
            mode='lines+markers',
            name='Cumulative Added + Matured',
            line=dict(color='#38bdf8', width=3, dash='dot'),
            marker=dict(size=7, color='#38bdf8', line=dict(color='#0f172a', width=1)),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_added'],
            mode='lines+markers',
            name='Cumulative Added',
            line=dict(color='#facc15', width=3, dash='dot'),
            marker=dict(size=6, color='#facc15', line=dict(color='#1f2937', width=1)),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_matured'],
            mode='lines+markers',
            name='Cumulative Matured',
            line=dict(color='#fb923c', width=3, dash='dot'),
            marker=dict(size=6, color='#fb923c', line=dict(color='#1f2937', width=1)),
            row=2,
            col=1,
            secondary_y=True,
        )
    fig.update_layout(
        title=title,
        barmode='relative',
        legend=dict(orientation='h'),
        bargap=0.18,
    )
    fig.update_yaxes(
        title_text='Total (EUR)',
        row=1,
        col=1,
        secondary_y=False,
        range=_maybe_flip_range(primary_top_min, primary_top_max, flip_y_axis),
        separatethousands=True,
        title_standoff=24,
        title_font=dict(size=12),
        automargin=True,
    )
    if include_top_cumulative_total:
        fig.update_yaxes(
            title_text='Cum Total (EUR)',
            row=1,
            col=1,
            secondary_y=True,
            range=_maybe_flip_range(sec_top_lo, sec_top_hi, flip_y_axis),
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
            title_standoff=24,
            title_font=dict(size=12),
            automargin=True,
        )
    fig.update_yaxes(
        title_text='Added/Matured (EUR)',
        row=2,
        col=1,
        secondary_y=False,
        range=_maybe_flip_range(primary_bottom_min, primary_bottom_max, flip_y_axis),
        zeroline=True,
        zerolinewidth=1,
        separatethousands=True,
        title_standoff=24,
        title_font=dict(size=12),
        automargin=True,
    )
    if include_bottom_cumulative:
        fig.update_yaxes(
            title_text='Cum A+M (EUR)',
            row=2,
            col=1,
            secondary_y=True,
            range=_maybe_flip_range(sec_bottom_lo, sec_bottom_hi, flip_y_axis),
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
            title_standoff=24,
            title_font=dict(size=12),
            automargin=True,
        )
    fig.update_xaxes(
        title='Date',
        range=[df['date'].min(), df['date'].max()],
        row=2,
        col=1,
    )
    y_axes = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4']
    fig = plot_axis_number_format(fig, y_axes=y_axes)
    st.plotly_chart(fig, use_container_width=True)


def render_daily_interest_chart(daily_t1: pd.DataFrame, daily_t2: pd.DataFrame, label_t1: str, label_t2: str) -> None:
    options = []
    if not daily_t1.empty:
        options.append(label_t1)
    if not daily_t2.empty:
        options.append(label_t2)
    if not options:
        st.info('No daily interest data available.')
        return

    view_key = 'daily_view_date'
    option_current = coerce_option(st.session_state.get(view_key, options[0]), options, options[0])
    st.session_state[view_key] = option_current
    option = st.radio(
        label='Daily view',
        options=options,
        index=options.index(option_current),
        horizontal=True,
        key=view_key,
    )
    chart_options = ['Daily Interest Decomposition', 'Daily Notional Decomposition']
    chart_key = 'daily_chart_view'
    chart_current = coerce_option(st.session_state.get(chart_key, chart_options[0]), chart_options, chart_options[0])
    st.session_state[chart_key] = chart_current
    chart_view = st.radio(
        label='Daily chart view',
        options=chart_options,
        index=chart_options.index(chart_current),
        horizontal=True,
        key=chart_key,
    )
    flip_y_axis_key = 'daily_flip_y_axis'
    flip_y_axis = st.checkbox(
        'Flip y-axis orientation',
        value=bool(st.session_state.get(flip_y_axis_key, False)),
        key=flip_y_axis_key,
        help='Visual-only orientation flip. Signed values are unchanged.',
    )
    selected_df = daily_t1 if option == label_t1 else daily_t2

    if chart_view == 'Daily Notional Decomposition':
        _plot_daily_metric(
            selected_df,
            title=f'Daily Notional ({option})',
            metric_prefix='notional',
            y_label='Daily Notional (EUR)',
            cumulative_label='Cumulative Notional (EUR)',
            include_bottom_cumulative=False,
            include_top_cumulative_total=False,
            flip_y_axis=flip_y_axis,
        )
        return

    if option == label_t1:
        _plot_daily_metric(
            daily_t1,
            title=f'Daily Interest ({label_t1})',
            metric_prefix='interest',
            y_label='Daily Interest (EUR)',
            cumulative_label='Cumulative Interest (EUR)',
            include_bottom_cumulative=True,
            include_top_cumulative_total=True,
            flip_y_axis=flip_y_axis,
        )
        st.caption(f'Month-End Cumulative Daily Interest Decomposition ({label_t1})')
        st.dataframe(style_numeric_table(_build_interest_cumulative_table(daily_t1)), use_container_width=True)
    else:
        _plot_daily_metric(
            daily_t2,
            title=f'Daily Interest ({label_t2})',
            metric_prefix='interest',
            y_label='Daily Interest (EUR)',
            cumulative_label='Cumulative Interest (EUR)',
            include_bottom_cumulative=True,
            include_top_cumulative_total=True,
            flip_y_axis=flip_y_axis,
        )
        st.caption(f'Month-End Cumulative Daily Interest Decomposition ({label_t2})')
        st.dataframe(style_numeric_table(_build_interest_cumulative_table(daily_t2)), use_container_width=True)
