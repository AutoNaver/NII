"""Daily interest time series chart with T1/T2 toggle."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.components.controls import coerce_option
from src.dashboard.components.formatting import plot_axis_number_format


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


def _plot_daily_metric(
    df: pd.DataFrame,
    title: str,
    metric_prefix: str,
    y_label: str,
    cumulative_label: str,
    include_cumulative: bool = True,
) -> None:
    df = df.sort_values('date').copy()
    total_col = f'{metric_prefix}_total'
    existing_col = f'{metric_prefix}_existing'
    added_col = f'{metric_prefix}_added'
    matured_col = f'{metric_prefix}_matured'
    df['cumulative_total'] = df[total_col].cumsum()
    df['cumulative_matured'] = df[matured_col].cumsum()

    primary_top_min, primary_top_max = _series_range([df[existing_col], df[added_col], df[total_col]])
    primary_top_min = min(primary_top_min, 0.0)
    primary_top_max = max(primary_top_max, 0.0)

    primary_bottom_min, primary_bottom_max = _series_range([df[matured_col]])
    primary_bottom_min = min(primary_bottom_min, 0.0)
    primary_bottom_max = max(primary_bottom_max, 0.0)

    sec_top_lo = sec_top_hi = None
    sec_bottom_lo = sec_bottom_hi = None
    if include_cumulative:
        sec_top_min, sec_top_max = _series_range([df['cumulative_total']])
        sec_top_lo, sec_top_hi = _aligned_secondary_range(
            primary_top_min, primary_top_max, sec_top_min, sec_top_max
        )
        sec_bottom_min, sec_bottom_max = _series_range([df['cumulative_matured']])
        sec_bottom_lo, sec_bottom_hi = _aligned_secondary_range(
            primary_bottom_min, primary_bottom_max, sec_bottom_min, sec_bottom_max
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": include_cumulative}], [{"secondary_y": include_cumulative}]],
    )
    fig.add_bar(
        x=df['date'],
        y=df[existing_col],
        name='Existing',
        marker=dict(color='#1f77b4'),
        row=1,
        col=1,
    )
    fig.add_bar(
        x=df['date'],
        y=df[added_col],
        name='Added (from deals starting this month)',
        marker=dict(color='#2ca02c'),
        row=1,
        col=1,
    )
    fig.add_bar(
        x=df['date'],
        y=df[matured_col],
        name='Matured',
        marker=dict(color='#ef4444'),
        row=2,
        col=1,
        secondary_y=False,
    )
    if include_cumulative:
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_matured'],
            mode='lines+markers',
            name='Cumulative Matured',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            row=2,
            col=1,
            secondary_y=True,
        )
    fig.add_scatter(
        x=df['date'],
        y=df[total_col],
        mode='lines+markers',
        name='Total',
        line=dict(color='#00e5ff', width=3),
        row=1,
        col=1,
        secondary_y=False,
    )
    if include_cumulative:
        fig.add_scatter(
            x=df['date'],
            y=df['cumulative_total'],
            mode='lines+markers',
            name='Cumulative Total',
            line=dict(color='#ffd166', width=2, dash='dot'),
            row=1,
            col=1,
            secondary_y=True,
        )
    fig.update_layout(
        title=title,
        barmode='relative',
        legend=dict(orientation='h'),
    )
    fig.update_yaxes(
        title_text=y_label,
        row=1,
        col=1,
        secondary_y=False,
        range=[primary_top_min, primary_top_max],
        zeroline=True,
        zerolinewidth=1,
        separatethousands=True,
    )
    if include_cumulative:
        fig.update_yaxes(
            title_text=cumulative_label,
            row=1,
            col=1,
            secondary_y=True,
            range=[sec_top_lo, sec_top_hi],
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        )
    fig.update_yaxes(
        title_text=f'Matured {y_label}',
        row=2,
        col=1,
        secondary_y=False,
        range=[primary_bottom_min, primary_bottom_max],
        zeroline=True,
        zerolinewidth=1,
        separatethousands=True,
    )
    if include_cumulative:
        fig.update_yaxes(
            title_text=f'Cumulative Matured {y_label}',
            row=2,
            col=1,
            secondary_y=True,
            range=[sec_bottom_lo, sec_bottom_hi],
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            separatethousands=True,
        )
    fig.update_xaxes(
        title='Date',
        range=[df['date'].min(), df['date'].max()],
        row=2,
        col=1,
    )
    y_axes = ['yaxis', 'yaxis3']
    if include_cumulative:
        y_axes.extend(['yaxis2', 'yaxis4'])
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
    selected_df = daily_t1 if option == label_t1 else daily_t2

    if chart_view == 'Daily Notional Decomposition':
        _plot_daily_metric(
            selected_df,
            title=f'Daily Notional ({option})',
            metric_prefix='notional',
            y_label='Daily Notional (EUR)',
            cumulative_label='Cumulative Notional (EUR)',
            include_cumulative=False,
        )
        return

    if option == label_t1:
        _plot_daily_metric(
            daily_t1,
            title=f'Daily Interest ({label_t1})',
            metric_prefix='interest',
            y_label='Daily Interest (EUR)',
            cumulative_label='Cumulative Interest (EUR)',
            include_cumulative=True,
        )
    else:
        _plot_daily_metric(
            daily_t2,
            title=f'Daily Interest ({label_t2})',
            metric_prefix='interest',
            y_label='Daily Interest (EUR)',
            cumulative_label='Cumulative Interest (EUR)',
            include_cumulative=True,
        )
