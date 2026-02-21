"""Plotly chart builders for monthly bucket metrics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_monthly_plots(monthly_df: pd.DataFrame, title_prefix: str = 'Monthly') -> None:
    """Render volume, coupon, and interest charts from monthly buckets."""
    if monthly_df.empty:
        st.info('No monthly data available for plotting.')
        return
    if len(monthly_df) < 2:
        st.info(f'{title_prefix} charts hidden because only one month is available.')
        return

    volume_fig = px.bar(
        monthly_df,
        x='month_end',
        y='total_active_notional',
        title=f'{title_prefix} Total Active Notional',
    )
    coupon_fig = px.line(
        monthly_df,
        x='month_end',
        y='weighted_avg_coupon',
        title=f'{title_prefix} Weighted Average Coupon',
        markers=True,
    )
    interest_fig = px.bar(
        monthly_df,
        x='month_end',
        y='interest_paid_eur',
        title=f'{title_prefix} Interest Paid (EUR)',
    )
    combined_fig = go.Figure()
    combined_fig.add_bar(
        x=monthly_df['month_end'],
        y=monthly_df['total_active_notional'],
        name='Total Active Notional',
        yaxis='y1',
    )
    combined_fig.add_scatter(
        x=monthly_df['month_end'],
        y=monthly_df['weighted_avg_coupon'],
        name='Weighted Avg Coupon',
        mode='lines+markers',
        yaxis='y2',
    )
    combined_fig.update_layout(
        title=f'{title_prefix} Combined Notional and Coupon',
        yaxis=dict(title='Total Active Notional'),
        yaxis2=dict(title='Weighted Avg Coupon', overlaying='y', side='right'),
        legend=dict(orientation='h'),
    )

    st.plotly_chart(volume_fig, use_container_width=True)
    st.plotly_chart(coupon_fig, use_container_width=True)
    st.plotly_chart(interest_fig, use_container_width=True)
    st.plotly_chart(combined_fig, use_container_width=True)


def render_monthly_comparison_overview(
    monthly_d1: pd.DataFrame,
    monthly_d2: pd.DataFrame,
    label_d1: str,
    label_d2: str,
) -> None:
    """Render a single multi-panel comparison chart for monthly bucket series."""
    if monthly_d1.empty and monthly_d2.empty:
        st.info('No monthly comparison data available.')
        return
    if max(len(monthly_d1), len(monthly_d2)) < 2:
        st.info('Comparison chart hidden because only one month is available.')
        return

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Total Active Notional',
            'Weighted Average Coupon',
            'Interest Paid (EUR)',
        ),
    )

    def _add_traces(df: pd.DataFrame, label: str, color: str) -> None:
        if df.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=df['month_end'],
                y=df['total_active_notional'],
                mode='lines+markers',
                name=f'{label} Notional',
                legendgroup=label,
                marker=dict(color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df['month_end'],
                y=df['weighted_avg_coupon'],
                mode='lines+markers',
                name=f'{label} Coupon',
                legendgroup=label,
                marker=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df['month_end'],
                y=df['interest_paid_eur'],
                name=f'{label} Interest',
                legendgroup=label,
                marker=dict(color=color),
                showlegend=False,
                opacity=0.7,
            ),
            row=3,
            col=1,
        )

    _add_traces(monthly_d1, label_d1, '#1f77b4')
    _add_traces(monthly_d2, label_d2, '#ff7f0e')
    fig.update_layout(
        title='Monthly Bucket Comparison Overview',
        barmode='group',
        legend=dict(orientation='h'),
        height=900,
    )
    fig.update_yaxes(title_text='Notional', row=1, col=1)
    fig.update_yaxes(title_text='Coupon', tickformat='.2%', row=2, col=1)
    fig.update_yaxes(title_text='EUR', row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
