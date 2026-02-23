"""Plot builders for monthly bucket activity views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.formatting import apply_plot_layout_hygiene

def render_deal_count_activity(activity_df: pd.DataFrame, title: str) -> None:
    """Render active and added deal counts by month bucket."""
    if activity_df.empty or len(activity_df) < 2:
        st.info('Deal count graph hidden because fewer than 2 month buckets are available.')
        return

    fig = go.Figure()
    fig.add_bar(
        x=activity_df['month_end'],
        y=activity_df['added_deal_count'],
        name='Added Deals',
    )
    fig.add_scatter(
        x=activity_df['month_end'],
        y=activity_df['active_deal_count'],
        mode='lines+markers',
        name='Active Deals',
    )
    fig.update_layout(title=title, barmode='group')
    fig.update_yaxes(title='Deal Count')
    fig = apply_plot_layout_hygiene(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_notional_coupon_activity(activity_df: pd.DataFrame, title: str) -> None:
    """Render active and added notional*coupon by month bucket."""
    if activity_df.empty or len(activity_df) < 2:
        st.info('Notional*coupon graph hidden because fewer than 2 month buckets are available.')
        return

    fig = go.Figure()
    fig.add_bar(
        x=activity_df['month_end'],
        y=activity_df['added_notional_coupon'],
        name='Added Notional*Coupon',
    )
    fig.add_scatter(
        x=activity_df['month_end'],
        y=activity_df['active_notional_coupon'],
        mode='lines+markers',
        name='Active Notional*Coupon',
    )
    fig.update_layout(title=title, barmode='group')
    fig.update_yaxes(title='Notional * Coupon')
    fig = apply_plot_layout_hygiene(fig)
    st.plotly_chart(fig, use_container_width=True)
