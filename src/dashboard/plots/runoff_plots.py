"""Plot renderers for monthly runoff analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_runoff_comparison_bar(
    compare_df: pd.DataFrame,
    label_d1: str,
    label_d2: str,
    mode: str,
) -> None:
    """Render one bar chart for runoff comparison with selectable mode."""
    if compare_df.empty or len(compare_df) < 2:
        st.info('Runoff comparison graph requires at least 2 month offsets.')
        return

    if mode == 'Absolute Remaining':
        chart_df = compare_df[
            ['month_offset', 'remaining_abs_notional_d1', 'remaining_abs_notional_d2']
        ].melt(id_vars='month_offset', var_name='series', value_name='amount')
        chart_df['series'] = chart_df['series'].map(
            {
                'remaining_abs_notional_d1': label_d1,
                'remaining_abs_notional_d2': label_d2,
            }
        )
        fig = px.bar(
            chart_df,
            x='month_offset',
            y='amount',
            color='series',
            barmode='group',
            title='Runoff Comparison: Remaining Abs Notional',
        )
    else:
        fig = go.Figure()
        fig.add_bar(
            x=compare_df['month_offset'],
            y=compare_df['added_remaining_abs_notional'],
            name='Added Deals (Positive)',
        )
        fig.add_bar(
            x=compare_df['month_offset'],
            y=-compare_df['matured_remaining_abs_notional'],
            name='Maturities (Negative)',
        )
        fig.add_scatter(
            x=compare_df['month_offset'],
            y=compare_df['remaining_abs_notional_delta'],
            mode='lines+markers',
            name='Observed Delta',
        )
        fig.update_layout(
            title='Runoff Delta Attribution: Added Deals vs Maturities',
            barmode='relative',
        )

    fig.update_xaxes(title='Month Offset')
    fig.update_yaxes(title='Abs Notional')
    st.plotly_chart(fig, use_container_width=True)
