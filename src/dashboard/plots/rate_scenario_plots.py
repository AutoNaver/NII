"""Plot builders for Overview rate scenario analysis."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.components.formatting import plot_axis_number_format


def build_scenario_matrix_table(yearly_summary: pd.DataFrame) -> pd.DataFrame:
    """Return scenario matrix table ordered for quick comparison."""
    if yearly_summary.empty:
        return pd.DataFrame(
            columns=['Scenario', 'Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta', '5Y Cumulative Delta']
        )
    cols = ['scenario_label', 'Y1 Delta', 'Y2 Delta', 'Y3 Delta', 'Y4 Delta', 'Y5 Delta', '5Y Cumulative Delta']
    out = yearly_summary.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    out = out[cols].rename(columns={'scenario_label': 'Scenario'})
    return out.reset_index(drop=True)


def build_selected_scenario_impact_figure(
    *,
    monthly_base: pd.DataFrame,
    monthly_scenarios: pd.DataFrame,
    scenario_id: str,
    scenario_label: str,
    show_totals: bool = True,
) -> go.Figure:
    """Build monthly impact chart for one selected scenario."""
    base = monthly_base.sort_values('calendar_month_end').copy()
    scen = monthly_scenarios[monthly_scenarios['scenario_id'] == str(scenario_id)].copy()
    scen = scen.sort_values('calendar_month_end')
    if base.empty or scen.empty:
        fig = go.Figure()
        fig.update_layout(title='Selected Scenario Monthly Impact')
        return fig

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_bar(
        x=scen['calendar_month_end'],
        y=scen['delta_vs_base'],
        name='Delta vs Base',
        marker=dict(color='rgba(59, 130, 246, 0.72)'),
        hovertemplate='Month: %{x|%Y-%m-%d}<br>Delta: %{y:,.2f}<extra></extra>',
        secondary_y=False,
    )
    if show_totals:
        fig.add_scatter(
            x=base['calendar_month_end'],
            y=base['base_total_interest'],
            mode='lines+markers',
            name='Base Total',
            line=dict(color='#e2e8f0', width=2),
            secondary_y=False,
        )
        fig.add_scatter(
            x=scen['calendar_month_end'],
            y=scen['shocked_total_interest'],
            mode='lines+markers',
            name='Shocked Total',
            line=dict(color='#22c55e', width=2),
            secondary_y=False,
        )
    fig.add_scatter(
        x=scen['calendar_month_end'],
        y=scen['cumulative_delta'],
        mode='lines+markers',
        name='Cumulative Delta',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        secondary_y=True,
    )

    fig.update_layout(
        title=f'Selected Scenario Monthly Impact ({scenario_label})',
        barmode='relative',
        hovermode='x unified',
        legend=dict(orientation='h'),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title_text='Monthly Interest / Delta (EUR)', secondary_y=False, zeroline=True, zerolinewidth=1, separatethousands=True)
    fig.update_yaxes(title_text='Cumulative Delta (EUR)', secondary_y=True, showgrid=False, zeroline=True, zerolinewidth=1, separatethousands=True)
    fig.update_xaxes(title='Calendar Month End')
    return plot_axis_number_format(fig, y_axes=['yaxis', 'yaxis2'])


def build_curve_comparison_figure(
    *,
    curve_points: pd.DataFrame,
    tenor_paths: pd.DataFrame,
    scenario_id: str,
    scenario_label: str,
) -> go.Figure:
    """Build anchor curve comparison + first-year tenor movement chart."""
    cp = curve_points[curve_points['scenario_id'] == str(scenario_id)].copy()
    cp = cp.sort_values(['state', 'tenor_months'])
    tp = tenor_paths[tenor_paths['scenario_id'] == str(scenario_id)].copy()
    if 'tenor_months' not in tp.columns and 'tenor_label' in tp.columns:
        label_map = {'1M': 1, '1Y': 12, '5Y': 60, '10Y': 120}
        tp['tenor_months'] = tp['tenor_label'].map(label_map).astype('Int64')
    tp = tp.sort_values(['tenor_months', 'calendar_month_end'])
    if 'base_rate' not in tp.columns and 'shocked_rate' in tp.columns:
        if 'shock_bps' in tp.columns:
            tp['base_rate'] = tp['shocked_rate'] - (pd.to_numeric(tp['shock_bps'], errors='coerce') / 10000.0)
        else:
            tp['base_rate'] = tp['shocked_rate']
    if cp.empty and tp.empty:
        fig = go.Figure()
        fig.update_layout(title='Base vs Shocked Curve Shape')
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['Anchor Curve (T2)', 'Tenor Movement (0-24M, Plateau After 12M)'],
        shared_yaxes=False,
        horizontal_spacing=0.08,
    )

    # Left panel: anchor base vs shocked curve shape.
    state_frames = {state: cp[cp['state'] == state].copy() for state in ['anchor', 'month6', 'month12']}
    sdf_anchor = state_frames['anchor']
    sdf_m6 = state_frames['month6']
    sdf_m12 = state_frames['month12']
    is_ramp = str(scenario_id).startswith('ramp_')

    if not sdf_anchor.empty:
        fig.add_scatter(
            x=sdf_anchor['tenor_months'],
            y=sdf_anchor['base_rate'] * 100.0,
            mode='lines+markers',
            name='Base Curve',
            line=dict(color='#60a5fa', width=2),
            showlegend=True,
            row=1,
            col=1,
        )
    if is_ramp:
        if not sdf_m6.empty:
            fig.add_scatter(
                x=sdf_m6['tenor_months'],
                y=sdf_m6['shocked_rate'] * 100.0,
                mode='lines+markers',
                name='Shocked Curve (6M)',
                line=dict(color='#f59e0b', width=2, dash='dot'),
                showlegend=True,
                row=1,
                col=1,
            )
        if not sdf_m12.empty:
            fig.add_scatter(
                x=sdf_m12['tenor_months'],
                y=sdf_m12['shocked_rate'] * 100.0,
                mode='lines+markers',
                name='Shocked Curve (12M)',
                line=dict(color='#ef4444', width=2),
                showlegend=True,
                row=1,
                col=1,
            )
    elif not sdf_anchor.empty:
        fig.add_scatter(
            x=sdf_anchor['tenor_months'],
            y=sdf_anchor['shocked_rate'] * 100.0,
            mode='lines+markers',
            name='Shocked Curve',
            line=dict(color='#ef4444', width=2),
            showlegend=True,
            row=1,
            col=1,
        )
    fig.update_xaxes(title='Tenor (Months)', row=1, col=1)

    # Right panel: base vs shocked path for selected key tenors over first year.
    tenor_palette = {'1M': '#22c55e', '6M': '#eab308', '1Y': '#f59e0b', '5Y': '#a855f7', '10Y': '#38bdf8'}
    for tenor_label in ['1M', '6M', '1Y', '5Y', '10Y']:
        tdf = tp[tp['tenor_label'] == tenor_label].copy()
        if tdf.empty:
            continue
        fig.add_scatter(
            x=tdf['calendar_month_end'],
            y=tdf['base_rate'] * 100.0,
            mode='lines+markers',
            name=f'Base {tenor_label}',
            line=dict(color=tenor_palette.get(tenor_label, '#94a3b8'), width=1.6, dash='dot'),
            row=1,
            col=2,
        )
        fig.add_scatter(
            x=tdf['calendar_month_end'],
            y=tdf['shocked_rate'] * 100.0,
            mode='lines+markers',
            name=f'Shocked {tenor_label}',
            line=dict(color=tenor_palette.get(tenor_label, '#94a3b8'), width=2),
            hovertemplate=(
                'Month: %{x|%Y-%m-%d}<br>'
                'Shocked: %{y:.3f}%<br>'
                'Base: %{customdata[0]:.3f}%<br>'
                'Delta: %{customdata[1]:.1f} bps<extra></extra>'
            ),
            customdata=pd.DataFrame(
                {
                    'base_pct': tdf['base_rate'].astype(float) * 100.0,
                    'delta_bps': (tdf['shocked_rate'].astype(float) - tdf['base_rate'].astype(float)) * 10000.0,
                }
            ).to_numpy(),
            row=1,
            col=2,
        )
    fig.update_xaxes(title='Calendar Month End', row=1, col=2)

    fig.update_layout(
        title=f'Base vs Shocked Curve Shape ({scenario_label})',
        hovermode='x unified',
        legend=dict(orientation='h'),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title='Rate (%)', separatethousands=True, row=1, col=1)
    fig.update_yaxes(title='Rate (%)', separatethousands=True, row=1, col=2)
    return plot_axis_number_format(fig, y_axes=['yaxis', 'yaxis2'])
