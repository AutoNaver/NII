"""UI controls for custom Overview rate scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.calculations.rate_scenarios import interpolate_curve_rate
from src.dashboard.scenario_store import make_custom_scenario_id

MANUAL_TENOR_GRID: list[tuple[str, int]] = [
    ('1M', 1),
    ('3M', 3),
    ('6M', 6),
    ('1Y', 12),
    ('2Y', 24),
    ('5Y', 60),
    ('10Y', 120),
    ('20Y', 240),
]


def _manual_editor_default() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'Tenor': [t for t, _ in MANUAL_TENOR_GRID],
            'Use Node': [True for _ in MANUAL_TENOR_GRID],
            'Shock (bps)': [0.0 for _ in MANUAL_TENOR_GRID],
        }
    )


def _manual_nodes_from_editor(df: pd.DataFrame) -> list[dict[str, float]]:
    tenor_map = {label: float(months) for label, months in MANUAL_TENOR_GRID}
    nodes: list[dict[str, float]] = []
    work = df.copy()
    if 'Tenor' not in work.columns or 'Shock (bps)' not in work.columns:
        return nodes
    for _, row in work.iterrows():
        tenor_label = str(row.get('Tenor', '')).strip()
        tenor_months = tenor_map.get(tenor_label)
        if tenor_months is None:
            continue
        if 'Use Node' in work.columns and not bool(row.get('Use Node', False)):
            continue
        shock = pd.to_numeric(row.get('Shock (bps)'), errors='coerce')
        nodes.append(
            {
                'tenor_months': float(tenor_months),
                'shock_bps': None if pd.isna(shock) else float(shock),
            }
        )
    return nodes


def _preview_tenor_grid(curve_df: pd.DataFrame | None) -> list[int]:
    curve_tenors: list[int] = []
    if curve_df is not None and not curve_df.empty and 'ir_tenor' in curve_df.columns:
        t = pd.to_numeric(curve_df['ir_tenor'], errors='coerce').dropna().astype(int)
        curve_tenors = [int(x) for x in t[t > 0].tolist()]
    max_t = max([240] + curve_tenors) if curve_tenors else 240
    max_t = int(min(max_t, 600))
    dense = list(range(1, max_t + 1))
    anchors = [months for _, months in MANUAL_TENOR_GRID]
    grid = sorted(set(dense + anchors + curve_tenors))
    return grid


def _build_manual_interpolation_preview_figure(
    manual_nodes: list[dict[str, float]],
    tenor_grid: list[int],
    *,
    mode: str,
    curve_df: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> go.Figure | None:
    nodes = []
    for n in manual_nodes:
        tenor = pd.to_numeric(n.get('tenor_months'), errors='coerce')
        shock = pd.to_numeric(n.get('shock_bps'), errors='coerce')
        if pd.isna(tenor) or pd.isna(shock):
            continue
        if float(tenor) <= 0:
            continue
        nodes.append((float(tenor), float(shock)))
    if not nodes:
        return None
    nodes = sorted(nodes, key=lambda x: x[0])
    x_nodes = np.array([p[0] for p in nodes], dtype=float)
    y_nodes = np.array([p[1] for p in nodes], dtype=float)
    x = np.array([float(t) for t in tenor_grid], dtype=float)
    y = np.interp(x, x_nodes, y_nodes)

    fig = go.Figure()
    mode_key = str(mode or 'Shock (bps)').strip()
    if mode_key == 'Base + Final Shocked Curve (%)':
        if curve_df is None or curve_df.empty:
            return None
        as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.today()
        base = np.array(
            [interpolate_curve_rate(curve_df, as_of_date=as_of, tenor_months=int(t)) for t in x],
            dtype=float,
        )
        shocked = base + (y / 10000.0)
        base_nodes = np.array(
            [interpolate_curve_rate(curve_df, as_of_date=as_of, tenor_months=int(t)) for t in x_nodes],
            dtype=float,
        )
        shocked_nodes = base_nodes + (y_nodes / 10000.0)

        fig.add_scatter(
            x=x,
            y=base * 100.0,
            mode='lines',
            name='Base Curve',
            line=dict(color='#60a5fa', width=2, dash='dot'),
        )
        fig.add_scatter(
            x=x,
            y=shocked * 100.0,
            mode='lines',
            name='Final Shocked Curve',
            line=dict(color='#22c55e', width=2),
        )
        fig.add_scatter(
            x=x_nodes,
            y=shocked_nodes * 100.0,
            mode='markers',
            name='Selected Nodes (Shocked)',
            marker=dict(color='#f59e0b', size=8, symbol='circle'),
        )
        title = 'Manual Tenor Interpolation Preview (Base + Final Shocked Curve)'
        y_title = 'Rate (%)'
    else:
        fig.add_scatter(
            x=x,
            y=y,
            mode='lines',
            name='Interpolated Shock',
            line=dict(color='#38bdf8', width=2),
        )
        fig.add_scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers',
            name='Selected Nodes',
            marker=dict(color='#f59e0b', size=8, symbol='circle'),
        )
        title = 'Manual Tenor Interpolation Preview (Shock Only)'
        y_title = 'Shock (bps)'
    fig.update_layout(
        title=title,
        height=320,
        hovermode='x unified',
        legend=dict(orientation='h'),
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(title='Tenor (Months)')
    fig.update_yaxes(title=y_title, zeroline=True, zerolinewidth=1)
    return fig


def render_rate_scenario_builder(
    *,
    scenario_universe_df: pd.DataFrame,
    custom_scenarios_df: pd.DataFrame,
    active_scenario_ids: list[str],
    curve_df: pd.DataFrame | None = None,
    preview_anchor_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Render builder and return actions for app-level processing."""
    actions: dict[str, Any] = {
        'add_scenario': None,
        'delete_scenario_id': None,
        'set_active_ids': None,
        'error': '',
    }

    universe = scenario_universe_df.copy()
    if universe.empty:
        return actions

    option_ids = universe['scenario_id'].astype(str).tolist()
    label_map = {
        str(r['scenario_id']): str(r['scenario_label'])
        for r in universe[['scenario_id', 'scenario_label']].to_dict(orient='records')
    }
    default_active_ids = [sid for sid in active_scenario_ids if sid in option_ids]
    if not default_active_ids:
        default_active_ids = option_ids

    with st.expander('Scenario Builder', expanded=False):
        st.caption('Create custom scenarios and control the active scenario set used by calculations.')

        active_key = 'overview_rate_active_scenarios'
        cur_state = st.session_state.get(active_key)
        if not isinstance(cur_state, list) or any(str(x) not in option_ids for x in cur_state):
            st.session_state[active_key] = default_active_ids
        selected_active_ids = st.multiselect(
            'Active Scenarios',
            options=option_ids,
            default=st.session_state.get(active_key, default_active_ids),
            key=active_key,
            format_func=lambda sid: label_map.get(str(sid), str(sid)),
        )
        selected_active_ids = [sid for sid in option_ids if sid in set(selected_active_ids)]
        if set(selected_active_ids) != set(default_active_ids):
            actions['set_active_ids'] = selected_active_ids

        st.markdown('---')
        name = st.text_input('Scenario Name', value='', key='overview_rate_builder_name')
        scenario_type = st.selectbox(
            'Scenario Type',
            options=['Parallel', 'Twist', 'Manual Tenors'],
            index=0,
            key='overview_rate_builder_type',
        )
        materialization = st.radio(
            'Materialization',
            options=['Instant', 'Linear 12M'],
            horizontal=True,
            key='overview_rate_builder_materialization',
        )

        shock_direction = 'Up'
        shock_magnitude = 0.0
        twist_pivot_tenor_months = 6.0
        manual_nodes: list[dict[str, float]] | None = None
        if scenario_type in {'Parallel', 'Twist'}:
            c1, c2 = st.columns([1, 2])
            with c1:
                shock_direction = st.radio(
                    'Direction',
                    options=['Up', 'Down'],
                    horizontal=False,
                    key='overview_rate_builder_direction',
                )
            with c2:
                shock_magnitude = float(
                    st.number_input(
                        'Shock (bps)',
                        min_value=0.0,
                        value=10.0 if scenario_type == 'Twist' else 50.0,
                        step=1.0,
                        key='overview_rate_builder_shock_bps',
                    )
                )
            if scenario_type == 'Twist':
                twist_pivot_tenor_months = float(
                    st.number_input(
                        'Twist Pivot Tenor (Months)',
                        min_value=1.0,
                        value=6.0,
                        step=1.0,
                        format='%.2f',
                        key='overview_rate_builder_twist_pivot',
                    )
                )
        else:
            st.caption('Select the tenor nodes to use for interpolation. Unselected rows are ignored.')
            editor_df = st.data_editor(
                _manual_editor_default(),
                num_rows='fixed',
                hide_index=True,
                column_config={
                    'Tenor': st.column_config.TextColumn(disabled=True),
                    'Use Node': st.column_config.CheckboxColumn(),
                    'Shock (bps)': st.column_config.NumberColumn(format='%.4f'),
                },
                key='overview_rate_builder_manual_editor',
            )
            manual_nodes = _manual_nodes_from_editor(editor_df)
            preview_mode = st.radio(
                'Interpolation Preview Mode',
                options=['Shock (bps)', 'Base + Final Shocked Curve (%)'],
                horizontal=True,
                key='overview_rate_builder_preview_mode',
            )
            preview_fig = _build_manual_interpolation_preview_figure(
                manual_nodes=manual_nodes or [],
                tenor_grid=_preview_tenor_grid(curve_df),
                mode=preview_mode,
                curve_df=curve_df,
                as_of_date=preview_anchor_date,
            )
            if preview_fig is None:
                st.caption('Interpolation preview unavailable: select nodes with numeric shocks and ensure base curve data exists for curve mode.')
            else:
                st.plotly_chart(
                    preview_fig,
                    width='stretch',
                    key='overview_rate_builder_manual_preview',
                )

        submitted = st.button('Add Scenario', key='overview_rate_add_scenario_button')

        if submitted:
            name_clean = str(name).strip()
            if not name_clean:
                actions['error'] = 'Scenario name is required.'
                return actions
            scenario_id = make_custom_scenario_id(name_clean)
            materialization_code = 'inst' if materialization == 'Instant' else 'ramp'
            if scenario_type == 'Parallel':
                sign = 1.0 if shock_direction == 'Up' else -1.0
                actions['add_scenario'] = {
                    'scenario_id': scenario_id,
                    'scenario_label': name_clean,
                    'shock_type': 'parallel',
                    'materialization': materialization_code,
                    'shock_bps': sign * float(shock_magnitude),
                    'pivot_tenor_months': None,
                    'manual_nodes': None,
                }
            elif scenario_type == 'Twist':
                sign = 1.0 if shock_direction == 'Up' else -1.0
                actions['add_scenario'] = {
                    'scenario_id': scenario_id,
                    'scenario_label': name_clean,
                    'shock_type': 'twist',
                    'materialization': materialization_code,
                    'shock_bps': sign * float(shock_magnitude),
                    'pivot_tenor_months': float(twist_pivot_tenor_months),
                    'manual_nodes': None,
                }
            else:
                actions['add_scenario'] = {
                    'scenario_id': scenario_id,
                    'scenario_label': name_clean,
                    'shock_type': 'manual',
                    'materialization': materialization_code,
                    'shock_bps': None,
                    'pivot_tenor_months': None,
                    'manual_nodes': manual_nodes or [],
                }

        st.markdown('---')
        custom = custom_scenarios_df.copy()
        if custom.empty:
            st.caption('No custom scenarios saved yet.')
        else:
            custom_options = custom['scenario_id'].astype(str).tolist()
            delete_id = st.selectbox(
                'Delete Custom Scenario',
                options=custom_options,
                format_func=lambda sid: label_map.get(str(sid), str(sid)),
                key='overview_rate_builder_delete_id',
            )
            if st.button('Delete Selected Custom Scenario', key='overview_rate_builder_delete_button'):
                actions['delete_scenario_id'] = str(delete_id)

    return actions

