"""Shared UI controls and state normalization helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

RUNOFF_DISPLAY_OPTIONS = ['Aligned Buckets (Remaining Maturity)', 'Calendar Months']
DEFAULT_RUNOFF_DISPLAY_MODE = 'Calendar Months'


def coerce_option(current: Any, options: list[Any], default: Any) -> Any:
    """Return a stable option value that is guaranteed to be in options."""
    if not options:
        return default
    if current in options:
        return current
    if default in options:
        return default
    return options[0]


def default_month_view_indices(month_ends: list[pd.Timestamp]) -> tuple[int, int]:
    """Return default indices for T1 and T2 selectors."""
    if not month_ends:
        return 0, 0
    t1_idx = 0
    t2_idx = 1 if len(month_ends) > 1 else 0
    return t1_idx, t2_idx


def _stable_radio(
    *,
    label: str,
    options: list[str],
    key: str,
    default: str,
    horizontal: bool = True,
) -> str:
    current = coerce_option(st.session_state.get(key, default), options, default)
    st.session_state[key] = current
    idx = options.index(current)
    return st.radio(label, options=options, index=idx, horizontal=horizontal, key=key)


def _stable_selectbox(
    *,
    label: str,
    options: list[Any],
    key: str,
    default: Any,
    format_func=None,
) -> Any:
    if not options:
        return default
    current = coerce_option(st.session_state.get(key, default), options, default)
    st.session_state[key] = current
    idx = options.index(current)
    if format_func is None:
        return st.selectbox(label, options, index=idx, key=key)
    return st.selectbox(label, options, index=idx, key=key, format_func=format_func)


def render_global_controls(
    month_ends: list[pd.Timestamp],
    products: list[str] | None = None,
    default_product: str | None = None,
) -> dict[str, Any]:
    """Render fixed sidebar controls and return normalized UI state."""
    products = products or []
    if products:
        product_default = coerce_option(
            default_product if default_product is not None else st.session_state.get('global_product', products[0]),
            products,
            products[0],
        )
    else:
        product_default = default_product

    t1_idx, t2_idx = default_month_view_indices(month_ends)
    t1_default = month_ends[t1_idx] if month_ends else None
    t2_default = month_ends[t2_idx] if month_ends else None

    with st.sidebar:
        st.subheader('Controls')
        input_path = st.text_input('Workbook path', value=st.session_state.get('global_input_path', 'Input.xlsx'), key='global_input_path')
        if st.button('Refresh Cached Calculations', key='global_refresh_cached_calculations'):
            st.cache_data.clear()
            st.rerun()
        if products:
            product = _stable_selectbox(
                label='Product',
                options=products,
                key='global_product',
                default=product_default,
            )
        else:
            product = product_default

        if month_ends:
            t1 = _stable_selectbox(
                label='Monthly View 1 (T1)',
                options=month_ends,
                key='global_t1',
                default=t1_default,
                format_func=lambda d: d.date().isoformat(),
            )
            t2_enabled = st.checkbox('Enable Monthly View 2 (T2)', value=st.session_state.get('global_t2_enabled', len(month_ends) > 1), key='global_t2_enabled')
            t2 = None
            if t2_enabled:
                t2 = _stable_selectbox(
                    label='Monthly View 2 (T2)',
                    options=month_ends,
                    key='global_t2',
                    default=t2_default,
                    format_func=lambda d: d.date().isoformat(),
                )
        else:
            st.caption('Load a valid workbook path to enable month-end controls.')
            t1 = None
            t2_enabled = False
            t2 = None

        runoff_display_mode = _stable_radio(
            label='Runoff Display Mode',
            options=RUNOFF_DISPLAY_OPTIONS,
            key='runoff_display_mode',
            default=DEFAULT_RUNOFF_DISPLAY_MODE,
            horizontal=False,
        )
        runoff_decomposition_basis = _stable_radio(
            label='Runoff decomposition basis',
            options=['T1', 'T2'],
            key='runoff_decomposition_basis',
            default='T2',
            horizontal=True,
        )
        growth_mode = _stable_radio(
            label='Growth mode',
            options=['constant', 'user_defined'],
            key='runoff_growth_mode',
            default='constant',
            horizontal=True,
        )
        growth_monthly_value = float(st.session_state.get('runoff_growth_monthly_value', 0.0))
        if growth_mode == 'user_defined':
            growth_monthly_value = float(
                st.number_input(
                    'User-defined monthly growth (EUR)',
                    min_value=0.0,
                    value=growth_monthly_value,
                    step=1000000.0,
                    format='%.2f',
                    key='runoff_growth_monthly_value',
                )
            )

    return {
        'input_path': input_path,
        'product': product,
        't1': t1,
        't2_enabled': t2_enabled,
        't2': t2,
        'runoff_display_mode': runoff_display_mode,
        'runoff_decomposition_basis': runoff_decomposition_basis,
        'growth_mode': growth_mode,
        'growth_monthly_value': growth_monthly_value,
    }


def render_runoff_controls(default_basis: str = 'T2') -> dict[str, Any]:
    """Render compact runoff controls above charts."""
    include_refill = bool(st.session_state.get('runoff_has_refill_views', False))
    options = [
        'Notional Decomposition',
        'Effective Interest Decomposition',
        'Effective Interest Contribution',
        'Deal Count Decomposition',
        'Cumulative Notional',
    ]
    if include_refill:
        options.extend(
            [
                'Effective Interest Decomposition (Refill/Growth)',
                'Cumulative Notional (Refill/Growth)',
                'Refill Allocation Heatmap',
            ]
        )

    c1, c2 = st.columns([2, 2])
    with c1:
        runoff_chart_view = _stable_radio(
            label='Runoff chart view',
            options=options,
            key='runoff_chart_view',
            default=options[0],
            horizontal=False,
        )
    with c2:
        st.caption(f'Decomposition basis: {st.session_state.get("runoff_decomposition_basis", default_basis)}')
        st.caption('Buckets = remaining maturity; Monthly View = T1/T2.')
        flip_y_axis = st.checkbox(
            'Flip y-axis orientation',
            value=bool(st.session_state.get('runoff_flip_y_axis', False)),
            key='runoff_flip_y_axis',
            help='Visual-only orientation flip. Signed values are unchanged.',
        )

    return {
        'runoff_chart_view': runoff_chart_view,
        'runoff_decomposition_basis': st.session_state.get('runoff_decomposition_basis', default_basis),
        'flip_y_axis': flip_y_axis,
    }
