"""Deal-level difference table components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components.formatting import style_numeric_table


def _show_table(title: str, df: pd.DataFrame) -> None:
    st.markdown(f'**{title} ({len(df)})**')
    if df.empty:
        st.caption('No rows')
    else:
        st.dataframe(style_numeric_table(df), use_container_width=True)


def render_deal_diff_tables(diff: dict[str, pd.DataFrame | float], compact_mode: bool = False) -> None:
    """Render all difference tables for comparison mode."""
    changes = diff.get('deal_changes', pd.DataFrame())
    if isinstance(changes, pd.DataFrame):
        if changes.empty:
            st.caption('No changed deals between dates.')
        else:
            def _status_color(value: str) -> str:
                if 'new' in value:
                    return 'background-color: #d1fae5'
                if 'matured' in value:
                    return 'background-color: #fee2e2'
                if 'notional_changed' in value or 'coupon_changed' in value:
                    return 'background-color: #fef3c7'
                return ''

            styled = changes.style.map(_status_color, subset=['status'])
            num_fmt: dict[str, str] = {}
            for col in changes.columns:
                if pd.api.types.is_numeric_dtype(changes[col]):
                    name = str(col).lower()
                    if 'count' in name:
                        num_fmt[col] = '{:,.0f}'
                    elif 'coupon' in name:
                        num_fmt[col] = '{:,.4f}'
                    else:
                        num_fmt[col] = '{:,.2f}'
            if num_fmt:
                styled = styled.format(num_fmt, na_rep='-')
            st.markdown('**Consolidated Changes**')
            st.dataframe(styled, use_container_width=True)

    if compact_mode:
        with st.expander('Added Deals', expanded=False):
            _show_table('Added Deals', diff['added'])
        with st.expander('Matured/Removed Deals', expanded=False):
            _show_table('Matured/Removed Deals', diff['matured'])
        with st.expander('Notional Changes', expanded=False):
            _show_table('Notional Changes', diff['notional_changed'])
        with st.expander('Coupon Changes', expanded=False):
            _show_table('Coupon Changes', diff['coupon_changed'])
    else:
        _show_table('Added Deals', diff['added'])
        _show_table('Matured/Removed Deals', diff['matured'])
        _show_table('Notional Changes', diff['notional_changed'])
        _show_table('Coupon Changes', diff['coupon_changed'])
