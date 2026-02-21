"""Deal-level difference table components."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _show_table(title: str, df: pd.DataFrame) -> None:
    st.markdown(f'**{title} ({len(df)})**')
    if df.empty:
        st.caption('No rows')
    else:
        st.dataframe(df, use_container_width=True)


def render_deal_diff_tables(diff: dict[str, pd.DataFrame | float]) -> None:
    """Render all difference tables for comparison mode."""
    st.subheader('Deal-Level Differences')
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
            st.markdown('**Consolidated Changes**')
            st.dataframe(styled, use_container_width=True)

    _show_table('Added Deals', diff['added'])
    _show_table('Matured/Removed Deals', diff['matured'])
    _show_table('Notional Changes', diff['notional_changed'])
    _show_table('Coupon Changes', diff['coupon_changed'])
