"""Summary card renderer for key KPIs."""

from __future__ import annotations

import streamlit as st


def render_summary_cards(
    realized_nii_prev_month: float,
    active_deals: int,
    accrued_interest: float,
    title: str = 'Primary Date Metrics',
) -> None:
    """Render top-level KPI cards."""
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric('Realized NII (Prev Month, EUR)', f'{realized_nii_prev_month:,.2f}')
    c2.metric('Active Deals', f'{active_deals:,d}')
    c3.metric('Accrued Interest (EUR)', f'{accrued_interest:,.2f}')
