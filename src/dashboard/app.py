"""Streamlit app entrypoint for NII dashboard MVP."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.calculations.nii import (
    accrued_interest_to_date,
    active_deals_snapshot,
    compare_month_ends,
    compute_monthly_realized_nii,
)
from src.calculations.volumes import compute_monthly_buckets
from src.calculations.volumes import (
    compare_monthly_bucket_series,
    compute_runoff_delta_attribution,
)
from src.dashboard.components.deal_diff_table import render_deal_diff_tables
from src.dashboard.components.summary_cards import render_summary_cards
from src.dashboard.plots.monthly_plots import render_monthly_comparison_overview, render_monthly_plots
from src.dashboard.plots.runoff_plots import render_runoff_comparison_bar
from src.data.loader import load_input_workbook
from src.utils.date_utils import month_end_sequence, previous_calendar_month_window

DEFAULT_INPUT_PATH = PROJECT_ROOT / 'Input.xlsx'


@st.cache_data
def _load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_input_workbook(path)


def _available_month_ends(deals_df: pd.DataFrame) -> list[pd.Timestamp]:
    start = deals_df['value_date'].min()
    end = deals_df['maturity_date'].max() - pd.Timedelta(days=1)
    return month_end_sequence(start, end)


def main() -> None:
    st.set_page_config(page_title='NII Dashboard', layout='wide')
    st.title('Net Interest Income Dashboard')

    input_path = st.text_input('Workbook path', value=str(DEFAULT_INPUT_PATH))

    deals_df, _curve_df = _load(input_path)

    month_ends = _available_month_ends(deals_df)
    if not month_ends:
        st.error('No month-end dates available from input data.')
        return

    default_t1_idx = 0
    default_t2_idx = 1 if len(month_ends) > 1 else 0
    t1 = st.selectbox('Primary month-end (T1)', month_ends, index=default_t1_idx, format_func=lambda d: d.date().isoformat())

    t2_enabled = st.checkbox('Enable comparison date (T2)', value=len(month_ends) > 1)
    t2 = None
    if t2_enabled:
        t2 = st.selectbox(
            'Comparison month-end (T2)',
            month_ends,
            index=default_t2_idx,
            format_func=lambda d: d.date().isoformat(),
        )

    prev_start, prev_end = previous_calendar_month_window(t1)
    realized_nii = compute_monthly_realized_nii(deals_df, prev_start, prev_end)
    active_t1 = active_deals_snapshot(deals_df, t1)
    active_count_t1 = int(len(active_t1))
    accrued_t1 = accrued_interest_to_date(deals_df, t1)

    render_summary_cards(realized_nii, active_count_t1, accrued_t1, title=f'Metrics at {t1.date()}')
    st.subheader(f'Active Deals at {t1.date()}')
    st.dataframe(
        active_t1[
            ['deal_id', 'trade_date', 'value_date', 'maturity_date', 'notional', 'coupon']
        ].sort_values(['maturity_date', 'deal_id']),
        use_container_width=True,
    )

    st.subheader('Monthly Buckets')
    monthly_t1 = compute_monthly_buckets(deals_df, t1)
    st.dataframe(monthly_t1, use_container_width=True)
    render_monthly_plots(monthly_t1, title_prefix='Primary Date')

    if t2 is not None:
        st.divider()
        st.header('Comparison View')

        prev_start_t2, prev_end_t2 = previous_calendar_month_window(t2)
        realized_nii_t2 = compute_monthly_realized_nii(deals_df, prev_start_t2, prev_end_t2)
        active_t2 = active_deals_snapshot(deals_df, t2)
        active_count_t2 = int(len(active_t2))
        accrued_t2 = accrued_interest_to_date(deals_df, t2)

        monthly_t2 = compute_monthly_buckets(deals_df, t2)
        row_t1 = monthly_t1[monthly_t1['month_end'] == t1].iloc[0]
        row_t2 = monthly_t2[monthly_t2['month_end'] == t2].iloc[0]

        st.subheader('Delta Summary (T2 - T1)')
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric('Realized NII Delta (EUR)', f'{(realized_nii_t2 - realized_nii):,.2f}')
        d2.metric('Active Deals Delta', f'{(active_count_t2 - active_count_t1):,d}')
        d3.metric('Accrued Interest Delta (EUR)', f'{(accrued_t2 - accrued_t1):,.2f}')
        d4.metric(
            'Volume Delta (EUR)',
            f'{(float(row_t2["total_active_notional"]) - float(row_t1["total_active_notional"])):,.2f}',
        )
        d5.metric(
            'Coupon Delta',
            f'{(float(row_t2["weighted_avg_coupon"]) - float(row_t1["weighted_avg_coupon"])):.6f}',
        )

        st.subheader(f'Active Deals at {t2.date()}')
        st.dataframe(
            active_t2[
                ['deal_id', 'trade_date', 'value_date', 'maturity_date', 'notional', 'coupon']
            ].sort_values(['maturity_date', 'deal_id']),
            use_container_width=True,
        )

        st.subheader('Comparison Date Monthly Buckets')
        st.dataframe(monthly_t2, use_container_width=True)
        render_monthly_plots(monthly_t2, title_prefix='Comparison Date')

        all_months = sorted(set(monthly_t1['month_end']).union(set(monthly_t2['month_end'])))
        excluded_months = st.multiselect(
            'Hide month-end points from comparison views',
            options=all_months,
            default=[],
            format_func=lambda d: pd.Timestamp(d).date().isoformat(),
        )
        excluded_set = set(pd.to_datetime(excluded_months))
        monthly_t1_filtered = monthly_t1[~monthly_t1['month_end'].isin(excluded_set)].copy()
        monthly_t2_filtered = monthly_t2[~monthly_t2['month_end'].isin(excluded_set)].copy()

        st.subheader('Combined Comparison Graph (T1 vs T2)')
        render_monthly_comparison_overview(
            monthly_t1_filtered,
            monthly_t2_filtered,
            label_d1=f'T1 {t1.date()}',
            label_d2=f'T2 {t2.date()}',
        )

        st.subheader('Monthly Bucket Differences (Calendar)')
        bucket_delta = compare_monthly_bucket_series(monthly_t1_filtered, monthly_t2_filtered)
        st.dataframe(bucket_delta, use_container_width=True)

        st.subheader('Runoff Comparison (Month Offset)')
        runoff_mode = st.radio(
            'Runoff view mode',
            options=['Absolute Remaining', 'Delta Attribution'],
            horizontal=True,
        )
        runoff_delta = compute_runoff_delta_attribution(deals_df, t1, t2)
        render_runoff_comparison_bar(
            runoff_delta,
            label_d1=f'T1 {t1.date()}',
            label_d2=f'T2 {t2.date()}',
            mode=runoff_mode,
        )
        st.dataframe(runoff_delta, use_container_width=True)

        diff = compare_month_ends(deals_df, t1, t2)
        render_deal_diff_tables(diff)


if __name__ == '__main__':
    main()
