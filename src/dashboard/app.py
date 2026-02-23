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
from src.calculations.volumes import (
    compute_calendar_month_runoff_view,
    compute_monthly_buckets,
    compute_runoff_delta_attribution,
)
from src.dashboard.components.controls import render_global_controls, render_runoff_controls
from src.dashboard.components.deal_diff_table import render_deal_diff_tables
from src.dashboard.components.formatting import style_numeric_table
from src.dashboard.components.summary_cards import render_summary_cards
from src.dashboard.plots.interest_daily import render_daily_interest_chart
from src.dashboard.plots.runoff_plots import render_calendar_runoff_charts, render_runoff_delta_charts
from src.data.loader import load_input_workbook
from src.utils.date_utils import month_end_sequence, previous_calendar_month_window

DEFAULT_INPUT_PATH = PROJECT_ROOT / 'Input.xlsx'


@st.cache_data
def _load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_input_workbook(path)


@st.cache_data
def _load_refill_logic(path: str) -> pd.DataFrame | None:
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return None
    sheet_name = next((s for s in xl.sheet_names if str(s).strip().lower() == 'refill_logic'), None)
    if sheet_name is None:
        return None
    try:
        return xl.parse(sheet_name)
    except Exception:
        return None


@st.cache_data
def _cached_monthly_buckets(path: str, month_end: pd.Timestamp) -> pd.DataFrame:
    deals_df, _ = _load(path)
    return compute_monthly_buckets(deals_df, pd.Timestamp(month_end))


@st.cache_data
def _cached_daily_interest(path: str, month_end: pd.Timestamp) -> pd.DataFrame:
    deals_df, _ = _load(path)
    return _compute_daily_interest(deals_df, pd.Timestamp(month_end))


@st.cache_data
def _cached_compare_month_ends(path: str, d1: pd.Timestamp, d2: pd.Timestamp) -> dict[str, pd.DataFrame | float]:
    deals_df, _ = _load(path)
    return compare_month_ends(deals_df, pd.Timestamp(d1), pd.Timestamp(d2))


@st.cache_data
def _cached_runoff_delta(path: str, d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    deals_df, _ = _load(path)
    return compute_runoff_delta_attribution(deals_df, pd.Timestamp(d1), pd.Timestamp(d2))


@st.cache_data
def _cached_calendar_runoff(path: str, d1: pd.Timestamp, d2: pd.Timestamp, include_effective: bool) -> pd.DataFrame:
    deals_df, _ = _load(path)
    runoff_delta = _cached_runoff_delta(path, pd.Timestamp(d1), pd.Timestamp(d2))
    effective_deals_df = deals_df if include_effective else None
    return compute_calendar_month_runoff_view(runoff_delta, pd.Timestamp(d1), pd.Timestamp(d2), deals_df=effective_deals_df)


def _available_month_ends(deals_df: pd.DataFrame) -> list[pd.Timestamp]:
    start = deals_df['value_date'].min()
    end = deals_df['maturity_date'].max() - pd.Timedelta(days=1)
    return month_end_sequence(start, end)


def _dual_view_metric_table(row1: pd.Series, row2: pd.Series, label_t1: str, label_t2: str) -> pd.DataFrame:
    """Three-column metric table (T1, T2, Delta)."""
    metrics = [
        ('total_active_notional', 'Total Active Notional (EUR)'),
        ('weighted_avg_coupon', 'Weighted Avg Coupon (pp)'),
        ('interest_paid_eur', 'Interest Paid EUR (30/360)'),
        ('active_deal_count', 'Active Deal Count'),
    ]
    data = []
    for field, label in metrics:
        v1 = row1[field]
        v2 = row2[field]
        if field == 'weighted_avg_coupon':
            v1 *= 100.0
            v2 *= 100.0
        data.append(
            {
                'metric': label,
                label_t1: v1,
                label_t2: v2,
                'Delta (T2-T1)': v2 - v1,
            }
        )
    return pd.DataFrame(data).set_index('metric')


def _compute_daily_interest(deals_df: pd.DataFrame, month_end: pd.Timestamp) -> pd.DataFrame:
    """Compute daily interest breakdown for the calendar month of month_end using 30/360 scaling.

    Interest components:
    - existing: deals from before month start that are still active on the day
    - added: deals booked during the month that are still active on the day
    - matured: cumulative negative effect from deals that have matured in the month up to the day
    """
    me = pd.Timestamp(month_end).normalize()
    start = me.replace(day=1)
    end = me
    all_days = pd.date_range(start=start, end=end, freq='D')
    days_in_month = len(all_days)

    if days_in_month == 31:
        days = all_days[:-1]
        weights = pd.Series(1.0, index=days)
    else:
        days = all_days
        weights = pd.Series(1.0, index=days)
        if days_in_month < 30:
            weights.iloc[-1] += float(30 - days_in_month)

    prior_cohort = deals_df[deals_df['value_date'] < start].copy()
    monthly_booked = deals_df[(deals_df['value_date'] >= start) & (deals_df['value_date'] <= end)].copy()
    matured_in_month_all = deals_df[
        (deals_df['maturity_date'] >= start)
        & (deals_df['maturity_date'] <= end)
        & (deals_df['value_date'] < deals_df['maturity_date'])
    ].copy()

    rows = []
    for d in days:
        existing = prior_cohort[
            (prior_cohort['value_date'] <= d) & (d < prior_cohort['maturity_date'])
        ]
        added = monthly_booked[
            (monthly_booked['value_date'] <= d) & (d < monthly_booked['maturity_date'])
        ]
        matured_cum = matured_in_month_all[matured_in_month_all['maturity_date'] <= d]

        interest_existing = (
            float((existing['notional'] * existing['coupon'] / 360.0).sum()) if not existing.empty else 0.0
        )
        interest_added = (
            float((added['notional'] * added['coupon'] / 360.0).sum()) if not added.empty else 0.0
        )
        matured_contrib = (
            float((matured_cum['notional'] * matured_cum['coupon'] / 360.0).sum()) if not matured_cum.empty else 0.0
        )
        interest_matured = -matured_contrib

        day_weight = float(weights.loc[d])
        interest_existing *= day_weight
        interest_added *= day_weight
        interest_matured *= day_weight
        interest_total = interest_existing + interest_added

        notional_existing = float(existing['notional'].sum()) if not existing.empty else 0.0
        notional_added = float(added['notional'].sum()) if not added.empty else 0.0
        matured_notional_contrib = (
            float(matured_cum['notional'].sum()) if not matured_cum.empty else 0.0
        )
        notional_matured = -matured_notional_contrib
        notional_total = notional_existing + notional_added

        rows.append(
            {
                'date': d,
                'interest_total': interest_total,
                'interest_existing': interest_existing,
                'interest_added': interest_added,
                'interest_matured': interest_matured,
                'notional_total': notional_total,
                'notional_existing': notional_existing,
                'notional_added': notional_added,
                'notional_matured': notional_matured,
                'day_weight_30_360': day_weight,
            }
        )
    return pd.DataFrame(rows)


def _build_runoff_bucket_table(runoff_delta: pd.DataFrame, t1: pd.Timestamp, t2: pd.Timestamp) -> pd.DataFrame:
    n1 = runoff_delta['signed_notional_d1'] if 'signed_notional_d1' in runoff_delta.columns else runoff_delta['abs_notional_d1']
    n2 = runoff_delta['signed_notional_d2'] if 'signed_notional_d2' in runoff_delta.columns else runoff_delta['abs_notional_d2']
    mask = ((n1.abs() + n2.abs() + runoff_delta['deal_count_d1'] + runoff_delta['deal_count_d2']) > 0)
    rd = runoff_delta.loc[mask].copy()
    if rd.empty:
        rd = runoff_delta.head(24).copy()

    rows = []
    for _, r in rd.iterrows():
        bucket = int(r['remaining_maturity_months'])
        notional_t1 = r['signed_notional_d1'] if 'signed_notional_d1' in r else r['abs_notional_d1']
        notional_t2 = r['signed_notional_d2'] if 'signed_notional_d2' in r else r['abs_notional_d2']
        notional_delta = r['signed_notional_delta'] if 'signed_notional_delta' in r else r['abs_notional_delta']
        effective_t1 = r['effective_interest_d1'] if 'effective_interest_d1' in r else r['notional_coupon_d1']
        effective_t2 = r['effective_interest_d2'] if 'effective_interest_d2' in r else r['notional_coupon_d2']
        effective_delta = r['effective_interest_delta'] if 'effective_interest_delta' in r else r['notional_coupon_delta']
        rows.append(
            {
                'Remaining Maturity (Months)': bucket,
                'Metric': 'Notional (EUR)',
                f'T1 {t1.date()}': notional_t1,
                f'T2 {t2.date()}': notional_t2,
                'Delta (T2-T1)': notional_delta,
            }
        )
        rows.append(
            {
                'Remaining Maturity (Months)': bucket,
                'Metric': 'Effective Interest',
                f'T1 {t1.date()}': effective_t1,
                f'T2 {t2.date()}': effective_t2,
                'Delta (T2-T1)': effective_delta,
            }
        )
        rows.append(
            {
                'Remaining Maturity (Months)': bucket,
                'Metric': 'Deal Count',
                f'T1 {t1.date()}': r['deal_count_d1'],
                f'T2 {t2.date()}': r['deal_count_d2'],
                'Delta (T2-T1)': r['deal_count_delta'],
            }
        )
    return pd.DataFrame(rows)


def _build_runoff_calendar_table(calendar_df: pd.DataFrame, t1: pd.Timestamp, t2: pd.Timestamp) -> pd.DataFrame:
    if calendar_df.empty:
        return pd.DataFrame()
    table = calendar_df.copy()
    table['calendar_month_end'] = table['calendar_month_end'].dt.date
    eff_t1_col = 'effective_interest_t1' if 'effective_interest_t1' in table.columns else 'notional_coupon_t1'
    eff_t2_col = 'effective_interest_t2' if 'effective_interest_t2' in table.columns else 'notional_coupon_t2'
    eff_delta_col = 'effective_interest_delta' if 'effective_interest_delta' in table.columns else 'notional_coupon_delta'
    eff_cum_t1_col = (
        'cumulative_effective_interest_t1'
        if 'cumulative_effective_interest_t1' in table.columns
        else 'cumulative_notional_coupon_t1'
    )
    eff_cum_t2_col = (
        'cumulative_effective_interest_t2'
        if 'cumulative_effective_interest_t2' in table.columns
        else 'cumulative_notional_coupon_t2'
    )
    n_t1_col = 'signed_notional_t1' if 'signed_notional_t1' in table.columns else 'abs_notional_t1'
    n_t2_col = 'signed_notional_t2' if 'signed_notional_t2' in table.columns else 'abs_notional_t2'
    n_delta_col = 'signed_notional_delta' if 'signed_notional_delta' in table.columns else 'abs_notional_delta'
    n_cum_t1_col = (
        'cumulative_signed_notional_t1'
        if 'cumulative_signed_notional_t1' in table.columns
        else 'cumulative_abs_notional_t1'
    )
    n_cum_t2_col = (
        'cumulative_signed_notional_t2'
        if 'cumulative_signed_notional_t2' in table.columns
        else 'cumulative_abs_notional_t2'
    )
    return table.rename(
        columns={
            'calendar_month_end': 'Calendar Month End',
            n_t1_col: f'Notional T1 {t1.date()}',
            n_t2_col: f'Notional T2 {t2.date()}',
            n_delta_col: 'Notional Delta (T2-T1)',
            eff_t1_col: f'Effective Interest T1 {t1.date()}',
            eff_t2_col: f'Effective Interest T2 {t2.date()}',
            eff_delta_col: 'Effective Interest Delta (T2-T1)',
            'deal_count_t1': f'Deal Count T1 {t1.date()}',
            'deal_count_t2': f'Deal Count T2 {t2.date()}',
            'deal_count_delta': 'Deal Count Delta (T2-T1)',
            n_cum_t1_col: f'Cumulative Notional T1 {t1.date()}',
            n_cum_t2_col: f'Cumulative Notional T2 {t2.date()}',
            eff_cum_t1_col: f'Cumulative Effective Interest T1 {t1.date()}',
            eff_cum_t2_col: f'Cumulative Effective Interest T2 {t2.date()}',
        }
    )


def _filter_runoff_bucket_table_by_view(table: pd.DataFrame, chart_view: str) -> pd.DataFrame:
    if table.empty:
        return table
    if chart_view == 'Deal Count Decomposition':
        keep = {'Deal Count'}
    elif 'Effective Interest' in chart_view:
        keep = {'Effective Interest'}
    else:
        keep = {'Notional (EUR)'}
    return table[table['Metric'].isin(keep)].copy()


def _filter_runoff_calendar_table_by_view(table: pd.DataFrame, chart_view: str) -> pd.DataFrame:
    if table.empty:
        return table

    cols = ['Calendar Month End']
    lower_cols = {c: str(c).lower() for c in table.columns}

    def _matches(key: str) -> list[str]:
        return [c for c, lc in lower_cols.items() if key in lc]

    if chart_view == 'Deal Count Decomposition':
        cols.extend(_matches('deal count'))
    elif chart_view in {'Cumulative Notional', 'Cumulative Notional (Refill/Growth)'}:
        cols.extend(_matches('cumulative notional'))
    elif chart_view in {'Effective Interest Contribution'}:
        cols.extend([c for c in _matches('effective interest') if 'cumulative' not in lower_cols[c]])
        cols.extend(_matches('cumulative effective interest'))
    elif 'Effective Interest' in chart_view:
        cols.extend([c for c in _matches('effective interest') if 'cumulative' not in lower_cols[c]])
    else:
        cols.extend([c for c in _matches('notional ') if 'cumulative' not in lower_cols[c]])

    cols = [c for c in cols if c in table.columns]
    seen = set()
    ordered = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return table[ordered].copy()


def main() -> None:
    st.set_page_config(page_title='NII Dashboard', layout='wide')
    st.title('Net Interest Income Dashboard')

    current_path = st.session_state.get('global_input_path', str(DEFAULT_INPUT_PATH))
    try:
        deals_probe, _ = _load(current_path)
        month_ends = _available_month_ends(deals_probe)
    except Exception:
        month_ends = []

    ui = render_global_controls(month_ends)
    input_path = ui['input_path']
    if input_path != current_path:
        st.rerun()

    try:
        deals_df, curve_df = _load(input_path)
    except Exception as exc:
        st.error(f'Failed to load workbook at `{input_path}`: {exc}')
        st.stop()

    refill_logic_df = _load_refill_logic(input_path)
    st.session_state['runoff_has_refill_views'] = refill_logic_df is not None and not refill_logic_df.empty

    month_ends = _available_month_ends(deals_df)
    if not month_ends:
        st.error('No month-end dates available from input data.')
        return

    t1 = ui['t1'] if ui['t1'] is not None else month_ends[0]
    t2 = ui['t2'] if ui['t2_enabled'] else None

    overview_tab, daily_tab, runoff_tab, diff_tab = st.tabs(['Overview', 'Daily', 'Runoff', 'Deal Differences'])

    prev_start, prev_end = previous_calendar_month_window(t1)
    realized_nii_t1 = compute_monthly_realized_nii(deals_df, prev_start, prev_end)
    active_t1 = active_deals_snapshot(deals_df, t1)
    active_count_t1 = int(len(active_t1))
    accrued_t1 = accrued_interest_to_date(deals_df, t1)
    monthly_t1 = _cached_monthly_buckets(input_path, t1)
    row_t1 = monthly_t1[monthly_t1['month_end'] == t1].iloc[0]

    if t2 is None:
        with overview_tab:
            render_summary_cards(realized_nii_t1, active_count_t1, accrued_t1, title=f'Metrics at {t1.date()}')
            st.info('Enable `Monthly View 2 (T2)` in the sidebar to unlock comparison tabs.')
        with daily_tab:
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        with runoff_tab:
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        with diff_tab:
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        return

    prev_start_t2, prev_end_t2 = previous_calendar_month_window(t2)
    realized_nii_t2 = compute_monthly_realized_nii(deals_df, prev_start_t2, prev_end_t2)
    active_t2 = active_deals_snapshot(deals_df, t2)
    active_count_t2 = int(len(active_t2))
    accrued_t2 = accrued_interest_to_date(deals_df, t2)
    monthly_t2 = _cached_monthly_buckets(input_path, t2)
    row_t2 = monthly_t2[monthly_t2['month_end'] == t2].iloc[0]

    with overview_tab:
        st.subheader('Monthly View Delta Summary (T2 - T1)')
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric('Realized NII Delta (EUR)', f'{(realized_nii_t2 - realized_nii_t1):,.2f}')
        d2.metric('Active Deals Delta', f'{(active_count_t2 - active_count_t1):,d}')
        d3.metric('Accrued Interest Delta (EUR)', f'{(accrued_t2 - accrued_t1):,.2f}')
        d4.metric(
            'Volume Delta (EUR)',
            f'{(float(row_t2["total_active_notional"]) - float(row_t1["total_active_notional"])):,.2f}',
        )
        d5.metric(
            'Coupon Delta (pp)',
            f'{((float(row_t2["weighted_avg_coupon"]) - float(row_t1["weighted_avg_coupon"])) * 100.0):.4f}',
        )

        st.subheader('Monthly View Metrics (T1 vs T2)')
        metrics_table = _dual_view_metric_table(row_t1, row_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')
        st.dataframe(style_numeric_table(metrics_table), use_container_width=True)

    with daily_tab:
        st.caption(
            'Top chart uses a stacked-style total bar: month-start base plus shaded delta versus first day (green up, red down). '
            'Interest view also overlays cumulative Total in the top chart. '
            'Bottom chart shows Added and Matured breakdown plus cumulative Added+Matured and cumulative Added/Matured lines '
            'and includes a month-end cumulative decomposition summary table below.'
        )
        daily_t1 = _cached_daily_interest(input_path, t1)
        daily_t2 = _cached_daily_interest(input_path, t2)
        render_daily_interest_chart(daily_t1, daily_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')

    with runoff_tab:
        runoff_ui = render_runoff_controls(default_basis=ui['runoff_decomposition_basis'])
        full_ui = {**ui, **runoff_ui}

        if ui['runoff_display_mode'] == 'Aligned Buckets (Remaining Maturity)':
            runoff_delta = _cached_runoff_delta(input_path, t1, t2)
            selected_view = render_runoff_delta_charts(
                runoff_delta,
                key_prefix='runoff_aligned',
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=refill_logic_df,
                curve_df=curve_df,
                ui_state=full_ui,
            )
            bucket_table = _build_runoff_bucket_table(runoff_delta, t1, t2)
            bucket_table = _filter_runoff_bucket_table_by_view(bucket_table, selected_view)
            st.dataframe(style_numeric_table(bucket_table), use_container_width=True)
        else:
            basis = ui['runoff_decomposition_basis']
            anchor = pd.Timestamp(t1 if basis == 'T1' else t2) + pd.offsets.MonthEnd(0)
            timeframe_end = anchor + pd.offsets.MonthEnd(240)
            needs_effective = 'Effective Interest' in str(full_ui.get('runoff_chart_view', ''))
            calendar_runoff = _cached_calendar_runoff(input_path, t1, t2, needs_effective)
            filtered = calendar_runoff[
                (calendar_runoff['calendar_month_end'] >= anchor)
                & (calendar_runoff['calendar_month_end'] <= timeframe_end)
            ].copy()

            selected_view = render_calendar_runoff_charts(
                filtered,
                label_t1=f'T1 {t1.date()}',
                label_t2=f'T2 {t2.date()}',
                key_prefix='runoff_calendar',
                runoff_compare_df=None,
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=refill_logic_df,
                curve_df=curve_df,
                ui_state=full_ui,
            )
            calendar_table = _build_runoff_calendar_table(filtered, t1, t2)
            calendar_table = _filter_runoff_calendar_table_by_view(calendar_table, selected_view)
            st.dataframe(style_numeric_table(calendar_table), use_container_width=True)

    with diff_tab:
        diff = _cached_compare_month_ends(input_path, t1, t2)
        render_deal_diff_tables(diff, compact_mode=True)


if __name__ == '__main__':
    main()
