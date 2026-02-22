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
    is_active,
)
from src.calculations.volumes import (
    compute_calendar_month_runoff_view,
    compute_monthly_buckets,
    compute_runoff_delta_attribution,
)
from src.dashboard.components.deal_diff_table import render_deal_diff_tables
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


def _stable_radio(
    *,
    label: str,
    options: list[str],
    key: str,
    default: str,
    horizontal: bool = True,
) -> str:
    current = st.session_state.get(key, default)
    if current not in options:
        current = default if default in options else options[0]
        st.session_state[key] = current
    idx = options.index(current)
    return st.radio(label, options=options, index=idx, horizontal=horizontal, key=key)



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


def _styled_numeric_table(df: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if df.empty:
        return df
    fmt: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            name = str(col).lower()
            if 'count' in name:
                fmt[col] = '{:,.0f}'
            elif 'coupon' in name:
                fmt[col] = '{:,.4f}'
            else:
                fmt[col] = '{:,.2f}'
    if not fmt:
        return df
    return df.style.format(fmt, na_rep='-')


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

    # 30/360 daily handling:
    # - 31-day month: exclude day 31 from the plotted daily series.
    # - 30-day month: each day weight is 1.
    # - short month (e.g., Feb): keep all days and add the top-up to last day so weights sum to 30.
    if days_in_month == 31:
        days = all_days[:-1]
        weights = pd.Series(1.0, index=days)
    else:
        days = all_days
        weights = pd.Series(1.0, index=days)
        if days_in_month < 30:
            weights.iloc[-1] += float(30 - days_in_month)

    # Monthly cohorts used for decomposition
    prior_cohort = deals_df[deals_df['value_date'] < start].copy()
    monthly_booked = deals_df[(deals_df['value_date'] >= start) & (deals_df['value_date'] <= end)].copy()
    matured_in_month_all = deals_df[
        (deals_df['maturity_date'] >= start)
        & (deals_df['maturity_date'] <= end)
        & (deals_df['value_date'] < deals_df['maturity_date'])
    ].copy()

    rows = []
    for d in days:
        # Existing and added are active-only contributions for the day.
        existing = prior_cohort[
            (prior_cohort['value_date'] <= d) & (d < prior_cohort['maturity_date'])
        ]
        added = monthly_booked[
            (monthly_booked['value_date'] <= d) & (d < monthly_booked['maturity_date'])
        ]
        # Matured effect is cumulative across the month (visible from maturity date onward).
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
        # Total line shows active daily interest only; matured is shown as separate negative component.
        interest_total = interest_existing + interest_added

        # Daily relevant notionals use signed values so positive/negative notionals
        # are handled consistently with monthly total active notional metrics.
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


def main() -> None:
    st.set_page_config(page_title='NII Dashboard', layout='wide')
    st.title('Net Interest Income Dashboard')

    input_path = st.text_input('Workbook path', value=str(DEFAULT_INPUT_PATH))

    deals_df, curve_df = _load(input_path)
    refill_logic_df = _load_refill_logic(input_path)

    month_ends = _available_month_ends(deals_df)
    if not month_ends:
        st.error('No month-end dates available from input data.')
        return

    default_t1_idx = 0
    default_t2_idx = 1 if len(month_ends) > 1 else 0
    t1 = st.selectbox('Monthly View 1 (T1)', month_ends, index=default_t1_idx, format_func=lambda d: d.date().isoformat())

    t2_enabled = st.checkbox('Enable Monthly View 2 (T2)', value=len(month_ends) > 1)
    t2 = None
    if t2_enabled:
        t2 = st.selectbox(
            'Monthly View 2 (T2)',
            month_ends,
            index=default_t2_idx,
            format_func=lambda d: d.date().isoformat(),
        )

    prev_start, prev_end = previous_calendar_month_window(t1)
    realized_nii = compute_monthly_realized_nii(deals_df, prev_start, prev_end)
    active_t1 = active_deals_snapshot(deals_df, t1)
    active_count_t1 = int(len(active_t1))
    accrued_t1 = accrued_interest_to_date(deals_df, t1)

    monthly_t1 = compute_monthly_buckets(deals_df, t1)
    row_t1 = monthly_t1[monthly_t1['month_end'] == t1].iloc[0]

    if t2 is not None:
        st.divider()
        st.header('Monthly View Comparison')

        prev_start_t2, prev_end_t2 = previous_calendar_month_window(t2)
        realized_nii_t2 = compute_monthly_realized_nii(deals_df, prev_start_t2, prev_end_t2)
        active_t2 = active_deals_snapshot(deals_df, t2)
        active_count_t2 = int(len(active_t2))
        accrued_t2 = accrued_interest_to_date(deals_df, t2)

        monthly_t2 = compute_monthly_buckets(deals_df, t2)
        row_t1 = monthly_t1[monthly_t1['month_end'] == t1].iloc[0]
        row_t2 = monthly_t2[monthly_t2['month_end'] == t2].iloc[0]

        st.subheader('Monthly View Delta Summary (T2 - T1)')
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

        st.subheader('Monthly View Metrics (T1 vs T2)')
        st.dataframe(
            _styled_numeric_table(
                _dual_view_metric_table(row_t1, row_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')
            ),
            use_container_width=True,
        )

        st.subheader('Daily Interest (Previous Calendar Month)')
        daily_t1 = _compute_daily_interest(deals_df, t1)
        daily_t2 = _compute_daily_interest(deals_df, t2)
        render_daily_interest_chart(daily_t1, daily_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')

        st.subheader('Runoff Comparison')
        runoff_delta = compute_runoff_delta_attribution(deals_df, t1, t2)
        runoff_mode_options = ['Aligned Buckets (Remaining Maturity)', 'Calendar Months']
        runoff_mode = _stable_radio(
            label='Runoff Display Mode',
            options=runoff_mode_options,
            key='runoff_display_mode',
            default=runoff_mode_options[0],
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
        st.caption('Switch between aligned maturity buckets and calendar-month aggregation over the same 240-month horizon.')

        if runoff_mode == 'Aligned Buckets (Remaining Maturity)':
            render_runoff_delta_charts(
                runoff_delta,
                key_prefix='runoff_aligned',
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=refill_logic_df,
                curve_df=curve_df,
                growth_mode=growth_mode,
                monthly_growth_amount=growth_monthly_value,
            )
            st.dataframe(_styled_numeric_table(_build_runoff_bucket_table(runoff_delta, t1, t2)), use_container_width=True)
        else:
            calendar_runoff = compute_calendar_month_runoff_view(runoff_delta, t1, t2, deals_df=deals_df)
            basis = st.session_state.get('runoff_decomposition_basis', 'T2')
            anchor = pd.Timestamp(t1 if basis == 'T1' else t2) + pd.offsets.MonthEnd(0)
            timeframe_end = anchor + pd.offsets.MonthEnd(240)
            filtered = calendar_runoff[
                (calendar_runoff['calendar_month_end'] >= anchor)
                & (calendar_runoff['calendar_month_end'] <= timeframe_end)
            ].copy()

            render_calendar_runoff_charts(
                filtered,
                label_t1=f'T1 {t1.date()}',
                label_t2=f'T2 {t2.date()}',
                key_prefix='runoff_calendar',
                runoff_compare_df=runoff_delta,
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=refill_logic_df,
                curve_df=curve_df,
                growth_mode=growth_mode,
                monthly_growth_amount=growth_monthly_value,
            )
            st.dataframe(_styled_numeric_table(_build_runoff_calendar_table(filtered, t1, t2)), use_container_width=True)

        with st.expander('Deal-Level Differences', expanded=False):
            diff = compare_month_ends(deals_df, t1, t2)
            render_deal_diff_tables(diff)
    else:
        render_summary_cards(realized_nii, active_count_t1, accrued_t1, title=f'Metrics at {t1.date()}')
        st.info('Enable `Monthly View 2 (T2)` to see the concise comparison dashboard.')


if __name__ == '__main__':
    main()
