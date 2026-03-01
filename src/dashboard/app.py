"""Streamlit app entrypoint for NII dashboard MVP."""

from __future__ import annotations

import json
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
from src.calculations.rate_scenarios import normalize_scenarios_df, simulate_rate_scenarios
from src.calculations.refill_growth import (
    compute_refill_growth_components_anchor_safe,
    growth_outstanding_profile,
)
from src.calculations.volumes import (
    compute_calendar_month_runoff_view,
    compute_monthly_buckets,
    compute_runoff_delta_attribution,
)
from src.dashboard.components.controls import coerce_option, render_global_controls, render_runoff_controls
from src.dashboard.components.deal_diff_table import render_deal_diff_tables
from src.dashboard.components.formatting import style_numeric_table
from src.dashboard.components.rate_scenario_builder import render_rate_scenario_builder
from src.dashboard.components.summary_cards import render_summary_cards
from src.dashboard.plots.interest_daily import render_daily_interest_chart
from src.dashboard.plots.rate_scenario_plots import (
    build_curve_comparison_figure,
    build_scenario_matrix_table,
    build_selected_scenario_impact_figure,
)
from src.dashboard.plots.runoff_plots import render_calendar_runoff_charts, render_runoff_delta_charts
from src.dashboard.reporting.export_pack import (
    build_export_context,
    build_export_workbook_bytes,
    default_export_filename,
)
from src.dashboard.scenario_store import (
    SCENARIO_STORE_FILENAME,
    STORE_VERSION,
    add_custom_scenario,
    build_active_scenarios_df,
    build_scenario_universe_df,
    delete_custom_scenario,
    load_scenario_store,
    save_scenario_store,
    validate_custom_scenario,
)
from src.data.loader import load_input_workbook
from src.utils.date_utils import month_end_sequence, previous_calendar_month_window

DEFAULT_INPUT_PATH = PROJECT_ROOT / 'Input.xlsx'


@st.cache_data
def _load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_input_workbook(path)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def _serialize_scenarios_for_cache(scenarios_df: pd.DataFrame) -> str:
    records = scenarios_df.to_dict(orient='records') if scenarios_df is not None else []
    safe_records = [_sanitize_for_json(r) for r in records]
    return json.dumps(safe_records, sort_keys=True)


def _normalize_products(deals_df: pd.DataFrame) -> pd.DataFrame:
    out = deals_df.copy()
    if 'product' not in out.columns:
        out['product'] = 'Default'
    out['product'] = out['product'].fillna('Default').astype(str).str.strip()
    out.loc[out['product'].isin(['', 'nan', 'None']), 'product'] = 'Default'
    return out


def _available_products(deals_df: pd.DataFrame) -> list[str]:
    normalized = _normalize_products(deals_df)
    products = normalized['product'].dropna().astype(str).tolist()
    if not products:
        return ['Default']
    return sorted(set(products))


def _filter_deals_by_product(deals_df: pd.DataFrame, product: str | None) -> pd.DataFrame:
    normalized = _normalize_products(deals_df)
    if product is None:
        return normalized
    return normalized[normalized['product'] == str(product)].copy()


@st.cache_data
def _cached_monthly_buckets(path: str, month_end: pd.Timestamp, product: str) -> pd.DataFrame:
    deals_df, _ = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)
    return compute_monthly_buckets(deals_df, pd.Timestamp(month_end))


@st.cache_data
def _cached_daily_interest(path: str, month_end: pd.Timestamp, product: str) -> pd.DataFrame:
    deals_df, _ = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)
    return _compute_daily_interest(deals_df, pd.Timestamp(month_end))


@st.cache_data
def _cached_compare_month_ends(path: str, d1: pd.Timestamp, d2: pd.Timestamp, product: str) -> dict[str, pd.DataFrame | float]:
    deals_df, _ = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)
    return compare_month_ends(deals_df, pd.Timestamp(d1), pd.Timestamp(d2))


@st.cache_data
def _cached_runoff_delta(path: str, d1: pd.Timestamp, d2: pd.Timestamp, product: str) -> pd.DataFrame:
    deals_df, _ = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)
    return compute_runoff_delta_attribution(deals_df, pd.Timestamp(d1), pd.Timestamp(d2))


@st.cache_data
def _cached_calendar_runoff(path: str, d1: pd.Timestamp, d2: pd.Timestamp, include_effective: bool, product: str) -> pd.DataFrame:
    deals_df, _ = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)
    runoff_delta = _cached_runoff_delta(path, pd.Timestamp(d1), pd.Timestamp(d2), product)
    effective_deals_df = deals_df if include_effective else None
    return compute_calendar_month_runoff_view(runoff_delta, pd.Timestamp(d1), pd.Timestamp(d2), deals_df=effective_deals_df)


@st.cache_data
def _cached_rate_scenario_projection(
    path: str,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    product: str,
    basis: str,
    growth_mode: str,
    growth_monthly_value: float,
    scenarios_payload_json: str,
    active_ids_json: str,
    horizon_months: int = 60,
) -> dict[str, pd.DataFrame | str]:
    deals_df, curve_df = _load(path)
    deals_df = _filter_deals_by_product(deals_df, product)

    if curve_df is None or curve_df.empty:
        return {
            'error': 'Interest_Curve is empty. Rate scenario analysis is unavailable.',
            'scenarios': pd.DataFrame(),
            'monthly_base': pd.DataFrame(),
            'monthly_scenarios': pd.DataFrame(),
            'yearly_summary': pd.DataFrame(),
            'curve_points': pd.DataFrame(),
        }

    t1_me = pd.Timestamp(t1) + pd.offsets.MonthEnd(0)
    t2_me = pd.Timestamp(t2) + pd.offsets.MonthEnd(0)
    horizon_months = int(max(1, min(int(horizon_months), 240)))
    projection_months = pd.DatetimeIndex([t2_me + pd.offsets.MonthEnd(k) for k in range(horizon_months)])

    runoff_delta = _cached_runoff_delta(path, t1_me, t2_me, product)
    calendar_runoff = _cached_calendar_runoff(path, t1_me, t2_me, True, product).copy()
    if calendar_runoff.empty:
        return {
            'error': 'No runoff data available for scenario projection.',
            'scenarios': pd.DataFrame(),
            'monthly_base': pd.DataFrame(),
            'monthly_scenarios': pd.DataFrame(),
            'yearly_summary': pd.DataFrame(),
            'curve_points': pd.DataFrame(),
        }

    calendar_runoff['calendar_month_end'] = pd.to_datetime(calendar_runoff['calendar_month_end'])
    calendar_runoff = (
        calendar_runoff.set_index('calendar_month_end')
        .reindex(projection_months)
        .fillna(0.0)
        .reset_index()
        .rename(columns={'index': 'calendar_month_end'})
    )

    existing_col = 'effective_interest_t2' if 'effective_interest_t2' in calendar_runoff.columns else 'notional_coupon_t2'
    cumulative_notional_col = (
        'cumulative_signed_notional_t2'
        if 'cumulative_signed_notional_t2' in calendar_runoff.columns
        else 'cumulative_abs_notional_t2'
    )

    existing_contractual_interest = calendar_runoff[existing_col].astype(float).reset_index(drop=True)
    cumulative_notional = calendar_runoff[cumulative_notional_col].astype(float).reset_index(drop=True)
    growth_components = compute_refill_growth_components_anchor_safe(
        cumulative_notional=cumulative_notional,
        growth_mode=growth_mode,
        monthly_growth_amount=float(growth_monthly_value),
    )
    refill_required = growth_components['refill_required'].astype(float).reset_index(drop=True)
    growth_required = growth_components['growth_required'].astype(float).reset_index(drop=True)
    growth_outstanding = growth_outstanding_profile(
        growth_flow=growth_required,
        runoff_compare_df=runoff_delta,
        basis=str(basis or 'T2'),
    ).astype(float).reset_index(drop=True)

    tenor_months = pd.Series(range(1, horizon_months + 1), dtype=int)
    scenario_df = pd.DataFrame()
    try:
        scenario_records = json.loads(str(scenarios_payload_json or '[]'))
        if isinstance(scenario_records, list) and scenario_records:
            scenario_df = normalize_scenarios_df(pd.DataFrame(scenario_records))
        active_ids = json.loads(str(active_ids_json or '[]'))
        if isinstance(active_ids, list) and not scenario_df.empty:
            active_set = {str(x) for x in active_ids}
            scenario_df = scenario_df[scenario_df['scenario_id'].astype(str).isin(active_set)].copy()
    except Exception:
        scenario_df = pd.DataFrame()

    try:
        return simulate_rate_scenarios(
            month_ends=pd.Series(projection_months),
            existing_contractual_interest=existing_contractual_interest,
            refill_notional=refill_required,
            growth_notional=growth_outstanding,
            tenor_months=tenor_months,
            curve_df=curve_df,
            anchor_date=t2_me,
            scenarios=scenario_df if not scenario_df.empty else None,
        )
    except Exception as exc:
        return {
            'error': f'Failed to compute rate scenarios: {exc}',
            'scenarios': pd.DataFrame(),
            'monthly_base': pd.DataFrame(),
            'monthly_scenarios': pd.DataFrame(),
            'yearly_summary': pd.DataFrame(),
            'curve_points': pd.DataFrame(),
        }


def _available_month_ends(deals_df: pd.DataFrame) -> list[pd.Timestamp]:
    if deals_df is None or deals_df.empty:
        return []
    start = deals_df['value_date'].min()
    end = deals_df['maturity_date'].max() - pd.Timedelta(days=1)
    if pd.isna(start) or pd.isna(end):
        return []
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
    elif chart_view in {'Cumulative Notional', 'Cumulative Notional (Refill/Growth)', 'Refill Allocation Heatmap'}:
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

    current_path = st.session_state.get('global_input_path', 'Input.xlsx')
    try:
        deals_probe, _ = _load(current_path)
        products_probe = _available_products(deals_probe)
        product_probe = coerce_option(
            st.session_state.get('global_product', products_probe[0] if products_probe else 'Default'),
            products_probe if products_probe else ['Default'],
            products_probe[0] if products_probe else 'Default',
        )
        month_ends = _available_month_ends(_filter_deals_by_product(deals_probe, product_probe))
    except Exception:
        products_probe = []
        product_probe = None
        month_ends = []

    ui = render_global_controls(month_ends, products=products_probe, default_product=product_probe)
    input_path = ui['input_path']
    selected_product = ui.get('product')
    if input_path != current_path:
        st.rerun()
    if selected_product != product_probe:
        st.rerun()

    try:
        deals_df_all, curve_df = _load(input_path)
    except Exception as exc:
        st.error(f'Failed to load workbook at `{input_path}`: {exc}')
        st.stop()
    products = _available_products(deals_df_all)
    if not products:
        st.error('No products available from input data.')
        st.stop()
    selected_product = coerce_option(
        selected_product,
        products,
        products[0],
    )
    if st.session_state.get('global_product') != selected_product:
        st.session_state['global_product'] = selected_product
        st.rerun()
    deals_df = _filter_deals_by_product(deals_df_all, selected_product)

    st.session_state['runoff_has_refill_views'] = True

    scenario_store_path = PROJECT_ROOT / SCENARIO_STORE_FILENAME
    scenario_store_warning = ''
    try:
        scenario_store_payload = load_scenario_store(str(scenario_store_path))
    except Exception as exc:
        scenario_store_warning = (
            f'Failed to read scenario store `{scenario_store_path.name}`: {exc}. '
            'Using built-in scenarios only for this run.'
        )
        scenario_store_payload = {
            'version': STORE_VERSION,
            'custom_scenarios': [],
            'active_scenario_ids': [],
        }
    scenario_universe_df = build_scenario_universe_df(scenario_store_payload)
    active_scenarios_df = build_active_scenarios_df(scenario_store_payload)
    active_ids = active_scenarios_df['scenario_id'].astype(str).tolist()
    scenario_payload_json = _serialize_scenarios_for_cache(active_scenarios_df)
    active_ids_json = json.dumps(active_ids, sort_keys=True)
    scenario_id_to_label = {
        str(r['scenario_id']): str(r['scenario_label'])
        for r in active_scenarios_df[['scenario_id', 'scenario_label']].to_dict(orient='records')
    }
    scenario_id_to_label['__base__'] = 'Base Case (No Shock)'

    month_ends = _available_month_ends(deals_df)
    if not month_ends:
        st.error(f'No month-end dates available for selected product `{selected_product}`.')
        return

    t1 = ui['t1'] if ui['t1'] is not None else month_ends[0]
    t2 = ui['t2'] if ui['t2_enabled'] else None

    section_options = ['Overview', 'Daily', 'Runoff', 'Deal Differences']
    section_current = st.session_state.get('main_section', 'Overview')
    if section_current not in section_options:
        section_current = 'Overview'
    selected_section = st.radio(
        label='Section',
        options=section_options,
        index=section_options.index(section_current),
        horizontal=True,
        key='main_section',
    )

    prev_start, prev_end = previous_calendar_month_window(t1)
    realized_nii_t1 = compute_monthly_realized_nii(deals_df, prev_start, prev_end)
    active_t1 = active_deals_snapshot(deals_df, t1)
    active_count_t1 = int(len(active_t1))
    accrued_t1 = accrued_interest_to_date(deals_df, t1)
    monthly_t1 = _cached_monthly_buckets(input_path, t1, selected_product)
    row_t1 = monthly_t1[monthly_t1['month_end'] == t1].iloc[0]

    if t2 is None:
        if selected_section == 'Overview':
            render_summary_cards(realized_nii_t1, active_count_t1, accrued_t1, title=f'Metrics at {t1.date()}')
            st.info('Enable `Monthly View 2 (T2)` in the sidebar to unlock comparison tabs.')
        elif selected_section == 'Daily':
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        elif selected_section == 'Runoff':
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        else:
            st.info('Comparison mode required. Enable `T2` in the sidebar.')
        return

    prev_start_t2, prev_end_t2 = previous_calendar_month_window(t2)
    realized_nii_t2 = compute_monthly_realized_nii(deals_df, prev_start_t2, prev_end_t2)
    active_t2 = active_deals_snapshot(deals_df, t2)
    active_count_t2 = int(len(active_t2))
    accrued_t2 = accrued_interest_to_date(deals_df, t2)
    monthly_t2 = _cached_monthly_buckets(input_path, t2, selected_product)
    row_t2 = monthly_t2[monthly_t2['month_end'] == t2].iloc[0]

    if selected_section == 'Overview':
        overview_delta_kpis = {
            'Realized NII Delta (EUR)': float(realized_nii_t2 - realized_nii_t1),
            'Active Deals Delta': float(active_count_t2 - active_count_t1),
            'Accrued Interest Delta (EUR)': float(accrued_t2 - accrued_t1),
            'Volume Delta (EUR)': float(row_t2['total_active_notional']) - float(row_t1['total_active_notional']),
            'Coupon Delta (pp)': (float(row_t2['weighted_avg_coupon']) - float(row_t1['weighted_avg_coupon'])) * 100.0,
        }
        st.subheader('Monthly View Delta Summary (T2 - T1)')
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric('Realized NII Delta (EUR)', f'{overview_delta_kpis["Realized NII Delta (EUR)"]:,.2f}')
        d2.metric('Active Deals Delta', f'{int(overview_delta_kpis["Active Deals Delta"]):,d}')
        d3.metric('Accrued Interest Delta (EUR)', f'{overview_delta_kpis["Accrued Interest Delta (EUR)"]:,.2f}')
        d4.metric(
            'Volume Delta (EUR)',
            f'{overview_delta_kpis["Volume Delta (EUR)"]:,.2f}',
        )
        d5.metric(
            'Coupon Delta (pp)',
            f'{overview_delta_kpis["Coupon Delta (pp)"]:.4f}',
        )

        st.subheader('Monthly View Metrics (T1 vs T2)')
        metrics_table = _dual_view_metric_table(row_t1, row_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')
        st.dataframe(style_numeric_table(metrics_table), width='stretch')

        st.subheader('Rate Scenario Analysis (5Y, vs Base Case)')
        if scenario_store_warning:
            st.warning(scenario_store_warning)

        scenarios_df = pd.DataFrame()
        monthly_base = pd.DataFrame()
        monthly_scenarios = pd.DataFrame()
        yearly_summary = pd.DataFrame()
        curve_points = pd.DataFrame()
        tenor_paths = pd.DataFrame()
        scenario_result = _cached_rate_scenario_projection(
            input_path,
            t1,
            t2,
            selected_product,
            'T2',
            ui.get('growth_mode', 'constant'),
            float(ui.get('growth_monthly_value', 0.0)),
            scenario_payload_json,
            active_ids_json,
            60,
        )
        scenario_error = str(scenario_result.get('error', '')) if isinstance(scenario_result, dict) else ''
        if scenario_error:
            st.warning(scenario_error)
        else:
            scenarios_df = scenario_result.get('scenarios', pd.DataFrame())
            monthly_base = scenario_result.get('monthly_base', pd.DataFrame())
            monthly_scenarios = scenario_result.get('monthly_scenarios', pd.DataFrame())
            yearly_summary = scenario_result.get('yearly_summary', pd.DataFrame())
            curve_points = scenario_result.get('curve_points', pd.DataFrame())
            tenor_paths = scenario_result.get('tenor_paths', pd.DataFrame())

            if scenarios_df.empty or monthly_base.empty or monthly_scenarios.empty or yearly_summary.empty:
                st.info('Rate scenario outputs are currently unavailable for this selection.')
            else:
                st.caption('Scenario Impact Matrix')
                matrix_options = ['Delta', 'Absolute']
                matrix_current = coerce_option(
                    st.session_state.get('overview_rate_matrix_view', matrix_options[0]),
                    matrix_options,
                    matrix_options[0],
                )
                matrix_view = st.radio(
                    'Matrix view',
                    options=matrix_options,
                    index=matrix_options.index(matrix_current),
                    horizontal=True,
                    key='overview_rate_matrix_view',
                )
                matrix_df = build_scenario_matrix_table(
                    yearly_summary,
                    view_mode=('absolute' if matrix_view == 'Absolute' else 'delta'),
                    monthly_base=monthly_base,
                    monthly_scenarios=monthly_scenarios,
                )
                numeric_cols = [c for c in matrix_df.columns if c != 'Scenario']
                matrix_cmap = 'YlGnBu' if matrix_view == 'Absolute' else 'RdYlGn'
                matrix_styler = (
                    matrix_df.style
                    .format({col: '{:,.2f}' for col in numeric_cols})
                    .background_gradient(cmap=matrix_cmap, subset=numeric_cols)
                )
                st.dataframe(matrix_styler, width='stretch')

                labels = scenarios_df['scenario_label'].astype(str).tolist()
                c_s1, c_s2 = st.columns([2, 2])
                with c_s2:
                    detail_options = ['Delta', 'Absolute']
                    detail_current = coerce_option(
                        st.session_state.get('overview_rate_detail_view', detail_options[0]),
                        detail_options,
                        detail_options[0],
                    )
                    detail_view = st.radio(
                        'Detail view',
                        options=detail_options,
                        index=detail_options.index(detail_current),
                        horizontal=True,
                        key='overview_rate_detail_view',
                    )

                scenario_label_to_id = {
                    str(r['scenario_label']): str(r['scenario_id'])
                    for r in scenarios_df[['scenario_id', 'scenario_label']].to_dict(orient='records')
                }
                base_case_label = 'BaseCase'
                with c_s1:
                    if detail_view == 'Absolute':
                        absolute_labels = [base_case_label] + labels
                        selected_label = coerce_option(
                            st.session_state.get('overview_rate_scenario', absolute_labels[0]),
                            absolute_labels,
                            absolute_labels[0],
                        )
                        selected_label = st.selectbox(
                            'Detail scenario',
                            options=absolute_labels,
                            index=absolute_labels.index(selected_label),
                            key='overview_rate_scenario',
                        )
                        selected_scenario_id = (
                            '__base__' if selected_label == base_case_label else scenario_label_to_id[selected_label]
                        )
                    else:
                        label_default = labels[0]
                        selected_label = coerce_option(
                            st.session_state.get('overview_rate_scenario', label_default),
                            labels,
                            label_default,
                        )
                        selected_label = st.selectbox(
                            'Detail scenario',
                            options=labels,
                            index=labels.index(selected_label),
                            key='overview_rate_scenario',
                        )
                        selected_scenario_id = scenario_label_to_id[selected_label]

                detail_fig = build_selected_scenario_impact_figure(
                    monthly_base=monthly_base,
                    monthly_scenarios=monthly_scenarios,
                    scenario_id=selected_scenario_id,
                    scenario_label=selected_label,
                    view_mode=('absolute' if detail_view == 'Absolute' else 'delta'),
                    show_totals=True,
                )
                st.plotly_chart(detail_fig, width='stretch', key='overview_rate_detail_chart')

                if curve_points.empty and tenor_paths.empty:
                    st.info('Curve visualization unavailable for current scenario selection.')
                else:
                    if selected_scenario_id == '__base__':
                        st.info('Curve comparison is available for shocked scenarios. Select a non-base scenario.')
                    else:
                        curve_fig = build_curve_comparison_figure(
                            curve_points=curve_points,
                            tenor_paths=tenor_paths,
                            scenario_id=selected_scenario_id,
                            scenario_label=selected_label,
                        )
                        st.plotly_chart(curve_fig, width='stretch', key='overview_rate_curve_chart')

        st.caption('Export Executive Pack')
        export_signature = (
            f'{input_path}|{selected_product}|{pd.Timestamp(t1).date().isoformat()}|'
            f'{pd.Timestamp(t2).date().isoformat()}|{ui.get("growth_mode", "constant")}|'
            f'{float(ui.get("growth_monthly_value", 0.0))}|{active_ids_json}'
        )
        if st.session_state.get('overview_export_signature') != export_signature:
            st.session_state['overview_export_signature'] = export_signature
            st.session_state.pop('overview_export_bytes', None)
            st.session_state.pop('overview_export_filename', None)

        c_export_1, c_export_2 = st.columns([1, 2])
        with c_export_1:
            generate_export = st.button('Generate Executive Excel', key='overview_generate_export')
        if generate_export:
            try:
                runoff_delta_export = _cached_runoff_delta(input_path, t1, t2, selected_product)
                calendar_runoff_export = _cached_calendar_runoff(input_path, t1, t2, True, selected_product)
                export_context = build_export_context(
                    path=input_path,
                    product=selected_product,
                    t1=t1,
                    t2=t2,
                    growth_mode=ui.get('growth_mode', 'constant'),
                    growth_monthly_value=float(ui.get('growth_monthly_value', 0.0)),
                    scenario_payload_json=scenario_payload_json,
                    active_ids_json=active_ids_json,
                    overview_metrics=metrics_table,
                    overview_delta_kpis=overview_delta_kpis,
                    yearly_summary=yearly_summary,
                    monthly_base=monthly_base,
                    monthly_scenarios=monthly_scenarios,
                    calendar_runoff=calendar_runoff_export,
                    runoff_delta=runoff_delta_export,
                    curve_df=curve_df,
                )
                workbook_bytes = build_export_workbook_bytes(
                    export_context,
                    workbook_title='NII Executive Export Pack',
                )
                st.session_state['overview_export_bytes'] = workbook_bytes
                st.session_state['overview_export_filename'] = default_export_filename(selected_product, t1, t2)
                st.success('Executive export generated. Use the download button to save the workbook.')
            except Exception as exc:
                st.error(f'Failed to generate executive export: {exc}')

        if st.session_state.get('overview_export_bytes') is not None:
            with c_export_2:
                st.download_button(
                    label='Download Executive Pack (.xlsx)',
                    data=st.session_state['overview_export_bytes'],
                    file_name=st.session_state.get('overview_export_filename', default_export_filename(selected_product, t1, t2)),
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='overview_download_export',
                )

        # Keep builder at the very bottom of Overview.
        st.markdown('---')
        custom_scenarios_df = scenario_universe_df[
            scenario_universe_df['scenario_id'].astype(str).str.startswith('custom_')
        ].copy()
        active_ids_default = active_scenarios_df['scenario_id'].astype(str).tolist()
        builder_actions = render_rate_scenario_builder(
            scenario_universe_df=scenario_universe_df,
            custom_scenarios_df=custom_scenarios_df,
            active_scenario_ids=active_ids_default,
            curve_df=curve_df,
            preview_anchor_date=pd.Timestamp(t2) if t2 is not None else pd.Timestamp(t1),
        )
        builder_error = str(builder_actions.get('error', '') or '')
        if builder_error:
            st.error(builder_error)

        add_spec = builder_actions.get('add_scenario')
        if isinstance(add_spec, dict):
            is_valid, msg = validate_custom_scenario(add_spec)
            labels_lower = {str(x).strip().lower() for x in scenario_universe_df['scenario_label'].astype(str).tolist()}
            ids = {str(x).strip() for x in scenario_universe_df['scenario_id'].astype(str).tolist()}
            if not is_valid:
                st.error(msg)
            elif str(add_spec['scenario_label']).strip().lower() in labels_lower:
                st.error(f'Scenario name `{add_spec["scenario_label"]}` already exists.')
            elif str(add_spec['scenario_id']).strip() in ids:
                st.error(
                    f'Scenario id `{add_spec["scenario_id"]}` already exists (name collision after slug sanitization).'
                )
            else:
                updated_payload = add_custom_scenario(scenario_store_payload, add_spec)
                try:
                    save_scenario_store(str(scenario_store_path), updated_payload)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f'Failed to save scenario store: {exc}')

        delete_id = builder_actions.get('delete_scenario_id')
        if delete_id:
            updated_payload = delete_custom_scenario(scenario_store_payload, str(delete_id))
            try:
                save_scenario_store(str(scenario_store_path), updated_payload)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f'Failed to update scenario store: {exc}')

        set_active_ids = builder_actions.get('set_active_ids')
        if isinstance(set_active_ids, list):
            if not set_active_ids:
                st.warning('At least one active scenario is required.')
            elif set(set_active_ids) != set(active_ids_default):
                updated_payload = dict(scenario_store_payload)
                updated_payload['active_scenario_ids'] = [str(x) for x in set_active_ids]
                try:
                    save_scenario_store(str(scenario_store_path), updated_payload)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f'Failed to persist active scenario set: {exc}')

    elif selected_section == 'Daily':
        st.caption(
            'Top chart uses a stacked-style total bar: month-start base plus shaded delta versus first day (green up, red down). '
            'Interest view also overlays cumulative Total in the top chart. '
            'Bottom chart shows Added and Matured breakdown plus cumulative Added+Matured and cumulative Added/Matured lines '
            'and includes a month-end cumulative decomposition summary table below.'
        )
        daily_t1 = _cached_daily_interest(input_path, t1, selected_product)
        daily_t2 = _cached_daily_interest(input_path, t2, selected_product)
        render_daily_interest_chart(daily_t1, daily_t2, label_t1=f'T1 {t1.date()}', label_t2=f'T2 {t2.date()}')

    elif selected_section == 'Runoff':
        runoff_ui = render_runoff_controls(
            default_basis=ui['runoff_decomposition_basis'],
            scenario_options=active_scenarios_df[['scenario_id', 'scenario_label']].to_dict(orient='records'),
        )
        full_ui = {**ui, **runoff_ui}
        scenario_monthly_for_runoff = pd.DataFrame()
        if (
            str(full_ui.get('runoff_chart_view', '')) == 'Effective Interest Decomposition (Refill/Growth)'
            and bool(full_ui.get('runoff_scenario_compare_enabled', False))
        ):
            if str(full_ui.get('runoff_decomposition_basis', 'T2')) != 'T2':
                st.info('Scenario comparison in this chart is available for T2 basis only.')
            else:
                scenario_result = _cached_rate_scenario_projection(
                    input_path,
                    t1,
                    t2,
                    selected_product,
                    'T2',
                    ui.get('growth_mode', 'constant'),
                    float(ui.get('growth_monthly_value', 0.0)),
                    scenario_payload_json,
                    active_ids_json,
                    240,
                )
                if isinstance(scenario_result, dict):
                    scenario_monthly_for_runoff = scenario_result.get('monthly_scenarios', pd.DataFrame())

        if ui['runoff_display_mode'] == 'Aligned Buckets (Remaining Maturity)':
            runoff_delta = _cached_runoff_delta(input_path, t1, t2, selected_product)
            selected_view = render_runoff_delta_charts(
                runoff_delta,
                key_prefix='runoff_aligned',
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=None,
                curve_df=curve_df,
                ui_state=full_ui,
                scenario_monthly=scenario_monthly_for_runoff,
                scenario_label_map=scenario_id_to_label,
            )
            bucket_table = _build_runoff_bucket_table(runoff_delta, t1, t2)
            bucket_table = _filter_runoff_bucket_table_by_view(bucket_table, selected_view)
            st.dataframe(style_numeric_table(bucket_table), width='stretch')
        else:
            basis = ui['runoff_decomposition_basis']
            anchor = pd.Timestamp(t1 if basis == 'T1' else t2) + pd.offsets.MonthEnd(0)
            timeframe_end = anchor + pd.offsets.MonthEnd(240)
            needs_effective = 'Effective Interest' in str(full_ui.get('runoff_chart_view', ''))
            needs_shifted_refill = 'Refill' in str(full_ui.get('runoff_chart_view', ''))
            calendar_runoff = _cached_calendar_runoff(input_path, t1, t2, needs_effective, selected_product)
            runoff_delta_for_refill = _cached_runoff_delta(input_path, t1, t2, selected_product) if needs_shifted_refill else None
            filtered = calendar_runoff[
                (calendar_runoff['calendar_month_end'] >= anchor)
                & (calendar_runoff['calendar_month_end'] <= timeframe_end)
            ].copy()

            selected_view = render_calendar_runoff_charts(
                filtered,
                label_t1=f'T1 {t1.date()}',
                label_t2=f'T2 {t2.date()}',
                key_prefix='runoff_calendar',
                runoff_compare_df=runoff_delta_for_refill,
                deals_df=deals_df,
                basis_t1=t1,
                basis_t2=t2,
                refill_logic_df=None,
                curve_df=curve_df,
                ui_state=full_ui,
                scenario_monthly=scenario_monthly_for_runoff,
                scenario_label_map=scenario_id_to_label,
            )
            calendar_table = _build_runoff_calendar_table(filtered, t1, t2)
            calendar_table = _filter_runoff_calendar_table_by_view(calendar_table, selected_view)
            st.dataframe(style_numeric_table(calendar_table), width='stretch')

    else:
        diff = _cached_compare_month_ends(input_path, t1, t2, selected_product)
        render_deal_diff_tables(diff, compact_mode=True)


if __name__ == '__main__':
    main()

