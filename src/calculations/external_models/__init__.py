"""External position modeling framework for Overview analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.calculations.rate_scenarios import (
    interpolate_curve_rate,
    normalize_scenarios_df,
    shock_path_bps_for_spec,
    summarize_yearly_delta,
)

MONTH_FRACTION_30_360 = 30.0 / 360.0
EXTERNAL_MODEL_KEY_MANUAL_PROFILE = 'manual_profile'
EXTERNAL_MODEL_KEY_DAILY_DUE_SAVINGS = 'daily_due_savings'
EXTERNAL_PROFILE_COLUMNS = [
    'product',
    'external_product_type',
    'calendar_month_end',
    'external_notional',
    'repricing_tenor_months',
    'manual_rate',
]
DEFAULT_DAILY_DUE_SAVINGS_SETTINGS = {
    'initial_client_rate': 0.01,
    'mean_reversion': 0.25,
    'a': 0.0,
    'b': 0.5,
    'c': 0.1,
}


def empty_external_profile() -> pd.DataFrame:
    """Return an empty external profile frame with stable columns."""
    return pd.DataFrame(columns=EXTERNAL_PROFILE_COLUMNS)


def normalize_external_profile(profile_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize optional external profile rows to a stable schema."""
    if profile_df is None or profile_df.empty:
        return empty_external_profile()

    out = profile_df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    for col in EXTERNAL_PROFILE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out['product'] = out['product'].fillna('Default').astype(str).str.strip()
    out.loc[out['product'].isin(['', 'nan', 'None']), 'product'] = 'Default'
    out['external_product_type'] = out['external_product_type'].fillna(EXTERNAL_MODEL_KEY_MANUAL_PROFILE).astype(str).str.strip()
    out.loc[out['external_product_type'].isin(['', 'nan', 'None']), 'external_product_type'] = EXTERNAL_MODEL_KEY_MANUAL_PROFILE
    out['calendar_month_end'] = pd.to_datetime(out['calendar_month_end'], errors='coerce')
    valid_dates = out['calendar_month_end'].notna()
    out.loc[valid_dates, 'calendar_month_end'] = out.loc[valid_dates, 'calendar_month_end'] + pd.offsets.MonthEnd(0)
    out['external_notional'] = pd.to_numeric(out['external_notional'], errors='coerce')
    out['repricing_tenor_months'] = pd.to_numeric(out['repricing_tenor_months'], errors='coerce')
    out['manual_rate'] = pd.to_numeric(out['manual_rate'], errors='coerce')
    return out[EXTERNAL_PROFILE_COLUMNS].reset_index(drop=True)


def filter_external_profile_by_product(profile_df: pd.DataFrame | None, product: str | None) -> pd.DataFrame:
    """Filter normalized external profile rows by product."""
    profile = normalize_external_profile(profile_df)
    if product is None:
        return profile
    return profile[profile['product'] == str(product)].copy().reset_index(drop=True)


def normalize_external_settings(
    model_key: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Normalize model settings to stable numeric values."""
    key = str(model_key or '').strip().lower()
    if key != EXTERNAL_MODEL_KEY_DAILY_DUE_SAVINGS:
        return {}
    merged = dict(DEFAULT_DAILY_DUE_SAVINGS_SETTINGS)
    for name, value in (settings or {}).items():
        if name in merged:
            try:
                merged[name] = float(value)
            except Exception:
                continue
    merged['mean_reversion'] = float(min(max(merged['mean_reversion'], 0.0), 1.0))
    return merged


def available_external_model_types(profile_df: pd.DataFrame | None) -> list[str]:
    """Return sorted model keys present in the profile."""
    profile = normalize_external_profile(profile_df)
    if profile.empty:
        return []
    types = profile['external_product_type'].dropna().astype(str).str.strip().tolist()
    return sorted({t for t in types if t})


def _sign_from_profile(profile_df: pd.DataFrame, *, default_sign: float) -> float:
    values = pd.to_numeric(profile_df.get('external_notional'), errors='coerce').dropna()
    values = values[values.abs() > 1e-12]
    if values.empty:
        return float(default_sign)
    return float(np.sign(values.iloc[0]))


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    val = pd.to_numeric(values, errors='coerce').fillna(0.0).astype(float)
    w = pd.to_numeric(weights, errors='coerce').fillna(0.0).astype(float).abs()
    total = float(w.sum())
    if total <= 1e-12:
        return 0.0
    return float((val * w).sum() / total)


def _empty_month_frame(month_ends: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    me = pd.to_datetime(pd.Series(month_ends)).reset_index(drop=True) + pd.offsets.MonthEnd(0)
    frame = pd.DataFrame({'calendar_month_end': me, 'month_idx': np.arange(len(me), dtype=int)})
    return frame


def _build_zero_result(month_ends: pd.Series | pd.DatetimeIndex, scenarios: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    frame = _empty_month_frame(month_ends)
    sdef = normalize_scenarios_df(scenarios if scenarios is not None else None)
    monthly_base = frame.copy()
    monthly_base['total_active_notional'] = 0.0
    monthly_base['weighted_avg_coupon'] = 0.0
    monthly_base['base_total_interest'] = 0.0
    monthly_base['active_deal_count'] = 0
    monthly_base['accrued_interest_eur'] = 0.0
    scenario_rows: list[pd.DataFrame] = []
    for rec in sdef[['scenario_id', 'scenario_label']].to_dict(orient='records'):
        out = frame.copy()
        out['scenario_id'] = str(rec['scenario_id'])
        out['scenario_label'] = str(rec['scenario_label'])
        out['shock_bps'] = 0.0
        out['base_total_interest'] = 0.0
        out['shocked_total_interest'] = 0.0
        out['delta_vs_base'] = 0.0
        out['cumulative_delta'] = 0.0
        scenario_rows.append(out)
    monthly_scenarios = pd.concat(scenario_rows, ignore_index=True) if scenario_rows else pd.DataFrame()
    yearly_summary = summarize_yearly_delta(monthly_scenarios, scenarios=sdef, years=5)
    return {
        'scenarios': sdef.reset_index(drop=True),
        'monthly_base': monthly_base.reset_index(drop=True),
        'monthly_scenarios': monthly_scenarios.reset_index(drop=True),
        'yearly_summary': yearly_summary.reset_index(drop=True),
        'metadata': {'model_types': [], 'volume_source': 'mirror_internal_absolute'},
    }


class ExternalPositionModel(Protocol):
    """Protocol for pluggable external position models."""

    model_key: str

    def simulate(
        self,
        *,
        month_ends: pd.Series | pd.DatetimeIndex,
        mirrored_notional: pd.Series,
        curve_df: pd.DataFrame,
        anchor_date: pd.Timestamp,
        scenarios: pd.DataFrame | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return monthly and yearly scenario outputs for the external side."""


@dataclass
class ManualExternalProfileModel:
    """Manual-rate external profile model."""

    profile_df: pd.DataFrame
    model_key: str = EXTERNAL_MODEL_KEY_MANUAL_PROFILE

    def simulate(
        self,
        *,
        month_ends: pd.Series | pd.DatetimeIndex,
        mirrored_notional: pd.Series,
        curve_df: pd.DataFrame,
        anchor_date: pd.Timestamp,
        scenarios: pd.DataFrame | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, pd.DataFrame]:
        frame = _empty_month_frame(month_ends)
        sdef = normalize_scenarios_df(scenarios if scenarios is not None else None)
        profile = normalize_external_profile(self.profile_df)
        profile = profile.dropna(subset=['calendar_month_end', 'repricing_tenor_months', 'manual_rate']).copy()
        if profile.empty:
            return _build_zero_result(frame['calendar_month_end'], sdef)

        sign = _sign_from_profile(profile, default_sign=1.0)
        mirrored_abs = pd.Series(mirrored_notional, dtype=float).abs().reset_index(drop=True)
        profile = profile.sort_values('calendar_month_end').drop_duplicates('calendar_month_end', keep='last')
        base = frame.merge(
            profile[['calendar_month_end', 'repricing_tenor_months', 'manual_rate']],
            on='calendar_month_end',
            how='left',
        )
        base['repricing_tenor_months'] = pd.to_numeric(base['repricing_tenor_months'], errors='coerce').round().clip(lower=1)
        base['manual_rate'] = pd.to_numeric(base['manual_rate'], errors='coerce')
        base['total_active_notional'] = sign * mirrored_abs.to_numpy(dtype=float)
        base['weighted_avg_coupon'] = base['manual_rate'].fillna(0.0).astype(float)
        base['base_total_interest'] = (base['total_active_notional'] * base['weighted_avg_coupon'] * MONTH_FRACTION_30_360).astype(float)
        base['active_deal_count'] = np.where(base['total_active_notional'].abs() > 1e-12, 1, 0)
        base['accrued_interest_eur'] = 0.0
        base['repricing_tenor_months'] = base['repricing_tenor_months'].fillna(1.0)

        scenario_rows: list[pd.DataFrame] = []
        for spec in sdef.to_dict(orient='records'):
            out = base[['month_idx', 'calendar_month_end', 'base_total_interest', 'total_active_notional']].copy()
            shock_bps = pd.Series(0.0, index=out.index, dtype=float)
            valid_rate_mask = base['manual_rate'].notna()
            if valid_rate_mask.any():
                shock_bps.loc[valid_rate_mask] = shock_path_bps_for_spec(
                    spec,
                    month_idx=base.loc[valid_rate_mask, 'month_idx'],
                    tenor_months=base.loc[valid_rate_mask, 'repricing_tenor_months'],
                ).astype(float).to_numpy()
            shocked_rate = base['weighted_avg_coupon'] + (shock_bps / 10000.0)
            out['scenario_id'] = str(spec['scenario_id'])
            out['scenario_label'] = str(spec['scenario_label'])
            out['shock_bps'] = shock_bps
            out['shocked_total_interest'] = out['total_active_notional'] * shocked_rate * MONTH_FRACTION_30_360
            out['delta_vs_base'] = out['shocked_total_interest'] - out['base_total_interest']
            out['cumulative_delta'] = out['delta_vs_base'].cumsum()
            scenario_rows.append(out.drop(columns=['total_active_notional']))

        monthly_scenarios = pd.concat(scenario_rows, ignore_index=True) if scenario_rows else pd.DataFrame()
        yearly_summary = summarize_yearly_delta(monthly_scenarios, scenarios=sdef, years=5)
        monthly_base = base[
            [
                'month_idx',
                'calendar_month_end',
                'total_active_notional',
                'weighted_avg_coupon',
                'base_total_interest',
                'active_deal_count',
                'accrued_interest_eur',
            ]
        ].reset_index(drop=True)
        return {
            'scenarios': sdef.reset_index(drop=True),
            'monthly_base': monthly_base,
            'monthly_scenarios': monthly_scenarios.reset_index(drop=True),
            'yearly_summary': yearly_summary.reset_index(drop=True),
            'metadata': {'model_type': self.model_key, 'volume_source': 'mirror_internal_absolute'},
        }


@dataclass
class DailyDueSavingsModel:
    """Daily due savings external model with equilibrium-rate mean reversion."""

    profile_df: pd.DataFrame
    model_key: str = EXTERNAL_MODEL_KEY_DAILY_DUE_SAVINGS

    def simulate(
        self,
        *,
        month_ends: pd.Series | pd.DatetimeIndex,
        mirrored_notional: pd.Series,
        curve_df: pd.DataFrame,
        anchor_date: pd.Timestamp,
        scenarios: pd.DataFrame | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, pd.DataFrame]:
        frame = _empty_month_frame(month_ends)
        sdef = normalize_scenarios_df(scenarios if scenarios is not None else None)
        profile = normalize_external_profile(self.profile_df)
        cfg = normalize_external_settings(self.model_key, settings)
        sign = _sign_from_profile(profile, default_sign=-1.0)
        mirrored_abs = pd.Series(mirrored_notional, dtype=float).abs().reset_index(drop=True)
        total_notional = sign * mirrored_abs.to_numpy(dtype=float)

        base_1m = float(interpolate_curve_rate(curve_df, pd.Timestamp(anchor_date), 1))
        base_10y = float(interpolate_curve_rate(curve_df, pd.Timestamp(anchor_date), 120))
        equilibrium_base = (
            float(cfg['a'])
            + float(cfg['b']) * base_1m
            + float(cfg['c']) * (base_10y - base_1m)
        )

        client_rate = np.zeros(len(frame), dtype=float)
        equilibrium_path = np.full(len(frame), equilibrium_base, dtype=float)
        client_rate[0] = float(cfg['initial_client_rate'])
        for i in range(1, len(frame)):
            prev = client_rate[i - 1]
            client_rate[i] = prev + (float(cfg['mean_reversion']) * (equilibrium_path[i] - prev))

        monthly_base = frame.copy()
        monthly_base['total_active_notional'] = total_notional
        monthly_base['equilibrium_rate'] = equilibrium_path
        monthly_base['client_rate'] = client_rate
        monthly_base['weighted_avg_coupon'] = client_rate
        monthly_base['base_total_interest'] = monthly_base['total_active_notional'] * monthly_base['client_rate'] * MONTH_FRACTION_30_360
        monthly_base['active_deal_count'] = np.where(np.abs(total_notional) > 1e-12, 1, 0)
        monthly_base['accrued_interest_eur'] = 0.0

        scenario_rows: list[pd.DataFrame] = []
        for spec in sdef.to_dict(orient='records'):
            shock_1m = shock_path_bps_for_spec(
                spec,
                month_idx=monthly_base['month_idx'],
                tenor_months=pd.Series(1.0, index=monthly_base.index),
            ).astype(float)
            shock_10y = shock_path_bps_for_spec(
                spec,
                month_idx=monthly_base['month_idx'],
                tenor_months=pd.Series(120.0, index=monthly_base.index),
            ).astype(float)
            one_m = base_1m + (shock_1m / 10000.0)
            ten_y = base_10y + (shock_10y / 10000.0)
            equilibrium_shocked = (
                float(cfg['a'])
                + float(cfg['b']) * one_m
                + float(cfg['c']) * (ten_y - one_m)
            ).astype(float)
            client_rate_shocked = np.zeros(len(frame), dtype=float)
            client_rate_shocked[0] = float(cfg['initial_client_rate'])
            for i in range(1, len(frame)):
                prev = client_rate_shocked[i - 1]
                client_rate_shocked[i] = prev + (
                    float(cfg['mean_reversion']) * (float(equilibrium_shocked.iloc[i]) - prev)
                )

            out = frame.copy()
            out['scenario_id'] = str(spec['scenario_id'])
            out['scenario_label'] = str(spec['scenario_label'])
            out['shock_bps'] = ((shock_1m + shock_10y) / 2.0).astype(float)
            out['equilibrium_rate_shocked'] = equilibrium_shocked.astype(float).to_numpy()
            out['client_rate_shocked'] = client_rate_shocked
            out['base_total_interest'] = monthly_base['base_total_interest'].astype(float).to_numpy()
            out['shocked_total_interest'] = total_notional * client_rate_shocked * MONTH_FRACTION_30_360
            out['delta_vs_base'] = out['shocked_total_interest'] - out['base_total_interest']
            out['cumulative_delta'] = out['delta_vs_base'].cumsum()
            scenario_rows.append(out)

        monthly_scenarios = pd.concat(scenario_rows, ignore_index=True) if scenario_rows else pd.DataFrame()
        yearly_summary = summarize_yearly_delta(monthly_scenarios, scenarios=sdef, years=5)
        return {
            'scenarios': sdef.reset_index(drop=True),
            'monthly_base': monthly_base.reset_index(drop=True),
            'monthly_scenarios': monthly_scenarios.reset_index(drop=True),
            'yearly_summary': yearly_summary.reset_index(drop=True),
            'metadata': {
                'model_type': self.model_key,
                'volume_source': 'mirror_internal_absolute',
                'settings': cfg,
            },
        }


MODEL_REGISTRY: dict[str, type[ExternalPositionModel]] = {
    EXTERNAL_MODEL_KEY_MANUAL_PROFILE: ManualExternalProfileModel,
    EXTERNAL_MODEL_KEY_DAILY_DUE_SAVINGS: DailyDueSavingsModel,
}


def build_external_model(type_name: str, profile_df: pd.DataFrame) -> ExternalPositionModel:
    """Build an external model instance from registry."""
    model_key = str(type_name or EXTERNAL_MODEL_KEY_MANUAL_PROFILE).strip().lower()
    model_cls = MODEL_REGISTRY.get(model_key)
    if model_cls is None:
        raise ValueError(f'Unsupported external model type `{type_name}`.')
    return model_cls(profile_df=normalize_external_profile(profile_df))


def _model_shares(profile_df: pd.DataFrame) -> pd.Series:
    profile = normalize_external_profile(profile_df)
    if profile.empty:
        return pd.Series(dtype=float)
    weights = profile.groupby('external_product_type')['external_notional'].apply(
        lambda s: float(pd.to_numeric(s, errors='coerce').abs().sum())
    )
    weights = weights.astype(float)
    total = float(weights.sum())
    if total <= 1e-12:
        unique_types = sorted(profile['external_product_type'].dropna().astype(str).unique().tolist())
        if not unique_types:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / float(len(unique_types)), index=unique_types, dtype=float)
    return weights / total


def simulate_external_portfolio(
    *,
    profile_df: pd.DataFrame | None,
    month_ends: pd.Series | pd.DatetimeIndex,
    mirrored_notional: pd.Series,
    curve_df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    scenarios: pd.DataFrame | None,
    settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate the full external portfolio, aggregated across model types."""
    profile = normalize_external_profile(profile_df)
    frame = _empty_month_frame(month_ends)
    sdef = normalize_scenarios_df(scenarios if scenarios is not None else None)
    if frame.empty or profile.empty:
        return _build_zero_result(frame['calendar_month_end'], sdef)

    shares = _model_shares(profile)
    if shares.empty:
        return _build_zero_result(frame['calendar_month_end'], sdef)

    base_parts: list[pd.DataFrame] = []
    scenario_parts: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    mirrored_abs = pd.Series(mirrored_notional, dtype=float).abs().reset_index(drop=True)

    for model_key, share in shares.items():
        model_profile = profile[profile['external_product_type'].astype(str) == str(model_key)].copy()
        model = build_external_model(str(model_key), model_profile)
        model_result = model.simulate(
            month_ends=frame['calendar_month_end'],
            mirrored_notional=(mirrored_abs * float(share)),
            curve_df=curve_df,
            anchor_date=anchor_date,
            scenarios=sdef if not sdef.empty else None,
            settings=(settings_by_model or {}).get(str(model_key), {}),
        )
        part_base = model_result.get('monthly_base', pd.DataFrame()).copy()
        part_base['model_type'] = str(model_key)
        base_parts.append(part_base)

        part_scen = model_result.get('monthly_scenarios', pd.DataFrame()).copy()
        if not part_scen.empty:
            part_scen['model_type'] = str(model_key)
            scenario_parts.append(part_scen)

        md = dict(model_result.get('metadata', {}))
        md['model_type'] = str(model_key)
        md['share'] = float(share)
        metadata_rows.append(md)

    base_all = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    if base_all.empty:
        return _build_zero_result(frame['calendar_month_end'], sdef)
    base_all['abs_notional'] = base_all['total_active_notional'].abs()
    base_all['rate_num'] = base_all['weighted_avg_coupon'] * base_all['abs_notional']
    monthly_base = (
        base_all.groupby(['month_idx', 'calendar_month_end'], as_index=False)
        .agg(
            total_active_notional=('total_active_notional', 'sum'),
            base_total_interest=('base_total_interest', 'sum'),
            active_deal_count=('active_deal_count', 'sum'),
            accrued_interest_eur=('accrued_interest_eur', 'sum'),
            abs_notional=('abs_notional', 'sum'),
            rate_num=('rate_num', 'sum'),
        )
    )
    monthly_base['weighted_avg_coupon'] = 0.0
    nz = monthly_base['abs_notional'].abs() > 1e-12
    monthly_base.loc[nz, 'weighted_avg_coupon'] = monthly_base.loc[nz, 'rate_num'] / monthly_base.loc[nz, 'abs_notional']
    monthly_base = monthly_base[
        [
            'month_idx',
            'calendar_month_end',
            'total_active_notional',
            'weighted_avg_coupon',
            'base_total_interest',
            'active_deal_count',
            'accrued_interest_eur',
        ]
    ].sort_values('calendar_month_end').reset_index(drop=True)

    scen_all = pd.concat(scenario_parts, ignore_index=True) if scenario_parts else pd.DataFrame()
    if scen_all.empty:
        monthly_scenarios = pd.DataFrame()
    else:
        monthly_scenarios = (
            scen_all.groupby(['scenario_id', 'scenario_label', 'month_idx', 'calendar_month_end'], as_index=False)
            .agg(
                shock_bps=('shock_bps', 'mean'),
                base_total_interest=('base_total_interest', 'sum'),
                shocked_total_interest=('shocked_total_interest', 'sum'),
                delta_vs_base=('delta_vs_base', 'sum'),
            )
            .sort_values(['scenario_id', 'calendar_month_end'])
            .reset_index(drop=True)
        )
        monthly_scenarios['cumulative_delta'] = monthly_scenarios.groupby('scenario_id')['delta_vs_base'].cumsum()

    yearly_summary = summarize_yearly_delta(monthly_scenarios, scenarios=sdef, years=5)
    return {
        'scenarios': sdef.reset_index(drop=True),
        'monthly_base': monthly_base,
        'monthly_scenarios': monthly_scenarios.reset_index(drop=True),
        'yearly_summary': yearly_summary.reset_index(drop=True),
        'metadata': {
            'model_types': shares.index.tolist(),
            'shares': {str(k): float(v) for k, v in shares.to_dict().items()},
            'volume_source': 'mirror_internal_absolute',
            'models': metadata_rows,
        },
    }


def compute_external_monthly_snapshot(
    profile_df: pd.DataFrame | None,
    *,
    month_ends: pd.Series | pd.DatetimeIndex,
    mirrored_notional: pd.Series,
    curve_df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    settings_by_model: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return aggregated external monthly base metrics from mirrored internal notionals."""
    result = simulate_external_portfolio(
        profile_df=profile_df,
        month_ends=month_ends,
        mirrored_notional=mirrored_notional,
        curve_df=curve_df,
        anchor_date=anchor_date,
        scenarios=None,
        settings_by_model=settings_by_model,
    )
    base = result.get('monthly_base', pd.DataFrame()).copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                'calendar_month_end',
                'total_active_notional',
                'weighted_avg_coupon',
                'interest_paid_eur',
                'active_deal_count',
                'accrued_interest_eur',
            ]
        )
    base = base.rename(columns={'base_total_interest': 'interest_paid_eur'})
    return base[
        [
            'calendar_month_end',
            'total_active_notional',
            'weighted_avg_coupon',
            'interest_paid_eur',
            'active_deal_count',
            'accrued_interest_eur',
        ]
    ].sort_values('calendar_month_end').reset_index(drop=True)
