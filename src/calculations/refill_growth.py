"""Refill and growth helper calculations shared by dashboard views."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shifted_portfolio_refill_weights(runoff_compare_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Infer refill allocation weights from one-month shifted portfolio delta."""
    if runoff_compare_df is None or runoff_compare_df.empty:
        return None
    needed = {'remaining_maturity_months', 'abs_notional_d1', 'abs_notional_d2'}
    if not needed.issubset(runoff_compare_df.columns):
        return None

    w = runoff_compare_df.loc[:, ['remaining_maturity_months', 'abs_notional_d1', 'abs_notional_d2']].copy()
    w['remaining_maturity_months'] = pd.to_numeric(w['remaining_maturity_months'], errors='coerce')
    w['abs_notional_d1'] = pd.to_numeric(w['abs_notional_d1'], errors='coerce')
    w['abs_notional_d2'] = pd.to_numeric(w['abs_notional_d2'], errors='coerce')
    w = w.dropna().sort_values('remaining_maturity_months').drop_duplicates('remaining_maturity_months', keep='last')
    if w.empty:
        return None

    d1 = w['abs_notional_d1'].astype(float).to_numpy()
    d2 = w['abs_notional_d2'].astype(float).to_numpy()
    d1_shift = np.roll(d1, -1)
    d1_shift[-1] = 0.0
    shift_delta = np.clip(d2 - d1_shift, a_min=0.0, a_max=None)

    total = float(shift_delta.sum())
    if total <= 1e-12:
        fallback = np.clip(d2, a_min=0.0, a_max=None)
        total = float(fallback.sum())
        if total <= 1e-12:
            return None
        shift_delta = fallback

    out = pd.DataFrame(
        {
            'tenor': w['remaining_maturity_months'].astype(int).to_numpy(),
            'shift_delta': shift_delta,
        }
    )
    out = out[out['shift_delta'] > 1e-12].copy()
    if out.empty:
        return None
    out['weight'] = out['shift_delta'] / float(out['shift_delta'].sum())
    return out.reset_index(drop=True)


def t0_portfolio_weights(runoff_compare_df: pd.DataFrame | None, *, basis: str) -> pd.Series | None:
    """Portfolio tenor distribution for the selected T0 basis."""
    if runoff_compare_df is None or runoff_compare_df.empty:
        return None
    col = 'abs_notional_d1' if str(basis).strip().upper() == 'T1' else 'abs_notional_d2'
    needed = {'remaining_maturity_months', col}
    if not needed.issubset(runoff_compare_df.columns):
        return None

    w = runoff_compare_df.loc[:, ['remaining_maturity_months', col]].copy()
    w['remaining_maturity_months'] = pd.to_numeric(w['remaining_maturity_months'], errors='coerce')
    w[col] = pd.to_numeric(w[col], errors='coerce').clip(lower=0.0)
    w = w.dropna().groupby('remaining_maturity_months', as_index=False).sum()
    if w.empty:
        return None
    total = float(w[col].sum())
    if total <= 1e-12:
        return None
    s = w.set_index('remaining_maturity_months')[col].astype(float)
    return s / total


def growth_outstanding_profile(
    *,
    growth_flow: pd.Series,
    runoff_compare_df: pd.DataFrame | None,
    basis: str,
) -> pd.Series:
    """Convert monthly growth injections into an outstanding growth profile."""
    flow = growth_flow.astype(float)
    if flow.empty:
        return flow
    if float(flow.abs().sum()) <= 1e-12:
        return pd.Series(0.0, index=flow.index, dtype=float)

    weights = t0_portfolio_weights(runoff_compare_df, basis=basis)
    if weights is None or weights.empty:
        return flow.cumsum()

    tenor = weights.index.astype(int).to_numpy()
    w = weights.astype(float).to_numpy()
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return flow.cumsum()
    w = w / w_sum

    n = len(flow)
    survival = np.zeros(n, dtype=float)
    for lag in range(n):
        survival[lag] = float(w[tenor >= lag].sum())

    outstanding = np.convolve(flow.to_numpy(dtype=float), survival, mode='full')[:n]
    return pd.Series(outstanding, index=flow.index, dtype=float)


def compute_refill_growth_components(
    *,
    cumulative_notional: pd.Series,
    growth_mode: str,
    monthly_growth_amount: float,
) -> dict[str, pd.Series]:
    """Compute signed refill and growth requirements against a cumulative notional profile."""
    if cumulative_notional.empty:
        empty = pd.Series(dtype=float, index=cumulative_notional.index)
        return {
            'target_total': empty,
            'total_required': empty,
            'refill_required': empty,
            'growth_required': empty,
        }

    base_level = float(cumulative_notional.iloc[0])
    idx = cumulative_notional.index
    if abs(base_level) <= 1e-12:
        non_zero = cumulative_notional[cumulative_notional.abs() > 1e-12]
        if not non_zero.empty:
            base_level = float(non_zero.iloc[0])
    direction = -1.0 if base_level < 0.0 else 1.0

    mode = str(growth_mode or 'constant').strip().lower()
    growth_per_step = float(monthly_growth_amount) if mode == 'user_defined' else 0.0
    growth_per_step = max(growth_per_step, 0.0)

    refill_magnitude = (
        direction * (pd.Series(base_level, index=idx, dtype=float) - cumulative_notional)
    ).clip(lower=0.0)
    refill_required = direction * refill_magnitude
    growth_required = pd.Series(direction * growth_per_step, index=idx, dtype=float)
    total_required = refill_required + growth_required
    target_total = cumulative_notional + total_required

    first = idx[0]
    refill_required.loc[first] = 0.0
    if mode == 'user_defined':
        growth_required.loc[first] = 0.0
    else:
        growth_required.loc[:] = 0.0
    total_required = refill_required + growth_required
    target_total = cumulative_notional + total_required

    return {
        'target_total': target_total,
        'total_required': total_required,
        'refill_required': refill_required,
        'growth_required': growth_required,
    }


def compute_refill_growth_components_anchor_safe(
    *,
    cumulative_notional: pd.Series,
    growth_mode: str,
    monthly_growth_amount: float,
) -> dict[str, pd.Series]:
    """Compute refill/growth requirements while tolerating leading anchor zeros."""
    series = cumulative_notional.astype(float)
    non_zero_pos = np.flatnonzero(series.abs().to_numpy() > 1e-12)
    if non_zero_pos.size == 0:
        return compute_refill_growth_components(
            cumulative_notional=series,
            growth_mode=growth_mode,
            monthly_growth_amount=monthly_growth_amount,
        )

    start = int(non_zero_pos[0])
    active = series.iloc[start:]
    components = compute_refill_growth_components(
        cumulative_notional=active,
        growth_mode=growth_mode,
        monthly_growth_amount=monthly_growth_amount,
    )

    out: dict[str, pd.Series] = {}
    for key, values in components.items():
        aligned = pd.Series(0.0, index=series.index, dtype=float)
        aligned.loc[active.index] = values.astype(float).to_numpy()
        out[key] = aligned
    return out
