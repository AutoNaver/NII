"""Input data validation for deals and curve tables."""

from __future__ import annotations

import pandas as pd

DEAL_REQUIRED_COLUMNS = [
    'deal_id',
    'trade_date',
    'value_date',
    'maturity_date',
    'notional',
    'coupon',
]

CURVE_REQUIRED_COLUMNS = ['ir_date', 'ir_tenor', 'rate']


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    cols = set(df.columns)
    return [col for col in required if col not in cols]


def validate_deals(df: pd.DataFrame) -> list[str]:
    """Validate normalized deals data and return non-fatal warnings."""
    missing = _missing_columns(df, DEAL_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f'Missing required deal columns: {missing}')

    warnings: list[str] = []

    if df['deal_id'].duplicated().any():
        raise ValueError('Duplicate deal_id values found.')

    for col in ['trade_date', 'value_date', 'maturity_date']:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f'Column {col} must be datetime64 dtype.')

    for col in ['notional', 'coupon']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f'Column {col} must be numeric dtype.')

    if df[DEAL_REQUIRED_COLUMNS].isna().any().any():
        raise ValueError('Deals contain nulls in required columns.')

    invalid_life = int((df['maturity_date'] <= df['value_date']).sum())
    if invalid_life:
        warnings.append(
            f'{invalid_life} deals have maturity_date <= value_date and will be excluded.'
        )

    zero_notional = (df['notional'] == 0).sum()
    if zero_notional:
        warnings.append(f'{zero_notional} deals have zero notional.')

    high_coupon = (df['coupon'].abs() > 1.0).sum()
    if high_coupon:
        warnings.append(f'{high_coupon} deals have coupon magnitude > 100%.')

    return warnings


def validate_curve(df: pd.DataFrame) -> list[str]:
    """Validate normalized interest curve data and return non-fatal warnings."""
    missing = _missing_columns(df, CURVE_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f'Missing required curve columns: {missing}')

    warnings: list[str] = []

    if df[CURVE_REQUIRED_COLUMNS].isna().any().any():
        raise ValueError('Curve contains nulls in required columns.')

    if not pd.api.types.is_datetime64_any_dtype(df['ir_date']):
        raise ValueError('Column ir_date must be datetime64 dtype.')

    for col in ['ir_tenor', 'rate']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f'Column {col} must be numeric dtype.')

    dupes = df.duplicated(subset=['ir_date', 'ir_tenor']).sum()
    if dupes:
        warnings.append(f'{dupes} duplicate curve points found (ir_date, ir_tenor).')

    return warnings
