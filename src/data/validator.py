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
EXTERNAL_PROFILE_REQUIRED_COLUMNS = [
    'product',
    'external_product_type',
    'calendar_month_end',
    'external_notional',
    'repricing_tenor_months',
    'manual_rate',
]
EXTERNAL_MODEL_MANUAL_PROFILE = 'manual_profile'
EXTERNAL_MODEL_DAILY_DUE_SAVINGS = 'daily_due_savings'


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    cols = set(df.columns)
    return [col for col in required if col not in cols]


def validate_deals(df: pd.DataFrame) -> list[str]:
    """Validate normalized deals data and return non-fatal warnings."""
    missing = _missing_columns(df, DEAL_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f'Missing required deal columns: {missing}')

    warnings: list[str] = []

    if 'product' in df.columns:
        if df.duplicated(subset=['product', 'deal_id']).any():
            raise ValueError('Duplicate deal_id values found within product.')
    elif df['deal_id'].duplicated().any():
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


def validate_external_profile(df: pd.DataFrame) -> list[str]:
    """Validate normalized external profile data and return non-fatal warnings."""
    missing = _missing_columns(df, EXTERNAL_PROFILE_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f'Missing required external profile columns: {missing}')

    warnings: list[str] = []
    work = df.copy()

    if work[['product', 'external_product_type']].isna().any().any():
        raise ValueError('External profile requires non-null product and external_product_type.')

    if not pd.api.types.is_datetime64_any_dtype(work['calendar_month_end']):
        raise ValueError('Column calendar_month_end must be datetime64 dtype.')
    for col in ['external_notional', 'repricing_tenor_months', 'manual_rate']:
        if not pd.api.types.is_numeric_dtype(work[col]):
            raise ValueError(f'Column {col} must be numeric dtype.')

    work['external_product_type'] = work['external_product_type'].astype(str).str.strip().str.lower()

    manual = work[work['external_product_type'] == EXTERNAL_MODEL_MANUAL_PROFILE].copy()
    if not manual.empty:
        required = ['product', 'external_product_type', 'calendar_month_end', 'repricing_tenor_months', 'manual_rate']
        if manual[required].isna().any().any():
            raise ValueError(
                'Manual-profile external rows require product, external_product_type, calendar_month_end, repricing_tenor_months, and manual_rate.'
            )
        if manual.duplicated(subset=['product', 'external_product_type', 'calendar_month_end']).any():
            raise ValueError(
                'Duplicate external profile rows found for (product, external_product_type, calendar_month_end).'
            )
        non_positive_tenor = int((pd.to_numeric(manual['repricing_tenor_months'], errors='coerce') <= 0).sum())
        if non_positive_tenor:
            warnings.append(f'{non_positive_tenor} manual-profile rows have repricing tenor <= 0 months.')
        high_rate = int((pd.to_numeric(manual['manual_rate'], errors='coerce').abs() > 1.0).sum())
        if high_rate:
            warnings.append(f'{high_rate} manual-profile rows have manual_rate magnitude > 100%.')

    savings = work[work['external_product_type'] == EXTERNAL_MODEL_DAILY_DUE_SAVINGS].copy()
    if not savings.empty:
        if savings.duplicated(subset=['product', 'external_product_type']).any():
            raise ValueError('Duplicate daily-due-savings external rows found for (product, external_product_type).')

    other = work[~work['external_product_type'].isin([EXTERNAL_MODEL_MANUAL_PROFILE, EXTERNAL_MODEL_DAILY_DUE_SAVINGS])].copy()
    if not other.empty:
        if other.duplicated(subset=['product', 'external_product_type', 'calendar_month_end']).any():
            raise ValueError(
                'Duplicate external profile rows found for (product, external_product_type, calendar_month_end).'
            )

    zero_notional = int((pd.to_numeric(work['external_notional'], errors='coerce').fillna(0.0) == 0).sum())
    if zero_notional:
        warnings.append(f'{zero_notional} external profile rows have zero or missing external_notional.')

    return warnings
