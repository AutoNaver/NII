"""Excel loader and schema normalization."""

from __future__ import annotations

import pandas as pd

from src.data.validator import validate_curve, validate_deals, validate_external_profile
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEALS_SHEET = 'Deal_Data'
CURVE_SHEET = 'Interest_Curve'
EXTERNAL_PROFILE_SHEET = 'External_Profile'

DEAL_COLUMN_MAP = {
    'trade_id': 'deal_id',
    'trade_date': 'trade_date',
    'value_date': 'value_date',
    'maturity_date': 'maturity_date',
    'notional': 'notional',
    'coupon': 'coupon',
}

CURVE_COLUMN_MAP = {
    'ir_date': 'ir_date',
    'ir_tenor': 'ir_tenor',
    'rate': 'rate',
}

EXTERNAL_PROFILE_COLUMN_MAP = {
    'product': 'product',
    'external_product_type': 'external_product_type',
    'calendar_month_end': 'calendar_month_end',
    'external_notional': 'external_notional',
    'repricing_tenor_months': 'repricing_tenor_months',
    'manual_rate': 'manual_rate',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _normalize_product_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'product' not in out.columns:
        out['product'] = 'Default'
    out['product'] = out['product'].fillna('Default').astype(str).str.strip()
    out.loc[out['product'].isin(['', 'nan', 'None']), 'product'] = 'Default'
    return out


def _load_external_profile(path: str) -> pd.DataFrame:
    try:
        external_raw = pd.read_excel(path, sheet_name=EXTERNAL_PROFILE_SHEET)
    except ValueError as exc:
        if EXTERNAL_PROFILE_SHEET.lower() in str(exc).lower() and 'worksheet' in str(exc).lower():
            return pd.DataFrame(columns=list(EXTERNAL_PROFILE_COLUMN_MAP.values()))
        raise

    external = _normalize_columns(external_raw).rename(columns=EXTERNAL_PROFILE_COLUMN_MAP)
    for col in EXTERNAL_PROFILE_COLUMN_MAP.values():
        if col not in external.columns:
            external[col] = pd.NA
    external = _normalize_product_column(external)
    if 'external_product_type' not in external.columns:
        external['external_product_type'] = 'manual_profile'
    external['external_product_type'] = external['external_product_type'].fillna('manual_profile').astype(str).str.strip()
    external.loc[external['external_product_type'].isin(['', 'nan', 'None']), 'external_product_type'] = 'manual_profile'
    external['calendar_month_end'] = pd.to_datetime(external['calendar_month_end'], errors='coerce')
    valid_dates = external['calendar_month_end'].notna()
    external.loc[valid_dates, 'calendar_month_end'] = external.loc[valid_dates, 'calendar_month_end'] + pd.offsets.MonthEnd(0)
    external['external_notional'] = pd.to_numeric(external['external_notional'], errors='coerce')
    external['repricing_tenor_months'] = pd.to_numeric(external['repricing_tenor_months'], errors='coerce')
    external['manual_rate'] = pd.to_numeric(external['manual_rate'], errors='coerce')

    warnings = validate_external_profile(external)
    for warning in warnings:
        LOGGER.warning(warning)
    return external


def load_input_workbook_with_external(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, normalize, and validate deal, curve, and optional external profile sheets."""
    deals_raw = pd.read_excel(path, sheet_name=DEALS_SHEET)
    curve_raw = pd.read_excel(path, sheet_name=CURVE_SHEET)

    deals = _normalize_columns(deals_raw).rename(columns=DEAL_COLUMN_MAP)
    curve = _normalize_columns(curve_raw).rename(columns=CURVE_COLUMN_MAP)

    deals['trade_date'] = pd.to_datetime(deals['trade_date'])
    deals['value_date'] = pd.to_datetime(deals['value_date'])
    deals['maturity_date'] = pd.to_datetime(deals['maturity_date'])
    deals['notional'] = pd.to_numeric(deals['notional'])
    deals['coupon'] = pd.to_numeric(deals['coupon'])
    deals = _normalize_product_column(deals)

    curve['ir_date'] = pd.to_datetime(curve['ir_date'])
    curve['ir_tenor'] = pd.to_numeric(curve['ir_tenor'])
    curve['rate'] = pd.to_numeric(curve['rate'])

    invalid_life_mask = deals['maturity_date'] <= deals['value_date']
    if invalid_life_mask.any():
        LOGGER.warning(
            '%s deals have maturity_date <= value_date and were excluded.',
            int(invalid_life_mask.sum()),
        )
        deals = deals.loc[~invalid_life_mask].copy()

    deal_warnings = validate_deals(deals)
    curve_warnings = validate_curve(curve)

    for warning in deal_warnings + curve_warnings:
        LOGGER.warning(warning)

    external = _load_external_profile(path)
    return deals, curve, external


def load_input_workbook(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, normalize, and validate deal and curve sheets from workbook."""
    deals, curve, _ = load_input_workbook_with_external(path)
    return deals, curve
