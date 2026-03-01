"""Excel loader and schema normalization."""

from __future__ import annotations

import pandas as pd

from src.data.validator import validate_curve, validate_deals
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEALS_SHEET = 'Deal_Data'
CURVE_SHEET = 'Interest_Curve'

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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_input_workbook(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, normalize, and validate deal and curve sheets from workbook."""
    deals_raw = pd.read_excel(path, sheet_name=DEALS_SHEET)
    curve_raw = pd.read_excel(path, sheet_name=CURVE_SHEET)

    deals = _normalize_columns(deals_raw).rename(columns=DEAL_COLUMN_MAP)
    curve = _normalize_columns(curve_raw).rename(columns=CURVE_COLUMN_MAP)

    deals['trade_date'] = pd.to_datetime(deals['trade_date'])
    deals['value_date'] = pd.to_datetime(deals['value_date'])
    deals['maturity_date'] = pd.to_datetime(deals['maturity_date'])
    deals['notional'] = pd.to_numeric(deals['notional'])
    deals['coupon'] = pd.to_numeric(deals['coupon'])
    if 'product' not in deals.columns:
        deals['product'] = 'Default'
    deals['product'] = deals['product'].fillna('Default').astype(str).str.strip()
    deals.loc[deals['product'].isin(['', 'nan', 'None']), 'product'] = 'Default'

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

    return deals, curve
