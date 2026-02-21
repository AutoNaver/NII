import pandas as pd
import pytest

from src.data.loader import load_input_workbook
from src.data.validator import validate_deals


def test_loader_normalizes_columns(tmp_path) -> None:
    path = tmp_path / 'test_input.xlsx'

    deals = pd.DataFrame(
        {
            'Trade_Id': [1],
            'Trade_Date': ['2025-01-01'],
            'Value_Date': ['2025-01-01'],
            'Maturity_Date': ['2025-02-01'],
            'Notional': [1000],
            'Coupon': [0.05],
        }
    )
    curve = pd.DataFrame({'IR_Date': ['2025-01-31'], 'IR_Tenor': [1], 'Rate': [0.02]})

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        deals.to_excel(writer, sheet_name='Deal_Data', index=False)
        curve.to_excel(writer, sheet_name='Interest_Curve', index=False)

    deals_df, curve_df = load_input_workbook(str(path))

    assert set(['deal_id', 'trade_date', 'value_date', 'maturity_date', 'notional', 'coupon']).issubset(deals_df.columns)
    assert set(['ir_date', 'ir_tenor', 'rate']).issubset(curve_df.columns)


def test_validator_rejects_duplicate_ids() -> None:
    df = pd.DataFrame(
        {
            'deal_id': [1, 1],
            'trade_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'value_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'maturity_date': pd.to_datetime(['2025-02-01', '2025-02-02']),
            'notional': [1000, 2000],
            'coupon': [0.05, 0.06],
        }
    )

    with pytest.raises(ValueError, match='Duplicate deal_id'):
        validate_deals(df)
