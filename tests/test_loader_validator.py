import pandas as pd
import pytest

from src.data.loader import load_input_workbook, load_input_workbook_with_external
from src.data.validator import validate_deals, validate_external_profile


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
    assert 'product' in deals_df.columns
    assert deals_df['product'].tolist() == ['Default']


def test_loader_normalizes_product_values(tmp_path) -> None:
    path = tmp_path / 'test_input_product.xlsx'

    deals = pd.DataFrame(
        {
            'Trade_Id': [1, 2, 3],
            'Trade_Date': ['2025-01-01', '2025-01-01', '2025-01-01'],
            'Value_Date': ['2025-01-01', '2025-01-01', '2025-01-01'],
            'Maturity_Date': ['2025-02-01', '2025-02-02', '2025-02-03'],
            'Notional': [1000, 2000, 3000],
            'Coupon': [0.05, 0.06, 0.07],
            'Product': ['Core', ' ', None],
        }
    )
    curve = pd.DataFrame({'IR_Date': ['2025-01-31'], 'IR_Tenor': [1], 'Rate': [0.02]})

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        deals.to_excel(writer, sheet_name='Deal_Data', index=False)
        curve.to_excel(writer, sheet_name='Interest_Curve', index=False)

    deals_df, _ = load_input_workbook(str(path))

    assert deals_df['product'].tolist() == ['Core', 'Default', 'Default']


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


def test_validator_allows_duplicate_ids_across_products() -> None:
    df = pd.DataFrame(
        {
            'deal_id': [1, 1],
            'product': ['A', 'B'],
            'trade_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'value_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'maturity_date': pd.to_datetime(['2025-02-01', '2025-02-02']),
            'notional': [1000, 2000],
            'coupon': [0.05, 0.06],
        }
    )
    warnings = validate_deals(df)
    assert isinstance(warnings, list)


def test_validator_rejects_duplicate_ids_within_product() -> None:
    df = pd.DataFrame(
        {
            'deal_id': [1, 1],
            'product': ['A', 'A'],
            'trade_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'value_date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'maturity_date': pd.to_datetime(['2025-02-01', '2025-02-02']),
            'notional': [1000, 2000],
            'coupon': [0.05, 0.06],
        }
    )
    with pytest.raises(ValueError, match='within product'):
        validate_deals(df)


def test_loader_reads_optional_external_profile_sheet(tmp_path) -> None:
    path = tmp_path / 'test_input_with_external.xlsx'

    deals = pd.DataFrame(
        {
            'Trade_Id': [1],
            'Trade_Date': ['2025-01-01'],
            'Value_Date': ['2025-01-01'],
            'Maturity_Date': ['2025-02-01'],
            'Notional': [1000],
            'Coupon': [0.05],
            'Product': ['Core'],
        }
    )
    curve = pd.DataFrame({'IR_Date': ['2025-01-31'], 'IR_Tenor': [1], 'Rate': [0.02]})
    external = pd.DataFrame(
        {
            'Product': ['Core'],
            'External_Product_Type': ['manual_profile'],
            'Calendar_Month_End': ['2025-01-31'],
            'External_Notional': [5000],
            'Repricing_Tenor_Months': [12],
            'Manual_Rate': [0.03],
        }
    )

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        deals.to_excel(writer, sheet_name='Deal_Data', index=False)
        curve.to_excel(writer, sheet_name='Interest_Curve', index=False)
        external.to_excel(writer, sheet_name='External_Profile', index=False)

    deals_df, curve_df, external_df = load_input_workbook_with_external(str(path))
    assert not deals_df.empty
    assert not curve_df.empty
    assert external_df.columns.tolist() == [
        'product',
        'external_product_type',
        'calendar_month_end',
        'external_notional',
        'repricing_tenor_months',
        'manual_rate',
    ]
    assert external_df['product'].tolist() == ['Core']


def test_loader_missing_external_profile_sheet_is_nonfatal(tmp_path) -> None:
    path = tmp_path / 'test_input_no_external.xlsx'
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

    _, _, external_df = load_input_workbook_with_external(str(path))
    assert external_df.empty


def test_validate_external_profile_rejects_duplicates() -> None:
    df = pd.DataFrame(
        {
            'product': ['Core', 'Core'],
            'external_product_type': ['manual_profile', 'manual_profile'],
            'calendar_month_end': pd.to_datetime(['2025-01-31', '2025-01-31']),
            'external_notional': [1000, 2000],
            'repricing_tenor_months': [12, 12],
            'manual_rate': [0.03, 0.04],
        }
    )
    with pytest.raises(ValueError, match='Duplicate external profile rows'):
        validate_external_profile(df)


def test_validate_external_profile_allows_minimal_daily_due_savings_row() -> None:
    df = pd.DataFrame(
        {
            'product': ['Savings'],
            'external_product_type': ['daily_due_savings'],
            'calendar_month_end': pd.to_datetime([pd.NaT]),
            'external_notional': [pd.NA],
            'repricing_tenor_months': [pd.NA],
            'manual_rate': [pd.NA],
        }
    )
    df['external_notional'] = pd.to_numeric(df['external_notional'], errors='coerce')
    df['repricing_tenor_months'] = pd.to_numeric(df['repricing_tenor_months'], errors='coerce')
    df['manual_rate'] = pd.to_numeric(df['manual_rate'], errors='coerce')
    warnings = validate_external_profile(df)
    assert isinstance(warnings, list)
