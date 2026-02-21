from src.calculations.accrual import accrued_interest_eur, accrued_interest_for_overlap


def test_accrued_interest_positive_notional() -> None:
    result = accrued_interest_eur(1000, 0.12, '2025-01-01', '2025-02-01')
    assert round(result, 6) == round(10.0, 6)


def test_accrued_interest_negative_notional() -> None:
    result = accrued_interest_eur(-1000, 0.12, '2025-01-01', '2025-02-01')
    assert round(result, 6) == round(-10.0, 6)


def test_accrual_overlap_partial_window() -> None:
    result = accrued_interest_for_overlap(
        notional=1000,
        annual_coupon=0.12,
        deal_value_date='2025-01-15',
        deal_maturity_date='2025-03-15',
        window_start='2025-01-01',
        window_end='2025-02-01',
    )
    assert round(result, 6) == round(5.3333333333, 6)
