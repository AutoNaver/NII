from src.calculations.accrual import is_active


def test_active_inclusive_value_date() -> None:
    assert is_active('2025-01-01', '2025-02-01', '2025-01-01')


def test_inactive_on_maturity_date() -> None:
    assert not is_active('2025-01-01', '2025-02-01', '2025-02-01')


def test_inactive_before_value_date() -> None:
    assert not is_active('2025-01-02', '2025-02-01', '2025-01-01')
