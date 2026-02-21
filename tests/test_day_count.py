from src.calculations.day_count import days_30_360


def test_same_day_returns_zero() -> None:
    assert days_30_360('2025-01-31', '2025-01-31') == 0


def test_end_of_month_handling() -> None:
    assert days_30_360('2025-01-31', '2025-02-28') == 28


def test_cross_year() -> None:
    assert days_30_360('2024-12-15', '2025-01-15') == 30


def test_february_to_march() -> None:
    assert days_30_360('2025-02-28', '2025-03-31') == 33
