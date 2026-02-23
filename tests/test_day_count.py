import pandas as pd

from src.calculations.day_count import days_30_360, days_30_360_vectorized


def test_same_day_returns_zero() -> None:
    assert days_30_360('2025-01-31', '2025-01-31') == 0


def test_end_of_month_handling() -> None:
    assert days_30_360('2025-01-31', '2025-02-28') == 28


def test_cross_year() -> None:
    assert days_30_360('2024-12-15', '2025-01-15') == 30


def test_february_to_march() -> None:
    assert days_30_360('2025-02-28', '2025-03-31') == 33


def test_vectorized_matches_scalar_for_mixed_cases() -> None:
    starts = pd.Series(pd.to_datetime(['2025-01-31', '2024-12-15', '2025-02-28']))
    ends = pd.Series(pd.to_datetime(['2025-02-28', '2025-01-15', '2025-03-31']))
    vec = days_30_360_vectorized(starts, ends)
    expected = pd.Series([days_30_360(s, e) for s, e in zip(starts, ends)], dtype=int)
    assert vec.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_vectorized_same_day_returns_zero() -> None:
    starts = pd.Series(pd.to_datetime(['2025-01-31', '2025-02-15']))
    ends = pd.Series(pd.to_datetime(['2025-01-31', '2025-02-15']))
    vec = days_30_360_vectorized(starts, ends)
    assert vec.tolist() == [0, 0]
