import pandas as pd

from src.dashboard.components.controls import coerce_option, default_month_view_indices


def test_coerce_option_prefers_existing_value() -> None:
    options = ['A', 'B', 'C']
    assert coerce_option('B', options, 'A') == 'B'


def test_coerce_option_falls_back_to_default_then_first() -> None:
    options = ['A', 'B', 'C']
    assert coerce_option('X', options, 'B') == 'B'
    assert coerce_option('X', options, 'Y') == 'A'


def test_default_month_view_indices() -> None:
    assert default_month_view_indices([]) == (0, 0)
    idx = pd.to_datetime(['2025-01-31'])
    assert default_month_view_indices(list(idx)) == (0, 0)
    idx2 = pd.to_datetime(['2025-01-31', '2025-02-28'])
    assert default_month_view_indices(list(idx2)) == (0, 1)
