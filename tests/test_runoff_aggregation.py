import pandas as pd

from src.dashboard.plots.runoff_plots import _build_aggregation_windows


def test_build_aggregation_windows_next_five_years_has_y1_to_y5() -> None:
    idx = pd.date_range('2025-01-31', periods=130, freq='ME')
    month_ends = pd.Series(idx)

    windows = _build_aggregation_windows(month_ends, 'Next 5 Years')
    by_name = {k: v for k, v in windows}

    assert 'All (Y1-Y5)' in by_name
    assert all(f'Y{i}' in by_name for i in range(1, 6))
    assert int(by_name['All (Y1-Y5)'].sum()) == 60
    assert int(by_name['Y1'].sum()) == 12
    assert int(by_name['Y5'].sum()) == 12
    assert bool(by_name['Y1'].iloc[0])
    assert not bool(by_name['Y1'].iloc[12])


def test_build_aggregation_windows_calendar_years() -> None:
    idx = pd.date_range('2025-03-31', periods=90, freq='ME')
    month_ends = pd.Series(idx)

    windows = _build_aggregation_windows(month_ends, '5 Calendar Years')
    labels = [k for k, _ in windows]
    by_name = {k: v for k, v in windows}
    years = month_ends[by_name['All (5 calendar years)']].dt.year.unique().tolist()

    assert labels[0] == 'All (5 calendar years)'
    assert labels[1:] == ['2025', '2026', '2027', '2028', '2029']
    assert years == [2025, 2026, 2027, 2028, 2029]
