"""Interest curve interfaces for future floating-rate extensions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class InterestCurve:
    """Base class for index curve access."""

    def get_rate(self, as_of_date: pd.Timestamp, tenor_months: int) -> float:
        raise NotImplementedError


@dataclass
class FlatCurve(InterestCurve):
    """Simple flat-rate curve implementation used for fallback/testing."""

    rate: float

    def get_rate(self, as_of_date: pd.Timestamp, tenor_months: int) -> float:
        return float(self.rate)


@dataclass
class TableCurve(InterestCurve):
    """Table-backed curve with (ir_date, ir_tenor, rate) rows."""

    curve_df: pd.DataFrame

    def get_rate(self, as_of_date: pd.Timestamp, tenor_months: int) -> float:
        as_of = pd.Timestamp(as_of_date) + pd.offsets.MonthEnd(0)
        subset = self.curve_df[
            (self.curve_df['ir_date'] == as_of) & (self.curve_df['ir_tenor'] == tenor_months)
        ]
        if subset.empty:
            raise KeyError(f'No curve point for date={as_of.date()} tenor={tenor_months}')
        return float(subset['rate'].iloc[0])
