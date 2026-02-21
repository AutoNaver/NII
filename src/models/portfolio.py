"""Portfolio model and convenience behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.calculations.accrual import is_active


@dataclass
class Portfolio:
    """Container for deal records with active-slice helpers."""

    deals: pd.DataFrame

    def active_deals(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """Return deals active at a given date."""
        mask = self.deals.apply(
            lambda row: is_active(row['value_date'], row['maturity_date'], as_of_date),
            axis=1,
        )
        return self.deals.loc[mask].copy()
