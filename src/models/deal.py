"""Deal domain model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Deal:
    """Represents a single fixed-rate deal in the MVP implementation."""

    deal_id: int
    trade_date: pd.Timestamp
    value_date: pd.Timestamp
    maturity_date: pd.Timestamp
    notional: float
    coupon: float
