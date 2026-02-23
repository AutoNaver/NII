import pandas as pd

from src.calculations.accrual import accrued_interest_for_overlap
from src.calculations.accrual import accrued_interest_for_overlap_vectorized


def test_accrual_overlap_vectorized_matches_scalar() -> None:
    df = pd.DataFrame(
        [
            {
                'notional': 100.0,
                'coupon': 0.12,
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-04-01'),
            },
            {
                'notional': 200.0,
                'coupon': 0.05,
                'value_date': pd.Timestamp('2025-01-15'),
                'maturity_date': pd.Timestamp('2025-02-10'),
            },
            {
                'notional': -150.0,
                'coupon': 0.07,
                'value_date': pd.Timestamp('2025-02-01'),
                'maturity_date': pd.Timestamp('2025-03-15'),
            },
        ]
    )
    ws = pd.Timestamp('2025-02-01')
    we = pd.Timestamp('2025-03-01')

    vec = accrued_interest_for_overlap_vectorized(
        notional=df['notional'],
        annual_coupon=df['coupon'],
        deal_value_date=df['value_date'],
        deal_maturity_date=df['maturity_date'],
        window_start=ws,
        window_end=we,
    )

    expected = pd.Series(
        [
            accrued_interest_for_overlap(
                row.notional,
                row.coupon,
                row.value_date,
                row.maturity_date,
                ws,
                we,
            )
            for row in df.itertuples(index=False)
        ],
        index=df.index,
        dtype=float,
    )
    assert (vec - expected).abs().max() < 1e-9


def test_accrual_overlap_vectorized_zero_overlap() -> None:
    df = pd.DataFrame(
        [
            {
                'notional': 100.0,
                'coupon': 0.1,
                'value_date': pd.Timestamp('2025-01-01'),
                'maturity_date': pd.Timestamp('2025-01-31'),
            },
            {
                'notional': 200.0,
                'coupon': 0.2,
                'value_date': pd.Timestamp('2025-03-01'),
                'maturity_date': pd.Timestamp('2025-04-01'),
            },
        ]
    )
    vec = accrued_interest_for_overlap_vectorized(
        notional=df['notional'],
        annual_coupon=df['coupon'],
        deal_value_date=df['value_date'],
        deal_maturity_date=df['maturity_date'],
        window_start=pd.Timestamp('2025-02-01'),
        window_end=pd.Timestamp('2025-03-01'),
    )
    assert vec.tolist() == [0.0, 0.0]
