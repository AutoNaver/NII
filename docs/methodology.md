# NII Methodology

## Scope
This MVP computes fixed-rate deal NII in EUR using the 30/360 convention.

## Core Definitions
- Active deal at date `t`: `value_date <= t < maturity_date`
- Accrued interest: `notional * coupon * days_30_360 / 360`
- Monthly realized NII: sum of accrued EUR interest over overlap between deal lifetime and month window

## Sign Convention
- Positive notional: asset contribution (positive interest)
- Negative notional: liability contribution (negative interest)

## Monthly Buckets
For each month-end:
- `total_active_notional`
- `weighted_avg_coupon` (weights = absolute notionals)
- `interest_paid_eur`

## Floating-Rate Extension Hook
`src/models/curves.py` contains curve interfaces for future floating-coupon integration.
