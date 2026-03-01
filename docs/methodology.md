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

## Runoff Refill/Growth Mechanics
- Refill allocation is model-driven from shifted one-month portfolio delta by tenor:
  - compare `T2[k]` to `T1[k+1]`, clip at zero
- User-defined growth is a fixed monthly flow (EUR) in comparison mode.
- Growth allocation uses the selected basis (`T1`/`T2`) T0 portfolio tenor distribution.
- Cumulative growth is represented as outstanding profile:
  - monthly injections accumulate
  - outstanding stock rolls off with tenor survival/maturities

## Floating-Rate Extension Hook
`src/models/curves.py` contains curve interfaces for future floating-coupon integration.

## Overview Rate Scenario Method
- Scenarios include:
  - parallel shocks (`±50/100/200 bps`)
  - twist shocks (`±5/10 bps`) around a `6M` pivot tenor
    - `6M` shock is `0`
    - tenors `< 6M` and `> 6M` receive opposite signed shocks
    - `twist up`: right side up, left side down
    - `twist down`: right side down, left side up
- Materialization paths:
  - instant at month 0
  - linear ramp to full shock over 12 months
  - for ramp scenarios, shocked levels stay flat after month 12
- Anchor date is `T2`; horizon is 60 monthly points (5 years).
- Tenor-movement visualization tracks `1M`, `6M`, `1Y`, `5Y`, `10Y` over month `0..24`:
  - base path is anchored at `T2`
  - scenario path applies instant/ramp shock on top of the anchored base
- Current pricing basis is `Contractual + Refill/Growth`:
  - refill and growth remuneration are re-priced using shocked curve rates
  - existing contractual interest remains unchanged by rate shocks
- Down shocks are not floored; negative rates are allowed.
