# Feature Overview

This document tracks feature status for the NII Dashboard and acts as the canonical snapshot of:
- Final target capabilities
- What is implemented
- What is still pending
- Where to request new features

## Status Legend
- `Implemented`: Available in current codebase
- `Planned`: Confirmed target, not fully delivered yet
- `Backlog`: Candidate enhancement, not scheduled

## Core Feature Status

| Area | Feature | Status | Notes |
|---|---|---|---|
| Data | Excel loader with sheet mapping and column normalization | Implemented | Uses `Input.xlsx` with `Deal_Data` and `Interest_Curve` |
| Data | Validation for required columns and datatypes | Implemented | Hard fails for malformed schema; logs warnings for suspicious data |
| Data | Invalid lifecycle handling (`maturity_date <= value_date`) | Implemented | Invalid rows are excluded with warnings |
| Calculations | 30/360 day-count | Implemented | Shared utility used by accrual and NII logic |
| Calculations | Active-deal rule (`value_date <= t < maturity_date`) | Implemented | Used across snapshots and comparisons |
| Calculations | Fixed-rate accrued interest (EUR) | Implemented | Sign follows notional |
| Calculations | Monthly realized NII (EUR) | Implemented | Overlap-based monthly accrual aggregation |
| Calculations | Monthly buckets (notional, weighted coupon, interest paid) | Implemented | Includes active deal count |
| Dashboard | End-of-month view | Implemented | KPI cards + monthly tables/charts |
| Dashboard | Comparison mode (T1 vs T2) | Implemented | Deltas for NII, active deals, accrued interest, volume, coupon |
| Dashboard | Deal-level differences (added/matured/changed) | Implemented | Includes consolidated highlighted change view |
| Dashboard | EUR-first display | Implemented | Interest shown as currency amounts |
| Dashboard | Combined chart (notional + coupon) | Implemented | Dual-axis Plotly chart |
| Dashboard | Monthly runoff cohort view (month-offset roll-down) | Implemented | Month 0 starts with full active cohort and runs off by maturity |
| Dashboard | Monthly bucket and runoff delta tables for T1 vs T2 | Implemented | Tables provide exact reconciliation next to charts |
| Testing | Unit tests for day-count, accrual, activation, NII, loader | Implemented | Pytest suite in `tests/` |

## Planned Features (Next)

| Area | Feature | Status | Notes |
|---|---|---|---|
| Rates | True floating-rate support (`coupon_type`, `index`, `spread`) | Planned | Current input is fixed-only; curve interfaces already exist |
| Dashboard | Better metric formatting and UX polish | Planned | Improve readability for large portfolios |
| Ops | CI workflow for automated pytest checks | Planned | Tests run locally; CI config not added yet |

## Backlog / Future Extensions

| Area | Feature | Status | Notes |
|---|---|---|---|
| Analytics | Forward-looking NII projections | Backlog | Scenario-ready projection engine |
| Analytics | Rate shock / scenario analysis | Backlog | Parallel runs across curve scenarios |
| UX | Deal-level drill-down pages | Backlog | Per-deal timeline and contribution detail |
| Reporting | Export to Excel/PDF | Backlog | Downloadable artifacts for stakeholders |

## Add a Feature Request

Developers should add new requests in:
- `docs/feature_requests.md`

Process:
1. Add a new row in the **Open Requests** table using the template there.
2. Assign a unique request ID (`FR-###`).
3. Set initial status to `Proposed`.
4. Open a PR with the request update and rationale.
