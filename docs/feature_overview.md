# Feature Overview

This is the canonical feature-status document for the NII dashboard.

## Status Legend
- `Implemented`: available in the current codebase
- `Planned`: agreed target, not fully delivered
- `Backlog`: candidate enhancement, not scheduled

## Core Feature Status

| Area | Feature | Status | Notes |
|---|---|---|---|
| Data | Excel loader with schema normalization | Implemented | Uses `Input.xlsx` with `Deal_Data` and `Interest_Curve` |
| Data | Validation for required columns and types | Implemented | Hard-fails malformed schema; warns on suspicious values |
| Data | Invalid lifecycle handling (`maturity_date <= value_date`) | Implemented | Invalid rows excluded with warnings |
| Calculations | Active-deal rule (`value_date <= t < maturity_date`) | Implemented | Used consistently in snapshots and decompositions |
| Calculations | 30/360 day-count and accrual | Implemented | Shared across monthly and daily logic |
| Calculations | Monthly realized NII (EUR) | Implemented | Overlap-based monthly accrual |
| Calculations | Monthly buckets (notional, weighted coupon, interest, deal count) | Implemented | Includes signed/absolute handling where relevant |
| Calculations | Runoff delta attribution (existing/added/matured) | Implemented | Available for aligned bucket and calendar views |
| Calculations | Refill logic with tenor-based curve remuneration | Implemented | Interpolates `Interest_Curve` by tenor and basis date |
| Calculations | Growth mode (`constant` / `user_defined`) | Implemented | Growth deals follow refill remuneration mechanics |
| Dashboard | Monthly comparison mode (`T1` vs `T2`) | Implemented | Delta cards + compact metrics table |
| Dashboard | Daily decomposition charts (interest/notional) | Implemented | View toggle for `T1`/`T2` and chart type |
| Dashboard | Runoff display mode toggle | Implemented | `Aligned Buckets` and `Calendar Months` |
| Dashboard | Runoff chart-view toggle | Implemented | Notional, effective interest, contribution, deals, cumulative, refill/growth |
| Dashboard | Runoff 5Y aggregation table with split selector | Implemented | `Next 5 Years` (`Y1..Y5`) or `5 Calendar Years` (year-by-year) |
| Dashboard | Deal-level difference tables | Implemented | Consolidated + category tables |
| Dashboard | Number formatting (thousand separators) | Implemented | Applied across major tables and chart axes |
| Dashboard | Clarity-first UI refresh (sidebar controls + tabs + compact runoff workflow) | Implemented | Global controls consolidated; chart/table pairing improved; aggregation collapsed by default |
| Testing | Unit tests for accrual, loader, runoff, refill curve, growth | Implemented | Pytest suite in `tests/` |

## Planned Features

| Area | Feature | Status | Notes |
|---|---|---|---|
| Rates | True floating-rate support (`coupon_type`, `index`, `spread`) | Planned | Current pipeline is fixed-rate |
| Ops | CI workflow for automated pytest checks | Planned | Local tests available; CI not yet added |
| UX | Pinned comparison presets in session | Planned | Save/restore `T1/T2` and view-mode preferences |
| UX | Quick horizon jump controls (`Y1`, `Y2`, `2025`, `2026`) | Planned | Faster navigation in runoff aggregation |
| Reporting | Export visible table to CSV | Planned | Export only what user currently sees |
| Data | Data health ribbon (rows loaded/filtered) | Planned | Lightweight data-quality visibility in UI |

## Backlog / Future Extensions

| Area | Feature | Status | Notes |
|---|---|---|---|
| Analytics | Forward-looking NII projections | Backlog | Scenario-ready projection engine |
| Analytics | Rate-shock / scenario analysis | Backlog | Multi-curve runs |
| UX | Deal-level drill-down pages | Backlog | Per-deal lifecycle and contribution views |
| Reporting | Export to Excel/PDF | Backlog | Downloadable reporting artifacts |

## Add a Feature Request

Add requests in `docs/feature_requests.md` using the template there:
1. Add a new `FR-###` row in Open Requests.
2. Set status to `Proposed`.
3. Link rationale/PR when implemented.

