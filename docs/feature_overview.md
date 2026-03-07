# Feature Overview

This is the canonical feature-status document for the NII dashboard.

## Status Legend
- `Implemented`: available in the current codebase
- `Planned`: agreed target, not fully delivered
- `Backlog`: candidate enhancement, not scheduled

## Core Feature Status

| Area | Feature | Status | Notes |
|---|---|---|---|
| Data | Excel loader with schema normalization | Implemented | Uses `Input.xlsx` with `Deal_Data`, `Interest_Curve`, and optional `External_Profile` |
| Data | Optional `product` column normalization | Implemented | Missing/blank product values are normalized to `Default` |
| Data | Validation for required columns and types | Implemented | Hard-fails malformed schema; warns on suspicious values |
| Data | Optional external-profile validation | Implemented | Validates `External_Profile` monthly profile rows, types, and duplicate month/type combinations |
| Data | Invalid lifecycle handling (`maturity_date <= value_date`) | Implemented | Invalid rows excluded with warnings |
| Calculations | Active-deal rule (`value_date <= t < maturity_date`) | Implemented | Used consistently in snapshots and decompositions |
| Calculations | 30/360 day-count and accrual | Implemented | Shared across monthly and daily logic |
| Calculations | Monthly realized NII (EUR) | Implemented | Overlap-based monthly accrual |
| Calculations | Monthly buckets (notional, weighted coupon, interest, deal count) | Implemented | Includes signed/absolute handling where relevant |
| Calculations | Runoff delta attribution (existing/added/matured) | Implemented | Available for aligned bucket and calendar views |
| Calculations | Model-driven refill allocation + curve remuneration | Implemented | Refill allocation derived from shifted one-month portfolio delta; rates interpolated from `Interest_Curve` |
| Calculations | Growth mode (`constant` / `user_defined`) | Implemented | `user_defined` is fixed monthly growth flow; growth allocation uses T0 tenor distribution |
| Calculations | Cumulative growth outstanding profile | Implemented | Growth accumulates over time and rolls off by tenor survival |
| Calculations | Rate-shock scenario engine (Overview, 5Y) | Implemented | Parallel (`+/-50/100/200 bps`) + twist around `6M` (`+/-5/10 bps`), instant/linear-12M shocks (ramp plateaus after month 12), contractual+refill/growth basis |
| Calculations | External product modeling framework | Implemented | Modular external model registry with mirrored-volume support for `manual_profile` and `daily_due_savings` |
| Calculations | Daily-due savings external model | Implemented | External savings rate follows `a + b*1M + c*(10Y-1M)` with monthly mean reversion and steady T2 base curve |
| Calculations | Internal / External / Net overview projection | Implemented | Overview scenario outputs can switch between internal replication portfolio, mirrored external model, and arithmetic net layer |
| Dashboard | Monthly comparison mode (`T1` vs `T2`) | Implemented | Delta cards + compact metrics table |
| Dashboard | Overview rate-scenario visuals | Implemented | Scenario matrix (`Delta`/`Absolute`) and selected-scenario monthly impact (`Delta`/`Absolute`, includes `BaseCase` in absolute view) support `Internal` / `External` / `Net`; curve visuals remain scenario-based |
| Dashboard | Overview executive Excel export pack | Implemented | Overview-triggered generate/download flow for `.xlsx` export with metadata, layer-tagged KPI metrics, layer-tagged scenario delta/absolute matrices, first-year refill/growth distribution sheets, and external model metadata |
| Dashboard | Custom rate scenario builder with persistence | Implemented | Create/delete custom scenarios, persist active set in workspace `.nii_custom_scenarios.json`, and recompute on change |
| Dashboard | Daily decomposition charts (interest/notional) | Implemented | View toggle for `T1`/`T2` and chart type |
| Dashboard | Runoff display mode toggle | Implemented | `Aligned Buckets` and `Calendar Months` |
| Dashboard | Product filter dropdown | Implemented | Sidebar product selector applies to all sections; single-product view by default |
| Dashboard | Runoff chart-view toggle | Implemented | Notional, effective interest, contribution, deals, cumulative, refill/growth, refill allocation heatmap |
| Dashboard | Refill allocation detail charting | Implemented | Heatmap by month/tenor plus refill/growth/total volume and interest line chart |
| Dashboard | Runoff 5Y aggregation table with split selector | Implemented | `Next 5 Years` (`Y1..Y5`) or `5 Calendar Years` (year-by-year) |
| Dashboard | Deal-level difference tables | Implemented | Consolidated + category tables |
| Dashboard | Number formatting (thousand separators) | Implemented | Applied across major tables and chart axes |
| Dashboard | Clarity-first UI refresh (sidebar controls + stateful section workflow + compact runoff workflow) | Implemented | Global controls consolidated; section selection persists across reruns; chart/table pairing improved; aggregation collapsed by default |
| Testing | Unit tests for accrual, loader, runoff, refill/growth, rate scenarios, and scenario store | Implemented | Pytest suite in `tests/` including scenario-path, persistence, and visualization helper coverage |

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
| Analytics | Multi-curve / advanced scenario analysis | Backlog | Beyond current parallel/twist shocks and materialization profiles |
| UX | Deal-level drill-down pages | Backlog | Per-deal lifecycle and contribution views |
| Reporting | Export to PDF | Backlog | Downloadable narrative reporting artifact |

## Add a Feature Request

Add requests in `docs/feature_requests.md` using the template there:
1. Add a new `FR-###` row in Open Requests.
2. Set status to `Proposed`.
3. Link rationale/PR when implemented.


