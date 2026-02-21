# Dashboard Guide

## Run
```bash
streamlit run src/dashboard/app.py
```

## Inputs
- Workbook path (default: `Input.xlsx`)
- Primary month-end (T1)
- Optional comparison month-end (T2)
- Month-end exclusion filter for comparison views
- Runoff mode toggle: `Absolute Remaining` or `Delta Attribution`

## Default Opening State
- T1 defaults to the first available valuation month-end.
- T2 defaults to the month-end immediately after T1 (if available).

## Outputs
- KPI cards (previous-month realized NII, active deals, accrued interest)
- Active deals table at T1 (and T2 in comparison mode)
- Monthly bucket table
- Plotly charts for notional, coupon, and interest
- Combined plot for notional and weighted coupon
- Deal-level difference tables in comparison mode
- Consolidated highlighted deal-change table in comparison mode
- Monthly runoff profile (cohort view) with month-offset roll-off
- Monthly bucket difference table for T1 vs T2 (calendar aligned deltas)
- Combined comparison graph for T1 vs T2 monthly buckets (single multi-panel figure)
- Month-end filter control to hide selected months from comparison chart/table
- Single combined runoff comparison bar chart for T1 vs T2
- Delta attribution mode for runoff showing contributions from added deals and maturities

## Visualization Guidance
- Use charts to detect trend/shape quickly:
  - Runoff curves for remaining notional by month offset
  - Monthly matured volume bars
  - Remaining percent of initial cohort
- Use tables for exact reconciliation:
  - Calendar monthly bucket deltas (notional/coupon/interest/deals)
  - Runoff comparison deltas by month offset

## Notes
- Current implementation is fixed-rate only.
- Interest amounts are shown in EUR.
