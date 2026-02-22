# Dashboard Guide

## Run
```bash
streamlit run src/dashboard/app.py
```

## Inputs
- Workbook path (default: `Input.xlsx`)
- Monthly View 1 month-end (T1)
- Monthly View 2 month-end (T2, optional)
- Month-end exclusion filter for comparison views
- Runoff mode toggle: `Absolute Remaining` or `Delta Attribution`

## Default Opening State
- T1 defaults to the first available valuation month-end.
- T2 defaults to the month-end immediately after T1 (if available).

## Outputs
- KPI cards (previous-month realized NII, active deals, accrued interest)
- Active deals table at T1 (and T2 in comparison mode)
- Monthly bucket tables for:
  - total active notional
  - weighted average coupon (percentage points)
  - interest paid EUR (30/360)
- Deal-level difference tables in comparison mode
- Consolidated highlighted deal-change table in comparison mode
- Monthly runoff profile (cohort view) with month-offset roll-off
- Monthly bucket difference table for T1 vs T2 (calendar aligned deltas)
- Month-end filter control to hide selected months from comparison chart/table
- Single combined runoff comparison bar chart for T1 vs T2
- Delta attribution mode for runoff showing contributions from added deals and maturities
- Deal activity graph: active deals vs added deals by month bucket
- Notional*coupon activity graph: active vs added by month bucket

## Visualization Guidance
- Terminology:
  - `Buckets` = remaining maturity buckets
  - `Monthly View` = comparison date selections (T1/T2)
- Use charts to detect trend/shape quickly:
  - Runoff bars by remaining maturity buckets
  - Added vs matured attribution in runoff delta mode
- Use tables for exact reconciliation:
  - Monthly view deltas shown in transposed format
  - Runoff comparison deltas by remaining maturity buckets (transposed)

## Notes
- Current implementation is fixed-rate only.
- Interest amounts are shown in EUR.
