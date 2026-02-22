# Dashboard Guide

## Run
```bash
streamlit run src/dashboard/app.py
```

## Terminology
- `Monthly View`: selected month-end comparison dates (`T1`, `T2`)
- `Bucket`: remaining maturity bucket within a selected monthly view

## Layout

### Sidebar (global controls)
- Workbook path (default `Input.xlsx`)
- Monthly View 1 (`T1`)
- Monthly View 2 (`T2`) toggle + selector
- Runoff display mode:
  - `Aligned Buckets (Remaining Maturity)`
  - `Calendar Months`
- Runoff decomposition basis (`T1` / `T2`)
- Growth mode:
  - `constant`
  - `user_defined` (shows monthly EUR growth input)

### Main tabs
- `Overview`
- `Daily`
- `Runoff`
- `Deal Differences`

## Default Opening State
- `T1` defaults to the first available month-end.
- `T2` defaults to the next available month-end (if present).

## Tab Behavior

### Overview
- With `T2` enabled:
  - delta KPI row (`T2 - T1`)
  - compact metrics table (`T1`, `T2`, `Delta`)
- With `T2` disabled:
  - compact single-view summary cards
  - hint to enable comparison mode

### Daily
- Available in comparison mode.
- Toggles:
  - date view (`T1` or `T2`)
  - chart view (`Daily Interest Decomposition` / `Daily Notional Decomposition`)
- Semantics:
  - `Existing`: active carry-in deals
  - `Added`: deals starting during selected calendar month
  - `Matured`: run-off effect shown separately

### Runoff
- Compact controls above chart include runoff chart view selection.
- Exactly one chart is shown at a time.
- Exactly one related table is shown below the chart (aligned to selected chart view).
- Optional refill/growth views are shown only when `refill_logic` exists.
- `Runoff 5Y Aggregation` is collapsed by default and includes:
  - horizon toggle: `Next 5 Years` or `5 Calendar Years`
  - split toggle:
    - `Y1..Y5` for `Next 5 Years`
    - explicit years (`2025`, `2026`, ...) for `5 Calendar Years`

### Deal Differences
- Consolidated change table shown first.
- Detailed categories are in nested expanders:
  - Added Deals
  - Matured/Removed Deals
  - Notional Changes
  - Coupon Changes

## Notes
- Current implementation is fixed-rate.
- Interest is reported in EUR with 30/360 conventions.
- If `refill_logic` is not present in the workbook, refill/growth chart options are hidden automatically.
