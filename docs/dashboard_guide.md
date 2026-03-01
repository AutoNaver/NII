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
- `Refresh Cached Calculations` button (clears Streamlit data cache and reloads computed datasets)
- Product selector (filters all sections to one product at a time)
- Monthly View 1 (`T1`)
- Monthly View 2 (`T2`) toggle + selector
- Runoff display mode:
  - `Aligned Buckets (Remaining Maturity)`
  - `Calendar Months`
- Runoff decomposition basis (`T1` / `T2`)
- Growth mode:
  - `constant`
  - `user_defined` (shows monthly EUR growth input)

### Main sections
- `Overview`
- `Daily`
- `Runoff`
- `Deal Differences`

Note: The main section is stateful. Reruns keep the selected section instead of resetting to `Overview`.

## Default Opening State
- Product defaults to one available product (first valid product in the loaded workbook).
- `T1` defaults to the first available month-end.
- `T2` defaults to the next available month-end (if present).
- `Runoff Display Mode` defaults to `Calendar Months` for new sessions.

## Tab Behavior

### Overview
- With `T2` enabled:
  - delta KPI row (`T2 - T1`)
  - compact metrics table (`T1`, `T2`, `Delta`)
  - `Rate Scenario Analysis (5Y, vs Base Case)`:
    - parallel scenarios: `+/-50/100/200 bps`
    - twist scenarios (pivot at `6M`): `+/-5/10 bps` with opposite signs left vs right of pivot
    - shock materialization: `Instant` and `Linear 12M`
    - pricing basis: contractual + refill/growth (existing contractual interest unchanged by shocks)
    - `Scenario Impact Matrix` with `Matrix view` toggle:
      - `Delta`: yearly (`Y1..Y5`) and 5Y cumulative deltas
      - `Absolute`: yearly (`Y1..Y5`) and 5Y cumulative absolute totals (includes `BaseCase`)
    - selected-scenario monthly chart with `Detail view` toggle:
      - `Delta`: `Delta vs Base` + optional totals + cumulative delta
      - `Absolute`: `BaseCase Total` and selected `Scenario Total` (includes `BaseCase` as selectable scenario)
    - anchor curve panel:
      - instant scenarios: base (`T2`) vs shocked (`T2`)
      - ramp scenarios: base (`T2`) plus shocked curves at month 6 and month 12
    - tenor movement chart for `1M`, `6M`, `1Y`, `5Y`, `10Y` over `0-24M`
      - ramp scenarios plateau after month 12
    - `Scenario Builder` expander:
      - located at the bottom of Overview
      - collapsed by default
      - create custom scenarios (`Parallel`, `Twist`, `Manual Tenors`)
      - select materialization (`Instant`, `Linear 12M`)
      - custom twist scenarios allow user-defined pivot tenor (months)
      - manual scenarios allow selecting interpolation nodes on fixed tenors
      - manual scenarios show interpolation preview before saving:
        - `Shock (bps)` mode
        - `Base + Final Shocked Curve (%)` mode
      - manage active scenario set used by calculations
      - delete custom scenarios
      - scenario changes trigger cache clear + full recomputation
  - `Export Executive Pack`:
    - `Generate Executive Excel` button in Overview
    - download button appears after generation
    - download includes:
      - `Summary_Metadata`
      - `Overview_Metrics`
      - `Scenario_Matrix_Delta`
      - `Scenario_Matrix_Absolute` (includes `BaseCase`)
      - `Distribution_Month_Tenor_Y1` (T2-anchored first 12 months)
      - `Distribution_Monthly_Summary_Y1`
- With `T2` disabled:
  - compact single-view summary cards
  - hint to enable comparison mode

### Daily
- Available in comparison mode.
- Toggles:
  - date view (`T1` or `T2`)
  - chart view (`Daily Interest Decomposition` / `Daily Notional Decomposition`)
- Semantics:
  - Top panel: stacked-style `Total` bar (`Month Start Base` + shaded `Delta vs first day`, green up/red down)
  - Interest view top panel also overlays `Cumulative Total` on a secondary y-axis
  - Bottom panel: `Added` and `Matured` breakdown
  - Interest chart bottom panel overlays cumulative lines for `Added`, `Matured`, and `Added + Matured`
  - Interest chart includes a month-end cumulative decomposition summary table below the chart

### Runoff
- Compact controls above chart include runoff chart view selection.
- Exactly one chart is shown at a time.
- Exactly one related table is shown below the chart (aligned to selected chart view).
- Refill/growth views are available in comparison mode.
- Refill and growth modeling:
  - Refill is derived from shifted one-month portfolio delta by tenor.
  - `user_defined` growth uses a fixed monthly flow.
  - Growth allocation uses T0 portfolio tenor distribution.
  - Cumulative growth is shown as outstanding profile (adds over time, rolls off with maturities).
- Refill allocation visuals:
  - `Refill Allocation Heatmap` shows tenor/month allocation.
  - In `user_defined` mode, heatmap allocation includes refill + growth impact.
  - A line chart below the heatmap shows refill/growth/total volume (left axis) and refill/growth/total annualized interest (right axis).
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
- If `Deal_Data.product` is missing, the loader assigns `Default` to preserve compatibility with legacy inputs.
- Cache refresh is manual: if workbook contents change at the same file path, click `Refresh Cached Calculations`.
- Custom scenario persistence file: `.nii_custom_scenarios.json` in workspace root.

