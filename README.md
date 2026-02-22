# NII Dashboard MVP

This repository contains a Streamlit-based MVP for Net Interest Income analytics.

## Project Tracking
- Feature status and roadmap: `docs/feature_overview.md`
## Doc map
- Overview & setup: README.md
- Usage guide: docs/dashboard_guide.md
- Feature status: docs/feature_overview.md
- Feature requests/backlog: docs/feature_requests.md

- Team feature request backlog: `docs/feature_requests.md`

## Features
- Excel loader with column normalization and validation
- 30/360 accrual calculations
- Monthly realized NII in EUR
- Month-end snapshots and comparison mode
- Active-deal tables at selected month-ends
- Comparison deltas for NII, active deals, accrued interest, volume, and coupon
- Deal-level difference tables
- Consolidated highlighted deal-change view (new, matured, notional/coupon changed)
- Monthly runoff cohort view (month 0 full active cohort, then maturity roll-off)
- Monthly bucket delta table and runoff delta table for T1 vs T2
- Single runoff comparison bar chart for both dates with mode toggle (absolute vs delta attribution)
- Table-first monthly buckets (active notional, weighted coupon in percentage points, interest EUR 30/360)
- Monthly activity graphs: active vs added deal count, and active vs added notional*coupon
- Default opening dates set to first valuation month-end and the following month-end
- Unit tests with pytest

## Input Workbook
Default file: `Input.xlsx`

Required sheets:
- `Deal_Data`
- `Interest_Curve`

## Run
```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Test
```bash
pytest -q
```

