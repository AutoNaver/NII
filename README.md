# NII Dashboard MVP

Streamlit app for month-end Net Interest Income analytics with runoff decomposition and deal-level attribution.

## Doc Map
- Setup and quick run: `README.md`
- Dashboard usage and controls: `docs/dashboard_guide.md`
- Feature status (implemented/planned/backlog): `docs/feature_overview.md`
- Team backlog / feature requests: `docs/feature_requests.md`

## UI Map
- Sidebar (global controls):
  - Workbook path
  - Refresh cached calculations
  - Product selector (single-product filter; defaults to one product)
  - Monthly View 1 (`T1`)
  - Monthly View 2 toggle + selector (`T2`)
  - Runoff display mode
  - Runoff decomposition basis (`T1` / `T2`)
  - Growth mode (`constant` / `user_defined`) and monthly growth value
- Main sections (stateful selector):
  - `Overview`: delta KPI row + compact `T1/T2/Delta` metrics table
  - `Daily`: daily interest/notional decomposition with `T1/T2` toggle
  - `Runoff`: selected runoff chart + related table + collapsed 5Y aggregation section
  - `Deal Differences`: consolidated changes and detailed category tables in expanders

## Recent Updates
- Clarity-first UI refresh:
  - sidebar-first controls
  - stateful main section selector (no reset to Overview on rerun)
  - compact runoff chart/table pairing
- Shared formatting and control helpers:
  - `src/dashboard/components/formatting.py`
  - `src/dashboard/components/controls.py`
- Number formatting unified across major tables and chart axes.
- Runoff refill/growth is now model-driven:
  - refill allocation from shifted one-month portfolio delta
  - user-defined growth allocation from T0 portfolio distribution
  - cumulative growth shown as outstanding profile (rolls off with maturities)
- Overview now includes rate-shock simulation (5Y vs base case):
  - parallel scenarios: `+/-50/100/200 bps`
  - twist scenarios (pivot 6M): `+/-5/10 bps` (left/right of 6M receive opposite signs)
  - profiles: `Instant` and `Linear 12M`
  - pricing basis: contractual + refill/growth impact (existing contractual interest unchanged by shocks)
  - visuals: impact matrix, selected-scenario detail chart, anchor-curve comparison (ramp: base + shocked 6M/12M), tenor movement (`1M/6M/1Y/5Y/10Y`, `0-24M`, plateau after month 12)

## Current Scope
- Fixed-rate portfolio analytics on a 30/360 basis.
- Multi-product input support via `Deal_Data.product` with per-product filtering.
- Dual monthly-view analysis (`T1`, `T2`) with KPI deltas.
- Daily decomposition charts (interest and notional).
- Runoff decomposition in two display modes:
  - aligned remaining-maturity buckets
  - calendar-month aggregation
- Refill + user-defined growth decomposition in comparison mode.
- Refill allocation heatmap plus refill/growth volume and interest line chart.
- Deal-level difference tables (added, matured, changed).
- 5-year aggregation table with horizon/split controls.

## Input Workbook
Default path: `Input.xlsx`

Sheets:
- Required: `Deal_Data`, `Interest_Curve`

## Run
```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Test
```bash
pytest -q
```

## Notes for Contributors
- Add new feature ideas to `docs/feature_requests.md`.
- Update `docs/feature_overview.md` when feature status changes.
- Keep this README concise; detailed usage belongs in `docs/dashboard_guide.md`.

