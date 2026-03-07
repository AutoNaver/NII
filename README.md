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
  - `Overview`: internal/external/net layer summary, layer-aware delta KPI row, compact `T1/T2/Delta` metrics table, rate-scenario analysis, export
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
  - visuals: impact matrix (`Delta`/`Absolute`), selected-scenario detail chart (`Delta`/`Absolute` with `BaseCase`), anchor-curve comparison (ramp: base + shocked 6M/12M), tenor movement (`1M/6M/1Y/5Y/10Y`, `0-24M`, plateau after month 12)
  - custom scenario builder in Overview:
    - shown at the bottom of Overview, collapsed by default
    - create custom `parallel`, `twist`, or `manual tenor` scenarios with `Instant` or `Linear 12M` materialization
    - for custom twist scenarios, pivot tenor is user-configurable (months)
    - for manual scenarios, selected tenor nodes drive interpolation (unselected nodes ignored)
    - manual scenarios include interpolation preview modes:
      - shock-only interpolation
      - base + final shocked curve (shock applied on selected anchor curve)
    - save/delete custom scenarios across restarts
    - persist active scenario set and recompute on changes
- Overview now supports executive Excel export:
  - `Generate Executive Excel` action in Overview (followed by download button)
  - downloadable `.xlsx` pack with:
    - metadata and overview metrics
    - scenario sensitivity matrices (`Delta` and `Absolute` with `BaseCase`) for `Internal`, `External`, and `Net`
    - first-year refill/growth distribution (`month x tenor`) and monthly summary
- External product modeling framework (manual-rate v1):
  - optional `External_Profile` workbook sheet
  - modular external model layer for future mortgage/deposit model plug-ins
  - Overview layer selector for `Internal`, `External`, and `Net`
  - scenario matrix/detail charts can switch between internal replication portfolio, external profile, and arithmetic net result
  - external volume now mirrors the internal projected notional path in absolute terms
  - external side supports:
    - `manual_profile`: mirrored external volume with workbook-provided monthly rates / repricing tenor
    - `daily_due_savings`: mirrored liability volume with UI-configurable equilibrium/reversion model
  - Daily/Runoff/Deal Differences remain internal-only

## Current Scope
- Fixed-rate portfolio analytics on a 30/360 basis.
- Multi-product input support via `Deal_Data.product` with per-product filtering.
- Optional external-position input support via `External_Profile` for Overview-only internal/external/net analysis.
- External Overview volume uses mirrored internal notionals; it does not rely on workbook external notionals for projected sizing.
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
- Optional: `External_Profile`

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
