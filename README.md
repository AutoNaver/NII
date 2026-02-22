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
  - Monthly View 1 (`T1`)
  - Monthly View 2 toggle + selector (`T2`)
  - Runoff display mode
  - Runoff decomposition basis (`T1` / `T2`)
  - Growth mode (`constant` / `user_defined`) and monthly growth value
- Main tabs:
  - `Overview`: delta KPI row + compact `T1/T2/Delta` metrics table
  - `Daily`: daily interest/notional decomposition with `T1/T2` toggle
  - `Runoff`: selected runoff chart + related table + collapsed 5Y aggregation section
  - `Deal Differences`: consolidated changes and detailed category tables in expanders

## Recent Updates
- Clarity-first UI refresh:
  - sidebar-first controls
  - tabbed main workflow
  - compact runoff chart/table pairing
- Shared formatting and control helpers:
  - `src/dashboard/components/formatting.py`
  - `src/dashboard/components/controls.py`
- Number formatting unified across major tables and chart axes.

## Current Scope
- Fixed-rate portfolio analytics on a 30/360 basis.
- Dual monthly-view analysis (`T1`, `T2`) with KPI deltas.
- Daily decomposition charts (interest and notional).
- Runoff decomposition in two display modes:
  - aligned remaining-maturity buckets
  - calendar-month aggregation
- Refill + optional user-defined growth decomposition (when `refill_logic` is present).
- Deal-level difference tables (added, matured, changed).
- 5-year aggregation table with horizon/split controls.

## Input Workbook
Default path: `Input.xlsx`

Sheets:
- Required: `Deal_Data`, `Interest_Curve`
- Optional: `refill_logic` (enables refill/growth decomposition charts)

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
