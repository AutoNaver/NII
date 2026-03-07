# Feature Requests

This file is the team-managed backlog for proposed features.

## Status Values
- `Proposed`
- `Accepted`
- `In Progress`
- `Done`
- `Rejected`

## Open Requests

| Request ID | Title | Requested By | Priority | Status | Target Release | Summary | Dependencies |
|---|---|---|---|---|---|---|---|
| FR-001 | Floating-rate deal support | Team | High | Proposed | TBD | Add support for `coupon_type`, `index`, and `spread` with curve lookup integration | Input schema extension |
| FR-003 | Pinned comparison presets | Team | Medium | Proposed | TBD | Save/restore common `T1/T2` and chart mode combinations in-session | UI state model |
| FR-004 | Export visible table to CSV | Team | Medium | Proposed | TBD | Allow exporting the currently visible table from each main tab | Streamlit download action |
| FR-005 | Data health ribbon | Team | Medium | Proposed | TBD | Show rows loaded, invalid rows dropped, and detected optional sheets | Loader validation summary |

## Completed Requests

| Request ID | Title | Requested By | Priority | Status | Target Release | Summary | Dependencies |
|---|---|---|---|---|---|---|---|
| FR-002 | Clarity-first dashboard refresh | Team | High | Done | Current | Sidebar controls, tabbed IA, runoff chart/table pairing, compact deal-diff section, and doc sync completed | None |
| FR-006 | Executive Excel export pack | Team | High | Done | Current | Overview-triggered generate/download `.xlsx` pack with metadata, metrics, scenario delta/absolute matrices, and first-year refill/growth distribution sheets | openpyxl |
| FR-007 | External product modeling framework (manual-rate v1) | Team | High | Done | Current | Optional `External_Profile` sheet plus modular external model layer; Overview can switch between `Internal`, `External`, and `Net` for metrics, scenario analysis, and export | External profile schema |
| FR-008 | External volume mirroring + daily due savings model | Team | High | Done | Current | External Overview volume now mirrors internal projected notional in absolute terms; added `daily_due_savings` model with UI-configurable equilibrium/reversion settings and export metadata | Interest curve, Overview external model layer |

## Request Template

Copy this row into **Open Requests** and fill values:

| Request ID | Title | Requested By | Priority | Status | Target Release | Summary | Dependencies |
|---|---|---|---|---|---|---|---|
| FR-XXX | <short feature title> | <name/team> | Low/Medium/High | Proposed | TBD | <what problem it solves and expected outcome> | <optional blockers/dependencies> |

## Notes
- Keep requests concise and implementation-oriented.
- If accepted, link the implementing PR in the row summary or as follow-up note.
- Keep `Open Requests` and `Completed Requests` separated for faster triage.
