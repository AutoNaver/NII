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

## Request Template

Copy this row into **Open Requests** and fill values:

| Request ID | Title | Requested By | Priority | Status | Target Release | Summary | Dependencies |
|---|---|---|---|---|---|---|---|
| FR-XXX | <short feature title> | <name/team> | Low/Medium/High | Proposed | TBD | <what problem it solves and expected outcome> | <optional blockers/dependencies> |

## Notes
- Keep requests concise and implementation-oriented.
- If accepted, link the implementing PR in the row summary or as follow-up note.
- Keep `Open Requests` and `Completed Requests` separated for faster triage.
