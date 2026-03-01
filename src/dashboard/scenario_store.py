"""Persistent store helpers for custom Overview rate scenarios."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.calculations.rate_scenarios import build_parallel_scenarios, normalize_scenarios_df

SCENARIO_STORE_FILENAME = '.nii_custom_scenarios.json'
STORE_VERSION = 1
_CUSTOM_ID_RE = re.compile(r'^custom_[a-z0-9_]+$')


def _default_payload() -> dict[str, Any]:
    return {
        'version': STORE_VERSION,
        'custom_scenarios': [],
        'active_scenario_ids': [],
    }


def _slugify(label: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(label).strip().lower())
    slug = slug.strip('_')
    return slug or 'scenario'


def make_custom_scenario_id(label: str) -> str:
    """Generate deterministic custom scenario id from label."""
    return f'custom_{_slugify(label)}'


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = _default_payload()
    if not isinstance(payload, dict):
        return out
    out['version'] = int(payload.get('version', STORE_VERSION))
    custom = payload.get('custom_scenarios', [])
    active = payload.get('active_scenario_ids', [])
    out['custom_scenarios'] = custom if isinstance(custom, list) else []
    out['active_scenario_ids'] = [str(x) for x in active] if isinstance(active, list) else []
    return out


def load_scenario_store(path: str) -> dict[str, Any]:
    """Load scenario store from disk. Missing file returns defaults."""
    p = Path(path)
    if not p.exists():
        return _default_payload()
    with p.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    return _normalize_payload(raw)


def save_scenario_store(path: str, payload: dict[str, Any]) -> None:
    """Persist scenario store payload to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_payload(payload)
    with p.open('w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=True)


def validate_custom_scenario(spec: dict[str, Any]) -> tuple[bool, str]:
    """Validate custom scenario payload before persistence."""
    sid = str(spec.get('scenario_id', '')).strip()
    label = str(spec.get('scenario_label', '')).strip()
    if not sid or not _CUSTOM_ID_RE.match(sid):
        return False, 'Custom scenario_id must match `custom_<slug>`.'
    if not label:
        return False, 'Scenario name is required.'
    shock_type = str(spec.get('shock_type', '')).strip().lower()
    if shock_type not in {'parallel', 'twist', 'manual'}:
        return False, 'Scenario type must be parallel, twist, or manual.'
    materialization = str(spec.get('materialization', '')).strip().lower()
    if materialization not in {'inst', 'ramp'}:
        return False, 'Materialization must be inst or ramp.'
    if shock_type in {'parallel', 'twist'}:
        shock_bps = pd.to_numeric(spec.get('shock_bps'), errors='coerce')
        if pd.isna(shock_bps):
            return False, 'Shock (bps) is required for parallel/twist scenarios.'
    if shock_type == 'twist':
        pivot = pd.to_numeric(spec.get('pivot_tenor_months'), errors='coerce')
        if pd.isna(pivot) or float(pivot) <= 0.0:
            return False, 'Twist pivot tenor must be a positive number.'
    if shock_type == 'manual':
        nodes = spec.get('manual_nodes')
        if not isinstance(nodes, list) or not nodes:
            return False, 'Manual scenario requires tenor nodes.'
        valid = 0
        for node in nodes:
            if not isinstance(node, dict):
                continue
            tenor = pd.to_numeric(node.get('tenor_months'), errors='coerce')
            shock = pd.to_numeric(node.get('shock_bps'), errors='coerce')
            if pd.isna(tenor) or pd.isna(shock):
                continue
            valid += 1
        if valid == 0:
            return False, 'Manual scenario nodes must contain at least one numeric shock value.'
    return True, ''


def build_scenario_universe_df(store_payload: dict[str, Any]) -> pd.DataFrame:
    """Return built-in + custom scenarios normalized for UI and simulation."""
    built_ins = normalize_scenarios_df(build_parallel_scenarios())
    custom_raw = store_payload.get('custom_scenarios', []) if isinstance(store_payload, dict) else []
    custom_df = pd.DataFrame(custom_raw) if isinstance(custom_raw, list) else pd.DataFrame()
    if not custom_df.empty:
        try:
            custom_df = normalize_scenarios_df(custom_df)
            custom_df = custom_df[custom_df['scenario_id'].astype(str).str.startswith('custom_')].copy()
        except Exception:
            custom_df = pd.DataFrame(columns=built_ins.columns)
    if custom_df.empty:
        universe = built_ins.copy()
    else:
        universe = pd.concat([built_ins, custom_df], ignore_index=True)
    universe = universe.drop_duplicates(subset=['scenario_id'], keep='first').reset_index(drop=True)
    return universe


def build_active_scenarios_df(store_payload: dict[str, Any]) -> pd.DataFrame:
    """Return active scenarios from store payload. Defaults to all if unset."""
    universe = build_scenario_universe_df(store_payload)
    if universe.empty:
        return universe
    active_ids = store_payload.get('active_scenario_ids', []) if isinstance(store_payload, dict) else []
    active_ids = [str(x) for x in active_ids if str(x) in set(universe['scenario_id'].astype(str).tolist())]
    if not active_ids:
        active_ids = universe['scenario_id'].astype(str).tolist()
    active = universe[universe['scenario_id'].astype(str).isin(active_ids)].copy()
    return active.reset_index(drop=True)


def add_custom_scenario(store_payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Append a custom scenario and mark it active."""
    out = deepcopy(_normalize_payload(store_payload))
    out['custom_scenarios'] = [s for s in out['custom_scenarios'] if str(s.get('scenario_id', '')) != str(spec.get('scenario_id', ''))]
    with_meta = deepcopy(spec)
    with_meta['created_at'] = datetime.now(timezone.utc).isoformat()
    out['custom_scenarios'].append(with_meta)
    active = [str(x) for x in out.get('active_scenario_ids', [])]
    sid = str(spec.get('scenario_id'))
    if sid not in active:
        active.append(sid)
    out['active_scenario_ids'] = active
    return out


def delete_custom_scenario(store_payload: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    """Delete custom scenario by id and remove it from active list."""
    sid = str(scenario_id)
    if not sid.startswith('custom_'):
        return deepcopy(_normalize_payload(store_payload))
    out = deepcopy(_normalize_payload(store_payload))
    out['custom_scenarios'] = [s for s in out['custom_scenarios'] if str(s.get('scenario_id', '')) != sid]
    out['active_scenario_ids'] = [str(x) for x in out.get('active_scenario_ids', []) if str(x) != sid]
    return out
