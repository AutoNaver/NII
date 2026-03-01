from __future__ import annotations

import json

from src.dashboard.scenario_store import (
    STORE_VERSION,
    add_custom_scenario,
    build_active_scenarios_df,
    build_scenario_universe_df,
    delete_custom_scenario,
    load_scenario_store,
    make_custom_scenario_id,
    save_scenario_store,
    validate_custom_scenario,
)


def test_load_scenario_store_missing_file_returns_defaults(tmp_path) -> None:
    p = tmp_path / '.nii_custom_scenarios.json'
    payload = load_scenario_store(str(p))
    assert payload['version'] == STORE_VERSION
    assert payload['custom_scenarios'] == []
    assert payload['active_scenario_ids'] == []


def test_save_and_load_roundtrip(tmp_path) -> None:
    p = tmp_path / '.nii_custom_scenarios.json'
    payload = {
        'version': STORE_VERSION,
        'custom_scenarios': [
            {
                'scenario_id': 'custom_my_scenario',
                'scenario_label': 'My Scenario',
                'shock_type': 'parallel',
                'materialization': 'inst',
                'shock_bps': 25.0,
                'pivot_tenor_months': None,
                'manual_nodes': None,
            }
        ],
        'active_scenario_ids': ['inst_up_50', 'custom_my_scenario'],
    }
    save_scenario_store(str(p), payload)
    out = load_scenario_store(str(p))
    assert out['version'] == STORE_VERSION
    assert out['custom_scenarios'][0]['scenario_id'] == 'custom_my_scenario'
    assert set(out['active_scenario_ids']) == {'inst_up_50', 'custom_my_scenario'}

    # Ensure file is valid JSON on disk.
    parsed = json.loads(p.read_text(encoding='utf-8'))
    assert parsed['version'] == STORE_VERSION


def test_validate_custom_scenario_rules() -> None:
    good = {
        'scenario_id': make_custom_scenario_id('Manual A'),
        'scenario_label': 'Manual A',
        'shock_type': 'manual',
        'materialization': 'ramp',
        'shock_bps': None,
        'pivot_tenor_months': None,
        'manual_nodes': [{'tenor_months': 1, 'shock_bps': 10.0}],
    }
    ok, msg = validate_custom_scenario(good)
    assert ok is True
    assert msg == ''

    bad_manual = dict(good)
    bad_manual['manual_nodes'] = [{'tenor_months': 1, 'shock_bps': None}]
    ok2, _ = validate_custom_scenario(bad_manual)
    assert ok2 is False

    bad_id = dict(good)
    bad_id['scenario_id'] = 'inst_up_50'
    ok3, _ = validate_custom_scenario(bad_id)
    assert ok3 is False

    good_twist = {
        'scenario_id': make_custom_scenario_id('Twist P12'),
        'scenario_label': 'Twist P12',
        'shock_type': 'twist',
        'materialization': 'inst',
        'shock_bps': 10.0,
        'pivot_tenor_months': 12.0,
        'manual_nodes': None,
    }
    ok4, _ = validate_custom_scenario(good_twist)
    assert ok4 is True


def test_custom_only_delete_behavior_and_active_set_persistence() -> None:
    base = {'version': STORE_VERSION, 'custom_scenarios': [], 'active_scenario_ids': []}
    custom = {
        'scenario_id': 'custom_alpha',
        'scenario_label': 'Alpha',
        'shock_type': 'parallel',
        'materialization': 'inst',
        'shock_bps': 12.0,
        'pivot_tenor_months': None,
        'manual_nodes': None,
    }
    p1 = add_custom_scenario(base, custom)
    p1['active_scenario_ids'] = ['custom_alpha']
    active = build_active_scenarios_df(p1)
    assert active['scenario_id'].astype(str).tolist() == ['custom_alpha']

    # Built-ins are not deleted via delete_custom_scenario.
    p2 = delete_custom_scenario(p1, 'inst_up_50')
    assert p2['custom_scenarios'][0]['scenario_id'] == 'custom_alpha'

    # Custom delete removes scenario and active selection.
    p3 = delete_custom_scenario(p1, 'custom_alpha')
    assert p3['custom_scenarios'] == []
    assert 'custom_alpha' not in p3['active_scenario_ids']


def test_invalid_custom_record_falls_back_to_builtins() -> None:
    payload = {
        'version': STORE_VERSION,
        'custom_scenarios': [{'scenario_id': 'custom_bad', 'scenario_label': 'Bad', 'shock_type': 'manual'}],
        'active_scenario_ids': [],
    }
    universe = build_scenario_universe_df(payload)
    ids = set(universe['scenario_id'].astype(str).tolist())
    assert 'inst_up_50' in ids
    assert 'custom_bad' not in ids
