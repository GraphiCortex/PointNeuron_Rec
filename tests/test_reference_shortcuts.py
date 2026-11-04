from pathlib import Path
import importlib.util
import sys

import numpy as np

from pn2.evaluation.swc import Morphology
from pn2.evaluation.reference_shortcuts import ReferencePaths


def morphology(points, parents):
    n = len(points)
    return Morphology(None, np.arange(n)+1, np.zeros(n, int), np.asarray(points, float),
                      np.ones(n), np.asarray(parents))


def score(m, points, edges, low=(-100, -100, -100), high=(100, 100, 100)):
    return ReferencePaths(m).score(points, edges, low, high)


def test_continuous_same_segment_is_not_shortcut():
    m = morphology([[0, 0, 0], [20, 0, 0]], [-1, 0])
    r = score(m, [[3, 0, 0], [17, 0, 0]], [[0, 1]])
    assert r['categories']['scorable']['count'] == 1 and r['flagged'] == 0
    assert r['edges'][0]['reference_path_length'] == 14


def test_annotated_detour_flags_without_changing_predicted_edges():
    m = morphology([[0, 0, 0], [20, 0, 0], [20, 4, 0], [0, 4, 0]], [-1, 0, 1, 2])
    points = np.array([[0., 0, 0], [0, 4, 0]])
    r = score(m, points, [[0, 1]])
    assert r['flagged'] == 1 and r['edges'][0]['reference_path_length'] == 44
    np.testing.assert_array_equal(points, [[0, 0, 0], [0, 4, 0]])
    assert r['used_for_edge_selection'] is False


def test_disconnected_trace_is_unknown_not_wrong_join():
    m = morphology([[0, 0, 0], [10, 0, 0], [0, 5, 0], [10, 5, 0]], [-1, 0, -1, 2])
    r = score(m, [[5, 0, 0], [5, 5, 0]], [[0, 1]])
    assert r['categories']['different_reference_components_unknown']['count'] == 1
    assert r['flagged'] == 0 and r['flag_fraction_of_scorable'] is None


def test_coincident_unwelded_crossing_is_ambiguous():
    m = morphology([[-10, 0, 0], [10, 0, 0], [0, -10, 0], [0, 10, 0]], [-1, 0, -1, 2])
    r = score(m, [[0, 0, 0], [5, 0, 0]], [[0, 1]])
    assert r['categories']['ambiguous']['count'] == 1 and r['flagged'] == 0


def test_path_leaving_box_is_unknown():
    m = morphology([[0, 0, 0], [20, 0, 0], [20, 4, 0], [0, 4, 0]], [-1, 0, 1, 2])
    r = score(m, [[0, 0, 0], [0, 4, 0]], [[0, 1]], high=(5, 5, 5))
    assert r['categories']['reference_path_leaves_box_unknown']['count'] == 1
    assert r['flagged'] == 0


def test_subdivision_invariance_off_reference_and_empty():
    m = morphology([[0, 0, 0], [4, 0, 0], [8, 0, 0], [20, 0, 0]], [-1, 0, 1, 2])
    r = score(m, [[3.9, 0, 0], [17, 0, 0], [0, 9, 0]], [[0, 1], [1, 2]])
    assert r['categories']['scorable']['count'] == 1
    assert np.isclose(r['edges'][0]['reference_path_length'], 13.1)
    assert r['categories']['off_reference']['count'] == 1
    assert score(m, np.empty((0, 3)), [])['flag_fraction_of_scorable'] is None


def test_existing_input_evidence_uses_exact_frog_identity(tmp_path, monkeypatch):
    # Regression for multiple frog cases sharing tile 5: source-group text is insufficient.
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('transfer_fixture', scripts/'measure_curve_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.e, 'BASE', tmp_path/'base')
    (tmp_path/'base/prepare').mkdir(parents=True)
    (tmp_path/'base/prepare/selection.json').write_text(json.dumps({'four': [['correct_frog', 5]]}))
    (tmp_path/'audit.json').write_text(json.dumps({'summary': {'partitions': {}}}))
    (tmp_path/'boxes.jsonl').write_text('\n'.join(json.dumps({'case_id': cid, 'tile_id': 5, 'input': {'marker': cid}})
                                                   for cid in ['correct_frog', 'other_frog']))
    r = module.existing_input_evidence({'input_audit_report': 'audit.json', 'input_audit_rows': 'boxes.jsonl'})
    assert r['frog_fit_box5']['input']['marker'] == 'correct_frog'


def test_three_policy_graph_scoring_and_readable_report(tmp_path, monkeypatch):
    # Exercise the actual new scorer/export/summary boundary, without a model or real cohort.
    from types import SimpleNamespace
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('transfer_graph_fixture', scripts/'measure_curve_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m = morphology([[0, 1, 1], [8, 1, 1]], [-1, 0])
    data = SimpleNamespace(boxes={('case', 1): {'box': {'start_xyz': [0, 0, 0], 'stop_xyz': [9, 3, 3]}}},
                           reference=lambda cid, tile: {'reference_runs': [m.xyz]})
    monkeypatch.setattr(module.e, 'References', lambda c, d: SimpleNamespace(get=lambda cid: {'morphology': m}))
    records = [{'policy': name, 'key': ('case', 1), 'points': m.xyz} for name in ['dense', 'raw_input', 'sato512']]
    packets = module.scaffold.build_graph_artifacts(records, {'case': np.ones((3, 3, 9))}, tmp_path/'graphs')
    result = module.score({'policies': ['dense', 'raw_input', 'sato512']}, {}, data, packets, tmp_path)
    assert len(result['rows']) == len(result['reference_shortcuts']) == 3
    for s in result['summaries'].values():
        assert s['decoded']['macro']['2.0']['f1'] == 1
    report = {'stage': 'fixture', 'measurement': result}
    assert '0/1 scorable edges flagged' in module.readable(report)
