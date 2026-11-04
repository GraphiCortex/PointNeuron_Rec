import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.evaluation.formulation import segment_agreement, graph_segments, clip_segments
from pn2.evaluation.vote_scaffold import export_forest


def module():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('scoring_resume_fixture', scripts/'score_saved_curve_transfer.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_interrupted_record_is_not_complete_and_completed_record_is_reused(tmp_path):
    m = module()
    p = tmp_path/'row.json'
    def interrupt():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        m.cached_record(p, 'binding', interrupt)
    assert not p.exists()
    first, reused = m.cached_record(p, 'binding', lambda: {'value': 7})
    assert not reused and first['value'] == 7
    second, reused = m.cached_record(p, 'binding', lambda: pytest.fail('Completed score must not rerun'))
    assert reused and second == first
    with pytest.raises(ValueError, match='different source'):
        m.cached_record(p, 'changed', lambda: {})


def test_actual_saved_packet_boundary_uses_same_curve_and_candidates(tmp_path):
    m = module()
    xyz = np.array([[0., 1, 1], [8., 1, 1]])
    edges = np.array([[0, 1]])
    export_forest(tmp_path/'scaffold.swc', xyz, edges, np.ones(2))
    np.savez_compressed(tmp_path/'graph.npz', xyz=xyz, forest_edges=edges,
                        candidate4=edges, candidate6=edges, candidate10=edges)
    box = {'start_xyz': [0, 0, 0], 'stop_xyz': [9, 3, 3]}
    data = SimpleNamespace(boxes={('case', 1): {'box': box}},
                           reference=lambda cid, tile: {'reference_runs': [xyz]})
    packet = {'key': ('case', 1), 'policy': 'dense', 'folder': str(tmp_path), 'inference': {}}
    row = m.metric_row(packet, data)
    reference = graph_segments(SimpleNamespace(xyz=xyz, edges=edges))
    expected = segment_agreement(reference, clip_segments(reference, np.array(box['start_xyz'])-.5,
                                                         np.array(box['stop_xyz'])-.5))
    assert row['segments'] == expected
    assert all(v == expected for v in row['candidate_unions'].values())
