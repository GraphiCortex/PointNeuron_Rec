import numpy as np
import pytest

from pn2.evaluation.vote_scaffold import (neighbor_counts, support_histogram, isolation_decision,
    supported_consolidation, knn_edges, guide_at, image_edge_costs, minimum_spanning_forest, export_forest)
from pn2.evaluation.formulation import segment_agreement, curve_samples
from scipy.spatial import cKDTree
from pn2.evaluation.swc import parse_swc


def test_support_includes_self_and_exact_radius():
    p = np.array([[0., 0, 0], [2, 0, 0], [2.01, 0, 0]])
    assert neighbor_counts(p, p[:1]).tolist() == [2]
    histogram = support_histogram([1, 2, 3], [False, False, True])
    assert histogram['isolated']['beyond_2'] == 2
    assert histogram['supported']['precision'] == 1


def test_trigger_conjunction_and_far_source_stop():
    assert isolation_decision([1, 2, 3, 4], [False, False, True, True], [1, 1, 1, 1])['eligible']
    assert not isolation_decision([1, 2, 3, 4], [False, False, True, True], [7, 7, 1, 1])['eligible']
    assert not isolation_decision([1, 3, 3, 4], [True, False, False, True], [1]*4)['eligible']
    assert not isolation_decision([1, 2], [False, False], [1, 1])['eligible']
    assert isolation_decision([1, 3, 4], [False, False, True], [1, 1, 1])['eligible']


def test_variant_counts_only_accepted_raw_votes_and_never_averages():
    p = np.array([[0., 0, 0], [.5, 0, 0], [1., 0, 0], [10., 0, 0], [10.1, 0, 0]])
    keep = supported_consolidation(p, [1, .8, .7, .9, .1], np.arange(5))
    assert keep.tolist() == [0]
    np.testing.assert_array_equal(p[keep][0], p[0])


def test_knn_nested_with_ties_and_no_self_edges():
    p = np.array([[0., 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    a, b = knn_edges(p, 1), knn_edges(p, 3)
    assert set(map(tuple, a)) <= set(map(tuple, b))
    assert (a[:, 0] < a[:, 1]).all()
    assert (0, 1) in set(map(tuple, a))


def test_cost_uses_image_not_references():
    guide = np.zeros((5, 5, 8), dtype=np.float32)
    guide[2, 2, :] = 1
    p = np.array([[0., 2, 2], [7, 2, 2], [0, 0, 0], [7, 0, 0]])
    cost, quality = image_edge_costs(p, [[0, 1], [2, 3]], guide, 1.)
    np.testing.assert_allclose(cost, [7, 35])
    np.testing.assert_allclose(quality, [1, 0])
    assert guide_at(guide, np.array([[-1., 2, 2]]), 1.).tolist() == [0]


def test_forest_export_round_trip_preserves_components_and_placeholder(tmp_path):
    p = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]])
    edges, components = minimum_spanning_forest(p, [[0, 1], [1, 2], [0, 2]], [1, 1, 2])
    assert components == 2 and len(edges) == 2
    path = tmp_path/'scaffold.swc'
    receipt = export_forest(path, p, edges, [0, 1, 0, 0])
    m = parse_swc(path)
    assert m.component_count == 2 and receipt['roots'] == [1, 3]
    np.testing.assert_array_equal(m.xyz, p)
    assert (m.radii == 1).all() and (m.types == 0).all()
    assert set(map(tuple, np.sort(m.edges, axis=1))) == set(map(tuple, edges))


def test_edge_union_not_node_proximity_is_recall_ceiling():
    reference = np.array([[[0., 0, 0], [20, 0, 0]]])
    # Two isolated endpoints cover only a small part at tolerance 2, but their
    # straight candidate edge covers the entire curve at arbitrarily fine tol.
    score = segment_agreement(reference, reference, tolerances=(.001, 2.))
    samples, weights = curve_samples([reference[0]])
    node_distance = cKDTree(reference[0]).query(samples)[0]
    assert weights[node_distance <= 2].sum()/weights.sum() == .2
    assert score['tolerances']['2.0']['recall'] == 1
    assert score['tolerances']['0.001']['recall'] == 1
    subset = np.array([[[0., 0, 0], [5, 0, 0]]])
    assert segment_agreement(reference, subset)['tolerances']['2.0']['recall'] <= 1


def test_empty_outputs_are_explicit(tmp_path):
    points = np.empty((0, 3))
    assert knn_edges(points, 6).shape == (0, 2)
    edges, n = minimum_spanning_forest(points, np.empty((0, 2), int), [])
    assert n == 0
    export_forest(tmp_path/'empty.swc', points, edges, [])
    assert parse_swc(tmp_path/'empty.swc').node_count == 0


def test_saved_graph_scaffold_round_trip_and_same_segment_scorer(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('scaffold_fixture', scripts/'measure_saved_vote_scaffold.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    guide = np.zeros((5, 5, 8), np.float32); guide[2, 2, :] = 1
    points = np.array([[0., 2, 2], [3, 3, 2], [7, 2, 2]])
    records = [{'policy': 'original', 'key': ('image-only-case', 0), 'points': points}]
    packets = driver.build_graph_artifacts(records, {'image-only-case': guide}, tmp_path/'graphs')
    assert packets[0]['inference']['reference_edges_used'] is False
    reference = {'reference_runs': [np.array([[0., 2, 2], [7, 2, 2]])]}
    class Data:
        boxes = {('image-only-case', 0): {'box': {'start_xyz': [0, 0, 0], 'stop_xyz': [8, 5, 5]}}}
        def reference(self, cid, tile):
            return reference
    classical = {'image-only-case': SimpleNamespace(xyz=reference['reference_runs'][0], edges=np.array([[0, 1]]))}
    scored = driver.score_graph_artifacts(packets, Data(), classical)
    assert scored['summaries']['original']['decoded']['macro']['2.0']['f1'] == 1
    assert scored['classical_summary']['macro']['2.0']['f1'] == 1
    assert all(scored['summaries']['original']['candidate_unions'][str(k)]['macro']['2.0']['recall'] == 1
               for k in (4, 6, 10))


def test_worker_normalizes_json_box_key_before_guide_lookup(tmp_path, monkeypatch):
    """Exercise the actual A worker path with a JSON-list identity, not just C."""
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('scaffold_key_fixture', scripts/'measure_saved_vote_scaffold.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    source = tmp_path/'terminal/eval_step002000_training'
    source.mkdir(parents=True)
    keys = [['image-case', 0]]
    (source/'metrics.json').write_text(json.dumps({'step': 2000, 'keys': keys}))
    points = np.array([[0., 0, 0], [3, 0, 0], [6, 0, 0]])
    np.savez(source/'predictions.npz', box0_points=points, box0_scores=np.ones(3),
             box0_ranked_indices=np.arange(3))
    class Data:
        boxes = {('image-case', 0): {}}
        def image(self, cid, tile, budget):
            return {'packet': {'flat_indices': np.arange(3)}}
    data = Data()
    monkeypatch.setattr(driver.e, 'control', lambda: {})
    monkeypatch.setattr(driver.e, 'prepared', lambda _: (data, {}, {'four': keys}))
    monkeypatch.setattr(driver.e, 'References', lambda *_: SimpleNamespace(
        get=lambda _: {'distances': lambda p: np.zeros(len(p))}))
    monkeypatch.setattr(driver.e, 'label_arrays', lambda *_: {'source_distance': np.zeros(3)})
    monkeypatch.setattr(driver, 'baseline_graphs', lambda *_: {})
    class ReachedGuide(Exception):
        pass
    def guide_lookup(dataset, key, packet, guides):
        assert isinstance(key, tuple)
        assert key in dataset.boxes
        raise ReachedGuide()
    monkeypatch.setattr(driver.e, 'sato_scores', guide_lookup)
    with pytest.raises(ReachedGuide):
        driver.worker({'terminal_root': 'terminal'}, tmp_path/'measurement')
