"""Fixed image-only k=6 scaffold shared by bounded CPU diagnostics/evaluation."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from pn2.inputs.node_cache import digest, write_once
from .vote_scaffold import knn_edges, image_edge_costs, minimum_spanning_forest, guide_at, export_forest
from .swc import parse_swc
from .formulation import graph_segments, clip_segments
from .bounded_agreement import BoundedAgreement


def build(points, guide, maximum, folder, key, policy, source_indices=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    points = np.asarray(points)
    edges = knn_edges(points, 6)
    costs, quality = image_edge_costs(points, edges, guide, maximum)
    forest, components = minimum_spanning_forest(points, edges, costs)
    q = guide_at(guide, points, maximum)
    receipt = export_forest(folder/'scaffold.swc', points, forest, q)
    reread = parse_swc(folder/'scaffold.swc')
    if not np.array_equal(reread.xyz, points) or set(map(tuple, np.sort(reread.edges, axis=1))) != set(map(tuple, forest)):
        raise ValueError('SWC round-trip changed fixed geometry')
    source_indices = np.arange(len(points)) if source_indices is None else np.asarray(source_indices)
    if source_indices.shape != (len(points),):
        raise ValueError('Unaligned selected source identities')
    np.savez_compressed(folder/'graph.npz', xyz=points, forest_edges=forest,
                        candidate6=edges, cost=costs, mean_guide=quality, source_indices=source_indices)
    receipt.update(case_id=key[0], tile_id=key[1], policy=policy, decoding_k=6,
        candidate_components=components, reference_edges_used=False, reference_roots_used=False,
        graph_sha256=digest(folder/'graph.npz'), swc_sha256=digest(folder/'scaffold.swc'))
    write_once(folder/'inference.json', receipt)
    return {'policy': policy, 'key': list(key), 'folder': str(folder), 'inference': receipt}


def score(packet, reference_segments, box):
    agreement = BoundedAgreement(reference_segments)
    low, high = np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5
    folder = Path(packet['folder'])
    for name, field in [('graph.npz', 'graph_sha256'), ('scaffold.swc', 'swc_sha256')]:
        if digest(folder/name) != packet['inference'][field]:
            raise ValueError('Saved graph/SWC changed')
    decoded = agreement.score(clip_segments(graph_segments(parse_swc(folder/'scaffold.swc')), low, high))
    with np.load(folder/'graph.npz', allow_pickle=False) as a:
        graph = SimpleNamespace(xyz=a['xyz'], edges=a['candidate6'])
        candidate = agreement.score(clip_segments(graph_segments(graph), low, high))
    if decoded['tolerances']['2.0']['recall'] > candidate['tolerances']['2.0']['recall']+1e-12:
        raise ValueError('Decoded recall exceeded candidate union')
    return {'policy': packet['policy'], 'case_id': packet['key'][0], 'tile_id': packet['key'][1],
            'segments': decoded, 'candidate_union_k6': candidate, 'inference': packet['inference']}


def summarize(rows, field='segments'):
    neurons = []
    for cid in sorted({r['case_id'] for r in rows}):
        group = [r[field] for r in rows if r['case_id'] == cid]
        rl, pl = sum(r['reference_length'] for r in group), sum(r['predicted_length'] for r in group)
        values = {}
        for t in ('2.0', '6.0'):
            pc = sum(r['tolerances'][t]['supported_prediction_length'] for r in group)
            rc = sum(r['tolerances'][t]['supported_reference_length'] for r in group)
            p, q = pc/pl if pl else 0., rc/rl if rl else 0.
            values[t] = {'precision': p, 'recall': q, 'f1': 2*p*q/(p+q) if p+q else 0.}
        neurons.append({'case_id': cid, 'reference_length': rl, 'predicted_length': pl, 'tolerances': values})
    return {'neurons': neurons, 'macro': {t: {k: float(np.mean([n['tolerances'][t][k] for n in neurons]))
            for k in ('precision', 'recall', 'f1')} for t in ('2.0', '6.0')}}


def gate(candidate, controls):
    """User-specified paired macro gain and >=5/7 strict per-neuron wins only."""
    checks = {}
    for t in ('2.0', '6.0'):
        name = min(controls, key=lambda k: (-controls[k]['macro'][t]['f1'], k))
        baseline = controls[name]
        a = {r['case_id']: r for r in candidate['neurons']}
        b = {r['case_id']: r for r in baseline['neurons']}
        if set(a) != set(b) or len(a) != 7:
            raise ValueError('Unmatched seven-neuron comparator')
        wins = {cid: a[cid]['tolerances'][t]['f1'] > b[cid]['tolerances'][t]['f1'] for cid in sorted(a)}
        floor = baseline['macro'][t]['f1']+.05
        passed_mean = candidate['macro'][t]['f1'] >= floor
        checks[t] = {'comparator': name, 'baseline_f1': baseline['macro'][t]['f1'],
            'required_f1': floor, 'actual_f1': candidate['macro'][t]['f1'], 'macro_gain_passed': passed_mean,
            'wins': wins, 'win_count': sum(wins.values()), 'wins_passed': sum(wins.values()) >= 5,
            'passed': bool(passed_mean and sum(wins.values()) >= 5)}
    return {'passed': all(x['passed'] for x in checks.values()), 'checks': checks, 'training_gated': False}
