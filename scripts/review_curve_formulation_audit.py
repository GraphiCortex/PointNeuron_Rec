"""Read-only result/graph integrity and existing-scorer cross-checks; no model."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import audit_curve_formulation as audit
from pn2.evaluation.geometry import geometry_metrics
from pn2.evaluation.formulation import runs_graph, graph_segments, clip_segments, segment_agreement
from pn2.inputs.node_cache import digest, fingerprint, write_once

ROOT = audit.ROOT


def main(args):
    c, data, _ = audit.load_control(audit.CONFIG)
    folder = args.output.resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Use PN2/tmp')
    folder.mkdir(parents=True, exist_ok=False)
    a = json.loads((args.audit/'report.json').read_text());b = json.loads((args.baseline/'report.json').read_text())
    if any(r['status'] != 'completed' or r['config_fingerprint'] != fingerprint(c) for r in (a, b)):
        raise ValueError('Result is incomplete or bound to another recipe')
    rows = [json.loads(line) for line in (args.audit/'boxes.jsonl').read_text().splitlines()]
    expected = {key for key, box in data.boxes.items() if box['target']['count']}
    observed = [(r['case_id'], r['tile_id']) for r in rows]
    if len(rows) != 611 or len(set(observed)) != len(rows) or set(observed) != expected:
        raise ValueError('Missing or duplicated reference-present box')
    for row in rows:
        for key in ('2.0', '6.0', '8.0'):
            if row['input']['8192']['supported_length'][key]+1e-9 < row['input']['2048']['supported_length'][key]:
                raise ValueError('Nested inputs produced nonmonotone spatial support')
        for budget in (2048, 8192):
            if row['input'][str(budget)]['actual_points'] != min(budget, data.boxes[(row['case_id'], row['tile_id'])]['source_voxels']):
                raise ValueError('Input budget denominator changed')
    if len(a['metric_controls']) != 40 or any(x['reference_geometry_with_new_subdivisions']['tolerances']['0.001']['f1'] != 1. for x in a['metric_controls']):
        raise ValueError('Reference phase/subdivision control failed')
    ids = {cid for cid, case in data.cases.items() if case['partition'] == 'development'
           or case['alias'] in c['baseline_training_aliases']}
    if {x['case_id'] for x in b['arms']} != ids or len(b['arms']) != 12:
        raise ValueError('Classical case selection changed')
    for i, arm in enumerate(b['arms']):
        if arm['extraction']['references_opened_for_extraction'] or not arm['image_only_extraction']:
            raise ValueError('Classical extraction used reference information')
        case = data.cases[arm['case_id']]
        if len(arm['boxes']) != len(case['boxes']):raise ValueError('Classical grid denominator changed')
        with np.load(args.baseline/f'case{i:02d}/graphs.npz', allow_pickle=False) as packet:
            for kind in ('raw', 'pruned'):
                xyz, edges = packet[kind+'_xyz'], packet[kind+'_edges']
                if (not np.isfinite(xyz).all() or xyz.shape[1:] != (3,) or edges.shape[1:] != (2,)
                        or not np.issubdtype(edges.dtype, np.integer)
                        or (len(edges) and (edges.min() < 0 or edges.max() >= len(xyz)))
                        or np.any(xyz < 0) or np.any(xyz >= np.array(case['dimensions_xyz']))):
                    raise ValueError('Invalid classical graph coordinates/indices')
    checks = []
    for alias, tile in ((32, 20), (94, 14)):
        i, arm = next((i, x) for i, x in enumerate(b['arms']) if x['alias'] == alias)
        cid = arm['case_id'];box = data.boxes[(cid, tile)];ref = data.reference(cid, tile)
        with np.load(args.baseline/f'case{i:02d}/graphs.npz') as packet:
            graph = SimpleNamespace(xyz=packet['pruned_xyz'], edges=packet['pruned_edges'])
        ps = clip_segments(graph_segments(graph), np.array(box['box']['start_xyz'])-.5, np.array(box['box']['stop_xyz'])-.5)
        rg = runs_graph(ref['reference_runs']);pg = runs_graph(list(ps))
        existing = geometry_metrics(rg, pg);current = segment_agreement(graph_segments(rg), ps)
        delta = max(abs(existing['tolerances'][str(int(t))][k]-current['tolerances'][str(t)][k])
                    for t in (2., 6.) for k in ('precision', 'recall', 'f1'))
        if delta > 1e-6:raise ValueError('Existing and audit curve scorers disagree')
        checks.append({'alias': alias, 'tile': tile, 'maximum_scorer_difference': delta})
    result = {'status': 'passed', 'config_fingerprint': fingerprint(c),
        'audit_report_sha256': digest(args.audit/'report.json'), 'audit_rows_sha256': digest(args.audit/'boxes.jsonl'),
        'classical_report_sha256': digest(args.baseline/'report.json'),
        'reference_present_boxes_verified': 611, 'reference_phase_controls_passed': 40,
        'classical_cases_verified': 12, 'full_classical_development_cases': 7,
        'real_box_scorer_comparison': checks, 'model_training_or_filter_executed': False,
        'curve_geometry_is_topology_certification': False}
    write_once(folder/'report.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True);p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
