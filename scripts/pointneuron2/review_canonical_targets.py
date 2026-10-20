"""Verify preparation evidence and interpret the strict fraction nonregression failure."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.pointneuron2.prepare_canonical_targets import read, write, digest
from pointneuron.data.canonical_curve_targets import CanonicalTargetBuilder
from pointneuron.data.swc import parse_swc
from pointneuron.graph.representation_oracle import target_graph, clipped_reference_graph, sample_arclength, distance_to_graph
import torch


def run():
    torch.set_num_threads(2)
    config_path = ROOT / 'configs/pointneuron2/canonical_targets.json'; config = read(config_path)
    directory = ROOT / config['output']; report = read(directory / 'summary.json')
    assert report['status'] == 'completed' and report['config'] == config
    for path, expected in report['protected_files_sha256'].items():
        assert digest(path) == expected
    for path, expected in report['source_files_sha256'].items():
        assert digest(ROOT / path) == expected
    assert len(report['records']) == 262 and all(r['topology']['passed'] and r['newly_unrepresented_required_edges'] == 0 for r in report['records'])
    assert sum(len(r['removed']) for r in report['records']) == 215
    assert report['preparation_gate']['passed'] is False
    assert report['preparation_gate']['four_crop_no_precision_recall_decline_at_2'] is False
    assert read(directory / 'four_crop_cache/manifest.json')['training_ready'] is False
    for row in report['four_crop_cache']:
        assert digest(directory / row['path']) == row['sha256']
        with np.load(directory / row['path'], allow_pickle=False) as saved, np.load(ROOT / row['source_path'], allow_pickle=False) as original:
            for name in original.files:
                if name not in ('nodes', 'roles', 'target_edges', 'target_components', 'junction_mask'):
                    assert np.array_equal(saved[name], original[name])
    source = read(ROOT / config['source_audit'])
    row = next(r for r in source['records'] if [r['sample_index'], r['patch_index']] == [86, 3])
    origin = next(s for s in source['sources'] if s['sample_index'] == 86)
    tree = parse_swc(ROOT.parent / 'data/gold166' / origin['path'])
    center, radius = row['patch_center'], config['patch_radius']
    builder = CanonicalTargetBuilder(tree, **config['target_policy'])
    old = target_graph(tree, [(center, radius)], builder.step); new = builder.patch(center, radius)
    assert all(old.identities[i] in {tuple(k) for k in new.removed} for i in (249, 406))
    reference = clipped_reference_graph(tree, [(center, radius)])
    before, after = [], []
    for graph, destination in ((old, before), (new.graph, after)):
        points, weights = sample_arclength(graph, config['quadrature_step'])
        bad = distance_to_graph(points, reference) > 2.
        destination.extend([points[bad], weights[bad], float(weights.sum()), float(weights[bad].sum())])
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    probes, _ = sample_arclength(reference, config['quadrature_step'])
    assert np.array_equal(distance_to_graph(probes, old) > 2., distance_to_graph(probes, new.graph) > 2.)
    stats = {'same_out_of_tolerance_prediction_quadrature_points_and_weights': True,
        'same_out_of_tolerance_reference_probe_mask': True,
        'old_export_length': before[2], 'canonical_export_length': after[2],
        'old_out_of_tolerance_length': before[3], 'canonical_out_of_tolerance_length': after[3],
        'precision_decline_percentage_points': ((1 - before[3] / before[2]) - (1 - after[3] / after[2])) * 100,
        'interpretation': 'The identical out-of-tolerance numerator is divided by a slightly shorter total target curve. This is post-hoc interpretation, not a gate change.'}
    assert not torch.cuda.is_initialized()
    result = {'schema': 'pointneuron2-canonical-target-review-v1', 'date': '2026-10-21', 'status': 'completed',
        'source_report': str((directory / 'summary.json').relative_to(ROOT)), 'source_report_sha256': digest(directory / 'summary.json'),
        'reviewer_sha256': digest(Path(__file__)), 'config': config, 'totals': report['totals'],
        'preparation_gate': report['preparation_gate'], 'known_dense_residual_samples_removed': True,
        'dense_geometry_interpretation': stats, 'four_crop_records': [r for r in report['records'] if [r['sample_index'], r['patch_index']] in config['training_crops']],
        'four_crop_cache': report['four_crop_cache'], 'source_files_sha256': report['source_files_sha256'],
        'cuda_initialized': False, 'model_forwards': 0, 'optimizer_updates': 0,
        'decision': 'Implementation and CPU audit complete; strict preparation gate remains failed. Review a prospective geometry criterion before any matched GPU fit.'}
    output = directory / 'review.json'
    if output.exists():
        raise FileExistsError('Preserve review evidence')
    write(output, result)
    print(json.dumps({'review': str(output), 'totals': result['totals'], 'dense_geometry_interpretation': stats,
        'preparation_gate': result['preparation_gate']}, indent=2))


if __name__ == '__main__':
    run()
