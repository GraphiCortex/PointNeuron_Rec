"""Verify saved dropout controls and summarize their two-crop geometry/score support."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pointneuron.data.query_cache import file_digest


def cutoff_metrics(centers, scores, nodes, mask, thresholds):
    rows = []
    for threshold in thresholds:
        crops = []
        for i in range(len(centers)):
            target = nodes[i, mask[i], 1:4].astype(np.float64)
            chosen = centers[i, scores[i] >= threshold].astype(np.float64)
            if not len(target):
                raise ValueError('This diagnostic requires annotated fitted crops')
            distance = np.linalg.norm(chosen[:, None] - target[None], axis=-1)
            crops.append({'accepted': len(chosen),
                          'target_coverage_6': float((distance.min(0) <= 6).mean()) if len(chosen) else 0.,
                          'proposal_precision_6': float((distance.min(1) <= 6).mean()) if len(chosen) else None})
        rows.append({'threshold': threshold, 'crops': crops,
                     'mean_target_coverage_6': float(np.mean([c['target_coverage_6'] for c in crops]))})
    return rows


def summarize(baseline_path, control_path, audit_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError('Use a fresh summary directory')
    baseline, control = [json.loads(p.read_text()) for p in (baseline_path, control_path)]
    for run in (baseline, control):
        if run['status'] != 'completed' or run['paired_checks'] != 'passed':
            raise ValueError('Expected completed paired checks')
    expected = copy.deepcopy(baseline['config']); expected['decoder']['dropout'] = 0.
    assert baseline['config']['decoder']['dropout'] == .1 and control['config'] == expected
    audit_path = audit_dir / 'audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['comparison_sha256'] == file_digest(control_path) and audit['device'] == 'cuda'
    result = {'date': '2026-10-12', 'scope': 'same two fitted training crops; no validation or SWC evaluation',
              'status': 'verified', 'config_difference': 'decoder.dropout: 0.1 -> 0.0',
              'baseline_sha256': file_digest(baseline_path), 'control_sha256': file_digest(control_path),
              'audit_sha256': file_digest(audit_path), 'arms': [],
              'diagnostic_precision_definition': 'fraction of accepted slots within six voxels of a sparse target; duplicates count separately; not reconstruction precision',
              'thresholds_are_diagnostic_only': True}
    for old, new, checked in zip(baseline['arms'], control['arms'], audit['arms']):
        assert old['arm'] == new['arm'] == checked['arm']
        for key in ('initial_encoder_sha256', 'inputs_sha256', 'manifest_sha256', 'partition_sha256',
                    'updates', 'crop_paths', 'pointwise_recipe'):
            assert old[key] == new[key], f'Unmatched {key}'
        assert new['updates'] == 500 and new['device'] == 'cuda' and new['roundtrip'] == 'exact'
        checkpoint = control_path.parent / new['arm'] / 'proposal.pt'
        assert checked['checkpoint_unchanged'] and file_digest(checkpoint) == checked['checkpoint_sha256']
        modes = checked['modes']
        for key in ('mean_target_distance', 'target_coverage_6'):
            assert abs(new['final_fit'][key] - modes['eval_pair']['geometry'][key]) < .001
        result['arms'].append({'arm': new['arm'], 'baseline_fit': old['final_fit'], 'control_fit': new['final_fit'],
                             'control_selection': new['final_selection'], 'checkpoint_sha256': checked['checkpoint_sha256'],
                             'median_gpu_update_seconds': new['median_update_seconds'],
                             'modes': {k: modes[k] for k in ('eval_pair', 'eval_single', 'global_batch_stats_pair', 'encoder_batch_stats_pair')},
                             'pointwise_geometry_reproduced_exactly': old['final_fit'] == new['final_fit'] if new['arm'] == 'pointwise' else None})
    with np.load(audit_dir / 'proposals.npz', allow_pickle=False) as z:
        result['query_threshold_diagnostic'] = cutoff_metrics(z['query_eval_pair_centers'], z['query_eval_pair_scores'],
              z['nodes'], z['mask'], (0., .05, .1, .2, .3, .4, .5, .6, .8))
        result['pointwise_threshold_0_5'] = cutoff_metrics(z['pointwise_eval_pair_centers'], z['pointwise_eval_pair_scores'],
              z['nodes'], z['mask'], (.5,))
    losses = next(a for a in control['arms'] if a['arm'] == 'query')['losses']
    result['query_training_windows'] = [
        {'updates': [start + 1, start + 50],
         **{k: float(np.mean([v[k] for v in losses[start:start + 50]])) for k in ('objectness', 'center', 'total')}}
        for start in (0, 100, 200, 300, 450)]
    output_dir.mkdir(parents=True)
    (output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'query_thresholds': result['query_threshold_diagnostic'],
                      'report': str((output_dir / 'summary.json').resolve())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.baseline, args.control, args.audit_dir, args.output_dir)


if __name__ == '__main__':
    main()
