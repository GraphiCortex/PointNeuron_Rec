"""Summarize completed reconstruction and audit saved proposal selection on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from scripts.audit_selection_stage import audit_sample, path_key
from scripts.pointneuron2.sampling_ablation import check_matches, checked_manifest, digest, read, write


def draw_comparison(path, sample_index, sample, arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    def segments(tree):
        nodes = {n.node_id: n for n in tree.nodes}
        return np.asarray([[[nodes[n.parent_id].x, nodes[n.parent_id].y, nodes[n.parent_id].z], [n.x, n.y, n.z]]
                           for n in tree.nodes if n.parent_id in nodes], dtype=float).reshape(-1, 2, 3)

    reference = segments(parse_swc(sample.swc_path))
    predictions = [segments(parse_swc(a['reconstruction']['swc_path'])) for a in arms]
    coordinates = np.concatenate([reference.reshape(-1, 3)] + [p.reshape(-1, 3) for p in predictions])
    lower, upper = coordinates.min(0), coordinates.max(0)
    margin = np.maximum((upper - lower) * .04, 2.)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    legend = [Line2D([0], [0], color='#e34a33', label='SWC reference'),
              Line2D([0], [0], color='#167ca5', label='Reconstructed SWC')]
    for column, arm in enumerate(arms):
        for row, (a, b) in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            ax.add_collection(LineCollection(reference[..., [a, b]], colors='#e34a33', linewidths=1., alpha=.8))
            ax.add_collection(LineCollection(predictions[column][..., [a, b]], colors='#167ca5', linewidths=.8, alpha=.7))
            ax.set_xlim(lower[a] - margin[a], upper[a] + margin[a]); ax.set_ylim(lower[b] - margin[b], upper[b] + margin[b])
            ax.set_aspect('equal'); ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels)')
            if row == 1 and upper[b] - lower[b] < .1 * (upper[a] - lower[a]):
                ax.set_aspect('auto')
                ax.set_ylabel('Z (voxels; display axis expanded)')
            ax.set_title(f"{arm['name']} | spatial F1 {arm['spatial']['f1']:.3f}\n"
                         f"topology F1 {arm['topology']['edge_f1']:.3f}; bridges {arm['topology']['bridge_edges']}")
            ax.legend(handles=legend, fontsize=8)
    fig.suptitle(f'Validation neuron {sample_index}: five-epoch matched reconstruction\n'
                 'XY / XZ projections, shared axes; no image background; projections can hide depth errors')
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory; preserve existing reports')
    plan = read(args.experiment / 'experiment.json')
    if plan['status'] != 'evaluated':
        raise ValueError('Requires all three completed evaluation arms')
    manifest = checked_manifest(plan, args.experiment, allow_historical_reader=True)
    check_matches([read(args.experiment / a['name'] / 'run.json') for a in plan['arms']])
    expected = sorted(r['sample_index'] for r in read(plan['partition'])['partitions']['val'])
    samples = scan_gold166(plan['data_root'])
    report = {'status': 'running', 'scope': 'seven internal validation neurons; three five-epoch arms; no reserved test data',
              'experiment': str(args.experiment.resolve()), 'config': plan['config'], 'expected_neurons': expected,
              'matching_checks': 'passed', 'manifest_sha256': plan['manifest_sha256'],
              'source_sha256': {name: digest(REPO_ROOT / name) for name in
                                ['scripts/pointneuron2/summarize_ablation.py', 'scripts/audit_selection_stage.py']},
              'input_sha256': {str((args.experiment / 'experiment.json').resolve()): digest(args.experiment / 'experiment.json')},
              'selection_audit_settings': {'hit_distance_voxels': 8., 'segment_step_voxels': 4., 'proposal_score_threshold': .85},
              'arms': []}
    for arm in plan['arms']:
        name = arm['name']; directory = args.experiment / f'{name}_reconstruction'
        if digest(args.experiment / name / 'proposal.pt') != arm['checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        payload = {kind: read(directory / f'{kind}.json') for kind in ('summary', 'spatial', 'topology')}
        if payload['summary'].get('failures'):
            raise ValueError('Reconstruction has recorded failures')
        for kind, data in payload.items():
            if sorted(r['sample_index'] for r in data['samples']) != expected:
                raise ValueError(f'Incomplete/duplicate {kind} denominator for {name}')
            report['input_sha256'][str((directory / f'{kind}.json').resolve())] = digest(directory / f'{kind}.json')
        rows = []
        topology_map = {r['sample_index']: r for r in payload['topology']['samples']}
        spatial_map = {r['sample_index']: r for r in payload['spatial']['samples']}
        graph_map = {path_key(Path(r['graph_path'])): topology_map[r['sample_index']] for r in payload['summary']['samples']}
        for row in payload['summary']['samples']:
            index = row['sample_index']
            cached = next(r for r in manifest['records'] if r['sample_index'] == index)
            if samples[index].sample_id != cached['sample_id']:
                raise ValueError('Source identity changed')
            with np.load(cached['resolved_path'], allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata']))
            if digest(samples[index].swc_path) != metadata['swc_sha256']:
                raise ValueError('Reference changed since caching')
            for path in [row['proposal_path'], row['graph_path'], row['swc_path'], samples[index].swc_path]:
                report['input_sha256'][str(Path(path).resolve())] = digest(path)
            audit = audit_sample(row, samples, graph_map, proposal_score_threshold=.85, hit_distance=8., segment_step=4.)
            rows.append({'name': name, 'sample_index': index, 'sample_id': samples[index].sample_id,
                         'reconstruction': row, 'spatial': spatial_map[index], 'topology': topology_map[index],
                         'selection_audit': audit})
            print(f"{name} neuron={index}: spatial={spatial_map[index]['f1']:.3f} "
                  f"topology={topology_map[index]['edge_f1']:.3f} "
                  f"proposal/selected segment coverage={audit['proposal_segment_coverage']:.3f}/{audit['selected_segment_coverage']:.3f}", flush=True)
        report['arms'].append({'name': name, 'checkpoint_sha256': arm['checkpoint_sha256'],
                               'spatial_summary': payload['spatial']['summary'], 'topology_summary': payload['topology']['summary'],
                               'reconstruction_summary': payload['summary']['summary'], 'samples': rows})
    args.output_dir.mkdir(parents=True)
    for index in (19, 79, 135):
        draw_comparison(args.output_dir / f'sample_{index:04d}_comparison.png', index, samples[index],
                        [next(r for r in arm['samples'] if r['sample_index'] == index) for arm in report['arms']])
    report['status'] = 'completed'; write(args.output_dir / 'summary.json', report)
    print(f'Report: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
