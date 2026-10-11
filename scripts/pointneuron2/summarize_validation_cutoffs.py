"""Verify and summarize saved seven-neuron cutoff pairs on CPU.

No inference, training, tracing, parameter changes or metric reruns. Compare
numerical graph/SWC geometry separately from file hashes containing metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.swc import parse_swc
from scripts.pointneuron2.compare_validation_cutoffs import completed_pair, complete_means
from scripts.pointneuron2.sampling_ablation import digest, read, write
from scripts.pointneuron2.summarize_graph_cutoffs import segments


def inspect_pair(plan):
    with np.load(plan['shared_proposals']['path'], allow_pickle=False) as proposals:
        scores, before = proposals['scores'], proposals['all_scores']
        metadata = json.loads(str(proposals['metadata']))
        row = {'sample_index': plan['sample_index'], 'sample_id': plan['sample_id'],
               'proposals': len(scores), 'post_local_pre_global_proposals': len(before),
               'minimum_final_score': float(scores.min()),
               'final_scores_below_085': int((scores < .85).sum()),
               'pre_global_scores_below_085': int((before < .85).sum()),
               'patches': metadata['patches'], 'arms': []}
    graphs = []
    trees = []
    for arm in plan['results']:
        with np.load(arm['reconstruction']['graph_path'], allow_pickle=False) as graph:
            graphs.append({k: graph[k].copy() for k in graph.files if k != 'metadata'})
            metadata = json.loads(str(graph['metadata']))
        tree = parse_swc(arm['reconstruction']['swc_path'])
        if tree.validate() or tree.root_count != 1:
            raise ValueError('Saved SWC is invalid')
        trees.append(np.asarray([[n.node_id, n.x, n.y, n.z, n.radius, n.parent_id] for n in tree.nodes]))
        row['arms'].append({'cutoff': arm['cutoff'], 'spatial': arm['metrics']['spatial'],
                            'topology': arm['metrics']['topology'],
                            'reconstruction': arm['reconstruction'],
                            'adaptive_selection': metadata['adaptive_selection']})
    row['same_graph_numeric_arrays'] = (graphs[0].keys() == graphs[1].keys()
        and all(np.array_equal(graphs[0][k], graphs[1][k], equal_nan=True) for k in graphs[0]))
    row['same_swc_numeric_nodes'] = np.array_equal(trees[0], trees[1])
    row['deltas_low_minus_high'] = {k: row['arms'][1]['spatial'][k] - row['arms'][0]['spatial'][k]
                                  for k in ['precision', 'recall', 'f1']}
    return row


def figures(output, rows, plans):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    y = np.arange(len(rows))
    for r, yy in zip(rows, y):
        values = [a['spatial']['f1'] for a in r['arms']]
        axes[0].plot(values, [yy, yy], color='#999999')
    for arm, color in [(0, '#8c5aad'), (1, '#167ca5')]:
        axes[0].scatter([r['arms'][arm]['spatial']['f1'] for r in rows], y,
                        label=f"cutoff {rows[0]['arms'][arm]['cutoff']}", color=color, s=45)
    axes[0].set_yticks(y, [str(r['sample_index']) for r in rows])
    axes[0].set_xlim(0, 1); axes[0].set_xlabel('Spatial F1 (saved full-SWC metrics)')
    axes[0].set_ylabel('Validation neuron'); axes[0].legend(loc='lower left')
    delta = [100 * r['deltas_low_minus_high']['f1'] for r in rows]
    axes[1].barh(y, delta, color=['#167ca5' if d > 0 else '#c84c36' if d < 0 else '#999999' for d in delta])
    axes[1].axvline(0, color='#555555', linewidth=.8)
    axes[1].set_yticks(y, [str(r['sample_index']) for r in rows])
    axes[1].set_xlabel('F1 change: 0.5 minus 0.85 (percentage points)')
    for yy, d in zip(y, delta):
        axes[1].annotate(f'{d:+.2f}', (d, yy), xytext=(5 if d >= 0 else -5, 0),
                         textcoords='offset points', va='center', ha='left' if d >= 0 else 'right')
    axes[1].set_xlim(-4, 6)
    fig.suptitle('Seven validation neurons: 2 improve, 1 worsens, 4 unchanged\nSame checkpoint; each pair shares proposals; no reserved test data')
    fig.savefig(output / 'paired_f1.png', dpi=160); plt.close(fig)

    legend = [Line2D([0], [0], color='#c84c36', label='Reference SWC'),
              Line2D([0], [0], color='#167ca5', label='Reconstruction')]
    for index in [44, 94, 135]:
        plan = plans[index]
        reference = segments(parse_swc(plan['reference']))
        predictions = [segments(parse_swc(a['reconstruction']['swc_path'])) for a in plan['results']]
        coordinates = np.concatenate([reference.reshape(-1, 3)] + [s.reshape(-1, 3) for s in predictions])
        lower, upper = coordinates.min(0), coordinates.max(0)
        margin = np.maximum((upper - lower) * .04, 2.)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for col, arm in enumerate(plan['results']):
            for rr, (a, b) in enumerate([(0, 1), (0, 2)]):
                ax = axes[rr, col]
                ax.add_collection(LineCollection(reference[..., [a, b]], colors='#c84c36', linewidths=1.1, alpha=.85))
                ax.add_collection(LineCollection(predictions[col][..., [a, b]], colors='#167ca5', linewidths=.8, alpha=.75))
                ax.set_xlim(lower[a]-margin[a], upper[a]+margin[a])
                ax.set_ylim(lower[b]-margin[b], upper[b]+margin[b])
                ax.set_aspect('equal' if rr == 0 else 'auto')
                ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels; display axis expanded)')
                s = arm['metrics']['spatial']; nodes = arm['metrics']['topology']['nodes']
                ax.set_title(f"Cutoff {arm['cutoff']} | F1 {s['f1']:.3f} | graph nodes {nodes}\n"
                             f"Precision {s['precision']:.3f}; recall {s['recall']:.3f}")
        fig.legend(handles=legend, loc='outside lower center', ncol=2)
        fig.suptitle(f'Validation neuron {index}: full SWC projections with shared axes\n'
                     'No image background; projections can hide depth errors')
        fig.savefig(output / f'neuron_{index:04d}_swcs.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier reports; use a fresh output directory')
    parent = read(args.comparison)
    if parent['status'] != 'completed' or parent['failures']:
        raise ValueError('Requires all seven completed pairs with no failures')
    if parent['expected_neurons'] != [19, 21, 35, 44, 79, 94, 135]:
        raise ValueError('Unexpected validation denominator')
    for name, sha in parent['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    plans = {}
    for case in parent['cases']:
        if digest(case['comparison']) != case['comparison_sha256']:
            raise ValueError('Child comparison changed')
        plan = completed_pair(case['comparison'], parent['checkpoint_sha256'])
        if plan['sample_index'] != case['sample_index'] or plan['sample_id'] != case['sample_id']:
            raise ValueError('Child identity differs')
        plans[case['sample_index']] = plan
    means = complete_means(parent['cases'], plans)
    if means is None or means != parent['means']:
        raise ValueError('Saved seven-neuron means differ from verified pairs')
    rows = [inspect_pair(plans[i]) for i in parent['expected_neurons']]
    delta = np.asarray([r['deltas_low_minus_high']['f1'] for r in rows])
    report = {'status': 'completed', 'method': 'read-only verification of saved outputs; no metric reruns',
              'comparison_path': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
              'diagnostic_source_sha256': digest(Path(__file__)), 'means': means, 'cases': rows,
              'improved_neurons': [r['sample_index'] for r in rows if r['deltas_low_minus_high']['f1'] > 0],
              'worsened_neurons': [r['sample_index'] for r in rows if r['deltas_low_minus_high']['f1'] < 0],
              'unchanged_neurons': [r['sample_index'] for r in rows if r['deltas_low_minus_high']['f1'] == 0],
              'median_f1_delta': float(np.median(delta)),
              'mean_f1_delta_excluding_21': float(np.mean([r['deltas_low_minus_high']['f1'] for r in rows if r['sample_index'] != 21])),
              'validation_comparison': parent}
    args.output_dir.mkdir(parents=True)
    figures(args.output_dir, rows, plans)
    write(args.output_dir / 'summary.json', report)
    print('Verified all seven pairs, frozen sources, proposal/graph/SWC hashes and recomputed means.')
    for row in rows:
        print(f"neuron={row['sample_index']} F1_delta={row['deltas_low_minus_high']['f1']:+.6f} "
              f"proposals={row['proposals']} below_085={row['final_scores_below_085']} "
              f"same_graph_arrays={row['same_graph_numeric_arrays']} same_SWC={row['same_swc_numeric_nodes']}")
    print(f'Report: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
