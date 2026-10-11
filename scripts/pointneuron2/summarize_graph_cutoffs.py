"""Verify and visualize a completed shared-proposal graph-cutoff comparison.

Reads saved outputs on CPU; no inference, tracing or training. Figures compare
full SWCs in XY/XZ with shared limits; projections can hide depth errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.swc import parse_swc
from scripts.pointneuron2.compare_graph_cutoffs import verify
from scripts.pointneuron2.sampling_ablation import digest, read, write


def segments(tree):
    nodes = {n.node_id: n for n in tree.nodes}
    return np.asarray([[[nodes[n.parent_id].x, nodes[n.parent_id].y, nodes[n.parent_id].z],
                        [n.x, n.y, n.z]] for n in tree.nodes if n.parent_id in nodes],
                      dtype=float).reshape(-1, 2, 3)


def draw(path, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    reference = segments(parse_swc(plan['reference']))
    predictions = [segments(parse_swc(a['reconstruction']['swc_path'])) for a in plan['results']]
    coordinates = np.concatenate([reference.reshape(-1, 3)] + [p.reshape(-1, 3) for p in predictions])
    lower, upper = coordinates.min(0), coordinates.max(0)
    margin = np.maximum((upper - lower) * .04, 2.)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    legend = [Line2D([0], [0], color='#c84c36', label='Reference SWC'),
              Line2D([0], [0], color='#167ca5', label='Reconstruction')]
    for column, arm in enumerate(plan['results']):
        spatial = arm['metrics']['spatial']
        for row, (a, b) in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            ax.add_collection(LineCollection(reference[..., [a, b]], colors='#c84c36', linewidths=1.1, alpha=.85))
            ax.add_collection(LineCollection(predictions[column][..., [a, b]], colors='#167ca5', linewidths=.8, alpha=.75))
            ax.set_xlim(lower[a] - margin[a], upper[a] + margin[a]); ax.set_ylim(lower[b] - margin[b], upper[b] + margin[b])
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels; display axis expanded)')
            ax.set_aspect('equal' if row == 0 else 'auto')
            ax.set_title(f"Graph cutoff {arm['cutoff']} | spatial F1 {spatial['f1']:.3f}\n"
                         f"precision {spatial['precision']:.3f}; recall {spatial['recall']:.3f}")
            ax.legend(handles=legend, fontsize=8)
    fig.suptitle(f"Validation neuron {plan['sample_index']}: same checkpoint/proposals, 128 graph nodes\n"
                 'Full SWC projections; shared axes; no image background; depth errors can be hidden', fontsize=11)
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path, help='Completed comparison.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve previous reports; choose a fresh output directory')
    plan = read(args.comparison)
    if (plan['status'] != 'completed' or plan.get('paired_checks') != 'passed'
            or [a['cutoff'] for a in plan['results']] != [.85, .5]):
        raise ValueError('Requires both completed cutoff arms with paired checks')
    verify(plan)
    proposal = plan['shared_proposals']
    if digest(proposal['path']) != proposal['sha256']:
        raise ValueError('Shared proposal file changed')
    inputs = {str(args.comparison.resolve()): digest(args.comparison)}
    for arm in plan['results']:
        if arm['shared_proposal_sha256'] != proposal['sha256']:
            raise ValueError('Arms have different proposals')
        row = arm['reconstruction']
        if Path(row['proposal_path']).resolve() != Path(proposal['path']).resolve():
            raise ValueError('Proposal paths differ')
        for kind in ['swc', 'graph']:
            path = row[kind + '_path']
            if digest(path) != arm[kind + '_sha256']:
                raise ValueError(f'{kind} output changed')
            inputs[path] = digest(path)
        tree = parse_swc(row['swc_path'])
        if tree.validate() or tree.root_count != 1:
            raise ValueError('Invalid saved SWC')
        root = Path(row['graph_path']).parents[1]
        summary = read(root / 'summary.json')
        if summary['failures'] or summary['samples'] != [row]:
            raise ValueError('Saved reconstruction summary disagrees')
        for kind in ['spatial', 'topology']:
            path = root / f'{kind}.json'
            if read(path)['samples'] != [arm['metrics'][kind]]:
                raise ValueError(f'Saved {kind} metric report disagrees')
            inputs[str(path.resolve())] = digest(path)
        if arm['metrics']['topology']['nodes'] != plan['reconstruction']['graph_max_nodes']:
            raise ValueError('This completed pair does not have the agreed equal graph-node counts')
    high, low = plan['results']
    deltas = {key: low['metrics']['spatial'][key] - high['metrics']['spatial'][key]
              for key in ['precision', 'recall', 'f1', 'gt_to_pred_mean', 'pred_to_gt_mean']}
    deltas['topology_f1'] = low['metrics']['topology']['edge_f1'] - high['metrics']['topology']['edge_f1']
    args.output_dir.mkdir(parents=True)
    report = {'status': 'completed', 'scope': 'one selected validation development case; no generalization claim',
              'method': 'read-only saved-output verification and full SWC projections; no metric rerun',
              'diagnostic_source_sha256': digest(Path(__file__)), 'input_sha256': inputs,
              'deltas_low_minus_high': deltas, 'comparison': plan}
    write(args.output_dir / 'summary.json', report)
    draw(args.output_dir / 'swc_comparison.png', plan)
    print('Verified shared proposals, frozen inputs/code, SWC/graph hashes and saved metric reports.')
    print('Deltas (0.5 minus 0.85):', deltas)
    print(f'Report: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
