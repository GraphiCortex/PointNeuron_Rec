"""CPU control: score versus coverage priority in saved global proposal selection.

Uses three capped validation neurons, frozen local proposals and matched global
candidate/output limits and sphere IoU. No labels enter selection. No production
settings, inference, graph tracing or training are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from scripts.initialize_geodesic_graph import coverage_truncate, filter_proposals
from scripts.visualize_proposals import select_proposals
from scripts.pointneuron2.audit_full_volume_selection import distances, record, reference_targets
from scripts.pointneuron2.compare_validation_cutoffs import completed_pair
from scripts.pointneuron2.sampling_ablation import digest, read, write


def select_coverage(centers, radii, objectness, budget, iou):
    """Same candidate count and sphere suppression; replace score priority by FPS.

    The synthetic rank tensor is only an ordering argument to the unchanged
    selector. Actual objectness is returned/saved without modification.
    """
    candidate_limit = max(budget * 4, budget)
    order = coverage_truncate(centers, objectness, np.arange(len(centers)), candidate_limit)
    priority = torch.arange(len(order), 0, -1, dtype=torch.float32)
    keep = select_proposals(centers=torch.from_numpy(centers[order]),
        radii=torch.from_numpy(radii[order]), scores=priority, score_threshold=0.,
        top_proposals=budget, nms_radius=0., nms_mode='sphere', iou_threshold=iou)
    return order[np.asarray(keep, dtype=np.int64)], order


def measure_arm(centers, radii, scores, targets, cfg):
    base, _ = record(centers, targets)
    base['rows_within_6_of_reference_curve'] = float(np.mean(distances(targets['curve'], centers) <= 6))
    base['objectness_quantiles'] = np.quantile(scores, [0, .1, .5, .9, 1]).tolist()
    base['score_fraction_at_least_085'] = float(np.mean(scores >= .85))
    gates = {}
    for cutoff in [.5, .85]:
        gate, _ = record(centers[scores >= cutoff], targets)
        pool_indices = filter_proposals(centers, radii, scores, np.zeros((len(centers), 1), dtype=np.float32),
            min_score=cutoff, nms_distance=cfg['graph_nms_distance'], nms_mode=cfg['graph_nms_mode'],
            iou_threshold=.1, max_nodes=cfg['graph_max_nodes'], selection_mode=cfg['graph_selection_mode'])[-1]
        pool, _ = record(centers[pool_indices], targets)
        gates[str(cutoff)] = {'score_gate': gate, 'adaptive_input_pool': pool}
    return {'global_selected': base, 'graph_preparation': gates}


def draw(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    titles = ['Reference curve support @6', 'SWC endpoint support @6', 'Selected centers near reference @6']
    x = np.arange(len(rows)); width = .35
    for offset, name, color in [(-.5, 'current_score', '#8c5aad'), (.5, 'coverage_priority', '#167ca5')]:
        for ax, target, title in zip(axes, ['curve', 'terminal', 'proximity'], titles):
            values = []
            for row in rows:
                r = row['arms'][name]['global_selected']
                v = r['rows_within_6_of_reference_curve'] if target == 'proximity' else r['support'][target]['6']['fraction']
                values.append(100 * v)
            bars = ax.bar(x + offset * width, values, width, color=color, label=name.replace('_', ' '))
            ax.bar_label(bars, fmt='%.1f', fontsize=8)
            ax.set_ylim(0, 112); ax.set_title(title, fontsize=10)
            ax.set_xticks(x, [str(r['sample_index']) for r in rows]); ax.set_xlabel('Validation neuron')
            ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Percent')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    fig.suptitle('Same 4,096 proposals and sphere rule; score priority versus spatial coverage\n'
                 'Saved-pool diagnostic; center proximity is not final reconstruction precision')
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--artifact-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or args.artifact_dir.exists():
        raise FileExistsError('Use fresh directories; preserve earlier diagnostics')
    parent = read(args.comparison)
    if parent['status'] != 'completed' or parent['failures']:
        raise ValueError('Requires completed validation comparison')
    for name, sha in parent['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    plans = {}
    for index in [19, 35, 79]:
        case = next(c for c in parent['cases'] if c['sample_index'] == index)
        if case['status'] != 'completed' or digest(case['comparison']) != case['comparison_sha256']:
            raise ValueError('Saved child comparison changed')
        plans[index] = completed_pair(case['comparison'], parent['checkpoint_sha256'])
        if plans[index]['sample_index'] != index:
            raise ValueError('Unexpected sample identity')
    args.output_dir.mkdir(parents=True); args.artifact_dir.mkdir(parents=True)
    sources = ['scripts/pointneuron2/compare_global_coverage.py',
               'scripts/pointneuron2/audit_full_volume_selection.py',
               'scripts/initialize_geodesic_graph.py', 'scripts/visualize_proposals.py',
               'scripts/evaluate_paper_style_swc.py', 'src/pointneuron/data/swc.py']
    report = {'status': 'running', 'scope': 'three capped internal validation neurons; geometry diagnostic only',
              'comparison': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
              'checkpoint_sha256': parent['checkpoint_sha256'],
              'source_sha256': {name: digest(REPO_ROOT / name) for name in sources},
              'settings': {'candidate_limit': 16384, 'output_budget': 4096, 'sphere_iou': .1,
                           'global_score_floor': 0., 'reference_sample_step': 1., 'diagnostic_radii': [6, 12, 18]},
              'selection_rule': 'highest-score seed, then existing farthest-point coverage order; identical production sphere selector',
              'labels_used_for_selection': False, 'new_inference_or_tracing': False, 'optimizer_steps': 0,
              'promoted_method': None, 'reserved_test_neurons_used': False,
              'limitations': ['Local pool has already been filtered at score 0.5 and sphere-suppressed.',
                  'Nearest-reference center proximity is a diagnostic, not semantic precision or final SWC precision.',
                  'Incomplete references can mark real unannotated neurites as far from reference.',
                  'Endpoint counts in reference forests describe annotation fragments.',
                  'Graph pools precede geodesic selection/tracing; no new SWC has been scored.',
                  'Diagnostic artifacts omit unavailable pre-global features and are not production-ready proposal files.'],
              'cases': []}
    start = time.perf_counter()
    for index in [19, 35, 79]:
        plan = plans[index]
        with np.load(plan['shared_proposals']['path'], allow_pickle=False) as z:
            xyz, radii, scores = (z[k].copy() for k in ['all_centers', 'all_radii', 'all_scores'])
            current = {k: z[k].copy() for k in ['centers', 'radii', 'scores']}
            metadata = json.loads(str(z['metadata']))
        if (len(current['centers']) != 4096 or len(xyz) <= 16384
                or metadata['global_top_proposals'] != 4096 or metadata['global_iou_threshold'] != .1
                or metadata['global_nms_mode'] != 'sphere' or metadata['local_score_threshold'] != .5):
            raise ValueError('Unexpected capped-neuron production settings')
        print(f'neuron={index}: spatial candidate ordering and unchanged sphere suppression on CPU', flush=True)
        selected, order = select_coverage(xyz, radii, scores, 4096, .1)
        if len(selected) != 4096 or len(np.unique(selected)) != 4096 or len(order) != 16384:
            raise ValueError('Coverage arm does not match fixed output/candidate budgets')
        targets, roles = reference_targets(plan['reference'])
        row = {'sample_index': index, 'sample_id': plan['sample_id'], **roles,
               'proposal_path': plan['shared_proposals']['path'], 'proposal_sha256': plan['shared_proposals']['sha256'],
               'reference_path': plan['reference'], 'reference_sha256': plan['reference_sha256'], 'arms': {
                   'current_score': measure_arm(current['centers'], current['radii'], current['scores'], targets, plan['reconstruction']),
                   'coverage_priority': measure_arm(xyz[selected], radii[selected], scores[selected], targets, plan['reconstruction'])}}
        artifact = args.artifact_dir / f'neuron_{index:04d}_coverage_selection.npz'
        np.savez_compressed(artifact, source_indices=selected, candidate_order=order,
                            centers=xyz[selected], radii=radii[selected], scores=scores[selected])
        row.update(artifact_path=str(artifact.resolve()), artifact_sha256=digest(artifact))
        report['cases'].append(row)
        values = [row['arms'][name]['global_selected']['support']['curve']['6']['fraction']
                  for name in ['current_score', 'coverage_priority']]
        print(f'neuron={index}: curve support @6 {values[0]:.4f} -> {values[1]:.4f}', flush=True)
        write(args.output_dir / 'summary.json', report)
    draw(args.output_dir / 'global_coverage_comparison.png', report['cases'])
    report.update(status='completed', elapsed_seconds=time.perf_counter()-start, matched_budget_checks='passed')
    write(args.output_dir / 'summary.json', report)
    print(f"Completed CPU comparison in {report['elapsed_seconds']:.1f}s. Report: {args.output_dir.resolve()}")


if __name__ == '__main__':
    main()
