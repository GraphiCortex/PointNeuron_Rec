"""Fixed 50/50 confidence-anchor and coverage-fill control on saved proposals.

CPU only; references never enter selection. Reuses verified score-only and
pure-coverage controls. Same sphere selector, candidate limit and output budget.
No production settings, model weights, training or tracing are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from scripts.visualize_proposals import select_proposals
from scripts.pointneuron2.audit_full_volume_selection import reference_targets
from scripts.pointneuron2.compare_global_coverage import measure_arm
from scripts.pointneuron2.compare_validation_cutoffs import completed_pair
from scripts.pointneuron2.sampling_ablation import digest, read, write


def source_indices(all_xyz, all_radii, all_scores, current):
    """Map the accepted score-control rows back to exact source rows."""
    lookup = {}
    for i, row in enumerate(np.column_stack((all_xyz, all_radii, all_scores))):
        lookup.setdefault(tuple(row), i)
    try:
        indices = np.asarray([lookup[tuple(row)] for row in
            np.column_stack((current['centers'], current['radii'], current['scores']))], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Current accepted proposal is absent from source pool') from exc
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Current accepted proposals do not map to distinct source rows')
    return indices


def seeded_coverage_order(centers, anchors, limit):
    """Farthest-point fill from retained anchors, with O(N) working memory.

    The inherited seeded coverage helper constructs N x anchors distances.
    Initialize nearest-anchor distances with a KD-tree to avoid that large array.
    Each subsequent fill uses the same minimum squared-distance principle.
    """
    xyz = np.asarray(centers, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.int64)
    if (not len(anchors) or len(np.unique(anchors)) != len(anchors)
            or np.any(anchors < 0) or np.any(anchors >= len(xyz))
            or limit < len(anchors) or limit > len(xyz) or not np.isfinite(xyz).all()):
        raise ValueError('Invalid anchor identities, geometry or candidate count')
    chosen = anchors.tolist()
    mask = np.zeros(len(xyz), dtype=bool); mask[anchors] = True
    d = cKDTree(xyz[anchors]).query(xyz)[0]
    minimum = (d * d).astype(np.float32); minimum[mask] = -np.inf
    while len(chosen) < limit:
        index = int(np.argmax(minimum))
        if not np.isfinite(minimum[index]) or minimum[index] < 0:
            raise ValueError('Cannot fill the fixed candidate count')
        chosen.append(index); mask[index] = True
        distance2 = np.sum((xyz - xyz[index]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance2); minimum[mask] = -np.inf
    return np.asarray(chosen, dtype=np.int64)


def select_hybrid(centers, radii, anchors, budget, iou):
    order = seeded_coverage_order(centers, anchors, max(budget * 4, budget))
    priority = torch.arange(len(order), 0, -1, dtype=torch.float32)
    keep = select_proposals(centers=torch.from_numpy(centers[order]),
        radii=torch.from_numpy(radii[order]), scores=priority, score_threshold=0.,
        top_proposals=budget, nms_radius=0., nms_mode='sphere', iou_threshold=iou)
    selected = order[np.asarray(keep, dtype=np.int64)]
    if len(selected) != budget or not np.array_equal(selected[:len(anchors)], anchors):
        raise ValueError('Output budget or protected-anchor retention failed')
    return selected, order


def draw(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(rows)); width = .25
    for offset, name, color in [(-1, 'current_score', '#8c5aad'),
                                (0, 'coverage_priority', '#167ca5'),
                                (1, 'hybrid_50_50', '#b67529')]:
        for ax, target in zip(axes, ['curve', 'terminal', 'proximity']):
            values = []
            for row in rows:
                r = row['arms'][name]['global_selected']
                v = r['rows_within_6_of_reference_curve'] if target == 'proximity' else r['support'][target]['6']['fraction']
                values.append(100 * v)
            bars = ax.bar(x + offset * width, values, width, color=color, label=name.replace('_', ' '))
            ax.bar_label(bars, fmt='%.1f', fontsize=7)
            ax.set_xticks(x, [str(r['sample_index']) for r in rows]); ax.set_xlabel('Validation neuron')
            ax.set_ylim(0, 112); ax.grid(axis='y', alpha=.2)
    for ax, title in zip(axes, ['Reference curve support @6', 'SWC endpoint support @6',
                               'Selected centers near reference @6']):
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('Percent')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    fig.suptitle('Matched 4,096-proposal controls: fixed 50/50 hybrid\n'
                 'Retain 2,048 confidence anchors, fill spatially; no new SWC reconstruction')
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--coverage-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.artifact_dir.exists():
        raise FileExistsError('Use fresh output/artifact directories; preserve prior results')
    parent = read(args.comparison); prior = read(args.coverage_report)
    if (parent['status'] != 'completed' or parent['failures'] or prior['status'] != 'completed'
            or prior['matched_budget_checks'] != 'passed'
            or prior['comparison_sha256'] != digest(args.comparison)
            or prior['checkpoint_sha256'] != parent['checkpoint_sha256']):
        raise ValueError('Requires the completed matched validation and spatial controls')
    for name, sha in {**parent['source_sha256'], **prior['source_sha256']}.items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    plans = {}; prior_rows = {}
    for index in [19, 35, 79]:
        case = next(c for c in parent['cases'] if c['sample_index'] == index)
        if case['status'] != 'completed' or digest(case['comparison']) != case['comparison_sha256']:
            raise ValueError('Saved cutoff comparison changed')
        plan = completed_pair(case['comparison'], parent['checkpoint_sha256'])
        row = next(r for r in prior['cases'] if r['sample_index'] == index)
        if (plan['sample_index'] != index or plan['sample_id'] != row['sample_id']
                or row['proposal_sha256'] != plan['shared_proposals']['sha256']
                or digest(row['artifact_path']) != row['artifact_sha256']):
            raise ValueError('Spatial control identity/artifact differs')
        plans[index] = plan; prior_rows[index] = row
    sources = ['scripts/pointneuron2/compare_hybrid_global.py',
               'scripts/pointneuron2/compare_global_coverage.py',
               'scripts/pointneuron2/audit_full_volume_selection.py',
               'scripts/visualize_proposals.py', 'scripts/initialize_geodesic_graph.py',
               'scripts/evaluate_paper_style_swc.py', 'src/pointneuron/data/swc.py']
    report = {'status': 'running', 'scope': 'three capped validation neurons; fixed hybrid representation control',
              'comparison': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
              'coverage_report': str(args.coverage_report.resolve()), 'coverage_report_sha256': digest(args.coverage_report),
              'checkpoint_sha256': parent['checkpoint_sha256'],
              'source_sha256': {name: digest(REPO_ROOT / name) for name in sources},
              'settings': {'candidate_limit': 16384, 'output_budget': 4096, 'anchors': 2048,
                           'anchor_fraction': .5, 'sphere_iou': .1, 'graph_cutoffs_measured': [.5, .85]},
              'selection_rule': 'protect first 2048 accepted score-control proposals; farthest-point fill then unchanged sphere selector',
              'labels_used_for_selection': False, 'new_inference_or_tracing': False, 'optimizer_steps': 0,
              'reserved_test_neurons_used': False, 'promoted_method': None,
              'limitations': prior['limitations'] + ['One fixed balance; no ratio sweep or statistical/generalization claim.',
                  'Protected anchors are a subset, not a guarantee of keeping current curve coverage or proximity.'],
              'cases': []}
    args.output_dir.mkdir(parents=True); args.artifact_dir.mkdir(parents=True)
    write(args.output_dir / 'summary.json', report)
    start = time.perf_counter()
    for index in [19, 35, 79]:
        plan = plans[index]; prior_row = prior_rows[index]
        with np.load(plan['shared_proposals']['path'], allow_pickle=False) as z:
            xyz, radii, scores = (z[k].copy() for k in ['all_centers', 'all_radii', 'all_scores'])
            current = {k: z[k].copy() for k in ['centers', 'radii', 'scores']}
            metadata = json.loads(str(z['metadata']))
        if (len(current['centers']) != 4096 or len(xyz) <= 16384
                or metadata['global_top_proposals'] != 4096 or metadata['global_iou_threshold'] != .1
                or metadata['global_nms_mode'] != 'sphere' or float(scores.min()) < .5
                or np.any(np.diff(current['scores']) > 0)):
            raise ValueError('Unexpected capped pool or score-control ordering')
        targets, roles = reference_targets(plan['reference'])
        arms = {'current_score': measure_arm(current['centers'], current['radii'], current['scores'], targets, plan['reconstruction'])}
        with np.load(prior_row['artifact_path'], allow_pickle=False) as old:
            idx = old['source_indices']
            if len(np.unique(idx)) != 4096 or len(np.unique(old['candidate_order'])) != 16384:
                raise ValueError('Prior spatial-control budgets differ')
            for key in ['centers', 'radii', 'scores']:
                source = {'centers': xyz, 'radii': radii, 'scores': scores}[key]
                if not np.array_equal(old[key], source[idx]):
                    raise ValueError('Prior spatial selection rows/scores changed')
            arms['coverage_priority'] = measure_arm(xyz[idx], radii[idx], scores[idx], targets, plan['reconstruction'])
        if arms != prior_row['arms']:
            raise ValueError('Recomputed saved-control measurements differ')
        current_indices = source_indices(xyz, radii, scores, current)
        anchors = current_indices[:2048]
        print(f'neuron={index}: retain 2048 score anchors; spatial fill and same sphere suppression on CPU', flush=True)
        selected, order = select_hybrid(xyz, radii, anchors, 4096, .1)
        if len(np.unique(selected)) != 4096 or len(np.unique(order)) != 16384:
            raise ValueError('Hybrid budgets differ')
        arms['hybrid_50_50'] = measure_arm(xyz[selected], radii[selected], scores[selected], targets, plan['reconstruction'])
        artifact = args.artifact_dir / f'neuron_{index:04d}_hybrid_selection.npz'
        np.savez_compressed(artifact, source_indices=selected, candidate_order=order, anchor_indices=anchors,
                            centers=xyz[selected], radii=radii[selected], scores=scores[selected])
        row = {'sample_index': index, 'sample_id': plan['sample_id'], **roles, 'arms': arms,
               'proposal_path': plan['shared_proposals']['path'], 'proposal_sha256': plan['shared_proposals']['sha256'],
               'reference_path': plan['reference'], 'reference_sha256': plan['reference_sha256'],
               'artifact_path': str(artifact.resolve()), 'artifact_sha256': digest(artifact),
               'retained_anchors': len(anchors), 'saved_control_checks': 'passed'}
        report['cases'].append(row)
        a = arms['hybrid_50_50']['global_selected']
        print(f"neuron={index}: hybrid curve_support_6={a['support']['curve']['6']['fraction']:.4f} "
              f"near_reference_6={a['rows_within_6_of_reference_curve']:.4f}", flush=True)
        write(args.output_dir / 'summary.json', report)
    draw(args.output_dir / 'hybrid_comparison.png', report['cases'])
    report.update(status='completed', elapsed_seconds=time.perf_counter()-start,
                  matched_budget_checks='passed', protected_anchor_checks='passed', saved_control_checks='passed')
    write(args.output_dir / 'summary.json', report)
    print(f"Completed CPU hybrid comparison in {report['elapsed_seconds']:.1f}s. Report: {args.output_dir.resolve()}")


if __name__ == '__main__':
    main()
