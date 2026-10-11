"""Measure branch support through saved full-volume proposal/graph selection.

CPU only: no volume decoding, network inference, geodesic tracing or training.
References measure support; they never select or modify proposals. Center
support is distinct from final traced-SWC recall, especially for sparse nodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from pointneuron.data.swc import parse_swc
from scripts.evaluate_paper_style_swc import resample_swc_points, deterministic_downsample
from scripts.initialize_geodesic_graph import filter_proposals
from scripts.pointneuron2.compare_validation_cutoffs import completed_pair, complete_means
from scripts.pointneuron2.sampling_ablation import digest, read, write

RADII = (6, 12, 18)
STAGES = ('post_local_pre_global', 'global_rank_candidates', 'global_selected', 'graph_score_gate',
          'adaptive_input_pool', 'selected_graph_centers', 'traced_swc_points')


def distances(centers, targets):
    centers = np.asarray(centers).reshape(-1, 3)
    targets = np.asarray(targets).reshape(-1, 3)
    if not np.isfinite(centers).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite geometry')
    return cKDTree(centers).query(targets)[0] if len(centers) else np.full(len(targets), np.inf)


def support(d, radius):
    return {'covered': int(np.count_nonzero(d <= radius)), 'total': len(d),
            'fraction': float(np.mean(d <= radius)) if len(d) else None}


def change(before, after, radius):
    if before.shape != after.shape:
        raise ValueError('Support comparisons must use identical reference targets')
    a, b = before <= radius, after <= radius
    return {'lost': int(np.count_nonzero(a & ~b)), 'gained': int(np.count_nonzero(~a & b)),
            'retained': int(np.count_nonzero(a & b)), 'absent_both': int(np.count_nonzero(~a & ~b)),
            'total': len(a)}


def record(centers, targets):
    d = {name: distances(centers, points) for name, points in targets.items()}
    return ({'rows': len(centers), 'support': {
        name: {str(r): support(values, r) for r in RADII} for name, values in d.items()}}, d)


def reference_targets(path):
    tree = parse_swc(path)
    if tree.validate():
        raise ValueError('Invalid reference SWC')
    nodes = {n.node_id: n for n in tree.nodes}
    children = Counter(n.parent_id for n in tree.nodes if n.parent_id in nodes)
    terminal = [n for n in tree.nodes if children[n.node_id] == 0 and n.parent_id in nodes]
    junction = [n for n in tree.nodes if children[n.node_id] > 1]
    points = deterministic_downsample(resample_swc_points(tree, 1.), 250000)
    return {'curve': points,
            'terminal': np.asarray([[n.x, n.y, n.z] for n in terminal], dtype=np.float32).reshape(-1, 3),
            'junction': np.asarray([[n.x, n.y, n.z] for n in junction], dtype=np.float32).reshape(-1, 3)}, {
        'reference_nodes': len(tree.nodes), 'reference_roots': tree.root_count,
        'terminal_ids': [n.node_id for n in terminal], 'junction_ids': [n.node_id for n in junction]}


def audit_pair(plan, artifact_dir):
    targets, roles = reference_targets(plan['reference'])
    with np.load(plan['shared_proposals']['path'], allow_pickle=False) as z:
        centers, radii, scores, features = (z[k].copy() for k in ['centers', 'radii', 'scores', 'features'])
        before = z['all_centers'].copy(); before_scores = z['all_scores'].copy()
        metadata = json.loads(str(z['metadata']))
    # Replay the exact score-only prefix in production select_proposals, without
    # rerunning sphere NMS. Production aggregation creates these tensors on CPU.
    rank_scores = torch.from_numpy(before_scores)
    eligible = torch.nonzero(rank_scores >= 0., as_tuple=False).flatten()
    limit = metadata['global_top_proposals']
    rank_indices = eligible[rank_scores[eligible].argsort(descending=True)[:max(limit * 4, limit)]].numpy()
    rank_centers = before[rank_indices]
    if np.max(distances(rank_centers, centers), initial=0.) != 0.:
        raise ValueError('Saved final centers are outside replayed global score prefix')
    base_records, base_distances = {}, {}
    for name, xyz in [('post_local_pre_global', before), ('global_rank_candidates', rank_centers),
                      ('global_selected', centers)]:
        base_records[name], base_distances[name] = record(xyz, targets)
    row = {'sample_index': plan['sample_index'], 'sample_id': plan['sample_id'],
           'reference_sha256': plan['reference_sha256'], 'reference': str(plan['reference']),
           **roles, 'proposal_metadata': metadata, 'arms': []}
    raw = {name + '_xyz': xyz for name, xyz in targets.items()}
    for arm in plan['results']:
        cutoff = arm['cutoff']; cfg = plan['reconstruction']
        gated = np.flatnonzero(scores >= cutoff)
        pool = filter_proposals(centers, radii, scores, features, min_score=cutoff,
            nms_distance=cfg['graph_nms_distance'], nms_mode=cfg['graph_nms_mode'],
            iou_threshold=.1, max_nodes=cfg['graph_max_nodes'],
            selection_mode=cfg['graph_selection_mode'])[-1]
        with np.load(arm['reconstruction']['graph_path'], allow_pickle=False) as graph:
            selected = graph['centers'].copy(); indices = graph['original_indices']
            graph_metadata = json.loads(str(graph['metadata']))
            if (not np.isin(indices, pool).all() or not np.array_equal(selected, centers[indices])
                    or len(pool) != graph_metadata['adaptive_selection']['selection_pool_nodes']):
                raise ValueError('Replayed input pool differs from saved graph provenance')
        prediction = parse_swc(arm['reconstruction']['swc_path'])
        swc_points = deterministic_downsample(resample_swc_points(prediction, 1.), 250000)
        stage_records = dict(base_records); stage_distances = dict(base_distances)
        for name, xyz in [('graph_score_gate', centers[gated]), ('adaptive_input_pool', centers[pool]),
                          ('selected_graph_centers', selected), ('traced_swc_points', swc_points)]:
            stage_records[name], stage_distances[name] = record(xyz, targets)
        recall = stage_records['traced_swc_points']['support']['curve']['6']['fraction']
        precision = float(np.mean(distances(targets['curve'], swc_points) <= 6))
        saved = arm['metrics']['spatial']
        if (len(targets['curve']) != saved['gt_points'] or len(swc_points) != saved['pred_points']
                or not np.isclose(recall, saved['recall'], atol=1e-7, rtol=0)
                or not np.isclose(precision, saved['precision'], atol=1e-7, rtol=0)):
            raise ValueError('KD-tree full-SWC measurement disagrees with saved spatial metrics')
        transitions = []
        for parent, stage in zip(STAGES, STAGES[1:]):
            transitions.append({'from': parent, 'to': stage,
                'changes': {name: {str(r): change(stage_distances[parent][name], stage_distances[stage][name], r)
                                   for r in RADII} for name in targets}})
        row['arms'].append({'cutoff': cutoff, 'stages': stage_records, 'transitions': transitions,
                            'adaptive_selection': graph_metadata['adaptive_selection'],
                            'saved_spatial': saved, 'reconstruction': arm['reconstruction'],
                            'replayed_pool_matches_saved_graph': True, 'swc_metrics_match': True})
        prefix = str(cutoff).replace('.', '_')
        for stage, values in stage_distances.items():
            for name, d in values.items():
                raw[f'cutoff_{prefix}_{stage}_{name}_distance'] = d
    artifact = artifact_dir / f"neuron_{plan['sample_index']:04d}_distances.npz"
    np.savez_compressed(artifact, **raw)
    row['distances_path'] = str(artifact.resolve()); row['distances_sha256'] = digest(artifact)
    return row


def draw(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Pre-global', 'Rank prefix', 'Global', 'Score gate', 'Graph pool', 'Graph nodes']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for col, index in enumerate([35, 79, 94]):
        row = next(r for r in rows if r['sample_index'] == index)
        arm = next(a for a in row['arms'] if a['cutoff'] == .5)
        for rr, target in enumerate(['curve', 'terminal']):
            ax = axes[rr, col]
            for radius, color in [(6, '#167ca5'), (18, '#b67529')]:
                values = [100 * arm['stages'][s]['support'][target][str(radius)]['fraction'] for s in STAGES[:-1]]
                ax.plot(range(6), values, marker='o', color=color, label=f'Centers within {radius} voxels')
            swc = 100 * arm['stages']['traced_swc_points']['support'][target]['6']['fraction']
            ax.scatter([6], [swc], marker='D', color='#33854b', label='Traced SWC within 6 voxels', zorder=3)
            ax.axvline(5.5, color='#888888', linestyle=':', linewidth=.8)
            ax.set_xticks(range(7), labels + ['Final SWC'], rotation=25, ha='right', fontsize=8)
            ax.set_ylim(0, 105); ax.grid(axis='y', alpha=.2)
            ax.set_title(f"Neuron {index}: {'reference curve' if target == 'curve' else 'SWC endpoints'}")
            ax.set_ylabel('Supported reference samples (%)' if target == 'curve' else 'Supported endpoints (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    fig.suptitle('Saved full-volume selection, cutoff 0.5: center support versus traced geometry\n'
                 'Same weights/settings; wider radius checks spacing effects; no new reconstruction')
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--artifact-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or args.artifact_dir.exists():
        raise FileExistsError('Use fresh directories; preserve prior audits')
    start = time.perf_counter()
    parent = read(args.comparison)
    if parent['status'] != 'completed' or parent['failures']:
        raise ValueError('Requires completed seven-neuron comparison')
    if parent['expected_neurons'] != [19, 21, 35, 44, 79, 94, 135]:
        raise ValueError('Unexpected validation set')
    for name, sha in parent['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    plans = {}
    for case in parent['cases']:
        if digest(case['comparison']) != case['comparison_sha256']:
            raise ValueError('Child plan changed')
        plan = completed_pair(case['comparison'], parent['checkpoint_sha256'])
        if plan['sample_index'] != case['sample_index'] or plan['sample_id'] != case['sample_id']:
            raise ValueError('Child identity differs')
        plans[case['sample_index']] = plan
    if complete_means(parent['cases'], plans) != parent['means']:
        raise ValueError('Saved means disagree')
    args.output_dir.mkdir(parents=True); args.artifact_dir.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'all seven internal validation neurons; saved full-volume outputs',
              'comparison': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
              'checkpoint_sha256': parent['checkpoint_sha256'], 'source_sha256': parent['source_sha256'],
              'input_cases': parent['cases'],
              'diagnostic_source_sha256': digest(Path(__file__)), 'device': 'cpu', 'optimizer_steps': 0,
              'new_inference_or_tracing': False, 'reserved_test_neurons_used': False,
              'radii_voxels': list(RADII), 'reference_sample_step': 1., 'max_reference_points': 250000,
              'limitations': ['Pre-global arrays are already locally filtered/suppressed.',
                  'Rank-prefix loss is isolated; subsequent global loss still combines sphere suppression and final cap.',
                  'Center proximity is not segmentation accuracy, node precision or final SWC recall.',
                  '6/12/18 are diagnostic radii, not tuned reconstruction settings.',
                  'SWC leaves are annotation endpoints; in reference forests they may be fragment ends, not biological terminals.',
                  'Saved proposals cannot isolate image masking, grid coverage or encoder input losses.'],
              'cases': []}
    for index in parent['expected_neurons']:
        row = audit_pair(plans[index], args.artifact_dir)
        report['cases'].append(row)
        arm = next(a for a in row['arms'] if a['cutoff'] == .5)
        values = [arm['stages'][s]['support']['curve']['6']['fraction'] for s in STAGES]
        print(f"neuron={index}: curve support @6 " + ' -> '.join(f'{v:.4f}' for v in values), flush=True)
        write(args.output_dir / 'summary.json', report)
    draw(args.output_dir / 'coverage_stages.png', report['cases'])
    report.update(status='completed', elapsed_seconds=time.perf_counter() - start,
                  provenance_checks='passed', replayed_pool_checks='passed', swc_metric_checks='passed')
    write(args.output_dir / 'summary.json', report)
    print(f"Completed CPU audit in {report['elapsed_seconds']:.1f}s. Report: {args.output_dir.resolve()}")


if __name__ == '__main__':
    main()
