"""Verify and inspect the saved neuron-79 hybrid SWC pair on CPU.

No inference, tracing, training, selection changes or metric preset changes.
Same reference targets measure retained/new/lost support and stage geometry.
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
from scripts.evaluate_paper_style_swc import resample_swc_points, deterministic_downsample
from scripts.initialize_geodesic_graph import filter_proposals
from scripts.pointneuron2.audit_full_volume_selection import reference_targets, record, change, distances
from scripts.pointneuron2.compare_hybrid_swc import verify_new, validate_selection, check_graph
from scripts.pointneuron2.sampling_ablation import digest, read, write
from scripts.pointneuron2.summarize_graph_cutoffs import segments


def inspect(plan):
    if (plan['status'] != 'completed' or plan.get('paired_checks') != 'passed'
            or plan.get('pool_and_feature_checks') != 'passed'
            or [r['name'] for r in plan['results']] != ['current_score', 'hybrid_50_50']):
        raise ValueError('Requires the complete fixed neuron-79 pair')
    verify_new(plan)
    with np.load(plan['inputs']['saved_pool']['path'], allow_pickle=False) as old, np.load(plan['inputs']['hybrid_selection']['path'], allow_pickle=False) as selection:
        selected = validate_selection(old, selection)
    targets, roles = reference_targets(plan['baseline']['reference'])
    rows, all_distances = [], []
    cfg = plan['baseline']['reconstruction']
    full_path = plan['results'][0]['reconstruction']['proposal_path']
    for arm in plan['results']:
        row = arm['reconstruction']
        for kind in ['proposal', 'graph', 'swc']:
            if digest(row[kind + '_path']) != arm[kind + '_sha256']:
                raise ValueError(f'Saved {kind} changed')
        folder = Path(row['graph_path']).parents[1]
        summary = read(folder / 'summary.json')
        if summary['failures'] or summary['samples'] != [row]:
            raise ValueError('Saved reconstruction summary differs')
        for kind in ['spatial', 'topology']:
            if read(folder / f'{kind}.json')['samples'] != [arm['metrics'][kind]]:
                raise ValueError(f'Saved {kind} metric report differs')
        tree = parse_swc(row['swc_path'])
        if tree.validate() or tree.root_count != 1:
            raise ValueError('Invalid saved SWC')
        with np.load(row['proposal_path'], allow_pickle=False) as p, np.load(row['graph_path'], allow_pickle=False) as g:
            check_graph(plan, row, Path(row['proposal_path']), json.loads(str(g['metadata'])))
            if len(p['centers']) != 4096 or len(g['centers']) != 128:
                raise ValueError('Unexpected proposal/graph counts')
            if arm['name'] == 'hybrid_50_50':
                with np.load(full_path, allow_pickle=False) as full:
                    for key, source in [('centers', 'all_centers'), ('radii', 'all_radii'), ('scores', 'all_scores'),
                            ('features', 'all_features'), ('patch_indices', 'all_patch_indices')]:
                        if not np.array_equal(p[key], full[source][selected]):
                            raise ValueError(f'Hybrid source-row {key} differs')
            pool = filter_proposals(p['centers'], p['radii'], p['scores'], p['features'],
                min_score=.5, nms_distance=cfg['graph_nms_distance'], nms_mode=cfg['graph_nms_mode'],
                iou_threshold=.1, max_nodes=cfg['graph_max_nodes'], selection_mode=cfg['graph_selection_mode'])[-1]
            if not np.isin(g['original_indices'], pool).all() or len(pool) != arm['graph_metadata']['adaptive_selection']['selection_pool_nodes']:
                raise ValueError('Replayed pool disagrees with graph')
            swc_points = deterministic_downsample(resample_swc_points(tree, 1.), 250000)
            geometry = [('global_selected', p['centers']), ('adaptive_input_pool', p['centers'][pool]),
                ('selected_graph_centers', g['centers']), ('traced_swc_points', swc_points)]
            stages, stage_distances = {}, {}
            for name, xyz in geometry:
                stages[name], stage_distances[name] = record(xyz, targets)
            saved = arm['metrics']['spatial']
            recall = stages['traced_swc_points']['support']['curve']['6']['fraction']
            precision = float(np.mean(distances(targets['curve'], swc_points) <= 6))
            if (len(targets['curve']) != saved['gt_points'] or len(swc_points) != saved['pred_points']
                    or not np.isclose(recall, saved['recall'], atol=1e-7, rtol=0)
                    or not np.isclose(precision, saved['precision'], atol=1e-7, rtol=0)):
                raise ValueError('Independent geometry check differs from saved spatial metrics')
        rows.append({'name': arm['name'], 'spatial': saved, 'topology': arm['metrics']['topology'],
            'stages': stages, 'graph_metadata': arm['graph_metadata'], 'reconstruction': row})
        all_distances.append(stage_distances)
    contrasts = {stage: {target: {str(r): change(all_distances[0][stage][target], all_distances[1][stage][target], r)
        for r in [6, 12, 18]} for target in targets} for stage in all_distances[0]}
    deltas = {key: rows[1]['spatial'][key] - rows[0]['spatial'][key] for key in ['precision', 'recall', 'f1']}
    if deltas != plan['deltas_hybrid_minus_current']:
        raise ValueError('Saved paired deltas differ')
    previous = next(r for r in plan['baseline']['results'] if r['cutoff'] == .5)['metrics']['spatial']
    repeats = {k: rows[0]['spatial'][k] - previous[k] for k in ['precision', 'recall', 'f1']}
    if repeats != plan['current_repeat_deltas_vs_saved']:
        raise ValueError('Saved repeatability deltas differ')
    return {'status': 'completed', 'method': 'saved-output provenance plus CPU geometry checks; no new inference/tracing',
        'scope': 'one selected validation case; not a generalization benchmark', **roles,
        'arms': rows, 'same_target_support_changes_hybrid_minus_current': contrasts,
        'spatial_deltas': deltas, 'current_repeat_deltas': repeats,
        'topology_f1_delta': rows[1]['topology']['edge_f1'] - rows[0]['topology']['edge_f1'],
        'metric_limitations': ['Spatial precision/recall use resampled SWCs within 6 voxels, not biological branch identity.',
            'Topology targets depend on each selected node set; 60 versus 59 target edges.',
            'Reference is a 191-root forest; its 191 endpoints include annotation fragment ends.',
            'Center support at sparse graph stages is not SWC recall; tracing interpolates between centers.',
            'No image background in projection figures; overlap can hide depth errors.']}


def figures(output, plan, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    ref = segments(parse_swc(plan['baseline']['reference']))
    predictions = [segments(parse_swc(r['reconstruction']['swc_path'])) for r in plan['results']]
    coords = np.concatenate([ref.reshape(-1, 3)] + [s.reshape(-1, 3) for s in predictions])
    lo, hi = coords.min(0), coords.max(0)
    margin = np.maximum((hi-lo)*.04, 2.)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, arm in enumerate(report['arms']):
        m = arm['spatial']
        for rr, (a, b) in enumerate([(0, 1), (0, 2)]):
            ax = axes[rr, col]
            ax.add_collection(LineCollection(ref[..., [a,b]], colors='#c84c36', linewidths=1.0, alpha=.8))
            ax.add_collection(LineCollection(predictions[col][..., [a,b]], colors='#167ca5', linewidths=.8, alpha=.8))
            ax.set_xlim(lo[a]-margin[a], hi[a]+margin[a]); ax.set_ylim(lo[b]-margin[b], hi[b]+margin[b])
            ax.set_aspect('equal' if rr == 0 else 'auto')
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels; display axis expanded)')
            ax.set_title(f"{arm['name'].replace('_',' ')} | F1 {m['f1']:.3f}\nPrecision {m['precision']:.3f}; recall {m['recall']:.3f}")
    fig.legend(handles=[Line2D([0],[0],color='#c84c36',label='Reference SWC'),
        Line2D([0],[0],color='#167ca5',label='Reconstruction')],loc='outside lower center',ncol=2)
    fig.suptitle('Neuron 79: same weights, 4,096 proposals, cutoff 0.5, 128 graph nodes\nFull SWC projections; shared axes; no image background; depth overlap can hide errors')
    fig.savefig(output/'swc_comparison.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4.8),constrained_layout=True)
    stages = ['global_selected','adaptive_input_pool','selected_graph_centers','traced_swc_points']
    for arm, color in zip(report['arms'], ['#8c5aad','#167ca5']):
        values = [100*arm['stages'][s]['support']['curve']['6']['fraction'] for s in stages]
        ax.plot(range(4),values,marker='o',color=color,label=arm['name'].replace('_',' '))
        for x, v in enumerate(values):
            ax.annotate(f'{v:.1f}',(x,v),xytext=(0,7 if color=='#167ca5' else -14),textcoords='offset points',ha='center')
    ax.axvline(2.5,color='#888888',linestyle=':')
    ax.set_xticks(range(4),['Global proposals','Graph input pool','128 graph centers','Final traced SWC'])
    ax.set_ylim(0,105); ax.set_ylabel('Reference curve samples supported within 6 voxels (%)')
    ax.set_title('Neuron 79: proposal support versus final traced recall\nSparse graph-center coverage is not SWC coverage')
    ax.grid(axis='y',alpha=.2); ax.legend()
    fig.savefig(output/'support_stages.png',dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier reports; use a fresh directory')
    plan = read(args.comparison)
    report = inspect(plan)
    report.update(comparison_path=str(args.comparison.resolve()),comparison_sha256=digest(args.comparison),
        diagnostic_source_sha256=digest(Path(__file__)))
    args.output_dir.mkdir(parents=True)
    figures(args.output_dir,plan,report)
    write(args.output_dir/'summary.json',report)
    for arm in report['arms']:
        print(arm['name'], json.dumps({'spatial': arm['spatial'], 'stage_support': arm['stages']}), flush=True)
    print('Final SWC curve support gained/lost:', report['same_target_support_changes_hybrid_minus_current']['traced_swc_points']['curve']['6'])
    print('Final SWC endpoints gained/lost:', report['same_target_support_changes_hybrid_minus_current']['traced_swc_points']['terminal']['6'])
    print('Verified frozen inputs/source, exact hybrid feature identities, graph/SWC hashes and saved reports.',flush=True)
    print(f'Report: {args.output_dir.resolve()}',flush=True)


if __name__ == '__main__':
    main()
