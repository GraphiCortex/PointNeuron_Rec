"""Inspect saved allocation SWCs and attribute their samples to graph paths.

CPU-only provenance, exporter replay and fixed-reference geometry. No tracing,
selection change, bridge pruning, alternate SWC or training is performed.
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
import scripts.generate_swc_from_graph as exporter
from scripts.evaluate_paper_style_swc import resample_swc_points
from scripts.pointneuron2.audit_full_volume_selection import reference_targets, distances, change
from scripts.pointneuron2.compare_component_allocation import verify_inputs
from scripts.pointneuron2.compare_hybrid_swc import check_graph
from scripts.pointneuron2.sampling_ablation import digest, read, write

CATEGORIES = ['root', 'ordinary_reachable', 'bridge_reachable', 'bridge_unreachable_fallback']


def exporter_labels(graph, tree):
    """Observe the unchanged exporter; verify every saved node before labeling."""
    edges = graph['edges']
    edge_map = {tuple(sorted(map(int, pair))):i for i,pair in enumerate(edges)}
    if len(edge_map) != len(edges):
        raise ValueError('Duplicate graph edges')
    paths = exporter.load_edge_paths(graph, edges)
    roots = exporter.choose_component_roots(graph['centers'], graph['radii'], graph['scores'], edges, 'endpoint_radius')
    if len(roots) != 1 or tree.root_count != 1:
        raise ValueError('This fixed comparison requires one root per saved SWC')
    original = exporter.oriented_edge_path
    calls = []
    def observed(paths, parent, neighbor):
        path = original(paths, parent, neighbor)
        calls.append((edge_map[tuple(sorted((parent, neighbor)))], len(path)-1 if path is not None and len(path)>1 else 1))
        return path
    try:
        exporter.oriented_edge_path = observed
        replay = exporter.graph_to_swc_nodes(graph['centers'], graph['radii'], graph['scores'], edges, roots, paths)
    finally:
        exporter.oriented_edge_path = original
    if len(replay) != len(tree.nodes) or len(calls) != len(edges):
        raise ValueError('Exporter replay changed node or edge counts')
    for actual, saved in zip(replay, tree.nodes):
        if ((actual.node_id, actual.node_type, actual.parent_id) != (saved.node_id, saved.node_type, saved.parent_id)
                or not np.allclose([actual.x,actual.y,actual.z,actual.radius],
                    [saved.x,saved.y,saved.z,saved.radius], atol=1e-6, rtol=0)):
            raise ValueError('Exporter replay differs from saved six-decimal SWC')
    node_labels = np.asarray([-1] + [edge for edge,count in calls for _ in range(count)], dtype=np.int64)
    if len(node_labels) != len(tree.nodes):
        raise ValueError('Incomplete node-to-graph-edge attribution')
    nodes = {n.node_id:n for n in tree.nodes}
    labels = []
    for node, edge in zip(tree.nodes, node_labels):
        count = 1
        if node.parent_id in nodes:
            parent = nodes[node.parent_id]
            a = np.asarray([node.x,node.y,node.z], dtype=np.float32)
            b = np.asarray([parent.x,parent.y,parent.z], dtype=np.float32)
            count = max(1, int(np.ceil(float(np.linalg.norm(a-b))))) + 1
        labels.extend([int(edge)] * count)
    points = resample_swc_points(tree, 1.)
    if len(labels) != len(points) or len(points) > 250000:
        raise ValueError('Resampled point labels differ; downsampled cases unsupported')
    return points, np.asarray(labels), node_labels


def inspect(path):
    plan = read(path)
    if (plan['status'] != 'completed' or plan.get('paired_checks') != 'passed'
            or [r['name'] for r in plan['results']] != ['original_allocation','remaining_budget']):
        raise ValueError('Requires the completed fixed allocation comparison')
    original, trace, comparison = verify_inputs(plan)
    targets, roles = reference_targets(comparison['baseline']['reference'])
    curve = targets['curve']
    with np.load(trace['matrix_artifact']['path'], allow_pickle=False) as pool:
        pool_proposals = pool['proposal_indices'].copy()
    proposal_component = {}
    for c in trace['component_trace']['components']:
        for i in c['candidate_indices']:
            proposal_component[int(pool_proposals[i])] = c['component_rank']
    rows, raw = [], []
    actual_trace = read(path.parent / 'actual_repaired_component_trace.json')
    cached = read(plan['cached_diagnostic']['path'])
    if (actual_trace['checks'] != 'passed' or actual_trace['component_trace'] != cached['repaired_component_trace']
            or actual_trace['adaptive_metadata'] != cached['repaired_adaptive_metadata']
            or actual_trace['graph_sha256'] != plan['results'][1]['graph_sha256']):
        raise ValueError('Actual repaired trace differs from cached preparation')
    for arm in plan['results']:
        row = arm['reconstruction']
        for kind in ['proposal','graph','swc']:
            if digest(row[kind+'_path']) != arm[kind+'_sha256']:
                raise ValueError(f'Saved {kind} changed')
        folder = Path(row['graph_path']).parents[1]
        summary = read(folder / 'summary.json')
        if summary['failures'] or summary['samples'] != [row]:
            raise ValueError('Reconstruction summary differs')
        for kind in ['spatial','topology']:
            if read(folder / f'{kind}.json')['samples'] != [arm['metrics'][kind]]:
                raise ValueError(f'Saved {kind} report differs')
        with np.load(row['graph_path'], allow_pickle=False) as z:
            graph = {k:z[k].copy() for k in z.files}
        meta = json.loads(str(graph['metadata']))
        if meta != arm['graph_metadata']:
            raise ValueError('Graph metadata differs from the recorded result')
        check_graph(comparison, row, Path(row['proposal_path']), meta)
        tree = parse_swc(row['swc_path'])
        points, labels, node_labels = exporter_labels(graph, tree)
        pred_d = distances(curve, points)
        ref_d = distances(points, curve)
        p, r = float(np.mean(pred_d<=6)), float(np.mean(ref_d<=6))
        f1 = 2*p*r/(p+r)
        saved = arm['metrics']['spatial']
        if (len(points) != saved['pred_points'] or len(curve) != saved['gt_points']
                or not np.allclose([p,r,f1],[saved[k] for k in ['precision','recall','f1']],atol=1e-7,rtol=0)):
            raise ValueError('Independent resampled spatial metrics disagree')
        bridges = {tuple(sorted(map(int,pair))) for pair in graph['bridge_edges']}
        edge_categories = [ ('bridge_reachable' if reachable else 'bridge_unreachable_fallback') if tuple(sorted(map(int,pair))) in bridges
            else 'ordinary_reachable' if reachable else 'invalid_ordinary_fallback'
            for pair,reachable in zip(graph['edges'],graph['edge_geodesic_reachable'])]
        if 'invalid_ordinary_fallback' in edge_categories or len(bridges) != meta['bridge_edges']:
            raise ValueError('Unexpected edge reachability/bridge flags')
        point_categories = np.asarray(['root' if edge<0 else edge_categories[edge] for edge in labels])
        groups = {}
        for category in CATEGORIES:
            mask = point_categories == category
            n, far = int(mask.sum()), int(np.count_nonzero(mask & (pred_d>6)))
            groups[category] = {'edges':edge_categories.count(category), 'resampled_points':n,
                'points_farther_than_6_from_reference':far,
                'within_6_fraction':float(np.mean(pred_d[mask]<=6)) if n else None}
        edge_rows = []
        component_groups = {}
        for i,pair in enumerate(graph['edges']):
            mask = labels==i
            s,e = graph['edge_path_offsets'][i:i+2]
            polyline = graph['edge_path_points'][s:e]
            owners = [proposal_component[int(graph['original_indices'][int(j)])] for j in pair]
            key = 'within_component_' + str(owners[0]) if owners[0] == owners[1] else 'between_components'
            stats = component_groups.setdefault(key, {'edges':0,'resampled_points':0,'points_farther_than_6_from_reference':0})
            stats['edges'] += 1
            stats['resampled_points'] += int(mask.sum())
            stats['points_farther_than_6_from_reference'] += int(np.count_nonzero(mask & (pred_d>6)))
            edge_rows.append({'edge_index':i,'graph_nodes':list(map(int,pair)), 'category':edge_categories[i],
                'pool_component_ranks':owners,
                'resampled_points':int(mask.sum()), 'points_farther_than_6_from_reference':int(np.count_nonzero(mask & (pred_d>6))),
                'polyline_length_voxels':float(np.linalg.norm(np.diff(polyline,axis=0),axis=1).sum()),
                'euclidean_length_voxels':float(graph['edge_euclidean_distance'][i]),
                'saved_geodesic_cost':float(graph['edge_tree_distance'][i]) if np.isfinite(graph['edge_tree_distance'][i]) else None})
        rows.append({'name':arm['name'],'spatial':saved,'topology':arm['metrics']['topology'], 'path_categories':groups,
            'endpoint_component_groups':component_groups,
            'reference_samples_supported_at_6':int(np.count_nonzero(ref_d<=6)),
            'predicted_samples_farther_than_6':int(np.count_nonzero(pred_d>6)),
            'edge_records':edge_rows, 'exporter_replay':'passed', 'spatial_recheck':'passed'})
        raw.append({'tree':tree,'node_labels':node_labels,'edge_categories':edge_categories,'ref_d':ref_d})
    deltas = {k:rows[1]['spatial'][k]-rows[0]['spatial'][k] for k in ['precision','recall','f1']}
    if deltas != plan['deltas_repaired_minus_original']:
        raise ValueError('Recorded paired deltas differ')
    report = {'status':'completed','verification':'passed','comparison_sha256':digest(path),
        'inspection_source_sha256':digest(Path(__file__)), 'scope':'one selected validation neuron; saved-output inspection only',
        'elapsed_seconds_user_run':plan['elapsed_seconds'], 'reference_samples':len(curve), 'reference_roots':roles['reference_roots'],
        'spatial_deltas':deltas,'same_reference_support_change_at_6':change(raw[0]['ref_d'],raw[1]['ref_d'],6),
        'arms':rows,'limitations':['Point-distance errors refer to this reference, not verified biological false positives.',
            'Path attribution partitions the evaluator resampling, including duplicate junction points.',
            'Reachable geodesics can follow unrelated bright structures or take long detours.',
            'No bridges are removed and no counterfactual SWC or pruning score is reported.',
            'Topology target sets differ (59 versus 52 edges); reference has 191 disconnected roots.',
            'XY projections can hide depth differences; no source image background is shown.']}
    return report, raw, curve


def figures(output, report, raw, curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    colors = {'ordinary_reachable':'#1489a5','bridge_reachable':'#9460b5','bridge_unreachable_fallback':'#db502f'}
    fig, axes = plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    all_xy = np.concatenate([curve[:,:2]] + [np.asarray([[n.x,n.y] for n in d['tree'].nodes]) for d in raw])
    lo,hi = all_xy.min(0)-25,all_xy.max(0)+25
    for ax,arm,data in zip(axes,report['arms'],raw):
        ax.scatter(curve[:,0],curve[:,1],s=1,c='#888888',alpha=.5,label='Reference')
        nodes = {n.node_id:n for n in data['tree'].nodes}
        for category,color in colors.items():
            segments=[]
            for node,edge in zip(data['tree'].nodes,data['node_labels']):
                if edge>=0 and node.parent_id in nodes and data['edge_categories'][edge]==category:
                    parent=nodes[node.parent_id]
                    segments.append([[parent.x,parent.y],[node.x,node.y]])
            ax.add_collection(LineCollection(segments,colors=color,linewidths=.8,label=category.replace('_',' ')))
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
        ax.set_xlabel('X (voxels)');ax.set_ylabel('Y (voxels)')
        ax.set_title(f"{arm['name'].replace('_',' ')}\nPrecision {arm['spatial']['precision']:.3f}; recall {arm['spatial']['recall']:.3f}")
    axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.suptitle('Neuron 79: saved SWC paths by source graph edge\nShared XY axes; depth overlap can hide errors; no image background')
    fig.savefig(output/'path_categories.png',dpi=160);plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4.5),constrained_layout=True)
    x=np.arange(3)
    for shift,arm,color in zip([-.18,.18],report['arms'],['#54728b','#dc8b38']):
        values=[arm['path_categories'][c]['points_farther_than_6_from_reference'] for c in CATEGORIES[1:]]
        bars=ax.bar(x+shift,values,width=.36,color=color,label=arm['name'].replace('_',' '))
        ax.bar_label(bars,padding=3)
    ax.set_xticks(x,['Ordinary reachable paths','Reachable bridge paths','Unreachable straight bridges'])
    ax.set_ylabel('SWC evaluation samples farther than 6 voxels from reference')
    ax.set_title('Which saved path categories contain reference-distance errors?\nSame evaluator resampling; duplicate points retained; no paths removed')
    ax.legend();ax.margins(y=.2)
    fig.savefig(output/'path_error_counts.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier reports; use a fresh output directory')
    report,raw,curve=inspect(args.comparison)
    args.output_dir.mkdir(parents=True)
    figures(args.output_dir,report,raw,curve)
    write(args.output_dir/'summary.json',report)
    for arm in report['arms']:
        print(json.dumps({k:arm[k] for k in ['name','path_categories','endpoint_component_groups','reference_samples_supported_at_6','predicted_samples_farther_than_6']},indent=2),flush=True)
    print('Same-reference gain/loss:',report['same_reference_support_change_at_6'],flush=True)
    print(f'Report: {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
