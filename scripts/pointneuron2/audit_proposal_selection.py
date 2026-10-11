"""Audit frozen validation proposals through existing selectors; no graph/training.

Uses the saved cache crops, not a full-volume inference grid. References are used
only for measurement. Final geodesic adaptive selection is deliberately omitted.
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
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from scripts.initialize_geodesic_graph import filter_proposals, nms_proposals
from scripts.visualize_proposals import select_proposals
from scripts.pointneuron2.sampling_ablation import checked_manifest, check_matches, digest, read, write


def measure(centers, reference, endpoint_ids=(), junction_ids=()):
    """Nearest SWC-node proximity in voxel coordinates; empty sets score zero."""
    result = {'rows': len(centers), 'unique_xyz': len(np.unique(centers, axis=0)),
              'reference_nodes': len(reference)}
    if not len(reference):
        return dict(result, coverage_3=None, coverage_6=None,
                    rows_near_reference_6=None, endpoint_coverage_6=None,
                    junction_coverage_6=None)
    distances = (cKDTree(centers).query(reference[:, 1:4])[0] if len(centers)
                 else np.full(len(reference), np.inf))
    result.update(coverage_3=float(np.mean(distances <= 3)),
                  coverage_6=float(np.mean(distances <= 6)),
                  rows_near_reference_6=(float(np.mean(
                      cKDTree(reference[:, 1:4]).query(centers)[0] <= 6)) if len(centers) else None))
    for name, identities in [('endpoint', endpoint_ids), ('junction', junction_ids)]:
        mask = np.isin(reference[:, 0], list(identities))
        result[f'{name}_nodes'] = int(mask.sum())
        result[f'{name}_coverage_6'] = float(np.mean(distances[mask] <= 6)) if mask.any() else None
    return result


def select_spheres(centers, radii, scores, cutoff, limit):
    # Call the same implementation and ordering as aggregate_proposals.py.
    return np.asarray(select_proposals(
        centers=torch.from_numpy(centers), radii=torch.from_numpy(radii),
        scores=torch.from_numpy(scores), score_threshold=cutoff,
        top_proposals=limit, nms_mode='sphere', iou_threshold=.1,
        nms_radius=0.), dtype=np.int64)


def graph_pool(centers, radii, scores, cutoff, reconstruction, mode):
    return filter_proposals(
        centers, radii, scores, np.zeros((len(centers), 1), dtype=np.float32),
        min_score=cutoff, nms_distance=reconstruction['graph_nms_distance'],
        nms_mode=reconstruction['graph_nms_mode'], iou_threshold=.1,
        max_nodes=reconstruction['graph_max_nodes'], selection_mode=mode)[-1]


def stage_record(name, parent, indices, centers, reference, endpoints, junctions,
                 *, production=True):
    return dict(stage=name, parent=parent, production_stage=production,
                **measure(centers[indices], reference, endpoints, junctions))


def summarize(samples):
    result = {}
    for cutoff in ['0.5', '0.85']:
        stages = samples[0]['variants'][cutoff]
        result[cutoff] = {}
        for stage in stages:
            records = [next(r for r in s['variants'][cutoff] if r['stage'] == stage['stage']) for s in samples]
            values = {}
            for key in ['rows', 'unique_xyz', 'coverage_3', 'coverage_6',
                        'rows_near_reference_6', 'endpoint_coverage_6', 'junction_coverage_6']:
                valid = [r[key] for r in records if r[key] is not None]
                values[key] = {'mean': float(np.mean(valid)) if valid else None, 'neurons': len(valid)}
            result[cutoff][stage['stage']] = values
    return result


def render(path, arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages = ['raw', 'local_sphere', 'global_sphere', 'graph_score_gate', 'adaptive_pool']
    labels = ['Raw\nrows', 'Local\nsphere NMS', 'Global\nsphere NMS', 'Graph\nscore gate', 'Adaptive\ninput pool']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, arm in zip(axes, arms):
        for cutoff, color in [('0.5', '#1876ab'), ('0.85', '#d46b26')]:
            means = arm['means'][cutoff]
            ax.plot(range(len(stages)), [100 * means[s]['coverage_6']['mean'] for s in stages],
                    marker='o', color=color, label=f'Graph cutoff {cutoff}')
        ax.set_xticks(range(len(stages)), labels, fontsize=9)
        ax.set_ylim(0, 100); ax.set_title(arm['name'].replace('repaired_', '').replace('_', ' '))
        ax.grid(axis='y', alpha=.25); ax.legend(fontsize=9)
    axes[0].set_ylabel('Reference nodes within 6 voxels (%)')
    fig.suptitle('Cached-crop selection audit: mean of 7 validation neurons', fontsize=12)
    fig.text(.5, .015, 'Adaptive input pool precedes geodesic selection; no edges or reconstructed SWC are scored.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93)); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--artifact-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.output_dir.exists() or args.artifact_dir.exists():
        raise FileExistsError('Choose fresh report/artifact directories; preserve earlier audits')
    plan = read(args.experiment / 'experiment.json')
    if plan['status'] not in ('trained', 'evaluated'):
        raise ValueError('Requires completed frozen training')
    for name, expected in plan['source_sha256'].items():
        if digest(REPO_ROOT / name) != expected:
            raise ValueError(f'Original source/configuration changed: {name}')
    manifest = checked_manifest(plan, args.experiment)
    check_matches([read(args.experiment / arm['name'] / 'run.json') for arm in plan['arms']])
    config = plan['config']['reconstruction']
    if (config['proposal_score_threshold'] != .5 or config['graph_nms_mode'] != 'distance'
            or config['graph_selection_mode'] != 'adaptive_connected_coverage_nms'):
        raise ValueError('This audit targets the agreed sphere aggregation / adaptive distance-anchor preset')
    samples = scan_gold166(plan['data_root'])
    val = [row for row in manifest['records'] if row['partition'] == 'val']
    identities = sorted({row['sample_index'] for row in val})
    reference_info = {}
    for index in identities:
        source = samples[index]
        rows = [row for row in val if row['sample_index'] == index]
        nodes_by_id = {}
        for row in rows:
            with np.load(row['resolved_path'], allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata']))
                if source.sample_id != row['sample_id'] or digest(source.swc_path) != metadata['swc_sha256']:
                    raise ValueError('Reference source changed')
                for node in data['skeleton_nodes']:
                    identity = int(node[0])
                    if identity in nodes_by_id and not np.array_equal(node, nodes_by_id[identity]):
                        raise ValueError('Overlapping crop reference nodes disagree')
                    nodes_by_id[identity] = node
        reference = np.asarray([nodes_by_id[k] for k in sorted(nodes_by_id)], dtype=np.float32).reshape(-1, 6)
        full = parse_swc(source.swc_path).nodes
        children = Counter(n.parent_id for n in full)
        reference_info[index] = (reference, {n.node_id for n in full if children[n.node_id] == 0 and n.parent_id >= 0},
                                 {n.node_id for n in full if children[n.node_id] > 1})
    args.output_dir.mkdir(parents=True); args.artifact_dir.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'same 56 cached validation crops; seven partial-volume unions; no graph or reconstruction',
              'experiment': str(args.experiment.resolve()), 'manifest_sha256': plan['manifest_sha256'],
              'diagnostic_source_sha256': digest(Path(__file__)), 'matching_checks': 'passed',
              'device': args.device, 'optimizer_steps': 0, 'reserved_test_neurons_used': False,
              'settings': {'local_score_cutoff': .5, 'sphere_iou': .1, 'graph_cutoffs': [.5, .85],
                           'graph_distance': config['graph_nms_distance'], 'graph_node_budget': config['graph_max_nodes'],
                           'local_limit': config['proposal_top_per_patch'], 'global_limit': config['proposal_global_top']},
              'means_definition': 'equal neuron weight; empty proposal sets have zero reference coverage; no-reference groups omitted',
              'arms': []}
    start = time.perf_counter()
    for arm in plan['arms']:
        checkpoint_path = args.experiment / arm['name'] / 'proposal.pt'
        if digest(checkpoint_path) != arm['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        recipe, pipeline = checkpoint['args'], checkpoint['sampling_pipeline']
        if pipeline['strategy'] != 'random':
            raise ValueError('The paired coordinate pilot must use random inputs')
        encoder = DGCNNEncoder(k=recipe['k']).to(args.device).eval()
        head = SkeletonProposalHead(in_channels=encoder.output_dim + 3,
                                    coordinate_mode=recipe['proposal_coordinate_mode']).to(args.device).eval()
        encoder.load_state_dict(checkpoint['encoder']); head.load_state_dict(checkpoint['proposal'])
        collections = {index: [] for index in identities}
        with torch.inference_mode():
            for number, row in enumerate(val, 1):
                with np.load(row['resolved_path'], allow_pickle=False) as data:
                    metadata = json.loads(str(data['metadata']))
                    points = torch.as_tensor(data['points'], device=args.device)[None]
                generator = torch.Generator(device=args.device).manual_seed(metadata['seed'] + pipeline['evaluation_seed'])
                chosen = torch.randperm(points.shape[1], device=args.device, generator=generator)[:pipeline['selected_points']]
                selected = points[:, chosen]; output = head(selected, encoder(selected))
                arrays = {'centers': output.center_proposals[0].cpu().numpy(),
                          'radii': output.radius[0, :, 0].cpu().numpy(),
                          'scores': output.objectness_logits.softmax(-1)[0, :, 1].cpu().numpy()}
                if any(not np.isfinite(a).all() for a in arrays.values()):
                    raise FloatingPointError('Nonfinite predictions')
                arrays['patch_index'] = np.full(len(chosen), row['patch_index'], dtype=np.int64)
                arrays['local_keep'] = select_spheres(arrays['centers'], arrays['radii'], arrays['scores'], .5,
                                                      config['proposal_top_per_patch'])
                collections[row['sample_index']].append(arrays)
                if number % 8 == 0:
                    print(f"{arm['name']}: inferred and locally suppressed {number}/{len(val)} crops", flush=True)
        arm_report = {'name': arm['name'], 'checkpoint_sha256': arm['checkpoint_sha256'],
                      'checkpoint_epoch': checkpoint['epoch'], 'samples': []}
        for index, patches in collections.items():
            reference, endpoints, junctions = reference_info[index]
            arrays = {k: np.concatenate([p[k] for p in patches]) for k in ['centers', 'radii', 'scores', 'patch_index']}
            offset = 0; local = []
            for patch in patches:
                local.extend((patch['local_keep'] + offset).tolist()); offset += len(patch['centers'])
            local = np.asarray(local, dtype=np.int64)
            global_relative = select_spheres(arrays['centers'][local], arrays['radii'][local], arrays['scores'][local],
                                              0., config['proposal_global_top'])
            global_keep = local[global_relative]
            stages = {'raw': np.arange(offset), 'local_score': np.flatnonzero(arrays['scores'] >= .5),
                      'local_sphere': local, 'global_sphere': global_keep}
            sample = {'sample_index': index, 'sample_id': samples[index].sample_id, 'cached_crops': len(patches),
                      'reference_nodes_in_cached_union': len(reference), 'full_reference_nodes': len(parse_swc(samples[index].swc_path).nodes),
                      'variants': {}}
            for cutoff in [.5, .85]:
                records = []; parent = None
                for name, keep in stages.items():
                    records.append(stage_record(name, parent, keep, arrays['centers'], reference, endpoints, junctions)); parent = name
                c, r, s = [arrays[k][global_keep] for k in ['centers', 'radii', 'scores']]
                gate = np.flatnonzero(s >= cutoff)
                pool = graph_pool(c, r, s, cutoff, config, config['graph_selection_mode'])
                if not set(pool).issubset(set(gate)):
                    raise AssertionError('Graph pool contains below-threshold proposals')
                records.append(stage_record('graph_score_gate', 'global_sphere', global_keep[gate], arrays['centers'], reference, endpoints, junctions))
                records.append(stage_record('adaptive_pool', 'graph_score_gate', global_keep[pool], arrays['centers'], reference, endpoints, junctions))
                anchors = nms_proposals(c[gate], r[gate], s[gate], 'distance', config['graph_nms_distance'], .1, gate)
                # These are controls, not the adaptive selector's final nodes.
                controls = {'distance_anchors': anchors,
                            'distance_anchors_budget': anchors[:config['graph_max_nodes']],
                            'coverage_budget': graph_pool(c, r, s, cutoff, config, 'coverage_nms')}
                for name, keep in controls.items():
                    records.append(stage_record(name, 'graph_score_gate', global_keep[keep], arrays['centers'], reference, endpoints, junctions, production=False))
                sample['variants'][str(cutoff)] = records
                for name, keep in {'gate': gate, 'pool': pool, **controls}.items():
                    arrays[f'{name}_{str(cutoff).replace(".", "_")}'] = global_keep[keep]
            arrays.update(reference=reference, local_keep=local, global_keep=global_keep)
            artifact = args.artifact_dir / f"{arm['name']}_neuron_{index}.npz"
            np.savez_compressed(artifact, **arrays)
            sample['artifact_path'] = str(artifact.resolve()); sample['artifact_sha256'] = digest(artifact)
            arm_report['samples'].append(sample)
            print(f"{arm['name']} neuron={index}: raw={offset} local={len(local)} global={len(global_keep)}", flush=True)
        arm_report['means'] = summarize(arm_report['samples']); report['arms'].append(arm_report)
    report.update(status='completed', elapsed_seconds=time.perf_counter() - start)
    write(args.output_dir / 'summary.json', report)
    render(args.output_dir / 'coverage.png', report['arms'])
    print(f"Completed in {report['elapsed_seconds']:.1f}s; report: {args.output_dir.resolve()}", flush=True)


if __name__ == '__main__':
    main()
