"""Verify the saved neuron-79 allocation trace and measure component support.

CPU-only saved-output inspection. Replay the unchanged selector on cached pool
matrices; no image reload, geodesic computation, alternative selector or SWC.
References measure diagnostic support only and never influence allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.swc import parse_swc
from scripts.evaluate_paper_style_swc import resample_swc_points, deterministic_downsample
from scripts.pointneuron2.audit_full_volume_selection import reference_targets, distances
from scripts.pointneuron2.record_graph_components import SelectionRecorder, verify_plan, initializer
from scripts.pointneuron2.sampling_ablation import digest, read, write


def inspect_trace(folder):
    plan = read(folder / 'trace_plan.json')
    trace_path = folder / 'component_trace.json'
    if plan['status'] != 'completed' or plan.get('recording_checks') != 'passed':
        raise ValueError('Requires a completed trace with passed recording checks')
    comparison = verify_plan(plan)
    if digest(trace_path) != plan['component_trace_sha256']:
        raise ValueError('Recorded trace changed')
    trace = read(trace_path)
    for key in ['matrix_artifact', 'repeated_graph']:
        if digest(trace[key]['path']) != trace[key]['sha256']:
            raise ValueError(f'Recorded {key} changed')
    if trace['comparison_sha256'] != plan['comparison_sha256']:
        raise ValueError('Trace comparison provenance differs')
    with np.load(plan['expected_graph']['path'], allow_pickle=False) as old, np.load(trace['repeated_graph']['path'], allow_pickle=False) as repeated:
        if set(old.files) != set(repeated.files) or any(not np.array_equal(old[k], repeated[k]) for k in old.files):
            raise ValueError('Repeated graph differs from the original')
    with np.load(plan['expected_pool']['path'], allow_pickle=False) as z:
        expected = {k: z[k].copy() for k in z.files}
    with np.load(trace['matrix_artifact']['path'], allow_pickle=False) as z:
        pool = {k: z[k].copy() for k in z.files}
    if any(not np.array_equal(pool[k], expected[k]) for k in ['centers', 'radii', 'scores', 'proposal_indices']):
        raise ValueError('Cached matrix pool differs from the prepared pool')
    meta = plan['graph_metadata']
    args = {k: pool[k] for k in ['centers', 'radii', 'scores', 'geodesic', 'euclidean']}
    args.update({k: meta[k] for k in ['nms_distance', 'nms_mode', 'iou_threshold', 'max_nodes',
        'max_geodesic_ratio', 'disconnected_multiplier', 'candidate_k', 'max_euclidean_distance',
        'allow_unreachable_fallback', 'bridge_max_geodesic_ratio', 'bridge_allow_unreachable_fallback']})
    recorder = SelectionRecorder(expected)
    with recorder.installed():
        selected, adaptive = initializer.adaptive_connected_coverage_selection(**args)
    if (recorder.component_trace != trace['component_trace'] or adaptive != trace['adaptive_metadata']
            or not np.array_equal(selected, pool['selected_pool_indices'])):
        raise ValueError('Cached unchanged selector replay differs from the actual trace')
    arm = next(r for r in comparison['results'] if r['name'] == 'hybrid_50_50')
    if digest(arm['reconstruction']['swc_path']) != arm['swc_sha256']:
        raise ValueError('Saved final SWC changed')
    targets, roles = reference_targets(comparison['baseline']['reference'])
    curve = targets['curve']
    tree = parse_swc(arm['reconstruction']['swc_path'])
    swc = deterministic_downsample(resample_swc_points(tree, 1.), 250000)
    missed = distances(swc, curve) > 6
    if not np.isclose(float(np.mean(~missed)), arm['metrics']['spatial']['recall'], atol=1e-7, rtol=0):
        raise ValueError('Fixed-target SWC recall differs from the saved metric')
    centers = pool['centers']
    nearest_d, nearest_i = cKDTree(centers).query(curve)
    components = trace['component_trace']['components']
    component_owner = np.full(len(centers), -1, dtype=np.int64)
    rows = []
    group_indices = {'allocated_components': [], 'starved_eligible_components': [], 'excluded_small_components': []}
    for c in components:
        idx = np.asarray(c['candidate_indices'], dtype=np.int64)
        if np.any(component_owner[idx] != -1):
            raise ValueError('Overlapping component membership')
        component_owner[idx] = c['component_rank']
        group = ('excluded_small_components' if not c['eligible'] else
            'allocated_components' if c['selected_count'] else 'starved_eligible_components')
        group_indices[group].extend(idx.tolist())
        d = distances(centers[idx], curve)
        assigned = np.isin(nearest_i, idx) & (nearest_d <= 18)
        candidate_d = distances(curve, centers[idx])
        rows.append({**{k: c[k] for k in ['component_rank', 'candidate_count', 'eligible', 'mean_score',
            'visited_in_allocation_loop', 'requested_budgets', 'selected_count']},
            'group': group,
            'candidates_within_6_of_reference': int(np.count_nonzero(candidate_d <= 6)),
            'candidates_within_18_of_reference': int(np.count_nonzero(candidate_d <= 18)),
            'reference_samples_within_6': int(np.count_nonzero(d <= 6)),
            'reference_samples_within_18': int(np.count_nonzero(d <= 18)),
            'missing_swc_samples_within_18': int(np.count_nonzero(missed & (d <= 18))),
            'nearest_pool_assigned_reference_samples_within_18': int(np.count_nonzero(assigned)),
            'nearest_pool_assigned_missing_swc_samples_within_18': int(np.count_nonzero(assigned & missed))})
    if np.any(component_owner < 0) or sum(r['selected_count'] for r in rows) != len(selected):
        raise ValueError('Incomplete component partition or selected count')
    groups = {}
    for name, values in group_indices.items():
        idx = np.asarray(values, dtype=np.int64)
        other = np.setdiff1d(np.arange(len(centers)), idx)
        d, other_d = distances(centers[idx], curve), distances(centers[other], curve)
        assigned = np.isin(nearest_i, idx) & (nearest_d <= 18)
        groups[name] = {'components': sum(r['group'] == name for r in rows), 'candidates': len(idx),
            'selected_anchors': sum(r['selected_count'] for r in rows if r['group'] == name),
            'reference_samples_within_6': int(np.count_nonzero(d <= 6)),
            'reference_samples_within_18': int(np.count_nonzero(d <= 18)),
            'missing_swc_samples_within_18': int(np.count_nonzero(missed & (d <= 18))),
            'missing_swc_samples_exclusively_within_18_of_this_group': int(np.count_nonzero(missed & (d <= 18) & (other_d > 18))),
            'nearest_pool_assigned_missing_swc_samples_within_18': int(np.count_nonzero(missed & assigned))}
    return {'status': 'completed', 'verification': 'passed', 'observed_neuron': 79,
        'method': 'unchanged selector replay using cached matrices; fixed-reference component support; no new reconstruction',
        'trace_sha256': digest(trace_path), 'plan_sha256': digest(folder / 'trace_plan.json'),
        'source_sha256': digest(Path(__file__)), 'reference_sha256': digest(comparison['baseline']['reference']),
        'swc_sha256': arm['swc_sha256'], 'reference_samples': len(curve), 'missing_swc_samples_at_6': int(missed.sum()),
        'actual_graph_elapsed_seconds': plan['elapsed_seconds'], 'groups': groups, 'components': rows,
        'adaptive_metadata': adaptive, 'reference_roots': roles['reference_roots'],
        'limitations': ['Components are candidate connectivity groups, not biological branches.',
            'Reference support uses Euclidean proximity in voxels, not continuous image paths or branch identity.',
            'Per-component support can overlap; nearest-pool assignment partitions supported samples.',
            'Zero anchors do not prove zero traced coverage: paths may traverse a component location.',
            'The allocation repair has not been implemented or evaluated; no new spatial/topology scores.',
            'One selected validation neuron; no reserved test data used.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier summaries; use a fresh output file')
    report = inspect_trace(args.trace_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, report)
    print(json.dumps({'verification': report['verification'], 'groups': report['groups']}, indent=2), flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
