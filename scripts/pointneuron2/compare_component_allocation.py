"""Prepare a fixed neuron-79 allocation repair, then run visibly on user CPU.

Prepare reuses cached matrices for selection diagnostics only. Run recomputes
one repaired graph, exports/evaluates its SWC, and compares the verified saved
original allocation. No proposal inference, training, threshold or size tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from scripts.pointneuron2.remaining_budget_selection import installed, VERSION
from scripts.pointneuron2.record_graph_components import SelectionRecorder, initializer, verify_plan, initializer_arguments, progress_logging
from scripts.pointneuron2.summarize_component_trace import inspect_trace
from scripts.pointneuron2.audit_full_volume_selection import reference_targets, distances
from scripts.pointneuron2.compare_graph_cutoffs import reconstruction_command
from scripts.pointneuron2.compare_hybrid_swc import check_graph
from scripts.pointneuron2.sampling_ablation import digest, read, write, live

FIXED_ARGUMENTS = ['nms_distance', 'nms_mode', 'iou_threshold', 'max_nodes', 'max_geodesic_ratio',
    'disconnected_multiplier', 'candidate_k', 'max_euclidean_distance', 'allow_unreachable_fallback',
    'bridge_max_geodesic_ratio', 'bridge_allow_unreachable_fallback']
VARIABLE_GRAPH_METADATA = {'adaptive_selection', 'bridge_edges', 'eligible_candidate_pairs', 'components',
    'mean_snap_distance', 'mean_edge_euclidean_distance', 'mean_edge_geodesic_distance', 'reachable_edge_fraction'}


def load_pool(trace):
    with np.load(trace['matrix_artifact']['path'], allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def cached_repair(plan, trace):
    pool = load_pool(trace)
    args = {k:pool[k] for k in ['centers', 'radii', 'scores', 'geodesic', 'euclidean']}
    args.update({k:plan['graph_metadata'][k] for k in FIXED_ARGUMENTS})
    with installed():
        recorder = SelectionRecorder(pool)
        with recorder.installed():
            selected, metadata = initializer.adaptive_connected_coverage_selection(**args)
    if metadata['selection_policy'] != 'connected_coverage' or len(np.unique(selected)) != 128:
        raise ValueError('Repaired selection changed the fixed branch or anchor budget')
    if recorder.component_trace['eligible_zero_anchor_components']:
        raise ValueError('Eligible components remain starved despite a feasible budget')
    if recorder.component_trace['excluded_components'] != trace['component_trace']['excluded_components']:
        raise ValueError('Component eligibility changed')
    return pool, selected, metadata, recorder.component_trace


def prepare(trace_folder, output):
    if output.exists():
        raise FileExistsError('Use a fresh comparison directory')
    observed = inspect_trace(trace_folder)
    original = read(trace_folder / 'trace_plan.json')
    trace = read(trace_folder / 'component_trace.json')
    comparison = verify_plan(original)
    baseline = next(r for r in comparison['results'] if r['name'] == 'hybrid_50_50')
    pool, selected, metadata, records = cached_repair(original, trace)
    curve = reference_targets(comparison['baseline']['reference'])[0]['curve']
    stages = {}
    for name, indices in [('original_allocation', pool['selected_pool_indices']), ('remaining_budget', selected)]:
        d = distances(pool['centers'][indices], curve)
        candidate_d = distances(curve, pool['centers'][indices])
        stages[name] = {'anchors':len(indices), 'reference_curve_samples':len(curve),
            'reference_curve_samples_within_6':int(np.count_nonzero(d <= 6)),
            'reference_curve_samples_within_18':int(np.count_nonzero(d <= 18)),
            'anchors_within_6_of_reference':int(np.count_nonzero(candidate_d <= 6)),
            'anchors_within_18_of_reference':int(np.count_nonzero(candidate_d <= 18))}
    output.mkdir(parents=True)
    selection = output / 'cached_selection.npz'
    np.savez_compressed(selection, original_indices=pool['selected_pool_indices'], repaired_indices=selected,
        proposal_indices=pool['proposal_indices'])
    diagnostic = {'status':'completed', 'allocation_version':VERSION, 'cached_replay_checks':'passed',
        'original_component_trace':trace['component_trace'], 'repaired_component_trace':records,
        'original_adaptive_metadata':trace['adaptive_metadata'], 'repaired_adaptive_metadata':metadata,
        'anchor_geometry':stages, 'new_swc_evaluated':False, 'reference_used_for_selection':False,
        'limitations':['Sparse anchor support is not traced SWC recall.',
            'Spreading anchors can increase bridging and unwanted connections.',
            'Size-one/two eligibility exclusion and the within-component FPS remain unchanged.']}
    write(output / 'cached_diagnostic.json', diagnostic)
    files = sorted((REPO_ROOT / 'src').rglob('*.py')) + sorted((REPO_ROOT / 'scripts').rglob('*.py'))
    files += sorted((REPO_ROOT / 'configs/pointneuron2').glob('*.json'))
    plan = {'status':'prepared', 'sample_index':79, 'allocation_version':VERSION,
        'scope':'one selected validation case; saved original control versus one repaired SWC',
        'trace_dir':str(trace_folder), 'trace_plan_sha256':digest(trace_folder / 'trace_plan.json'),
        'component_trace_sha256':digest(trace_folder / 'component_trace.json'),
        'cached_selection':{'path':str(selection), 'sha256':digest(selection)},
        'cached_diagnostic':{'path':str(output / 'cached_diagnostic.json'), 'sha256':digest(output / 'cached_diagnostic.json')},
        'source_sha256':{f.relative_to(REPO_ROOT).as_posix():digest(f) for f in files},
        'baseline_result':dict(baseline, name='original_allocation'), 'results':[],
        'optimizer_steps':0, 'new_proposal_inference':False, 'reserved_test_neurons_used':False,
        'device':'CPU; saved proposals reused; SciPy graph and geodesics',
        'expected_repaired_graph':str(output / 'remaining_budget/sample_0079/sample_0079_geodesic_graph.npz')}
    write(output / 'comparison.json', plan)
    verify_inputs(plan)
    print('Prepared: same 625 candidates, 128 anchors, size eligibility, foreground and weights.', flush=True)
    print('Original allocation:', [r['selected_count'] for r in trace['component_trace']['components'] if r['eligible']], flush=True)
    print('Repaired allocation:', [r['selected_count'] for r in records['components'] if r['eligible']], flush=True)
    print(json.dumps({'anchor_geometry':stages, 'repaired_graph_pressure':metadata}, indent=2), flush=True)
    print('No new SWC or geodesics launched. Visible run will reuse the verified saved original control.', flush=True)
    print(f'Prepared comparison: {output}', flush=True)


def verify_inputs(plan):
    if plan['allocation_version'] != VERSION or plan['sample_index'] != 79:
        raise ValueError('Unsupported allocation experiment')
    folder = Path(plan['trace_dir'])
    for filename, key in [('trace_plan.json','trace_plan_sha256'), ('component_trace.json','component_trace_sha256')]:
        if digest(folder / filename) != plan[key]:
            raise ValueError(f'Original {filename} changed')
    original = read(folder / 'trace_plan.json')
    comparison = verify_plan(original)
    trace = read(folder / 'component_trace.json')
    for entry in [trace['matrix_artifact'], trace['repeated_graph'], plan['cached_selection'], plan['cached_diagnostic']]:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Frozen comparison artifact changed')
    for name, sha in plan['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Comparison source changed: {name}; prepare a fresh comparison')
    baseline = plan['baseline_result']
    arm = next(r for r in comparison['results'] if r['name'] == 'hybrid_50_50')
    if baseline != dict(arm, name='original_allocation'):
        raise ValueError('Saved original control differs')
    for kind in ['proposal', 'graph', 'swc']:
        if digest(baseline['reconstruction'][kind + '_path']) != baseline[kind + '_sha256']:
            raise ValueError(f'Saved control {kind} changed')
    report_dir = Path(baseline['reconstruction']['graph_path']).parents[1]
    for kind in ['spatial', 'topology']:
        if read(report_dir / f'{kind}.json')['samples'] != [baseline['metrics'][kind]]:
            raise ValueError(f'Saved control {kind} metrics changed')
    return original, trace, comparison


def graph_worker(output):
    plan = read(output / 'comparison.json')
    if plan['status'] != 'running':
        raise ValueError('Graph worker must be invoked by the visible run stage')
    original, trace, comparison = verify_inputs(plan)
    pool = load_pool(trace)
    with np.load(plan['cached_selection']['path'], allow_pickle=False) as z:
        expected_selected = z['repaired_indices'].copy()
    graph_path = Path(plan['expected_repaired_graph'])
    if graph_path.exists():
        raise FileExistsError('Do not overwrite an existing graph')
    graph_path.parent.mkdir(parents=True)
    args = initializer_arguments(original, comparison, output)
    args[args.index('--output') + 1] = str(graph_path)
    old_argv = sys.argv
    try:
        sys.argv = [str(REPO_ROOT / 'scripts/initialize_geodesic_graph.py')] + args
        with installed():
            recorder = SelectionRecorder(pool)
            with recorder.installed(), progress_logging():
                if initializer.main():
                    raise ValueError('Repaired graph initializer failed')
    finally:
        sys.argv = old_argv
    for key in ['centers','radii','scores','geodesic','euclidean']:
        if recorder.pool is None or not np.array_equal(recorder.pool[key], pool[key]):
            raise ValueError(f'Recomputed {key} differs from the frozen matrix pool')
    diagnostic = read(plan['cached_diagnostic']['path'])
    if (not np.array_equal(recorder.selected, expected_selected)
            or recorder.component_trace != diagnostic['repaired_component_trace']
            or recorder.adaptive_metadata != diagnostic['repaired_adaptive_metadata']):
        raise ValueError('Actual repaired selection differs from cached preparation')
    with np.load(graph_path, allow_pickle=False) as g:
        arrays = {k:g[k].copy() for k in g.files}
    meta = json.loads(str(arrays['metadata']))
    for key, value in original['graph_metadata'].items():
        if key not in VARIABLE_GRAPH_METADATA and meta[key] != value:
            raise ValueError(f'Fixed graph metadata differs: {key}')
    if not np.array_equal(arrays['original_indices'], pool['proposal_indices'][expected_selected]):
        raise ValueError('Actual graph source identities differ')
    meta['component_allocation_version'] = VERSION
    arrays['metadata'] = np.array(json.dumps(meta))
    np.savez_compressed(graph_path, **arrays)
    write(output / 'actual_repaired_component_trace.json', {'checks':'passed',
        'component_trace':recorder.component_trace, 'adaptive_metadata':recorder.adaptive_metadata,
        'allocation_version':VERSION, 'graph_sha256':digest(graph_path)})
    print('Actual repaired graph: exact same pool and matrices; cached allocation reproduced.', flush=True)


def run(output):
    path = output / 'comparison.json'
    plan = read(path)
    if plan['status'] != 'prepared' or (output / 'remaining_budget').exists():
        raise ValueError('Preserve prior/partial runs; prepare a fresh comparison')
    original, trace, comparison = verify_inputs(plan)
    start = time.perf_counter()
    plan.update(status='running', results=[plan['baseline_result']])
    write(path, plan)
    try:
        print('CPU-only neuron 79: one repaired graph/SWC; original saved control reused. No inference/training.', flush=True)
        live([sys.executable, '-u', str(Path(__file__)), '--stage', 'graph-worker', '--output-dir', str(output)], output / 'graph.log')
        graph_sha = digest(plan['expected_repaired_graph'])
        arm = output / 'remaining_budget'
        proposals = Path(original['proposal']['path'])
        live(reconstruction_command(comparison['baseline'], arm, proposals.parent, .5, 'cpu'), output / 'export.log')
        summary = read(arm / 'summary.json')
        if summary['failures'] or len(summary['samples']) != 1:
            raise ValueError('Reconstruction changed the denominator or failed')
        row = summary['samples'][0]
        if Path(row['graph_path']).resolve() != Path(plan['expected_repaired_graph']) or digest(row['graph_path']) != graph_sha:
            raise ValueError('Exporter changed or replaced the prepared repaired graph')
        with np.load(row['graph_path'], allow_pickle=False) as g:
            meta = json.loads(str(g['metadata']))
        check_graph(comparison, row, proposals, meta)
        metrics = {}
        for kind, script in [('spatial','evaluate_paper_style_swc.py'), ('topology','evaluate_geodesic_baseline.py')]:
            live([sys.executable, '-u', str(REPO_ROOT / 'scripts' / script), '--root', comparison['baseline']['data_root'],
                '--summary', str(arm / 'summary.json'), '--json-output', str(arm / f'{kind}.json'),
                '--csv-output', str(arm / f'{kind}.csv')], output / f'{kind}.log')
            report = read(arm / f'{kind}.json')
            if len(report['samples']) != 1 or report['samples'][0]['sample_index'] != 79:
                raise ValueError('Metric denominator changed')
            metrics[kind] = report['samples'][0]
        values = [metrics['spatial'][k] for k in ['precision','recall','f1']] + [metrics['topology']['edge_f1']]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid reconstruction metrics')
        plan['results'].append({'name':'remaining_budget', 'reconstruction':row, 'graph_metadata':meta, 'metrics':metrics,
            'proposal_sha256':digest(proposals), 'graph_sha256':graph_sha, 'swc_sha256':digest(row['swc_path'])})
        verify_inputs(plan)
        if digest(row['graph_path']) != graph_sha:
            raise ValueError('Repaired graph changed during evaluation')
        plan['deltas_repaired_minus_original'] = {k:metrics['spatial'][k] - plan['baseline_result']['metrics']['spatial'][k]
            for k in ['precision','recall','f1']}
        plan.update(status='completed', paired_checks='passed', elapsed_seconds=time.perf_counter() - start)
        m = metrics['spatial']
        print(f"remaining_budget: spatial_F1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} topology_F1={metrics['topology']['edge_f1']:.4f}", flush=True)
    except BaseException as exc:
        plan.update(status='failed_or_interrupted', error=str(exc), elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        write(path, plan)
    print(f"Status: completed; paired checks: passed; elapsed={plan['elapsed_seconds']:.1f}s", flush=True)
    print(f'Results: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','check','run','graph-worker'], required=True)
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == 'prepare':
        if args.trace_dir is None:
            parser.error('prepare requires --trace-dir')
        prepare(args.trace_dir.resolve(), output)
    elif args.stage == 'check':
        verify_inputs(read(output / 'comparison.json'))
        print('Prepared comparison provenance passed; no reconstruction launched.', flush=True)
    elif args.stage == 'graph-worker':
        graph_worker(output)
    else:
        run(output)


if __name__ == '__main__':
    main()
