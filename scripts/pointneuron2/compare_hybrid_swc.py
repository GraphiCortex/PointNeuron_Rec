"""Prepare/run neuron 79's fixed score-versus-hybrid reconstruction control.

Preparation and checks are CPU only. Run is explicitly launched by the user:
one feature-preserving inference, two reconstructions, spatial/topology metrics.
No training, ratio tuning, production changes or test-set use.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from scripts.pointneuron2.compare_graph_cutoffs import aggregation_command, reconstruction_command, verify
from scripts.pointneuron2.compare_hybrid_global import source_indices
from scripts.pointneuron2.compare_validation_cutoffs import completed_pair
from scripts.pointneuron2.sampling_ablation import digest, live, read, write


def check_snapshot():
    """Prove the copied aggregator differs only in path and two saved arrays."""
    original = (REPO_ROOT / 'scripts/aggregate_proposals.py').read_text(encoding='utf-8')
    snapshot = (REPO_ROOT / 'scripts/pointneuron2/aggregate_with_features.py').read_text(encoding='utf-8')
    tree = ast.parse(snapshot)
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Constant):
        raise ValueError('Aggregation snapshot documentation missing')
    tree.body.pop(0)
    snapshot = ast.unparse(tree)
    # Normalize only the declared serialization additions and location change.
    snapshot = snapshot.replace('Path(__file__).resolve().parents[2]', 'Path(__file__).resolve().parents[1]')
    tree = ast.parse(snapshot)
    removed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'savez_compressed':
            removed = [k for k in node.keywords if k.arg in ('all_features', 'all_patch_indices')]
            node.keywords = [k for k in node.keywords if k not in removed]
    expected = ast.parse('f(all_features=features_tensor.numpy(), all_patch_indices=np.asarray(all_patch_indices, dtype=np.int32))').body[0].value.keywords
    if ([ast.dump(k) for k in removed] != [ast.dump(k) for k in expected]
            or ast.dump(tree) != ast.dump(ast.parse(original))):
        raise ValueError('Snapshot differs beyond the approved feature serialization')


def validate_selection(old, selection):
    xyz, radii, scores = (old[k] for k in ('all_centers', 'all_radii', 'all_scores'))
    indices = selection['source_indices']
    order = selection['candidate_order']
    anchors = selection['anchor_indices']
    current = {k: old[k] for k in ('centers', 'radii', 'scores')}
    if (indices.dtype.kind not in 'iu' or order.dtype.kind not in 'iu' or anchors.dtype.kind not in 'iu'
            or indices.shape != (4096,) or order.shape != (16384,) or anchors.shape != (2048,)
            or len(np.unique(indices)) != 4096 or len(np.unique(order)) != 16384
            or np.any(order < 0) or np.any(order >= len(xyz))
            or not np.isin(indices, order).all()
            or not np.array_equal(indices[:2048], anchors)
            or not np.array_equal(order[:2048], anchors)
            or not np.array_equal(anchors, source_indices(xyz, radii, scores, current)[:2048])):
        raise ValueError('Saved hybrid budgets, source identities or anchors differ')
    for key, source in [('centers', xyz), ('radii', radii), ('scores', scores)]:
        if not np.array_equal(selection[key], source[indices]):
            raise ValueError(f'Saved hybrid {key} differs from frozen source rows')
    return indices.copy()


def verify_new(plan):
    verify(plan['baseline'])
    check_snapshot()
    for name, sha in plan['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Experiment source changed: {name}; prepare a fresh run')
    for entry in plan['inputs'].values():
        if digest(entry['path']) != entry['sha256']:
            raise ValueError(f"Frozen comparison input changed: {entry['path']}")


def prepare(baseline_path, hybrid_report_path, output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    report = read(hybrid_report_path)
    baseline = completed_pair(baseline_path, report['checkpoint_sha256'])
    if (baseline['sample_index'] != 79 or report['status'] != 'completed'
            or report['settings'] != {'candidate_limit': 16384, 'output_budget': 4096, 'anchors': 2048,
                'anchor_fraction': .5, 'sphere_iou': .1, 'graph_cutoffs_measured': [.5, .85]}):
        raise ValueError('Requires the completed fixed neuron-79 hybrid control')
    for name, sha in report['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Hybrid diagnostic source changed: {name}')
    row = next(r for r in report['cases'] if r['sample_index'] == 79)
    shared = baseline['shared_proposals']
    if row['proposal_sha256'] != shared['sha256'] or digest(row['artifact_path']) != row['artifact_sha256']:
        raise ValueError('Hybrid selection does not match the frozen baseline proposals')
    with np.load(shared['path'], allow_pickle=False) as old, np.load(row['artifact_path'], allow_pickle=False) as selection:
        validate_selection(old, selection)
    files = sorted((REPO_ROOT / 'src').rglob('*.py')) + sorted((REPO_ROOT / 'scripts').rglob('*.py'))
    files += sorted((REPO_ROOT / 'configs/pointneuron2').glob('*.json'))
    plan = {'status': 'prepared', 'scope': 'one selected validation diagnostic; no generalization claim',
        'sample_index': 79, 'baseline': baseline, 'settings': dict(report['settings'], graph_cutoff=.5),
        'inputs': {'baseline_comparison': {'path': str(baseline_path), 'sha256': digest(baseline_path)},
            'hybrid_report': {'path': str(hybrid_report_path), 'sha256': digest(hybrid_report_path)},
            'saved_pool': shared,
            'hybrid_selection': {'path': row['artifact_path'], 'sha256': row['artifact_sha256']}},
        'source_sha256': {f.relative_to(REPO_ROOT).as_posix(): digest(f) for f in files},
        'arms': ['current_score', 'hybrid_50_50'], 'optimizer_steps': 0,
        'reserved_test_neurons_used': False, 'results': []}
    verify_new(plan)
    output.mkdir(parents=True)
    write(output / 'comparison.json', plan)
    print('Prepared neuron 79: unchanged checkpoint; one inference, two SWCs; no training.', flush=True)
    print('Both arms: 4096 proposals, cutoff=0.5, graph nodes<=128; protected anchors=2048.', flush=True)
    print('Checks passed: historical provenance, exact saved hybrid identities, aggregation snapshot equivalence.', flush=True)
    print(f'Plan and future outputs: {output}', flush=True)


def make_hybrid(full, historical, selection, destination):
    """Require exact frozen geometry and recover each selected row's real feature."""
    indices = validate_selection(historical, selection)
    for key in ['all_centers', 'all_radii', 'all_scores', 'centers', 'radii', 'scores', 'features', 'patch_indices']:
        if not np.array_equal(full[key], historical[key]):
            raise ValueError(f'Regenerated {key} differs from saved control; stopping before reconstruction')
    n = len(full['all_centers'])
    if (full['all_features'].ndim != 2 or full['all_features'].shape[0] != n
            or full['all_features'].shape[1] != full['features'].shape[1]
            or full['all_patch_indices'].shape != (n,) or not np.isfinite(full['all_features']).all()):
        raise ValueError('Invalid complete encoder feature/patch arrays')
    current = {k: full[k] for k in ('centers', 'radii', 'scores')}
    current_indices = source_indices(full['all_centers'], full['all_radii'], full['all_scores'], current)
    if (not np.array_equal(full['all_features'][current_indices], full['features'])
            or not np.array_equal(full['all_patch_indices'][current_indices], full['patch_indices'])):
        raise ValueError('Feature or patch identities do not match current accepted rows')
    metadata = json.loads(str(full['metadata']))
    metadata.pop('metrics', None)  # Score-control proposal metrics do not describe the hybrid.
    metadata.update(global_selection='hybrid_50_50', protected_anchors=2048,
        selection_reference_used=False, feature_source='same full inference, exact source-row indices')
    payload = {k: full[k] for k in full.files if k not in ['centers', 'radii', 'scores', 'features', 'patch_indices', 'metadata']}
    for key, source in [('centers', 'all_centers'), ('radii', 'all_radii'), ('scores', 'all_scores'),
                        ('features', 'all_features'), ('patch_indices', 'all_patch_indices')]:
        payload[key] = full[source][indices]
    payload.update(metadata=np.array(json.dumps(metadata)), source_indices=indices)
    np.savez_compressed(destination, **payload)


def check_graph(plan, row, proposal, graph):
    config = plan['baseline']['reconstruction']
    if (row['sample_index'] != 79 or not row['swc_valid'] or row['reconstruction_roots'] != 1
            or Path(row['proposal_path']).resolve() != proposal.resolve()
            or not row['foreground_cap_satisfied']):
        raise ValueError('Reconstruction identity, single-root SWC or foreground cap checks failed')
    for key, value in [('min_proposal_score', .5), ('max_nodes', config['graph_max_nodes']),
            ('selection_mode', config['graph_selection_mode']), ('nms_mode', config['graph_nms_mode']),
            ('nms_distance', config['graph_nms_distance'])]:
        if graph[key] != value:
            raise ValueError(f'Graph setting differs: {key}')
    with np.load(row['graph_path'], allow_pickle=False) as g, np.load(proposal, allow_pickle=False) as p:
        if len(g['centers']) > config['graph_max_nodes']:
            raise ValueError('Graph node cap exceeded')
        idx = g['original_indices']
        for key in ['centers', 'radii', 'scores', 'features']:
            if not np.array_equal(g[key], p[key][idx]):
                raise ValueError(f'Graph changed proposal {key} identities')


def run(output, device):
    path = output / 'comparison.json'
    plan = read(path)
    if plan['status'] != 'prepared':
        raise ValueError('Preserve partial/completed runs; prepare a fresh directory')
    verify_new(plan)
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; refusing an unintended CPU reconstruction')
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}; no optimizer steps", flush=True)
    start = time.perf_counter()
    plan.update(status='running', device=device)
    write(path, plan)
    try:
        directories = {name: output / f'{name}_proposals' for name in plan['arms']}
        for directory in directories.values():
            directory.mkdir()
        proposals = {name: directory / 'sample_0079_proposals.npz' for name, directory in directories.items()}
        print('Stage 1/3: regenerate the frozen pool once, preserving all encoder features.', flush=True)
        command = aggregation_command(plan['baseline'], proposals['current_score'], device)
        command[2] = str(REPO_ROOT / 'scripts/pointneuron2/aggregate_with_features.py')
        live(command, output / 'aggregation.log')
        with np.load(proposals['current_score'], allow_pickle=False) as full, np.load(plan['inputs']['saved_pool']['path'], allow_pickle=False) as old, np.load(plan['inputs']['hybrid_selection']['path'], allow_pickle=False) as selection:
            make_hybrid(full, old, selection, proposals['hybrid_50_50'])
        plan['proposal_sha256'] = {name: digest(p) for name, p in proposals.items()}
        plan['pool_and_feature_checks'] = 'passed'
        write(path, plan)
        print('Exact pool, accepted baseline rows, protected anchors and real feature identities: passed.', flush=True)
        for number, name in enumerate(plan['arms'], 2):
            arm = output / name
            print(f'Stage {number}/3: reconstruct {name}; graph cutoff=0.5, then evaluate.', flush=True)
            live(reconstruction_command(plan['baseline'], arm, directories[name], .5, device), output / f'{name}.log')
            summary = read(arm / 'summary.json')
            if summary['failures'] or len(summary['samples']) != 1:
                raise ValueError('Reconstruction failed or changed the denominator')
            row = summary['samples'][0]
            with np.load(row['graph_path'], allow_pickle=False) as graph:
                meta = json.loads(str(graph['metadata']))
            check_graph(plan, row, proposals[name], meta)
            metrics = {}
            for kind, script in [('spatial', 'evaluate_paper_style_swc.py'), ('topology', 'evaluate_geodesic_baseline.py')]:
                live([sys.executable, '-u', str(REPO_ROOT / 'scripts' / script), '--root', plan['baseline']['data_root'],
                    '--summary', str(arm / 'summary.json'), '--json-output', str(arm / f'{kind}.json'),
                    '--csv-output', str(arm / f'{kind}.csv')], output / f'{name}_{kind}.log')
                report = read(arm / f'{kind}.json')
                if len(report['samples']) != 1 or report['samples'][0]['sample_index'] != 79:
                    raise ValueError('Metric denominator differs')
                metrics[kind] = report['samples'][0]
            values = [metrics['spatial'][k] for k in ['precision', 'recall', 'f1']] + [metrics['topology']['edge_f1']]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError('Invalid reconstruction metric')
            result = {'name': name, 'reconstruction': row, 'graph_metadata': meta, 'metrics': metrics,
                'proposal_sha256': plan['proposal_sha256'][name], 'graph_sha256': digest(row['graph_path']),
                'swc_sha256': digest(row['swc_path'])}
            plan['results'].append(result)
            write(path, plan)
            m = metrics['spatial']
            print(f"{name}: spatial_F1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} topology_F1={metrics['topology']['edge_f1']:.4f}", flush=True)
        verify_new(plan)
        if any(digest(proposals[n]) != sha for n, sha in plan['proposal_sha256'].items()):
            raise ValueError('Proposal artifacts changed during reconstruction')
        plan['deltas_hybrid_minus_current'] = {
            k: plan['results'][1]['metrics']['spatial'][k] - plan['results'][0]['metrics']['spatial'][k]
            for k in ['precision', 'recall', 'f1']}
        previous = next(r for r in plan['baseline']['results'] if r['cutoff'] == .5)['metrics']['spatial']
        plan['current_repeat_deltas_vs_saved'] = {k: plan['results'][0]['metrics']['spatial'][k] - previous[k]
            for k in ['precision', 'recall', 'f1']}
        plan.update(status='completed', paired_checks='passed', elapsed_seconds=time.perf_counter()-start)
    except BaseException as exc:
        plan.update(status='failed_or_interrupted', error=str(exc), elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        write(path, plan)
    print(f"Status: completed; paired checks: passed; elapsed={plan['elapsed_seconds']:.1f}s", flush=True)
    print(f'Results: {path}', flush=True)
    print('One selected validation case; inspect precision/recall and SWCs before any broader change.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'check', 'run'], required=True)
    parser.add_argument('--baseline-comparison', type=Path)
    parser.add_argument('--hybrid-report', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == 'prepare':
        if args.baseline_comparison is None or args.hybrid_report is None:
            parser.error('prepare requires --baseline-comparison and --hybrid-report')
        prepare(args.baseline_comparison.resolve(), args.hybrid_report.resolve(), output)
    elif args.stage == 'check':
        verify_new(read(output / 'comparison.json'))
        print('Prepared-run provenance and aggregation equivalence checks passed; no GPU work launched.')
    else:
        run(output, args.device)


if __name__ == '__main__':
    main()
