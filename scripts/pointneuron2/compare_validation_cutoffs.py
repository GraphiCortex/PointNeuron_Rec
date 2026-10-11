"""Extend the paired graph-cutoff check to frozen validation, reusing neuron 21.

Prepare only verifies provenance and writes plans. Run is a separate visible
stage with one proposal inference/two reconstructions per new neuron. Every
failure is retained; mean scores are withheld unless all seven pairs complete.
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
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.vaa3d_raw import read_header
from scripts.pointneuron2.compare_graph_cutoffs import run as run_pair, verify
from scripts.pointneuron2.sampling_ablation import checked_manifest, check_matches, digest, read, write


def completed_pair(path, expected_checkpoint):
    plan = read(path)
    if (plan['status'] != 'completed' or plan.get('paired_checks') != 'passed'
            or plan['checkpoint_sha256'] != expected_checkpoint
            or [r['cutoff'] for r in plan['results']] != [.85, .5]):
        raise ValueError('Requires a complete matched cutoff pair from the same checkpoint')
    verify(plan)
    shared = plan['shared_proposals']
    if digest(shared['path']) != shared['sha256']:
        raise ValueError('Shared proposals changed')
    for arm in plan['results']:
        row = arm['reconstruction']
        if (arm['shared_proposal_sha256'] != shared['sha256']
                or Path(row['proposal_path']).resolve() != Path(shared['path']).resolve()
                or row['sample_index'] != plan['sample_index'] or not row['swc_valid']
                or row['reconstruction_roots'] != 1):
            raise ValueError('Saved arm identity/proposal/SWC checks failed')
        root = Path(row['graph_path']).parents[1]
        summary = read(root / 'summary.json')
        if summary['failures'] or summary['samples'] != [row]:
            raise ValueError('Saved reconstruction summary changed')
        for kind in ['graph', 'swc']:
            if digest(row[kind + '_path']) != arm[kind + '_sha256']:
                raise ValueError(f'Saved {kind} changed')
        for kind in ['spatial', 'topology']:
            if read(root / f'{kind}.json')['samples'] != [arm['metrics'][kind]]:
                raise ValueError(f'Saved {kind} metric report changed')
        values = [arm['metrics']['spatial'][k] for k in ['precision', 'recall', 'f1']]
        values.append(arm['metrics']['topology']['edge_f1'])
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Nonfinite or invalid saved metric values')
        if arm['metrics']['topology']['nodes'] > plan['reconstruction']['graph_max_nodes']:
            raise ValueError('Graph exceeds the fixed node budget')
    return plan


def complete_means(cases, results):
    """Return no aggregate if any requested pair failed or is absent."""
    expected = {c['sample_index'] for c in cases}
    if set(results) != expected or any(c['status'] != 'completed' for c in cases):
        return None
    means = {}
    for cutoff in [.85, .5]:
        rows = [next(a for a in results[index]['results'] if a['cutoff'] == cutoff) for index in sorted(expected)]
        means[str(cutoff)] = {
            **{key: float(np.mean([a['metrics']['spatial'][key] for a in rows])) for key in ['precision', 'recall', 'f1']},
            'topology_f1': float(np.mean([a['metrics']['topology']['edge_f1'] for a in rows])),
            'bridges': float(np.mean([a['reconstruction']['bridge_edges'] for a in rows])),
            'neurons': len(rows),
            'foreground_cap_failed_neurons': [results[index]['sample_index'] for index in sorted(expected)
                if not next(a for a in results[index]['results'] if a['cutoff'] == cutoff)['reconstruction']['foreground_cap_satisfied']]}
    return means


def prepare(experiment, reuse_path, output):
    if output.exists():
        raise FileExistsError('Use a fresh validation comparison directory')
    original = read(experiment / 'experiment.json')
    if original['status'] not in ('trained', 'evaluated'):
        raise ValueError('Requires the completed coordinate experiment')
    for name, sha in original['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Original pipeline changed: {name}')
    manifest = checked_manifest(original, experiment)
    check_matches([read(experiment / a['name'] / 'run.json') for a in original['arms']])
    local = next(a for a in original['arms'] if a['name'] == 'repaired_local')
    prototype = completed_pair(reuse_path, local['checkpoint_sha256'])
    partition = read(original['partition'])
    expected = sorted(r['sample_index'] for r in partition['partitions']['val'])
    if len(expected) != 7 or prototype['sample_index'] != 21 or 21 not in expected:
        raise ValueError('Requires seven frozen validation neurons and the completed neuron-21 pair')
    if (prototype['reconstruction'] != original['config']['reconstruction']
            or Path(prototype['data_root']).resolve() != Path(original['data_root']).resolve()
            or prototype['manifest_sha256'] != original['manifest_sha256']):
        raise ValueError('Reused experiment differs from the frozen comparison settings/data')
    files = sorted((REPO_ROOT / 'src').rglob('*.py')) + sorted((REPO_ROOT / 'scripts').rglob('*.py'))
    files += sorted((REPO_ROOT / 'configs/pointneuron2').glob('*.json'))
    sources = {p.relative_to(REPO_ROOT).as_posix(): digest(p) for p in files}
    samples = scan_gold166(original['data_root'])
    shared_fields = ['data_root', 'checkpoint', 'checkpoint_sha256', 'checkpoint_epoch', 'partition',
                     'partition_sha256', 'manifest', 'manifest_sha256', 'reconstruction', 'cutoffs', 'sampling_pipeline']
    child_plans = {}; cases = []
    for index in expected:
        rows = [r for r in manifest['records'] if r['partition'] == 'val' and r['sample_index'] == index]
        if not rows:
            raise ValueError(f'No corrected validation cache for neuron {index}')
        with np.load(rows[0]['resolved_path'], allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata']))
        sample = samples[index]
        if (sample.sample_id != rows[0]['sample_id'] or digest(sample.volume_path) != metadata['volume_sha256']
                or digest(sample.swc_path) != metadata['swc_sha256']):
            raise ValueError(f'Validation source changed: {index}')
        if index == 21:
            cases.append({'sample_index': index, 'sample_id': sample.sample_id, 'reused': True,
                          'status': 'completed', 'comparison': str(reuse_path.resolve()),
                          'comparison_sha256': digest(reuse_path)})
        else:
            child = {key: prototype[key] for key in shared_fields}
            child.update(status='prepared', scope='one case in seven-neuron internal validation cutoff control; no training',
                         original_experiment=str(experiment.resolve()), sample_index=index, sample_id=sample.sample_id,
                         volume=str(sample.volume_path.resolve()), volume_sha256=metadata['volume_sha256'],
                         reference=str(sample.swc_path.resolve()), reference_sha256=metadata['swc_sha256'],
                         dimensions_xyzc=list(read_header(sample.volume_path).dimensions),
                         source_sha256=sources, results=[], optimizer_steps=0)
            path = output / f'neuron_{index:04d}' / 'comparison.json'
            cases.append({'sample_index': index, 'sample_id': sample.sample_id, 'reused': False,
                          'status': 'prepared', 'comparison': str(path.resolve())})
            child_plans[path] = child
        print(f"verified validation neuron={index}; {'reuse completed pair' if index == 21 else 'prepared new pair'}", flush=True)
    parent = {'status': 'prepared', 'scope': 'seven-neuron internal validation; selected neuron 21 reused; no training/test data',
              'original_experiment': str(experiment.resolve()), 'checkpoint_sha256': local['checkpoint_sha256'],
              'reconstruction': prototype['reconstruction'], 'expected_neurons': expected,
              'source_sha256': sources, 'cases': cases, 'failures': [], 'means': None,
              'optimizer_steps': 0, 'reserved_test_neurons_used': False,
              'mean_definition': 'equal weight for all seven requested neurons; no mean if a pair is incomplete'}
    output.mkdir(parents=True)
    for path, child in child_plans.items():
        path.parent.mkdir(); write(path, child)
    write(output / 'validation_comparison.json', parent)
    print(f'Prepared seven validation pairs: six new, one reused. Directory: {output.resolve()}')


def run(output, device):
    path = output / 'validation_comparison.json'; plan = read(path)
    if plan['status'] != 'prepared':
        raise ValueError('Requires a fresh prepared validation comparison; preserve prior outputs')
    for name, sha in plan['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Source/settings changed: {name}; prepare a fresh comparison')
    for case in plan['cases']:
        case_path = Path(case['comparison'])
        if case['reused']:
            if digest(case_path) != case['comparison_sha256']:
                raise ValueError('Reused neuron-21 comparison changed')
            completed_pair(case_path, plan['checkpoint_sha256'])
        else:
            child = read(case_path); verify(child)
            if child['status'] != 'prepared':
                raise ValueError('A child run is not fresh/prepared')
    plan.update(status='running', device=device); write(path, plan)
    start = time.perf_counter(); results = {}
    try:
        for case in plan['cases']:
            case_path = Path(case['comparison']); index = case['sample_index']
            try:
                if case['reused']:
                    print(f'neuron={index}: reuse verified completed outputs; no inference/reconstruction', flush=True)
                else:
                    print(f'neuron={index}: one new proposal set and two graph cutoffs on {device}', flush=True)
                    run_pair(case_path.parent, device)
                results[index] = completed_pair(case_path, plan['checkpoint_sha256'])
                case.update(status='completed', comparison_sha256=digest(case_path))
            except Exception as exc:
                case.update(status='failed', error=str(exc))
                plan['failures'].append({'sample_index': index, 'error': str(exc), 'comparison': str(case_path)})
                print(f'FAILED neuron={index}: {exc}; preserving failure and continuing remaining neurons', flush=True)
            write(path, plan)
        for case in plan['cases']:
            if case['status'] != 'completed':
                continue
            try:
                case_path = Path(case['comparison'])
                if digest(case_path) != case['comparison_sha256']:
                    raise ValueError('Completed comparison changed during the validation run')
                results[case['sample_index']] = completed_pair(case_path, plan['checkpoint_sha256'])
            except Exception as exc:
                case.update(status='failed', error=str(exc))
                results.pop(case['sample_index'], None)
                plan['failures'].append({'sample_index': case['sample_index'], 'error': str(exc), 'comparison': case['comparison']})
        plan['means'] = complete_means(plan['cases'], results)
        plan['status'] = 'completed' if plan['means'] is not None and not plan['failures'] else 'failed_or_incomplete'
    except BaseException as exc:
        plan.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        plan['elapsed_seconds_this_run'] = time.perf_counter() - start
        write(path, plan)
    print(f"Status: {plan['status']}; completed pairs: {len(results)}/7; reused: 1; new pairs requested: 6")
    if plan['means'] is not None:
        for cutoff, mean in plan['means'].items():
            print(f"cutoff={cutoff} seven-neuron mean: spatial_F1={mean['f1']:.4f} precision={mean['precision']:.4f} "
                  f"recall={mean['recall']:.4f} topology_F1={mean['topology_f1']:.4f} bridges={mean['bridges']:.2f} "
                  f"foreground_cap_failed_neurons={mean['foreground_cap_failed_neurons']}", flush=True)
    else:
        print('No aggregate scores: at least one requested paired comparison failed or is incomplete.')
    print(f'Results: {path.resolve()}')
    return 0 if plan['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    parser.add_argument('--experiment', type=Path, help='Original trained coordinate experiment (prepare only)')
    parser.add_argument('--reuse-comparison', type=Path, help='Completed neuron-21 comparison.json (prepare only)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.experiment is None or args.reuse_comparison is None:
            parser.error('prepare requires --experiment and --reuse-comparison')
        prepare(args.experiment.resolve(), args.reuse_comparison.resolve(), args.output_dir.resolve())
        return 0
    return run(args.output_dir.resolve(), args.device)


if __name__ == '__main__':
    raise SystemExit(main())
