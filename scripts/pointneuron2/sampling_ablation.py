"""Visible, staged three-arm sampler pilot; no stage automatically starts the next."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.gold166 import scan_gold166


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def live(command, log):
    print(subprocess.list2cmdline(command), flush=True)
    with Path(log).open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True); handle.write(line); handle.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()


def check_matches(runs):
    """Require identical initialization, pools, batch order and augmentation."""
    first = runs[0]
    for run in runs:
        if run['status'] != 'completed':
            raise ValueError('An arm did not complete')
        for key in ('manifest_sha256', 'partition_sha256', 'initial_model_sha256', 'random_streams', 'source_sha256'):
            if run[key] != first[key]:
                raise ValueError(f'Unmatched training arms: {key}')
        if len(run['epochs']) != len(first['epochs']):
            raise ValueError('Unmatched epoch counts')
        for actual, expected in zip(run['epochs'], first['epochs']):
            for role in ('train', 'val'):
                for key in ('records', 'annotated_records', 'batch_order_sha256', 'augmented_candidates_sha256'):
                    if actual[role][key] != expected[role][key]:
                        raise ValueError(f'Unmatched {role} {key} at epoch {actual["epoch"]}')


def checked_manifest(plan, experiment, *, allow_historical_reader=False):
    path = experiment / 'cache/manifest.json'
    if plan.get('manifest_sha256') != digest(path):
        raise ValueError('Shared cache manifest changed')
    manifest = load_sampling_manifest(path, Path(plan['partition']), allow_historical_reader=allow_historical_reader)
    if manifest['status'] != 'completed' or manifest['failures']:
        raise ValueError('Cache has failures')
    expected = {x['sample_index'] for role in ('train', 'val') for x in read(plan['partition'])['partitions'][role]}
    present = {x['sample_index'] for x in manifest['records']}
    if present != expected:
        raise ValueError('Every frozen train/validation neuron must have cached patches')
    return manifest


def check_evaluation_sources(plan, manifest):
    """Reconstruction must use the same images/references as cached validation."""
    samples = scan_gold166(plan['data_root'])
    checked = set()
    for row in manifest['records']:
        index = row['sample_index']
        if row['partition'] != 'val' or index in checked:
            continue
        sample = samples[index]
        with np.load(row['resolved_path'], allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
        if (sample.sample_id != row['sample_id'] or digest(sample.volume_path) != metadata['volume_sha256']
                or digest(sample.swc_path) != metadata['swc_sha256']):
            raise ValueError(f'Validation image/reference changed since caching: {index}')
        checked.add(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/pointneuron2/sampling_ablation.json',
                        help='Used only at preparation; later stages restore the frozen experiment config')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.output_dir is None or args.artifact_root is None:
            parser.error('prepare requires --output-dir and --artifact-root')
        experiment = args.output_dir.resolve()
        experiment.mkdir(parents=True, exist_ok=False)
        config = read(args.config)
        files = sorted((REPO_ROOT / 'src').rglob('*.py')) + sorted((REPO_ROOT / 'scripts').rglob('*.py'))
        files += sorted((REPO_ROOT / 'configs/pointneuron2').glob('*.json'))
        plan = {'status': 'preparing', 'scope': 'single-seed five-epoch internal pilot; no test neurons',
                'config': config, 'data_root': str((args.artifact_root / 'data/gold166').resolve()),
                'partition': str(REPO_ROOT / config['partition']),
                'source_sha256': {p.relative_to(REPO_ROOT).as_posix(): digest(p) for p in files}}
    else:
        if args.experiment is None:
            parser.error('train/evaluate requires --experiment')
        experiment = args.experiment.resolve(); plan = read(experiment / 'experiment.json')
        required_status = 'prepared' if args.stage == 'train' else 'trained'
        if plan['status'] != required_status:
            raise ValueError(f'{args.stage} requires status={required_status}; got {plan["status"]}')
        for name, expected in plan['source_sha256'].items():
            if digest(REPO_ROOT / name) != expected:
                raise ValueError(f'Code/settings changed since preparation: {name}; prepare a fresh experiment')
        manifest = checked_manifest(plan, experiment)
        if args.stage == 'evaluate':
            check_evaluation_sources(plan, manifest)
        config = plan['config']
        plan['status'] = 'training' if args.stage == 'train' else 'evaluating'
    plan_path = experiment / 'experiment.json'; write(plan_path, plan)
    python = [sys.executable, '-u']
    try:
        if args.stage == 'prepare':
            live(python + [str(REPO_ROOT / 'scripts/pointneuron2/build_sampling_cache.py'),
                 '--root', plan['data_root'], '--config', str(REPO_ROOT / config['cache_config']),
                 '--partition', plan['partition'], '--output-dir', str(experiment / 'cache')], experiment / 'cache.log')
            plan['manifest_sha256'] = digest(experiment / 'cache/manifest.json')
            manifest = checked_manifest(plan, experiment)
            plan['records'] = {role: sum(r['partition'] == role for r in manifest['records']) for role in ('train', 'val')}
            plan['status'] = 'prepared'
        elif args.stage == 'train':
            plan['training_device'] = args.device; plan['arms'] = []
            for arm in config['arms']:
                name = arm['name']; print(f'\nStarting {name}: {config["epochs"]} epochs on {args.device}', flush=True)
                live(python + [str(REPO_ROOT / 'scripts/pointneuron2/train_sampling.py'),
                     '--manifest', str(experiment / 'cache/manifest.json'), '--partition', plan['partition'],
                     '--output-dir', str(experiment / name), '--epochs', str(config['epochs']),
                     '--strategy', arm['strategy'], '--chamfer-weight', str(arm['chamfer_weight']),
                     '--device', args.device, '--log-every', '1'] +
                     (['--proposal-coordinate-mode', arm['proposal_coordinate_mode']] if 'proposal_coordinate_mode' in arm else []),
                     experiment / f'{name}.log')
                result = read(experiment / name / 'run.json')
                if result['strategy'] != arm['strategy'] or result['config']['chamfer_weight'] != arm['chamfer_weight']:
                    raise ValueError('Arm settings disagree with plan')
                if ('proposal_coordinate_mode' in arm and
                        result['recipe']['proposal_coordinate_mode'] != arm['proposal_coordinate_mode']):
                    raise ValueError('Proposal coordinate mode disagrees with plan')
                plan['arms'].append({'name': name, 'checkpoint_sha256': digest(experiment / name / 'proposal.pt')})
                write(plan_path, plan)
            check_matches([read(experiment / x['name'] / 'run.json') for x in config['arms']])
            plan.update(status='trained', matching_checks='passed')
        else:
            validation = read(plan['partition'])['partitions']['val']
            plan['evaluation_device'] = args.device; plan['evaluation'] = []
            for arm in plan['arms']:
                name = arm['name']; checkpoint = experiment / name / 'proposal.pt'
                if digest(checkpoint) != arm['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed since training')
                output = experiment / f'{name}_reconstruction'
                if output.exists():
                    raise FileExistsError('Evaluation output already exists; preserve it')
                command = python + [str(REPO_ROOT / 'scripts/run_end_to_end.py'), '--root', plan['data_root'],
                          '--checkpoint', str(checkpoint), '--output-root', str(output), '--device', args.device]
                for key, value in config['reconstruction'].items():
                    command += ['--' + key.replace('_', '-'), str(value)]
                for row in validation:
                    command += ['--sample-index', str(row['sample_index'])]
                # The underlying runner records every neuron failure. Preserve that
                # report and stop here; never silently compute means over survivors.
                live(command, experiment / f'{name}_reconstruction.log')
                summary = read(output / 'summary.json')
                if summary.get('failures') or {r['sample_index'] for r in summary['samples']} != {r['sample_index'] for r in validation}:
                    raise ValueError('Incomplete reconstruction denominator')
                metrics = {}
                for kind, script in [('spatial', 'evaluate_paper_style_swc.py'), ('topology', 'evaluate_geodesic_baseline.py')]:
                    live(python + [str(REPO_ROOT / 'scripts' / script), '--root', plan['data_root'],
                         '--summary', str(output / 'summary.json'), '--json-output', str(output / f'{kind}.json'),
                         '--csv-output', str(output / f'{kind}.csv')], experiment / f'{name}_{kind}.log')
                    report = read(output / f'{kind}.json')
                    if {r['sample_index'] for r in report['samples']} != {r['sample_index'] for r in validation}:
                        raise ValueError('Incomplete metric denominator')
                    metrics[kind] = report
                plan['evaluation'].append({'name': name, 'metrics': metrics}); write(plan_path, plan)
            plan['status'] = 'evaluated'
    except BaseException as exc:
        plan.update(status='failed_or_interrupted', failed_stage=args.stage, error=str(exc))
        raise
    finally:
        write(plan_path, plan)
    print(f'Experiment: {experiment}\nStatus: {plan["status"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
