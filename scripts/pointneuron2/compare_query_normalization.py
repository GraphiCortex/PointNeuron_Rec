"""Fresh pooled-LayerNorm query fit versus a saved, matched BatchNorm fit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from pointneuron.data.query_cache import file_digest, load_query_manifest
from pointneuron.models.query_pipeline import build_models
from scripts.pointneuron2.train_query_comparison import run_arm, state_digest, evaluation_steps, check_pairs


def initialization_check(baseline_config):
    control_config = copy.deepcopy(baseline_config)
    control_config['pooled_normalization'] = 'layernorm'
    old_encoder, old_head = build_models(baseline_config, 'query', baseline_config['seed'])
    new_encoder, new_head = build_models(control_config, 'query', control_config['seed'])
    old, new = old_encoder.state_dict(), new_encoder.state_dict()
    removed = {'global_mlp.1.running_mean', 'global_mlp.1.running_var', 'global_mlp.1.num_batches_tracked'}
    if set(old) - set(new) != removed or set(new) - set(old):
        raise ValueError('Normalization control changed unexpected encoder state')
    for key in new:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    if state_digest(old_head) != state_digest(new_head):
        raise ValueError('Normalization control changed query-head initialization')
    return control_config, {'unchanged_encoder_tensors': 'exact', 'query_head_initialization': 'exact',
                            'removed_state': sorted(removed), 'baseline_encoder_sha256': state_digest(old_encoder),
                            'query_head_sha256': state_digest(old_head)}


def verify_baseline(models_dir, manifest_path, manifest, steps, requested, device):
    path = models_dir / 'comparison.json'
    run = json.loads(path.read_text(encoding='utf-8'))
    if run['status'] != 'completed' or run['paired_checks'] != 'passed':
        raise ValueError('Requires a completed paired baseline')
    if sorted(a['arm'] for a in run['arms']) != ['pointwise', 'query']:
        raise ValueError('Requires the saved pointwise/query baseline')
    check_pairs(run['arms'])
    baseline = next(a for a in run['arms'] if a['arm'] == 'query')
    if baseline['status'] != 'completed' or baseline['scope'] != 'fixed-training-crop execution check' or baseline['roundtrip'] != 'exact':
        raise ValueError('Requires an exactly reloaded fixed-crop baseline')
    if baseline['config']['decoder']['dropout'] != 0 or baseline['config'].get('pooled_normalization', 'batchnorm') != 'batchnorm':
        raise ValueError('Baseline must use zero decoder dropout and pooled BatchNorm')
    if baseline['updates'] != steps or baseline['device'] != device:
        raise ValueError('Baseline/control update budgets and devices must match')
    if baseline['manifest_sha256'] != file_digest(manifest_path) or baseline['partition_sha256'] != manifest['partition_sha256']:
        raise ValueError('Baseline/control cache and partitions must match')
    milestones = evaluation_steps(steps, requested or baseline['evaluation_steps'], True)
    if milestones != baseline['evaluation_steps']:
        raise ValueError('Baseline/control evaluation milestones must match')
    for record in baseline['evaluations']:
        for key in ('checkpoint', 'proposals'):
            filename = Path(record[key])
            if filename.name != record[key] or filename.is_absolute():
                raise ValueError('Invalid baseline artifact filename')
            if file_digest(models_dir / 'query' / filename) != record[key + '_sha256']:
                raise ValueError('Baseline snapshot hash mismatch')
    last = baseline['evaluations'][-1]
    checkpoint = torch.load(models_dir / 'query' / last['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['config'] != baseline['config'] or checkpoint['arm'] != 'query' or checkpoint['provenance']['optimizer_updates'] != steps:
        raise ValueError('Baseline checkpoint settings/updates mismatch')
    # Decoder, labels, loss, input selection and pointwise recipe source must be unchanged.
    allowed = {'src/pointneuron/models/dgcnn.py', 'src/pointneuron/models/query_pipeline.py',
               'scripts/pointneuron2/train_query_comparison.py'}
    for filename, digest in baseline['source_sha256'].items():
        relative = filename.replace('\\', '/')
        if relative not in allowed and file_digest(ROOT / relative) != digest:
            raise ValueError(f'Baseline/control source changed: {relative}')
    config, initialized = initialization_check(baseline['config'])
    if initialized['baseline_encoder_sha256'] != baseline['initial_encoder_sha256']:
        raise ValueError('Original BatchNorm encoder initialization was not reproduced')
    return baseline, config, milestones, initialized


def check_control(baseline, control):
    expected = copy.deepcopy(baseline['config']); expected['pooled_normalization'] = 'layernorm'
    if control['config'] != expected:
        raise ValueError('Control changed settings beyond pooled normalization')
    for key in ('inputs_sha256', 'manifest_sha256', 'partition_sha256', 'crop_paths', 'updates', 'device', 'pointwise_recipe', 'evaluation_steps'):
        if baseline[key] != control[key]:
            raise ValueError(f'Unmatched normalization comparison: {key}')
    actual = [(r['step'], r['optimizer_updates']) for r in control['evaluations']]
    expected_steps = [(r['step'], r['optimizer_updates']) for r in baseline['evaluations']]
    if actual != expected_steps or any('single_crop' not in r for r in control['evaluations']):
        raise ValueError('Missing/mismatched control inference checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-models', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--partition', type=Path, default=ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-steps', type=int, nargs='+', default=[])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.log_every, args.cpu_threads) < 1:
        parser.error('Update budget, logging frequency and threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(args.cpu_threads)
    manifest = load_query_manifest(args.manifest, args.partition)
    baseline, config, milestones, initialized = verify_baseline(args.baseline_models, args.manifest, manifest,
                                                               args.steps, args.eval_steps, args.device)
    recipe = json.loads((ROOT / config['pointwise_recipe']).read_text())['training']
    if recipe != baseline['pointwise_recipe']:
        raise ValueError('Baseline/control pointwise recipe changed')
    args.smoke = True; args.eval_steps = milestones; args.inspect_single = True
    args.output_dir.mkdir(parents=True)
    summary = {'status': 'running', 'scope': 'same two fitted training crops; no validation or SWC evaluation',
               'comparison': 'saved query BatchNorm versus fresh query pooled LayerNorm',
               'baseline_comparison_sha256': file_digest(args.baseline_models / 'comparison.json'),
               'baseline_models': str(args.baseline_models.resolve()), 'baseline': baseline,
               'config': config, 'initialization_checks': initialized, 'update_budget': args.steps,
               'evaluation_steps': milestones, 'source_sha256': file_digest(Path(__file__))}
    path = args.output_dir / 'normalization_comparison.json'
    print(f'Only the pooled encoder normalization changes; fresh query fit, {args.steps} updates; device={args.device}.', flush=True)
    print(f'Initialization checks: passed; reuse baseline: {args.baseline_models.resolve()}', flush=True)
    try:
        control = run_arm('query', args, manifest, config, recipe, args.output_dir / 'query')
        check_control(baseline, control)
        summary.update(status='completed', matched_checks='passed', control=control)
    except BaseException as exc:
        summary.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        # Source artifacts are immutable throughout training.
        if file_digest(args.baseline_models / 'comparison.json') != summary['baseline_comparison_sha256']:
            summary.update(status='failed_or_interrupted', error='Baseline comparison changed during the control')
        path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if summary['status'] != 'completed':
        raise RuntimeError(summary['error'])
    print(f'Status: completed; matched checks: passed; report: {path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
