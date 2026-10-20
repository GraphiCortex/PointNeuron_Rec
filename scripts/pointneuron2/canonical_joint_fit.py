"""Matched canonical-target fit; explicit CPU check, throwaway timing, capped train/resume."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.pointneuron2 import stable_joint_learnability as fit
from pointneuron.models.partial_joint_loss import partial_joint_loss
from pointneuron.models.query_fit_state import cpu_snapshot, capture_rng, restore_rng

control, runner, base, write = fit.control, fit.runner, fit.base, fit.write
CONFIG = ROOT / 'configs/pointneuron2/canonical_joint_fit.json'
SCHEMA = 'pointneuron2-canonical-joint-fit-v1'


def validate_config(config):
    fixed = {'schema': SCHEMA, 'eligibility_policy': 'structural_bounds_absolute_bad_arclength_and_99pct_v1',
        'training_crops': [[0, 0], [0, 1], [92, 4], [86, 3]], 'target_counts': [6, 11, 15, 408],
        'input_counts': [848, 2048, 1219, 2048], 'query_count': 512, 'parent_stage': 1000,
        'updates': 3000, 'learning_rate': 1e-5, 'timing_updates': 8, 'snapshots': [0, 8, 64, 500, 1000, 2000, 3000],
        'minimum_precision_recall_at_2': .99, 'maximum_absolute_bad_length_increase': 1e-9, 'automatic_extension': False}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('Different prospective canonical comparison contract')
    for key, value in config.items():
        if key.endswith('_sha256') and (not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Expected exact lowercase SHA-256 pins')


def lengths(geometry):
    values = geometry['tolerances']['2.0']
    return {'extra': geometry['export_length'] * (1 - values['precision']) if geometry['export_length'] else 0.,
        'missed': geometry['reference_length'] * (1 - values['recall']) if geometry['reference_length'] else 0.}


def geometry_noninferior(actual, reference, config):
    values = actual['tolerances']['2.0']; a, b = lengths(actual), lengths(reference)
    return bool(all(values[k] is not None and values[k] >= config['minimum_precision_recall_at_2'] for k in ('precision', 'recall'))
        and np.isclose(actual['reference_length'], reference['reference_length'], rtol=0, atol=1e-9)
        and all(a[k] <= b[k] + config['maximum_absolute_bad_length_increase'] for k in a))


def eligibility(preparation, config):
    four = [r for r in preparation['records'] if [r['sample_index'], r['patch_index']] in config['training_crops']]
    if len(four) != 4:
        raise ValueError('Incomplete canonical preparation evidence')
    checks = {k: preparation['preparation_gate'][k] for k in ('structural_topology_all_crops', 'no_new_candidate_losses',
        'four_crop_candidate_recall_complete', 'all_targets_fit_capacity', 'exact_input_clouds_preserved')}
    checks['bounded_added_geometry_and_radius'] = all(c['maximum_curve_deviation'] <= .500001 and c['maximum_radius_deviation'] <= .500001
        for r in preparation['records'] for c in r['certificates'])
    checks['absolute_geometry_noninferiority_and_99pct_floor'] = all(geometry_noninferior(r['canonical_geometry'], r['legacy_geometry'], config) for r in four)
    checks['passed'] = all(checks.values())
    return checks


def load_inputs(config):
    validate_config(config)
    pinned = {str(ROOT / config[k]): config[k + '_sha256'] for k in ('baseline_report', 'preparation_report', 'preparation_review', 'canonical_manifest')}
    base.verify_hashes(pinned)
    preparation = base.read(ROOT / config['preparation_report']); review = base.read(ROOT / config['preparation_review'])
    if preparation['status'] != 'completed' or review['status'] != 'completed' or review['source_report_sha256'] != config['preparation_report_sha256']:
        raise ValueError('Incomplete or changed canonical preparation review')
    checked = eligibility(preparation, config)
    if not checked['passed']:
        raise ValueError('Canonical cache fails the prospective absolute-geometry eligibility policy')
    # Preserve the original failed preparation decision; do not edit or relabel it.
    if preparation['preparation_gate'] != review['preparation_gate'] or preparation['preparation_gate']['passed']:
        raise ValueError('Expected preserved strict normalized-fraction preparation failure')
    pinned.update(preparation['protected_files_sha256'])
    pinned.update({str(ROOT / p): h for p, h in preparation['source_files_sha256'].items()})
    saved_config = base.read(ROOT / config['source_recipe'])
    recipe, original, old_records, oracle, parent, _, protected, _ = fit.load_inputs(saved_config)
    pinned.update(protected)
    baseline = base.read(ROOT / config['baseline_report'])
    if (baseline['status'] != 'completed' or baseline['config'] != saved_config or baseline['proposal_updates'] != 3000
        or not baseline['sources_and_parent_unchanged'] or not baseline['prefix_model_optimizer_rng_exact']):
        raise ValueError('Different completed matched baseline')
    pinned.update(baseline['provenance']['sources'])
    pinned[baseline['last_checkpoint']] = config['baseline_terminal_sha256']
    if baseline['last_checkpoint_sha256'] != config['baseline_terminal_sha256']:
        raise ValueError('Different saved terminal comparator')
    manifest_path = ROOT / config['canonical_manifest']; manifest = base.read(manifest_path)
    if manifest['schema'] != 'pointneuron2-canonical-curve-targets-v2-experimental-cache' or manifest['records'] != preparation['four_crop_cache']:
        raise ValueError('Different versioned canonical cache')
    records = []
    for i, (cached, old) in enumerate(zip(manifest['records'], old_records)):
        path = (ROOT / preparation['config']['output'] / cached['path']).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or cached['crop'] != config['training_crops'][i]:
            raise ValueError('Canonical cache identity/path differs')
        pinned[str(path)] = cached['sha256']; pinned[str(ROOT / cached['source_path'])] = cached['source_sha256']
        base.verify_hashes({str(path): cached['sha256'], str(ROOT / cached['source_path']): cached['source_sha256']})
        with np.load(path, allow_pickle=False) as archive, np.load(ROOT / cached['source_path'], allow_pickle=False) as previous:
            for key in previous.files:
                if key not in ('nodes', 'roles', 'target_edges', 'target_components', 'junction_mask') and not np.array_equal(archive[key], previous[key]):
                    raise ValueError('Non-target cache content changed')
            record = {k: archive[k].copy() for k in ('points', 'nodes', 'target_edges', 'target_components', 'junction_mask')}
        if (not np.array_equal(record['points'], old['points']) or record['points'].shape != (config['input_counts'][i], 4)
            or record['nodes'].shape != (config['target_counts'][i], 6) or not np.isfinite(record['nodes']).all()):
            raise ValueError('Canonical input/target arrays differ')
        record.update(meta=old['meta'], geometry=old['geometry'], identity=dict(old['identity'], canonical_sha256=cached['sha256']))
        records.append(record)
    if len(records) != 4:
        raise ValueError('Four supervised training crops required')
    for evaluation in baseline['evaluations']:
        for i, row in enumerate(evaluation['crops']):
            path = Path(config['baseline_report']).parent / f"step_{evaluation['fork_update']:04d}_crop_{i}.npz"
            pinned[str(ROOT / path)] = row['prediction_sha256']
    base.verify_hashes(pinned)
    return recipe, original, records, oracle, parent, baseline, pinned, checked


def validate_state(model, optimizer, q, g, recipe):
    fit.validate_state(model, optimizer, q, g, base.read(fit.CONFIG), recipe)


def fingerprint(model, optimizer):
    def encode(value):
        if isinstance(value, torch.Tensor):
            return ['tensor', base.tensor_digest(value)]
        if isinstance(value, dict):
            return ['dict', [(str(k), encode(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0]))]]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [encode(v) for v in value]]
        return value
    state = encode((model.state_dict(), optimizer.state_dict(), capture_rng()))
    return hashlib.sha256(json.dumps(state, allow_nan=False).encode()).hexdigest()


def checkpoint(model, optimizer, report, config, recipe, provenance, purpose):
    q, g = report['proposal_updates'], report['graph_updates']; validate_state(model, optimizer, q, g, recipe)
    if purpose not in ('fit', 'timing', 'check', 'geometry_stop'):
        raise ValueError('Different checkpoint purpose')
    prefix = report.get('timing_prefix_verified', False)
    if type(prefix) is not bool or (q < 8 and prefix) or (q > 8 and not prefix) or (purpose == 'check' and q != 0) or (purpose == 'timing' and q != 8):
        raise ValueError('Checkpoint purpose or timing replay proof differs')
    return {'schema': SCHEMA, 'purpose': purpose, 'resumable': purpose == 'fit' and q < 3000,
        'config': config, 'recipe': recipe, 'provenance': provenance, 'proposal_updates': q, 'graph_updates': g,
        'historical_stage': 1000 + q, 'next_crop': q % 4, 'timing_prefix_verified': report.get('timing_prefix_verified', False),
        'parameter_names': [n for n, _ in model.named_parameters()], 'proposal_spec': model.proposal_model.model_spec(),
        'graph_spec': model.graph_model.model_spec(), 'model': cpu_snapshot(model.state_dict()),
        'optimizer': cpu_snapshot(optimizer.state_dict()), 'rng': capture_rng()}


def restore(saved, model, config, recipe, provenance, *, resume):
    q, g = saved.get('proposal_updates'), saved.get('graph_updates'); purpose = saved.get('purpose')
    if (saved.get('schema') != SCHEMA or saved.get('config') != config or saved.get('recipe') != recipe or saved.get('provenance') != provenance
        or type(q) is not int or type(g) is not int or not 0 <= g <= q <= 3000 or saved.get('next_crop') != q % 4
        or saved.get('historical_stage') != 1000 + q or purpose not in ('fit', 'check', 'timing', 'geometry_stop')
        or saved.get('resumable') != (purpose == 'fit' and q < 3000)
        or saved.get('parameter_names') != [n for n, _ in model.named_parameters()]
        or saved.get('proposal_spec') != model.proposal_model.model_spec() or saved.get('graph_spec') != model.graph_model.model_spec()
        or type(saved.get('timing_prefix_verified')) is not bool or (q > 8 and not saved['timing_prefix_verified'])
        or (q < 8 and saved['timing_prefix_verified']) or (purpose == 'check' and q != 0) or (purpose == 'timing' and q != 8)):
        raise ValueError('Checkpoint source/options/cursors/model differ')
    if resume and not saved['resumable']:
        raise ValueError('Terminal, timing, check or geometry-stop checkpoint cannot resume')
    model.load_state_dict(saved['model']); optimizer = runner.optimizer_for(model, dict(recipe, learning_rate=1e-5))
    optimizer.load_state_dict(cpu_snapshot(saved['optimizer'])); validate_state(model, optimizer, q, g, recipe)
    restore_rng(saved['rng'])
    return optimizer, q, g


def evaluate(model, records, oracle, original, recipe, device, q, directory, baseline):
    result = control.evaluate(model, records, oracle, original, recipe, device, q, directory)
    for row, comparator in zip(result['crops'], baseline['evaluations'][-1]['crops']):
        row['absolute_bad_lengths'] = lengths(row['geometry'])
        row['baseline_terminal_absolute_bad_lengths'] = lengths(comparator['geometry'])
        row['paired_geometry_noninferior'] = geometry_noninferior(row['geometry'], comparator['geometry'], base.read(CONFIG))
    result['historical_stage'] = 1000 + q
    result['paired_geometry_noninferior'] = all(r['paired_geometry_noninferior'] for r in result['crops'])
    return result


def save_state(model, optimizer, report, config, recipe, provenance, directory, purpose):
    path = directory / f"stage_{1000 + report['proposal_updates']:04d}.pt"
    if path.exists():
        raise FileExistsError('Preserve checkpoint evidence')
    temporary = path.with_suffix('.pt.writing')
    if temporary.exists():
        raise FileExistsError('Preserve incomplete checkpoint evidence')
    torch.save(checkpoint(model, optimizer, report, config, recipe, provenance, purpose), temporary); temporary.replace(path)
    report.update(last_checkpoint=str(path), last_checkpoint_sha256=base.file_digest(path))
    write(directory / 'report.json', report)


def verify_initial_prediction(directory, config):
    baseline_dir = (ROOT / config['baseline_report']).parent
    for i in range(4):
        actual = control.read_arrays(directory / f'step_0000_crop_{i}.npz')
        expected = control.read_arrays(baseline_dir / f'step_0000_crop_{i}.npz')
        for key in ('centers', 'radii', 'logits', 'decoder_features', 'candidate_edges', 'edge_logits', 'retained_node_ids', 'retained_edges', 'retained_edge_logits'):
            if not np.array_equal(actual[key], expected[key]):
                raise ValueError(f'GPU initial prediction differs from matched baseline: {key}')


def run(args):
    config = base.read(CONFIG); validate_config(config); torch.set_num_threads(2); runner.configure()
    if (args.action == 'resume') != (args.checkpoint is not None):
        raise ValueError('Only explicit resume accepts a checkpoint')
    device = torch.device('cpu' if args.action == 'check' else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    recipe, original, records, oracle, parent, baseline, protected, eligible = load_inputs(config)
    root = (args.directory or ROOT / config['output']).resolve()
    if not root.is_relative_to((ROOT / 'tmp/pointneuron2').resolve()):
        raise ValueError('Generated outputs belong in ignored tmp')
    sources = base.source_hashes(SimpleNamespace(recipe=CONFIG))
    launcher = ROOT / 'scripts/pointneuron2/run_canonical_joint_fit.ps1'; sources[str(launcher)] = base.file_digest(launcher)
    provenance = {'sources': sources, 'backend': runner.backend(device), 'root': str(root), 'baseline_sha256': config['baseline_report_sha256']}
    timing_reference = None
    if device.type == 'cuda':
        checked = base.read(root / 'check/report.json')
        if checked['status'] != 'completed' or not checked['cpu_check_passed'] or checked['provenance']['sources'] != sources or checked['config'] != config:
            raise ValueError('Matching completed zero-update CPU check required')
        if provenance['backend'] != baseline['provenance']['backend']:
            raise ValueError('GPU backend differs from saved matched baseline')
    if args.action in ('train', 'resume'):
        timing_reference = base.read(root / 'timing/report.json')
        if (timing_reference['status'] != 'completed' or not timing_reference['timing_gate'] or timing_reference['config'] != config
            or timing_reference['provenance'] != provenance):
            raise ValueError('Matching completed eight-update GPU timing required')
    directory = root / ('check' if args.action == 'check' else 'timing' if args.action == 'timing' else 'model')
    if args.action == 'resume':
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_relative_to(root) or args.checkpoint.suffix != '.pt' or not (args.checkpoint.parent.name == 'model' or args.checkpoint.parent.name.startswith('resume_')):
            raise ValueError('Resume checkpoint outside this fit')
        directory = root / f'resume_{args.checkpoint.stem}'; suffix = 2
        while directory.exists():
            directory = root / f'resume_{args.checkpoint.stem}_{suffix}'; suffix += 1
    if directory.exists():
        raise FileExistsError('Preserve prior action output')
    directory.mkdir(parents=True); started = time.perf_counter()
    report = {'schema': SCHEMA, 'status': 'running', 'action': args.action, 'config': config, 'provenance': provenance,
        'protected_files_sha256': protected, 'eligibility': eligible, 'historical_preparation_gate_passed': False,
        'proposal_updates': 0, 'graph_updates': 0, 'timing_prefix_verified': False, 'updates': [], 'probes': [], 'evaluations': []}
    write(directory / 'report.json', report)
    try:
        model = runner.warm_start(recipe, device); optimizer = control.imported_state(model, parent, recipe, 1e-5, device)
        if args.action == 'resume':
            saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            optimizer, q, g = restore(saved, model, config, recipe, provenance, resume=True)
            previous = base.read(args.checkpoint.parent / 'report.json')
            if previous['provenance'] != provenance:
                raise ValueError('Resume history source differs')
            report.update(proposal_updates=q, graph_updates=g, timing_prefix_verified=saved['timing_prefix_verified'], resumed_from=str(args.checkpoint))
            report['updates'] = [a for a in previous['updates'] if a['fork_update'] <= q]
            report['probes'] = [p for p in previous['probes'] if p['new_updates'] <= q]
        print(f"ACTION={args.action} device={provenance['backend']['name']} canonical_targets=6/11/15/408 new_updates={report['proposal_updates']}/3000", flush=True)
        if args.action == 'check':
            before = fingerprint(model, optimizer)
            report['evaluations'].append(evaluate(model, records, oracle, original, recipe, device, 0, directory, baseline))
            control.probe(model, records, original, recipe, device)
            signature = runner.prediction_signature(model, records, original, device)
            path = directory / 'temporary_check.pt'
            try:
                torch.save(checkpoint(model, optimizer, report, config, recipe, provenance, 'check'), path)
                optimizer, q, g = restore(torch.load(path, map_location='cpu', weights_only=True), model, config, recipe, provenance, resume=False)
                if q or g or fingerprint(model, optimizer) != before:
                    raise ValueError('CPU observation/state/Adam/RNG import or reload differs')
                runner.require_identical_predictions(signature, runner.prediction_signature(model, records, original, device))
            finally:
                path.unlink(missing_ok=True)
            if torch.cuda.is_initialized():
                raise ValueError('CPU check initialized CUDA')
            report.update(cpu_check_passed=True, state_optimizer_rng_unchanged=True, exact_reload_prediction_delta=0., cuda_initialized=False)
        else:
            cap = 8 if args.action == 'timing' else 3000
            start = report['proposal_updates']; seconds = []
            if start in config['snapshots']:
                if args.action != 'timing':
                    save_state(model, optimizer, report, config, recipe, provenance, directory, 'fit')
                report['evaluations'].append(evaluate(model, records, oracle, original, recipe, device, start, directory, baseline))
                if start == 0:
                    verify_initial_prediction(directory, config); report['initial_gpu_predictions_identical_to_baseline'] = True
                if start == 8 and not report['timing_prefix_verified']:
                    if fingerprint(model, optimizer) != timing_reference['terminal_state_fingerprint']:
                        raise ValueError('Resumed eight-update timing prefix differs')
                    report['timing_prefix_verified'] = True
            for q in range(start + 1, cap + 1):
                one = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
                record = records[(q - 1) % 4]; output = runner.forward(model, record, original, device)
                incoming = fit.timing.describe(output, record, original, recipe)
                total, _, edge, _ = partial_joint_loss(output, record, original['query_loss'], edge_weight=recipe['edge_weight'], match_tolerance=2.)
                active = bool(edge.supervised_crops)
                before_graph = control.graph_state(model, optimizer) if not active else None
                attempt = {'fork_update': q, 'stage_update': 1000 + q, 'crop': config['training_crops'][(q - 1) % 4],
                    'incoming': incoming, 'edge_active': active, 'optimizer_step_returned': False}
                persist = lambda: write(directory / 'active_update.json', attempt)
                persist(); fit.apply_update(model, optimizer, total, active, attempt, persist, recipe)
                report['proposal_updates'] = q; report['graph_updates'] += int(active)
                validate_state(model, optimizer, q, report['graph_updates'], recipe)
                if not active and not control.equal(before_graph, control.graph_state(model, optimizer)):
                    raise ValueError('Inactive GAT state changed')
                attempt['cursors_after'] = fit.timing.stability.cursors(model, optimizer); persist()
                compact = {k: v for k, v in attempt.items() if k != 'incoming'}
                compact['incoming'] = {k: v for k, v in incoming.items() if not k.startswith('matched_') or k in ('matched_distance_mean', 'matched_distance_max')}
                if timing_reference and q <= 8:
                    control.compare_values(compact, timing_reference['updates'][q - 1])
                report['updates'].append(compact)
                with (directory / 'updates.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(compact, allow_nan=False) + '\n')
                seconds.append(time.perf_counter() - one)
                stopped = False
                if fit.is_probe_step(q):
                    rows = control.probe(model, records, original, recipe, device)
                    report['probes'].append({'new_updates': q, 'historical_stage': 1000 + q, 'crops': rows})
                    stopped = fit.geometry_stop(report['probes'])
                    print(f"PROBE stage={1000 + q} trusted={[r['edge_supervision']['trusted_matches'] for r in rows]}", flush=True)
                if q == 8:
                    report['terminal_state_fingerprint'] = fingerprint(model, optimizer)
                    if timing_reference:
                        if report['terminal_state_fingerprint'] != timing_reference['terminal_state_fingerprint']:
                            raise ValueError('First eight updates differ from timing model/Adam/RNG')
                        report['timing_prefix_verified'] = True
                if q in config['snapshots'] or stopped:
                    if args.action != 'timing':
                        save_state(model, optimizer, report, config, recipe, provenance, directory, 'geometry_stop' if stopped else 'fit')
                    report['evaluations'].append(evaluate(model, records, oracle, original, recipe, device, q, directory, baseline))
                if args.action == 'timing' or q % 50 == 0:
                    print(f"UPDATE new={q}/{cap} loss={incoming['loss']:.6f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
                if fit.is_probe_step(q) or q % 50 == 0:
                    write(directory / 'report.json', report)
                if stopped:
                    report['status'] = 'stopped_geometry'; break
            if args.action == 'timing':
                path = directory / 'temporary_timing.pt'; before = fingerprint(model, optimizer)
                signature = runner.prediction_signature(model, records, original, device)
                try:
                    torch.save(checkpoint(model, optimizer, report, config, recipe, provenance, 'timing'), path)
                    optimizer, _, _ = restore(torch.load(path, map_location='cpu', weights_only=True), model, config, recipe, provenance, resume=False)
                    if fingerprint(model, optimizer) != before:
                        raise ValueError('Timing state reload differs')
                    runner.require_identical_predictions(signature, runner.prediction_signature(model, records, original, device))
                finally:
                    path.unlink(missing_ok=True)
                report.update(timing_gate=report['proposal_updates'] == 8, reload_prediction_delta=0.,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    estimated_update_seconds=float(np.mean(seconds[4:])) * 3000)
            else:
                report['terminal_stable_tail'] = fit.stable_tail(report, base.read(fit.CONFIG))
                final = report['evaluations'][-1]
                report['automatic_gate'] = final['automatic_gate']
                report['learnability_gate'] = report['proposal_updates'] == 3000 and final['automatic_gate'] and report['terminal_stable_tail']
                report['paired_geometry_noninferior'] = final['paired_geometry_noninferior']
                report['paired_improvement_gate'] = bool(report['learnability_gate'] and final['paired_geometry_noninferior']
                    and any(r['absolute_bad_lengths'][k] < r['baseline_terminal_absolute_bad_lengths'][k] - 1e-9 for r in final['crops'] for k in ('extra', 'missed')))
                signature = runner.prediction_signature(model, records, original, device); before = fingerprint(model, optimizer)
                optimizer, _, _ = restore(torch.load(report['last_checkpoint'], map_location='cpu', weights_only=True), model, config, recipe, provenance, resume=False)
                if fingerprint(model, optimizer) != before:
                    raise ValueError('Terminal model/Adam/RNG reload differs')
                runner.require_identical_predictions(signature, runner.prediction_signature(model, records, original, device))
                report['reload_prediction_delta'] = 0.
        base.verify_hashes(protected); base.verify_hashes(sources)
        if report['status'] == 'running':
            report['status'] = 'completed'
        report.update(elapsed_seconds=time.perf_counter() - started, sources_unchanged=True,
            next_step='Review execution, training learnability and paired geometry separately. No automatic extension, threshold or loss change.')
        write(directory / 'report.json', report)
        print(f"FINISHED {args.action} status={report['status']} learnability={report.get('learnability_gate')} paired_improvement={report.get('paired_improvement_gate')} report={directory / 'report.json'}", flush=True)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write(directory / 'report.json', report); (directory / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('check', 'timing', 'train', 'resume'), required=True)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    run(parser.parse_args())
