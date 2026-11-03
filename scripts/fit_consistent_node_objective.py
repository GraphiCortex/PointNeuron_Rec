"""Manual paired five-epoch continuation; immutable original study and objective.

Each arm forks the same terminal 2048 model/Adam/RNG state, then receives 2300
updates. Only the objective differs. Full terminal training/development grids
are evaluated. Recovery restores a complete epoch state into a fresh output.
"""
import argparse
import io
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import check_consistent_node_objective as candidate
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.training.process_control import launch_worker

ROOT = candidate.ROOT
CONFIG = ROOT/'configs/consistent_node_fit.json'
FILES = ('scripts/fit_consistent_node_objective.py', 'scripts/run_consistent_node_fit.ps1',
         'tests/test_consistent_node_fit.py')


def make_schedule(data):
    ids, original = data.schedule()
    schedule = original[:2300].copy()
    eligible = {(i, tile) for i, cid in enumerate(ids) for (case, tile), b in data.boxes.items()
                if case == cid and b['target']['gradient_training_allowed']}
    if len(ids) != 33 or len(eligible) != 460 or schedule.shape != (2300, 3):
        raise ValueError('Expected 33 training neurons, 460 supervised boxes and 2300 new updates')
    for epoch in range(5):
        rows = schedule[epoch*460:(epoch+1)*460]
        if not np.all(rows[:, 0] == epoch) or set(map(tuple, rows[:, 1:])) != eligible:
            raise ValueError('Every continuation epoch must expose every supervised box exactly once')
    return ids, schedule


def load_control(path):
    c = json.loads(Path(path).read_text())
    if (c['schema_version'] != 1 or c['methods'] != ['baseline', 'consistent']
            or c['new_epochs'] != 5 or c['updates_per_epoch'] != 460 or c['new_updates'] != 2300
            or c['origin_updates'] != 9200 or c['budget'] != 2048
            or c['worker_timeout_seconds'] != 14400 or c['automatic_extension'] is not False
            or set(c['source_sha256']) != set(FILES)):
        raise ValueError('Only the agreed five-epoch paired continuation is supported')
    for name, sha in c['source_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError(f'Continuation source changed: {name}')
    if digest(ROOT/c['objective_config']) != c['objective_config_sha256']:
        raise ValueError('Objective settings changed since GPU timing')
    oc, recipe, data, payload, panel = candidate.load_control(ROOT/c['objective_config'])
    candidate.require_cpu(oc)
    gpu = candidate.common.checked_json(c['timing_report'], c['timing_report_sha256'])
    if (gpu['status'] != 'passed' or gpu['config_fingerprint'] != fingerprint(oc)
            or [a['method'] for a in gpu['arms']] != c['methods']
            or any(a['status'] != 'passed' or a['config_fingerprint'] != fingerprint(oc)
                   or not a['next_update_model_adam_rng_exact'] or a['throwaway_updates'] != 8
                   or a['trained_checkpoint_retained'] for a in gpu['arms'])
            or gpu['arms'][0]['initial_state'] != gpu['arms'][1]['initial_state']):
        raise ValueError('Passed matched GPU timing and exact reload are required')
    old, _, _, profiles, _ = candidate.baseline.load_control(ROOT/oc['baseline_config'])
    for package, version in old['runtime_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Historical fit runtime changed: {package}')
    ids, schedule = make_schedule(data)
    if candidate.common.state_digest(payload['model']) != gpu['arms'][0]['initial_state']['model']:
        raise ValueError('GPU timing and continuation origins differ')
    return c, oc, recipe, data, profiles, payload, ids, schedule, gpu


def bindings(c, oc, method):
    if method not in c['methods']: raise ValueError('Unknown continuation method')
    return {'maximum_updates': 2300, 'origin_updates': 9200, 'method': method,
            'config_fingerprint': fingerprint(c), 'objective_config_fingerprint': fingerprint(oc),
            'origin_checkpoint_sha256': oc['checkpoint_sha256']}


def validate_adam(payload, stage):
    import torch
    opt = payload['optimizer']; params = [p for g in opt['param_groups'] for p in g['params']]
    clocks = opt['state']
    if (set(clocks) != set(params) or any(float(v['step']) != 9200+stage
            or not all(bool(torch.isfinite(v[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))
            for v in clocks.values())):
        raise ValueError('Complete finite saved Adam moments and origin-plus-stage clocks required')


def resume_payload(path, expected):
    import torch
    from pn2.training import node_fit_state as state
    path = Path(path); marker = json.loads(path.with_suffix('.complete.json').read_text())
    if marker['checkpoint'] != path.name or marker['sha256'] != digest(path):
        raise ValueError('Incomplete or changed continuation checkpoint')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    stage = payload['step']
    if (payload.get('schema') != state.SCHEMA or payload['bindings'] != expected
            or stage != marker['step'] or type(stage) is not int
            or not 0 <= stage <= 2300 or stage % 460):
        raise ValueError('Wrong continuation method/config or incomplete epoch cursor')
    validate_adam(payload, stage)
    return payload


def signature(model, opt):
    return {'model': candidate.common.state_digest(model.state_dict()),
            'adam': candidate.common.nested_digest(opt.state_dict()),
            'rng': candidate.common.nested_digest(candidate.common.rng_state())}


def count_strata(rows):
    result = []
    for low, high in [(1, 16), (17, 64), (65, 128), (129, 512)]:
        group = [r for r in rows if low <= r['target_count'] <= high]
        reference = sum(r['target_count'] for r in group)
        predicted = sum(r['accepted_count'] for r in group)
        matched = sum(r['tolerances']['2.0']['nodes']['matched'] for r in group)
        result.append({'target_count_range': [low, high], 'boxes': len(group),
            'reference_nodes': reference, 'accepted_nodes': predicted, 'matched_nodes_2': matched,
            'pooled_precision_2': matched/predicted if predicted else 0.,
            'pooled_recall_2': matched/reference if reference else None})
    return result


def measures(evaluation, partition):
    summary = evaluation['summary']; neurons = summary['neurons']
    key = 'reference_present_only' if partition == 'training' else 'full_grid'
    values = [n['tolerances']['2.0'] for n in neurons]
    return {'precision': float(np.mean([x[key]['precision'] for x in values])),
            'recall': float(np.mean([x[key]['recall'] for x in values])),
            'f1': float(np.mean([x[key]['f1'] for x in values])),
            'curve_support': float(np.mean([x['curve_to_accepted_node_support'] for x in values]))}


def comparison(arms, origin):
    if [a['method'] for a in arms] != ['baseline', 'consistent']:
        raise ValueError('A paired verdict requires both methods in fixed order')
    result = {'origin_epoch20': origin, 'terminal': {}, 'consistent_minus_baseline': {},
              'terminal_minus_origin': {}}
    for partition in ('training', 'development'):
        result['terminal'][partition] = {a['method']: measures(a['evaluations'][partition], partition) for a in arms}
        a, b = (result['terminal'][partition][m] for m in ('baseline', 'consistent'))
        result['consistent_minus_baseline'][partition] = {k: b[k]-a[k] for k in a}
        result['terminal_minus_origin'][partition] = {
            m: {k: v-origin[partition][k] for k, v in values.items()}
            for m, values in result['terminal'][partition].items()}
    train = result['consistent_minus_baseline']['training']
    dev = result['consistent_minus_baseline']['development']
    result['prospective_candidate_screening'] = {
        'training_precision_and_recall_improve': train['precision'] > 0 and train['recall'] > 0,
        'development_f1_improves_without_recall_loss': dev['f1'] > 0 and dev['recall'] >= 0,
        'development_curve_support_does_not_decline': dev['curve_support'] >= 0,
        'meaning': 'Warm-start pilot screening only; no biological, convergence or SWC certification'}
    result['candidate_screening_passed'] = all(v for k, v in result['prospective_candidate_screening'].items()
                                                if k != 'meaning')
    return result


def evaluate_terminal(model, data, profiles, method, folder):
    from pn2.training.node_fit_eval import evaluate
    result = {}
    for partition, expected_count in [('development', 415), ('training', 1964)]:
        ids = sorted(cid for cid, c in data.cases.items() if c['partition'] == partition)
        if sum(len(data.cases[cid]['boxes']) for cid in ids) != expected_count:
            raise ValueError('Full-grid evaluation denominator changed')
        print(f'EVALUATION method={method} partition={partition} expected_boxes={expected_count}', flush=True)
        evidence = evaluate(model, data, profiles, 2048, 'cuda:0', ids,
                            folder/f'eval_stage002300_{partition}', 2300)
        raw = json.loads(Path(evidence['folder']).joinpath('metrics.json').read_text())
        evidence['count_strata'] = count_strata(raw['boxes'])
        result[partition] = evidence
    return result


def worker(context, args):
    import torch
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import training_batch
    c, oc, recipe, data, profiles, payload, ids, schedule, gpu = context
    method = args.worker; folder = Path(args.output); began = time.perf_counter()
    expected = bindings(c, oc, method); start = 0
    if args.checkpoint:
        payload = resume_payload(args.checkpoint, expected); start = payload['step']
    candidate.common.runtime(recipe, 'cuda:0')
    model, opt = candidate.build(recipe, payload, 'cuda:0')
    validate_adam({'optimizer': opt.state_dict()}, start)
    initial = signature(model, opt)
    if start == 0 and initial != gpu['arms'][0]['initial_state']:
        raise ValueError('Forked model/Adam/RNG do not match passed GPU timing')
    write_once(folder/'launch.json', {'method': method, 'bindings': expected,
        'initial_state': initial, 'starting_stage': start, 'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'dataset': '33 training / seven exposed development neurons; no reserved cases',
        'optimizer': recipe['optimizer'], 'cutoff': .5, 'training_box_exposures': 5,
        'maximum_new_updates': 2300, 'automatic_extension': False})
    if start == 0: state.save(folder/'checkpoint_stage000000.pt', model, opt, expected, 0)
    with (folder/'updates.jsonl').open('x', encoding='utf-8') as journal:
        for index in range(start, 2300):
            epoch, case_index, tile = map(int, schedule[index]); cid = ids[case_index]
            record = training_batch(data.training(cid, tile, 2048), 'cuda:0')
            metrics = candidate.step(model, opt, record, recipe, method)
            stage = index+1
            journal.write(json.dumps({'stage': stage, 'history_updates': 9200+stage,
                'new_epoch': epoch+1, 'case_id': cid, 'tile_id': tile, **metrics})+'\n');journal.flush()
            if stage == start+1 or stage % 50 == 0:
                print(f'UPDATE method={method} stage={stage}/2300 new_epoch={epoch+1}/5 '
                      f'loss={metrics["loss"]:.5f} grad={metrics["gradient_norm_before_clip"]:.2f} '
                      f'elapsed={time.perf_counter()-began:.1f}s', flush=True)
            if stage % 460 == 0:
                checkpoint = folder/f'checkpoint_stage{stage:06d}.pt'
                state.save(checkpoint, model, opt, expected, stage)
                print(f'SAVED complete stage checkpoint: {checkpoint}', flush=True)
    # A complete-stage recovery does zero SGD and only regenerates evaluation.
    evaluations = evaluate_terminal(model, data, profiles, method, folder)
    result = {'status': 'completed', 'method': method, 'bindings': expected,
        'stage': 2300, 'history_updates': 11500, 'new_updates_this_run': 2300-start,
        'starting_stage': start, 'initial_state': initial, 'evaluations': evaluations,
        'seconds': time.perf_counter()-began, 'development_gradient_updates': 0,
        'absolute_screening': candidate.baseline.screening(evaluations['training'], evaluations['development']),
        'automatic_extension': False, 'final_graph_or_swc_evaluated': False}
    write_once(folder/'worker_report.json', result)
    print(f'COMPLETED method={method} stage=2300; '
          f'train={measures(evaluations["training"],"training")} '
          f'development={measures(evaluations["development"],"development")}', flush=True)
    return result


def require_cpu(c):
    proof = json.loads((ROOT/c['cpu_report']).read_text())
    if (proof['status'] != 'passed' or proof['config_fingerprint'] != fingerprint(c)
            or proof['cuda_initialized'] or not proof['next_update_reload_exact_both_methods']
            or proof['supervised_boxes_per_epoch'] != 460 or proof['new_updates'] != 2300):
        raise ValueError('Current passed continuation CPU proof required')


def cpucheck(context, folder):
    import torch
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import training_batch
    c, oc, recipe, data, _, payload, ids, schedule, _ = context
    candidate.common.runtime(recipe, 'cpu'); rows = []
    _, ci, tile = map(int, schedule[0]);record = training_batch(data.training(ids[ci], tile, 2048), 'cpu')
    for method in c['methods']:
        model, opt = candidate.build(recipe, payload, 'cpu')
        first = candidate.step(model, opt, record, recipe, method)
        expected_bindings = bindings(c, oc, method)
        path = folder/f'{method}_probe.pt'
        state.save(path, model, opt, expected_bindings, 460)
        # Above one-update state is deliberately NOT a valid epoch recovery:
        # stage 460 needs Adam clock 9660. Verify that the clock guard catches it.
        try: resume_payload(path, expected_bindings)
        except ValueError as exc:
            if 'Adam' not in str(exc): raise
        else: raise ValueError('Invalid epoch/Adam clock was accepted')
        # In-memory replay checks actual next update; no trained state retained.
        buffer = io.BytesIO();torch.save(state.capture(model, opt, expected_bindings, 1), buffer)
        loss_expected = candidate.step(model, opt, record, recipe, method);sig_expected = signature(model, opt)
        buffer.seek(0);saved = torch.load(buffer, map_location='cpu', weights_only=False)
        state.restore(saved, model, opt, expected_bindings)
        loss_actual = candidate.step(model, opt, record, recipe, method);sig_actual = signature(model, opt)
        if loss_actual != loss_expected or sig_actual != sig_expected:
            raise ValueError('Full-model CPU next update replay failed')
        validate_adam({'optimizer': opt.state_dict()}, 2)
        path.unlink();path.with_suffix('.complete.json').unlink()
        rows.append({'method': method, 'first_update': first, 'next_update_exact': True,
                     'real_training_box_throwaway_optimizer_calls': 3})
    if torch.cuda.is_initialized(): raise RuntimeError('CPU check initialized CUDA')
    proof = {'status': 'passed', 'config_fingerprint': fingerprint(c), 'source_sha256': c['source_sha256'],
        'new_updates': 2300, 'supervised_boxes_per_epoch': 460, 'training_neurons': len(ids),
        'next_update_reload_exact_both_methods': True, 'cuda_initialized': False,
        'trained_checkpoint_retained': False, 'methods': rows,
        'quality_improvement_established': False}
    write_once(folder/'report.json', proof)
    return proof


def controller(context, args):
    c, oc, _, _, _, _, _, _, _ = context
    require_cpu(c); folder = Path(args.output)
    methods = c['methods'] if args.action == 'train' else [args.method]
    arms = []
    for method in methods:
        target = folder/method; target.mkdir()
        command = [sys.executable, '-B', __file__, '--action', args.action, '--config', str(args.config),
                   '--output', str(target), '--worker', method]
        if args.checkpoint: command += ['--checkpoint', str(args.checkpoint)]
        code, status = launch_worker(command, target, c['worker_timeout_seconds'])
        if code != 0 or status != 'completed':
            write_once(folder/'report.json', {'status': status, 'failed_method': method,
                'completed_arms': arms, 'automatic_retry': False, 'automatic_extension': False})
            return 1
        report = json.loads((target/'worker_report.json').read_text())
        if report['status'] != 'completed' or report['stage'] != 2300 or report['bindings'] != bindings(c, oc, method):
            raise ValueError('Worker report is incomplete or bound to a different experiment')
        arms.append(report)
    result = {'status': 'completed', 'config_fingerprint': fingerprint(c), 'arms': arms,
              'automatic_extension': False, 'automatic_retry': False, 'reserved_files_used': False}
    if args.action == 'train':
        if arms[0]['initial_state'] != arms[1]['initial_state']:
            raise ValueError('Paired arms did not start from identical full states')
        prior = json.loads((ROOT/oc['completed_fit']).read_text())
        old = next(a for a in prior['arms'] if a['budget'] == 2048)
        origin = {partition: measures(next(e for e in old['evaluations'] if e['step'] == 9200
                    and partition in e['summary']['partitions']), partition)
                    for partition in ('training', 'development')}
        result['comparison'] = comparison(arms, origin)
    else:
        result['comparison'] = None  # One recovered arm cannot certify a paired verdict.
    write_once(folder/'report.json', result)
    print(f'FINISHED {args.action} status=completed; report={folder/"report.json"}', flush=True)
    return 0


def main(args):
    if args.action == 'cpucheck' and args.worker:
        raise ValueError('CPU checks cannot launch a GPU worker')
    context = load_control(args.config)
    folder = Path(args.output).resolve();args.output = folder
    if not folder.is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Output must stay under PN2/tmp')
    if args.worker:
        require_cpu(context[0]);worker(context, args);return 0
    folder.mkdir(parents=True, exist_ok=False)
    if args.action == 'cpucheck':
        result = cpucheck(context, folder);print(f'CPU proof {result["status"]}: {folder/"report.json"}');return 0
    return controller(context, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['cpucheck', 'train', 'resume'], required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['baseline', 'consistent'])
    parser.add_argument('--method', choices=['baseline', 'consistent'])
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.action == 'cpucheck' and args.worker:
        parser.error('CPU checks cannot launch a GPU worker')
    if args.method and args.worker and args.method != args.worker:
        parser.error('Recovery method and worker must agree')
    if args.action == 'resume' and (not args.checkpoint or not (args.method or args.worker)):
        parser.error('Recovery requires explicit method and complete epoch checkpoint')
    if args.action != 'resume' and (args.checkpoint or args.method):
        parser.error('Only explicit recovery accepts method/checkpoint')
    try: sys.exit(main(args))
    except Exception:
        traceback.print_exc();sys.exit(1)
