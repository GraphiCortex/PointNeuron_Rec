"""CPU audit and manual GPU timing for the candidate objective; no fit action.

The historical fit, clouds, targets, model and optimizer recipe remain frozen.
GPU timing forks the completed 2048 model AND its Adam/RNG state for each arm.
Only the objective changes. CPU checks never restore CUDA RNG or initialize it.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

import node_budget_fit as baseline
import sampler_node_common as common
import geometric_query_experiment as geometric
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.training.process_control import launch_worker

ROOT = common.ROOT
CONFIG = ROOT/'configs/consistent_node_objective.json'
FILES = ('src/pn2/models/consistent_node_objective.py',
         'scripts/check_consistent_node_objective.py',
         'scripts/run_consistent_node_objective.ps1',
         'tests/test_consistent_node_objective.py')


def load_control(path):
    c = json.loads(Path(path).read_text())
    if (c['schema_version'] != 1 or c['budget'] != 2048 or c['timing_updates'] != 8
            or c['worker_timeout_seconds'] != 600 or c['methods'] != ['baseline', 'consistent']
            or c['full_training_authorized'] is not False or set(c['source_sha256']) != set(FILES)):
        raise ValueError('Expected CPU audit / eight-update timing only')
    for name, sha in c['source_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError(f'Candidate source changed: {name}')
    if digest(ROOT/c['baseline_config']) != c['baseline_config_sha256']:
        raise ValueError('Frozen historical configuration changed')
    old, recipe, data, _, _ = baseline.load_control(ROOT/c['baseline_config'])
    report = common.checked_json(c['completed_fit'], c['completed_fit_sha256'])
    if report['status'] != 'completed' or any(a['total_updates'] != 9200 for a in report['arms']):
        raise ValueError('Expected the completed paired study')
    expected = baseline.bindings(old, recipe, data, 2048)
    payload, path = baseline.resume_payload(ROOT/c['checkpoint'], expected)
    if payload['step'] != 9200 or digest(path) != c['checkpoint_sha256']:
        raise ValueError('Expected the completed 2048 checkpoint and all Adam clocks')
    panel = []
    # Fixed training-only target-count strata. No error-based case selection.
    for aim, low, high in [(8, 1, 16), (32, 17, 64), (96, 65, 128), (180, 129, 512)]:
        eligible = [(abs(b['target']['count']-aim), cid, tile)
                    for (cid, tile), b in data.boxes.items()
                    if data.cases[cid]['partition'] == 'training'
                    and low <= b['target']['count'] <= high
                    and b['target']['gradient_training_allowed']]
        if not eligible: raise ValueError('Missing training target-count stratum')
        _, cid, tile = min(eligible)
        panel.append((cid, tile))
    if [list(x) for x in panel] != c['training_panel']:
        raise ValueError('Training-only diagnostic panel changed')
    return c, recipe, data, payload, panel


def objective(method):
    if method == 'baseline':
        from pn2.models.node_objective import node_set_loss
        return node_set_loss
    if method == 'consistent':
        from pn2.models.consistent_node_objective import consistent_node_set_loss
        return consistent_node_set_loss
    raise ValueError('Unknown objective')


def build(recipe, payload, device):
    import torch
    model = geometric.make_model(recipe, 'cpu')
    model.load_state_dict(payload['model'], strict=True)
    model.to(device)
    opt = common.make_optimizer(model, recipe)
    opt.load_state_dict(copy.deepcopy(payload['optimizer']))
    # Adam state follows the parameters' device; weights-only is never a resume.
    if any(v['exp_avg'].device != p.device for p, v in opt.state.items()):
        raise ValueError('Optimizer device mismatch')
    model.train(payload['model_training'])
    if str(device).startswith('cuda'):
        common.restore_rng(payload['rng'])
    else:
        rng = dict(payload['rng'], torch_cuda=None)
        common.restore_rng(rng)
        if torch.cuda.is_initialized(): raise RuntimeError('CPU audit initialized CUDA')
    return model, opt


def group_norms(model, grads):
    result = {}
    for (name, _), grad in zip(model.named_parameters(), grads):
        if grad is None or not bool(grad.isfinite().all()): raise FloatingPointError('Invalid gradient')
        key = 'encoder' if name.startswith('encoder.') else 'head'
        result[key] = result.get(key, 0.)+float(grad.double().square().sum())
    result['all'] = sum(result.values())
    return {k: v**.5 for k, v in result.items()}


def frozen_audit(c, data):
    import torch
    from types import SimpleNamespace
    folder = (ROOT/c['completed_fit']).parent/'points2048/eval_step009200_training'
    study = json.loads((ROOT/c['completed_fit']).read_text())
    arm = next(a for a in study['arms'] if a['budget'] == 2048)
    evidence = next(e for e in arm['evaluations'] if e['step'] == 9200
                    and 'training' in e['summary']['partitions'])
    if (digest(folder/'metrics.json') != evidence['metrics_sha256']
            or digest(folder/'predictions.npz') != evidence['predictions_sha256']):
        raise ValueError('Saved training predictions/metrics changed')
    saved = json.loads((folder/'metrics.json').read_text()); groups = []
    for lo, hi in [(1, 16), (17, 64), (65, 128), (129, 512)]:
        rows = [x for x in saved['boxes'] if lo <= x['target_count'] <= hi]
        groups.append({'target_count_range': [lo, hi], 'boxes': len(rows),
            'mean_targets': float(np.mean([x['target_count'] for x in rows])),
            'mean_accepted': float(np.mean([x['accepted_count'] for x in rows])),
            'pooled_recall_2': sum(x['tolerances']['2.0']['nodes']['matched'] for x in rows)
                              /sum(x['target_count'] for x in rows)})
    # Replay matching only on all positive training boxes; no classifier fit.
    changes = []; losses = []; negative_ratios = []
    with np.load(folder/'predictions.npz', allow_pickle=False) as packets:
        for i, row in enumerate(saved['boxes']):
            if not row['target_count']: continue
            target = data.reference(row['case_id'], row['tile_id'])['target_xyz']
            scores = packets[f'box{i}_scores'].astype(np.float32)
            scores = np.clip(scores, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
            # Only softmax probabilities were saved. Reconstructed logits are
            # probability-equivalent up to clipping, not bitwise original logits.
            logits = np.log(np.column_stack((1-scores, scores)))
            out = SimpleNamespace(center_proposals=torch.from_numpy(packets[f'box{i}_centers'])[None],
                                  objectness_logits=torch.from_numpy(logits)[None])
            targets = torch.from_numpy(target.astype(np.float32))
            a = objective('baseline')(out, targets)
            b = objective('consistent')(out, targets)
            first = set(zip(a.prediction_indices, a.target_indices))
            second = set(zip(b.prediction_indices, b.target_indices))
            changes.append(1-len(first & second)/len(first))
            losses.append(float(b.total))
            m, q = len(target), len(scores)
            negative_ratios.append(m/(q-m))
    return {'count_strata': groups, 'positive_training_boxes_replayed': len(changes),
        'mean_fraction_assignments_changed': float(np.mean(changes)),
        'baseline_per_negative_to_per_positive_weight_range': [min(negative_ratios), max(negative_ratios)],
        'candidate_per_negative_to_per_positive_weight': 1.,
        'all_candidate_losses_finite': bool(np.isfinite(losses).all()),
        'prediction_geometry_or_scores_changed': False,
        'warning': 'Matching/objective replay only; no evidence of learned improvement'}


def target_phase_counterexample():
    """Identical local curves can inherit different full-chain sampling phases.

    Constructed geometry only; no claim about its prevalence in the cohort.
    The active target builder and cached targets are not modified.
    """
    from pn2.evaluation.swc import Morphology
    from pn2.supervision.curve_nodes import rectangular_curve_nodes
    def targets(start):
        tree = Morphology(None, np.array([1, 2]), np.array([3, 3]),
            np.array([[start, 4., 4.], [40., 4., 4.]]), np.ones(2), np.array([-1, 0]))
        return rectangular_curve_nodes(tree, [0, 0, 0], [17, 9, 9])
    a, b = targets(-8.), targets(-10.)
    same_curve = (len(a.reference_runs) == len(b.reference_runs)
                  and all(np.allclose(x, y) for x, y in zip(a.reference_runs, b.reference_runs)))
    same_targets = a.xyz.shape == b.xyz.shape and np.allclose(a.xyz, b.xyz)
    if not same_curve or same_targets: raise ValueError('Target ambiguity fixture failed')
    return {'same_local_curve': True, 'same_target_nodes': False,
            'target_x_first': a.xyz[:, 0].tolist(), 'target_x_second': b.xyz[:, 0].tolist(),
            'dataset_failure_attribution': 'Untested; constructed counterexample only',
            'active_targets_changed': False}


def cpucheck(c, recipe, data, payload, panel, folder):
    import torch
    from pn2.training.node_runtime import training_batch
    common.runtime(recipe, 'cpu')
    began = time.perf_counter()
    result = frozen_audit(c, data)
    result['target_phase_counterexample'] = target_phase_counterexample()
    model, _ = build(recipe, payload, 'cpu')
    state_before = common.state_digest(model.state_dict())
    rows = []
    for cid, tile in panel:
        record = data.training(cid, tile, 2048);batch = training_batch(record, 'cpu')
        model.train(); out = model(batch)
        model.eval()
        with torch.no_grad(): ordinary = model(batch)
        if not (torch.equal(out.center_proposals, ordinary.center_proposals)
                and torch.equal(out.objectness_logits, ordinary.objectness_logits)):
            raise ValueError('Train/eval consistency failed')
        row = {'case_id': cid, 'tile_id': tile, 'targets': len(batch['target']), 'methods': {}}
        for method in c['methods']:
            loss = objective(method)(out, batch['target'], **recipe['loss'])
            grads = torch.autograd.grad(loss.total, tuple(model.parameters()), retain_graph=True)
            row['methods'][method] = {'loss': float(loss.total.detach()),
                'gradient_norms_before_clip': group_norms(model, grads),
                'matched_targets': len(loss.prediction_indices)}
        rows.append(row)
        print(f'CPU targets={row["targets"]} gradient norms='
              f'{ {k: round(v["gradient_norms_before_clip"]["all"], 4) for k,v in row["methods"].items()} }', flush=True)
        del out, ordinary, grads
    if state_before != common.state_digest(model.state_dict()):
        raise ValueError('CPU gradient diagnostic mutated model state')
    if torch.cuda.is_initialized(): raise RuntimeError('CPU audit initialized CUDA')
    result.update(status='passed', config_fingerprint=fingerprint(c), source_sha256=c['source_sha256'],
        checkpoint_sha256=c['checkpoint_sha256'], full_model_gradient_panel=rows,
        model_state_unchanged=True, optimizer_updates=0, cuda_initialized=False,
        seconds=time.perf_counter()-began,
        learnability_or_reconstruction_improvement_established=False)
    write_once(folder/'report.json', result)
    return result


def require_cpu(c):
    report = json.loads((ROOT/c['cpu_report']).read_text())
    if (report['status'] != 'passed' or report['config_fingerprint'] != fingerprint(c)
            or report['source_sha256'] != c['source_sha256'] or report['optimizer_updates'] != 0
            or report['cuda_initialized'] or not report['model_state_unchanged']):
        raise ValueError('Current passed CPU audit required before GPU timing')


def step(model, opt, record, recipe, method):
    import torch
    model.train();opt.zero_grad(set_to_none=True)
    loss = objective(method)(model(record), record['target'], **recipe['loss'])
    loss.total.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True)
    opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameters')
    return {'loss': float(loss.total.detach()), 'gradient_norm_before_clip': float(norm),
            'matched_targets': len(loss.prediction_indices)}


def timing(c, recipe, data, payload, panel, folder, method):
    import torch
    from pn2.training.node_runtime import training_batch
    from pn2.training import node_fit_state as state
    common.runtime(recipe, 'cuda:0');model, opt = build(recipe, payload, 'cuda:0')
    initial = {'model': common.state_digest(model.state_dict()),
               'adam': common.nested_digest(opt.state_dict()), 'rng': common.nested_digest(common.rng_state())}
    rows = [];torch.cuda.reset_peak_memory_stats()
    for i in range(c['timing_updates']):
        cid, tile = panel[i % len(panel)]
        record = training_batch(data.training(cid, tile, 2048), 'cuda:0')
        torch.cuda.synchronize(); began = time.perf_counter()
        row = step(model, opt, record, recipe, method)
        torch.cuda.synchronize();row.update(update=i+1, seconds=time.perf_counter()-began)
        rows.append(row);print(f'THROWAWAY {method} {i+1}/8 {row}', flush=True)
    # Exact next update after a full in-memory model/Adam/RNG serialization.
    bindings = {'maximum_updates': 9, 'config': fingerprint(c), 'method': method}
    memory = io.BytesIO();torch.save(state.capture(model, opt, bindings, 8), memory)
    expected_loss = step(model, opt, record, recipe, method)
    expected = (common.state_digest(model.state_dict()), common.nested_digest(opt.state_dict()),
                common.nested_digest(common.rng_state()))
    memory.seek(0);restored = torch.load(memory, map_location='cpu', weights_only=False)
    state.restore(restored, model, opt, bindings)
    actual_loss = step(model, opt, record, recipe, method)
    actual = (common.state_digest(model.state_dict()), common.nested_digest(opt.state_dict()),
              common.nested_digest(common.rng_state()))
    if actual != expected or actual_loss != expected_loss: raise ValueError('Next update reload mismatch')
    result = {'status': 'passed', 'method': method, 'config_fingerprint': fingerprint(c),
        'initial_state': initial, 'throwaway_updates': 8, 'reload_probe_updates': 2,
        'next_update_model_adam_rng_exact': True, 'rows': rows,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'trained_checkpoint_retained': False, 'quality_improvement_established': False}
    write_once(folder/'report.json', result)
    return result


def main(args):
    c, recipe, data, payload, panel = load_control(args.config)
    folder = Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Outputs belong in PN2/tmp')
    folder.mkdir(parents=True, exist_ok=False)
    if args.action == 'cpucheck':
        result = cpucheck(c, recipe, data, payload, panel, folder)
    else:
        require_cpu(c)
        if args.worker:
            result = timing(c, recipe, data, payload, panel, folder, args.worker)
        else:
            del payload
            arms = []
            for method in c['methods']:
                worker = folder/(method+'_process');worker.mkdir()
                command = [sys.executable, '-B', __file__, '--action', 'timing', '--config', str(args.config),
                           '--output', str(folder/method), '--worker', method]
                code, status = launch_worker(command, worker, c['worker_timeout_seconds'])
                if status != 'completed' or code != 0:
                    raise RuntimeError(f'{method} timing {status}; preserve logs, no retry')
                arms.append(json.loads((folder/method/'report.json').read_text()))
            if arms[0]['initial_state'] != arms[1]['initial_state']:
                raise ValueError('Timing arms did not fork identical model/Adam/RNG states')
            result = {'status': 'passed', 'arms': arms, 'config_fingerprint': fingerprint(c),
                      'full_training_started': False, 'quality_improvement_established': False}
            write_once(folder/'report.json', result)
    print(f'COMPLETED {args.action}; status={result["status"]}; report={folder/"report.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['cpucheck', 'timing'], required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['baseline', 'consistent'])
    arguments = parser.parse_args()
    if arguments.action == 'cpucheck' and arguments.worker: parser.error('CPU checks have no GPU worker')
    main(arguments)
