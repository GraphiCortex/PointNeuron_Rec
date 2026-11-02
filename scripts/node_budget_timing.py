"""Prepare and time two input budgets in the unchanged geometric-query model.

Preparation is image-only CPU work using retained guides. Model CPU checks and
manual CUDA timing use separate runtimes. Timing has no training/resume action.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import sampler_node_common as common
import geometric_query_experiment as geometric
import groupnorm_node_experiment as gn
from sampler_node_experiment import launch_worker

ROOT = common.ROOT
CONFIG = ROOT / 'configs/node_budget_timing.json'
ORDER = [['fruitfly32', 27], ['fruitfly32', 20], ['frog22', 2], ['frog22', 5]]


def load_control(path):
    control = json.loads(Path(path).read_text())
    if (control['schema_version'] != 1 or control['budgets'] != [2048, 8192]
            or control['box_order'] != ORDER or control['updates_per_arm'] != 8
            or control['script_sha256'] != common.digest(__file__)
            or control['short_box_policy'] != 'min_budget_source_voxels_without_padding'):
        raise ValueError('Expected the declared two-budget execution diagnostic')
    if common.digest(ROOT / control['geometric_config']) != control['geometric_config_sha256']:
        raise ValueError('Geometric model configuration changed')
    recipe = geometric.load_recipe(ROOT / control['geometric_config'])
    if common.fingerprint(recipe) != control['model_recipe_fingerprint']:
        raise ValueError('Unchanged geometric model recipe required')
    geometric.require_cpu(recipe)
    return control, recipe


def identity(control, recipe):
    return {'control_fingerprint': common.fingerprint(control),
            'script_sha256': common.digest(__file__), 'model_identity': geometric.identity(recipe)}


def output_folder(path):
    folder = Path(path).resolve()
    if not folder.is_relative_to((ROOT / 'tmp').resolve()):
        raise ValueError('Generated outputs must stay under PN2/tmp')
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def point_count(budget, source_voxels):
    if type(budget) is not int or type(source_voxels) is not int or min(budget, source_voxels) < 256:
        raise ValueError('At least 256 actual source voxels required; no padding')
    return min(budget, source_voxels)


def prepare(control, recipe, output):
    # This path must run in the pinned Phase 1 runtime, not the Torch runtime.
    import node_cloud_cache as cache_driver
    from pn2.inputs.node_cache import verify_case, case_key, point_arrays
    from pn2.inputs.image import decode_image
    from pn2.inputs.disk_guide import read_block
    cache_control, cases = cache_driver.load_config(ROOT / control['cache_config'])
    proof = common.checked_json(control['completion_evidence'], control['completion_evidence_sha256'])
    if proof['status'] != 'passed' or proof['counts']['packets_verified'] != 2379:
        raise ValueError('Completed cohort review is required')
    manifest = common.checked_json(recipe['inputs'], recipe['inputs_sha256'])
    audit = common.checked_json(recipe['target_audit'], recipe['target_audit_sha256'])
    folder = output_folder(output); rows = []; receipts = {}; decoded = {}
    for alias, tile_id in ORDER:
        original = next(r for r in manifest['records'] if r['method'] == 'pn2_sato'
                        and r['alias'] == alias and r['tile_id'] == tile_id)
        case = next(c for c in cases if c.case_id == original['case_id'])
        if case.partition != 'training':
            raise ValueError('Only the four exposed training boxes enter timing')
        cache = ROOT / control['cache_root'] / case_key(case)
        if case.case_id not in receipts:
            receipts[case.case_id] = verify_case(cache, case, cache_driver.cache_identity(cache_control))
            decoded[case.case_id] = decode_image(case)
        receipt = receipts[case.case_id]; image = decoded[case.case_id]
        if any(image.provenance[k] != receipt['image'][k]
               for k in ('source_sha256', 'decoded_sha256', 'dimensions_xyzc')):
            raise ValueError('Original image and completed cache differ')
        row = receipt['records'][tile_id]
        start, stop = row['box']['start_xyz'], row['box']['stop_xyz']
        tile = tuple(slice(a, b) for a, b in zip(start[::-1], stop[::-1]))
        guide = read_block(cache / receipt['guide']['path'], tile)
        target = next(t for t in audit['target_records'] if t['case_id'] == case.case_id and t['tile_id'] == tile_id)
        target_path = (ROOT / recipe['target_audit']).parent / target['packet']
        if common.digest(target_path) != target['packet_sha256']:
            raise ValueError('Frozen training targets changed')
        with np.load(target_path, allow_pickle=False) as target_packet:
            xyz = target_packet['xyz'].copy()
        lower = None
        for budget in control['budgets']:
            count = point_count(budget, int(image.array_zyx[tile].size))
            settings = {**cache_control['settings'], 'points_per_box': count}
            arrays, meta = point_arrays(image.array_zyx, guide, tile, tile_id, settings, receipt['normalization'])
            if budget == 2048:
                with np.load(cache / row['packet'], allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or not all(np.array_equal(saved[k], v)
                            and saved[k].dtype == v.dtype for k, v in arrays.items()):
                        raise ValueError('2048 view does not reproduce the successful cache')
                lower = arrays['flat_indices']
            elif not np.isin(lower, arrays['flat_indices']).all():
                raise ValueError('Larger cloud must retain every baseline source location')
            name = f'{alias}_tile{tile_id}_points{budget}.npz'
            np.savez_compressed(folder / name, **arrays, target_xyz=xyz)
            rows.append({'alias': alias, 'case_id': case.case_id, 'partition': 'training',
                'tile_id': tile_id, 'budget': budget, 'actual_points': count,
                'targets': len(xyz), 'target_packet_sha256': target['packet_sha256'],
                'metadata': {**meta, 'alias': alias}, 'packet': name,
                'packet_sha256': common.digest(folder / name),
                'source_sha256': receipt['image']['source_sha256'], 'guide_sha256': receipt['guide']['sha256']})
            print(f'PREPARE {alias}/{tile_id} points={count} targets={len(xyz)} baseline_replay/nesting verified', flush=True)
    common.write(folder / 'manifest.json', {'status': 'prepared', **identity(control, recipe),
        'records': rows, 'filters_rerun': 0, 'references_opened': 0,
        'frozen_target_packets_read': 4, 'models_executed': 0, 'gpu_executed': False})


def load_records(control, recipe, budget):
    path = ROOT / control['packet_manifest']; manifest = json.loads(path.read_text())
    if manifest['status'] != 'prepared' or any(manifest.get(k) != v for k, v in identity(control, recipe).items()):
        raise ValueError('Input manifest is stale or incompatible')
    rows = [r for r in manifest['records'] if r['budget'] == budget]
    if [[r['alias'], r['tile_id']] for r in rows] != ORDER:
        raise ValueError('Timing inputs must keep the declared four-box order')
    records = []
    for row in rows:
        p = path.parent / row['packet']
        if row['partition'] != 'training' or common.digest(p) != row['packet_sha256']:
            raise ValueError('Training input identity changed')
        with np.load(p, allow_pickle=False) as data:
            packet = {k: data[k].copy() for k in data.files if k != 'target_xyz'}
            targets = data['target_xyz'].copy()
        if (len(packet['flat_indices']) != row['actual_points'] or len(targets) != row['targets']
                or len(np.unique(packet['flat_indices'])) != row['actual_points']
                or row['actual_points'] > budget or not np.isfinite(targets).all()):
            raise ValueError('Input counts, distinct observations or targets changed')
        records.append({'metadata': row['metadata'], 'packet': packet, 'target_xyz': targets})
    return records


def data_identity(control):
    return common.digest(ROOT / control['packet_manifest'])


def cpu_check(control, recipe, output):
    folder = output_folder(output); torch = common.runtime(recipe, 'cpu')
    from pn2.models.node_objective import node_set_loss
    model = geometric.make_model(recipe, 'cpu'); initial = common.state_digest(model.state_dict())
    checks = []
    for budget in control['budgets']:
        records = load_records(control, recipe, budget)
        record = common.tensors(records[0], torch, 'cpu')
        with torch.no_grad():
            model.train(); a = model(record)
            model.eval(); b = model(record)
        check = gn.prediction_consistency(a, b)
        if not check['passed'] or b.center_proposals.shape != (1, 512, 3):
            raise ValueError('Whole-model budget interface/mode check failed')
        model.train(); model.zero_grad(set_to_none=True)
        loss = node_set_loss(model(record), record['target'], **recipe['loss']); loss.total.backward()
        if not loss.supervised or any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
            raise ValueError('Budget interface has missing/nonfinite gradients')
        checks.append({'points': budget, 'queries': 512, 'consistency': check, 'gradients_finite': True})
        print(f'CPU BUDGET {budget}: complete model mode/shape/gradient checks pass', flush=True)
    # Synthetic only: image-derived short-count policy plus the cohort's 419-target stress count.
    short = geometric.synthetic_record(torch, recipe, point_count(8192, 7650), 'cpu')
    short['target'] = short['points'][0, :419, :3].clone()
    model.zero_grad(set_to_none=True); model.train()
    prediction = model(short)
    stress = node_set_loss(prediction, short['target'], **recipe['loss']); stress.total.backward()
    if (prediction.center_proposals.shape != (1, 512, 3) or len(stress.prediction_indices) != 419
            or any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters())):
        raise ValueError('7650-point / 419-target synthetic interface failed')
    if common.state_digest(model.state_dict()) != initial:
        raise ValueError('CPU forward/backward checks changed parameters or buffers')
    fixture = geometric.synthetic_record(torch, recipe, 256, 'cpu')
    optimizer = common.make_optimizer(model, recipe)
    common.update(model, optimizer, fixture, recipe)
    restored, _ = geometric.reload_probe(model, optimizer, recipe, 1, 'cpu')
    with torch.no_grad():
        model.eval(); restored.eval()
        reloaded = gn.prediction_consistency(model(fixture), restored(fixture))
    if not reloaded['passed']:
        raise ValueError('In-memory serialization reload failed')
    common.write(folder / 'report.json', {'status': 'passed', **identity(control, recipe),
        'packet_manifest_sha256': data_identity(control), 'budget_checks': checks,
        'short_synthetic_points': 7650, 'synthetic_target_stress': 419,
        'synthetic_optimizer_updates': 1, 'real_optimizer_updates': 0,
        'reload_passed': True, 'gpu_executed': False})


def require_cpu(control, recipe):
    report = json.loads((ROOT / control['cpu_preflight']).read_text())
    if (report.get('status') != 'passed' or not report.get('reload_passed')
            or report.get('packet_manifest_sha256') != data_identity(control)
            or [r['points'] for r in report['budget_checks']] != [2048, 8192]
            or report.get('short_synthetic_points') != 7650 or report.get('synthetic_target_stress') != 419
            or any(report.get(k) != v for k, v in identity(control, recipe).items())):
        raise ValueError('Current two-budget CPU verification is required')


def worker(control, recipe, args):
    require_cpu(control, recipe); torch = common.runtime(recipe, 'cuda:0')
    records = [common.tensors(r, torch, 'cuda:0') for r in load_records(control, recipe, args.budget)]
    model = geometric.make_model(recipe, 'cuda:0'); optimizer = common.make_optimizer(model, recipe)
    initial = common.state_digest(model.state_dict()); durations = []; eval_durations = []
    torch.cuda.reset_peak_memory_stats(); folder = Path(args.output)
    print(f'TIMING points={args.budget} device={torch.cuda.get_device_name(0)} fresh_init={initial} updates=8', flush=True)
    with (folder / 'updates.jsonl').open('x', encoding='utf-8') as journal:
        for step in range(8):
            record = records[step % 4]; torch.cuda.synchronize(); one = time.perf_counter()
            metrics = common.update(model, optimizer, record, recipe)
            torch.cuda.synchronize(); seconds = time.perf_counter() - one; durations.append(seconds)
            journal.write(json.dumps({'step': step+1, 'seconds': seconds, **metrics})+'\n'); journal.flush()
            print(f'THROWAWAY points={args.budget} update={step+1}/8 seconds={seconds:.3f} loss={metrics["loss"]:.5f}', flush=True)
    model.eval()
    with torch.no_grad():
        for record in records:
            torch.cuda.synchronize(); one = time.perf_counter(); prediction = model(record)
            torch.cuda.synchronize(); eval_durations.append(time.perf_counter()-one)
            if not all(bool(getattr(prediction, k).isfinite().all()) for k in
                       ('center_proposals', 'objectness_logits', 'radius', 'decoder_features')):
                raise ValueError('Nonfinite inference output')
        model.train(); a = model(records[0]); model.eval(); b = model(records[0])
        consistency = gn.prediction_consistency(a, b)
    model, optimizer = geometric.reload_probe(model, optimizer, recipe, 8, 'cuda:0')
    with torch.no_grad():
        model.eval(); reload_check = gn.prediction_consistency(b, model(records[0]))
    if not consistency['passed'] or not reload_check['passed']:
        raise ValueError('Mode consistency or full model/Adam/RNG reload failed')
    mean_update = float(np.mean(durations[4:])); mean_eval = float(np.mean(eval_durations))
    result = {'status': 'completed', **identity(control, recipe), 'budget': args.budget,
        'packet_manifest_sha256': data_identity(control), 'updates': 8,
        'initial_state_sha256': initial, 'update_seconds': durations, 'inference_seconds': eval_durations,
        'warm_update_mean_seconds': mean_update, 'inference_mean_seconds': mean_eval,
        'projected_9200_update_hours': mean_update*9200/3600,
        'projected_3624_forward_hours': mean_eval*3624/3600,
        'projection_limit': 'Four exposed positive training boxes only; no checkpoint IO, CPU matching/curve/profile metrics or data loading; not a full-run bound',
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'mode_consistency_passed': True, 'model_adam_rng_reload_passed': True,
        'trained_checkpoint_retained': False, 'quality_or_generalization_gate': None}
    common.write(folder / 'worker_report.json', result)
    print(f'COMPLETED timing points={args.budget} update_projection_hours={result["projected_9200_update_hours"]:.2f}; excludes overhead', flush=True)


def controller(control, recipe, args):
    require_cpu(control, recipe)
    for budget in control['budgets']: load_records(control, recipe, budget)
    folder = output_folder(args.output); results = []; status = 'completed'
    common.write(folder / 'control.json', control)
    for budget in control['budgets']:
        dest = folder / f'points{budget}'; dest.mkdir()
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--action', '_worker',
                   '--config', str(Path(args.config).resolve()), '--budget', str(budget), '--output', str(dest)]
        code, state = launch_worker(command, dest, control['gpu_worker_timeout_seconds'])
        path = dest / 'worker_report.json'; row = json.loads(path.read_text()) if path.exists() else {}
        if (state != 'completed' or row.get('status') != 'completed' or row.get('budget') != budget
                or row.get('updates') != 8 or not row.get('mode_consistency_passed')
                or not row.get('model_adam_rng_reload_passed')
                or row.get('packet_manifest_sha256') != data_identity(control)
                or any(row.get(k) != v for k, v in identity(control, recipe).items())):
            status = 'failed'; results.append({'budget': budget, 'status': state,
                'exit_code': code, 'worker_report_valid': False}); break
        results.append(row)
    if status == 'completed' and results[0]['initial_state_sha256'] != results[1]['initial_state_sha256']:
        status = 'failed_initialization_mismatch'
    result = {'status': status, **identity(control, recipe), 'arms': results,
        'automatic_retry': False, 'training_action_exists': False}
    if status == 'completed':
        result['paired_update_plus_forward_projection_hours'] = sum(
            r['projected_9200_update_hours'] + r['projected_3624_forward_hours'] for r in results)
        result['projection_excludes'] = 'Data loading, checkpoint IO and CPU evaluation metrics; timing sample is not cohort representative'
    common.write(folder / 'report.json', result)
    print(f'FINISHED two-budget timing status={status} report={folder / "report.json"}', flush=True)
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'cpucheck', 'timing', '_worker'), required=True)
    parser.add_argument('--config', default=str(CONFIG)); parser.add_argument('--output', required=True)
    parser.add_argument('--budget', type=int, choices=(2048, 8192))
    args = parser.parse_args(); control, recipe = load_control(args.config)
    if args.action == 'prepare': prepare(control, recipe, args.output); return 0
    if args.action == 'cpucheck': cpu_check(control, recipe, args.output); return 0
    if args.action == '_worker':
        if args.budget is None: raise ValueError('Worker requires its explicit point budget')
        worker(control, recipe, args); return 0
    return controller(control, recipe, args)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
