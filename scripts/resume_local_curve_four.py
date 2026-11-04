"""User-launched recovery of the existing four-box dense fit, 1000 -> 2000.

Default invocation launches only this bounded recovery. --check validates the
existing checkpoint on CPU without creating outputs, CUDA context or updates.
The scientific driver and its original configuration remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

import local_curve_experiment as experiment
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree
from pn2.training.local_curve_recovery import (continuation_steps, prediction_replay,
                                               metric_replay, validate_payload)

ROOT = experiment.ROOT
CONFIG = ROOT/'configs/local_curve_recovery.json'
SOURCES = ('scripts/resume_local_curve_four.py', 'src/pn2/training/local_curve_recovery.py',
           'src/pn2/training/local_curve_control.py', 'tests/test_local_curve_recovery.py',
           'docs/local_curve_recovery_ready.md')


def plan():
    recovery = experiment.read(CONFIG)
    if (recovery['schema'] != 'pn2-local-curve-recovery-v1'
            or recovery['start_update'] != 1000 or recovery['maximum_update'] != 2000
            or recovery['wall_seconds'] != 1800 or recovery['checkpoint_every'] != 250
            or recovery['automatic_next_rung'] is not False
            or set(recovery['source_sha256']) != set(SOURCES)):
        raise ValueError('Recovery differs from the declared bounded command')
    for name, sha in recovery['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Recovery source changed: {name}')
    c = experiment.control()
    experiment.require_cpu(c)
    if fingerprint(c) != recovery['scientific_config_fingerprint']:
        raise ValueError('Original scientific recipe changed')
    data, manifest, selection = experiment.prepared(c)
    checkpoint = ROOT/recovery['checkpoint']
    marker = experiment.read(checkpoint.with_suffix('.complete.json'))
    if (marker['checkpoint'] != checkpoint.name or marker['step'] != 1000
            or marker['sha256'] != recovery['checkpoint_sha256']
            or digest(checkpoint) != recovery['checkpoint_sha256']):
        raise ValueError('Original recovery checkpoint changed or is incomplete')
    for key in ('predictions', 'metrics', 'updates'):
        if digest(ROOT/recovery[key]) != recovery[key+'_sha256']:
            raise ValueError(f'Original recovery evidence changed: {key}')
    source_eval = experiment.read(ROOT/recovery['metrics'])
    if source_eval['step'] != 1000 or source_eval['keys'] != selection['four']:
        raise ValueError('Original evaluation cursor/box order changed')
    bindings = {'maximum_updates': 2000, 'config_fingerprint': fingerprint(c), 'arm': 'dense',
                'rung': '4', 'data_sha256': manifest['data_sha256'],
                'selection_sha256': manifest['selection_sha256']}
    return recovery, c, data, manifest, selection, bindings


def checked_payload(recovery, c, bindings):
    import torch
    payload = torch.load(ROOT/recovery['checkpoint'], map_location='cpu', weights_only=False)
    states = validate_payload(payload, bindings, 1000, c['optimizer'])
    if payload['rng']['torch_cuda'] is None or len(payload['rng']['torch_cuda']) != 1:
        raise ValueError('Expected the original single-device CUDA RNG state')
    return payload, states


def check():
    import torch
    recovery, c, _, _, selection, bindings = plan()
    _, states = checked_payload(recovery, c, bindings)
    print(json.dumps({'status': 'ready', 'check_device': 'cpu', 'optimizer_updates': 0,
        'cuda_context_initialized': torch.cuda.is_initialized(), 'restore_update': 1000,
        'finish_update': 2000, 'additional_model_updates': 1000, 'Adam_states_verified': states,
        'boxes': len(selection['four']), 'wall_seconds': recovery['wall_seconds'],
        'scientific_config_fingerprint': fingerprint(c), 'automatic_next_rung': False}), flush=True)


def worker(output):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import image_batch
    recovery, c, data, manifest, selection, bindings = plan()
    payload, adam_states = checked_payload(recovery, c, bindings)
    experiment.runtime(c, 'cuda:0')
    expected_device = experiment.read(experiment.BASE/'timing_dense/report.json')['device']
    if torch.cuda.get_device_name(0) != expected_device:
        raise ValueError('GPU differs from the timed/saved run; no automatic device substitution')
    records = experiment.train_records(c, data, manifest, selection['four'], 'cuda:0')
    model = experiment.model_factory(c, 'dense', 'cuda:0')
    optimizer = common.make_optimizer(model, c)
    start = state.restore(payload, model, optimizer, bindings)
    if start != 1000:
        raise ValueError('Recovery must restore the complete update-1000 state')
    del payload
    output.mkdir(parents=True, exist_ok=False)
    write_once(output/'recovery_plan.json', {'recovery_control_fingerprint': fingerprint(recovery),
        'scientific_bindings': bindings, 'source_checkpoint': recovery['checkpoint'],
        'source_checkpoint_sha256': recovery['checkpoint_sha256'], 'starting_update': start,
        'terminal_update': 2000, 'wall_seconds': 1800, 'automatic_next_rung': False})
    shutil.copyfile(ROOT/recovery['checkpoint'], output/'checkpoint_step001000.pt')
    shutil.copyfile((ROOT/recovery['checkpoint']).with_suffix('.complete.json'),
                    output/'checkpoint_step001000.complete.json')
    print(f'RESTORED update=1000 Adam_states={adam_states} all_clocks=1000 full_RNG=restored '
          f'device={torch.cuda.get_device_name(0)}', flush=True)
    before_rng = common.nested_digest(common.rng_state())
    mode = model.training
    model.eval()
    with np.load(ROOT/recovery['predictions'], allow_pickle=False) as original, torch.inference_mode():
        for i, (key, record, _, _) in enumerate(records):
            batch = image_batch(data.image(*key, c['budget']), 'cuda:0')
            points, scores = experiment.unpack(model(batch), 'dense')
            prediction_replay(points, scores, original[f'box{i}_points'], original[f'box{i}_scores'])
    model.train(mode)
    if common.nested_digest(common.rng_state()) != before_rng:
        raise ValueError('Prediction replay changed RNG state')
    print('PREDICTION_REPLAY_EXACT boxes=4; continuing original optimizer trajectory', flush=True)
    previous = {}
    with (ROOT/recovery['updates']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['step'] > start:
                previous[row['step']] = row
    if not all(step in previous for step in range(1001, 1009)):
        raise ValueError('Missing original optimizer prefix for recovery verification')
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    prefix_matches = 0
    with (output/'updates.jsonl').open('x', encoding='utf-8') as log:
        for step, index in continuation_steps(1000, 2000, len(records)):
            metrics = experiment.update(c, model, optimizer, records[index][2], 'dense')
            if step in previous:
                metric_replay(step, metrics, previous[step])
                prefix_matches += 1
            log.write(json.dumps({'step': step, **metrics})+'\n')
            if step == 1008:
                print('OPTIMIZER_PREFIX_EXACT replayed_updates=8', flush=True)
            if step % 50 == 0:
                log.flush()
                print(f'RESUME step={step}/2000 new_updates={step-1000}/1000 loss={metrics["loss"]:.5f} '
                      f'elapsed={time.perf_counter()-began:.1f}s', flush=True)
            if step % recovery['checkpoint_every'] == 0:
                before = common.nested_digest(common.rng_state())
                state.save(output/f'checkpoint_step{step:06d}.pt', model, optimizer, bindings, step)
                if common.nested_digest(common.rng_state()) != before:
                    raise ValueError('Checkpoint serialization changed RNG state')
    terminal = experiment.evaluate(c, model, 'dense', data, manifest, selection['four'],
                                   output/'eval_step002000_training', 2000)
    gate = experiment.four_gate(c, terminal)
    report = {'status': 'completed', 'rung': '4', 'arm': 'dense', 'updates': 2000,
        'restored_update': 1000, 'new_model_updates': 1000, 'exposures_per_box_total': 500,
        'recovery_control_fingerprint': fingerprint(recovery), 'config_fingerprint': fingerprint(c),
        'checkpoint_sha256': recovery['checkpoint_sha256'], 'prediction_replay_exact': True,
        'matching_original_optimizer_updates': prefix_matches,
        'prior_persisted_optimizer_trace_last_step': max(previous),
        'replayed_logged_compute_is_extra_cost': True, 'gate': gate, 'training': terminal['summary'],
        'development': None, 'seconds_after_replay': time.perf_counter()-began,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'device': torch.cuda.get_device_name(0),
        'automatic_next_rung': False, 'interpretation': 'terminal four-box training fit; no development or graph-quality claim'}
    write_once(output/'report.json', report)
    print('RECOVERY COMPLETE '+json.dumps({'gate': gate, 'report': str(output/'report.json'),
                                         'automatic_next_rung': False}), flush=True)


def launch(output):
    import psutil
    recovery, _, _, _, _, _ = plan()
    if output.exists():
        raise FileExistsError('Preserve existing recovery evidence; choose an unused output path')
    logs = output.with_name(output.name+'_wall_guard')
    logs.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--output', str(output)]
    print(f'OUTPUT {output}\nWALL_LIMIT 1800 seconds; update 1000 -> 2000; no automatic sixteen-box run', flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        def relay():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        try:
            receipt = wait_wall(process, recovery['wall_seconds'], lambda: terminate_owned_tree(owner))
        except BaseException:
            terminate_owned_tree(owner)
            process.wait(timeout=5)
            receipt = {'status': 'interrupted', 'maximum_seconds': 1800}
            write_once(logs/'interruption.json', receipt)
            raise
        finally:
            reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError('Owned recovery worker output pipe did not close')
    receipt.update(output=str(output), command=command, recovery_control_fingerprint=fingerprint(recovery),
                   recovery_sources=recovery['source_sha256'])
    write_once(logs/'report.json', receipt)
    if receipt['status'] != 'completed':
        raise SystemExit(f'Recovery {receipt["status"]}; stopped without automatic retry or expansion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='CPU-only checkpoint/config check; no training')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check:
        if args.worker or args.output:
            parser.error('--check is a read-only action without output/worker options')
        check()
        return
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = experiment.output_path(args.output or experiment.BASE/f'recovery4_dense_{suffix}')
    if args.worker:
        try:
            worker(output)
        except BaseException:
            if output.is_dir() and not (output/'error.json').exists():
                write_once(output/'error.json', {'status': 'failed', 'traceback': traceback.format_exc()})
            raise
    else:
        launch(output)


if __name__ == '__main__':
    main()
