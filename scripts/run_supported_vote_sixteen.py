"""Conditional USER-launched sixteen-box candidate; never launched by CPU audit.

Requires the sole post-failure consolidation variant to pass the original
four-box numerical gates. Fresh model/Adam, unchanged dense curve supervision,
fixed r=2/min-support=3 inference. No GATv2 or automatic subsequent run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import local_curve_experiment as e
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.vote_scaffold import supported_consolidation
from pn2.evaluation.local_curve import summarize
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree


def ready(path):
    from measure_saved_vote_scaffold import control
    c = e.control(); scaffold = control(); m = e.read(path)
    data, prepared, selection = e.prepared(c)
    expected = {'support_cutoff': .25, 'support_radius': 2., 'minimum_support': 3, 'suppression_radius': 2.}
    proof = e.read(Path(path).parent/'B_result.json')
    complete = e.read(Path(path).parent/'report.json')
    guard = e.read(Path(path).parent.with_name(Path(path).parent.name+'_wall_guard')/'report.json')
    if (m['schema'] != 'pn2-supported-vote-sixteen-v1' or m['scientific_config_fingerprint'] != fingerprint(c)
            or m['scaffold_config_fingerprint'] != fingerprint(scaffold) or m['consolidation'] != expected
            or m['sixteen'] != selection['sixteen'] or m['development'] != selection['development']
            or m['updates'] != 4000 or m['wall_seconds'] != 14400 or m['automatic_launch'] is not False
            or m['preparation_manifest_sha256'] != digest(e.BASE/'prepare/manifest.json')
            or proof['status'] != 'variant_passed' or not proof['variant']['gate']['passed']
            or proof['variant']['gate'] != m['four_box_posthoc_gate']
            or complete['status'] != 'completed' or complete['config_fingerprint'] != fingerprint(scaffold)
            or complete['B']['status'] != 'variant_passed' or guard['status'] != 'completed'):
        raise ValueError('Missing/changed post-hoc gate, policy or sixteen-box manifest; no run')
    e.require_cpu(c)
    return c, m, data, prepared


def evaluate(c, model, data, prepared, keys, folder, step):
    import torch
    import sampler_node_common as common
    from pn2.training.node_runtime import image_batch
    from pn2.evaluation.node_profile_metrics import ProfileCache
    folder.mkdir(); refs = e.References(c, data)
    profiles = ProfileCache(e.ROOT, e.ROOT/c['profile_root'], data)
    mode = model.training; model.eval(); rows = []; arrays = {}; began = time.perf_counter()
    old_state = common.state_digest(model.state_dict()); old_rng = common.nested_digest(common.rng_state())
    with torch.inference_mode():
        for i, key in enumerate(keys):
            cid, tile = key
            image = data.image(cid, tile, 2048)
            output = model(image_batch(image, 'cuda:0'))
            points, scores = e.unpack(output, 'dense')
            keep = supported_consolidation(points, scores, image['packet']['flat_indices'])
            row = e.score(data, refs, key, points[keep])
            if row['metrics']['known']:
                row['thin_dim'] = e.thin_metrics(points[keep], profiles.load(cid, tile), profiles.manifest['cutoffs'])
                row['incident'] = e.incident_metrics(points[keep], e.incident_probes(refs.get(cid)['morphology']),
                                                     data.boxes[tuple(key)]['box'])
                row['longest_uncovered_length_2'] = e.gap_metrics(points[keep], data.reference(cid, tile))
                row['offset_bands'] = e.offset_metrics(output, e.label_arrays(prepared, key), refs.get(cid)['distances'])
            rows.append(row)
            arrays[f'box{i}_points'] = points; arrays[f'box{i}_scores'] = scores
            arrays[f'box{i}_retained_indices'] = keep
            if (i+1) % 32 == 0:
                print(f'EVAL supported-r2-min3 step={step} {i+1}/{len(keys)} elapsed={time.perf_counter()-began:.1f}s', flush=True)
    model.train(mode)
    if common.state_digest(model.state_dict()) != old_state or common.nested_digest(common.rng_state()) != old_rng:
        raise ValueError('Evaluation changed model/RNG')
    report = {'step': step, 'keys': keys, 'rows': rows, 'summary': summarize(rows),
        'consolidation': 'post-failure .25/r2/min3; no sweep', 'seconds': time.perf_counter()-began,
        'scope': 'isolated proposal geometry; no graph or final reconstruction verdict'}
    write_once(folder/'metrics.json', report); np.savez_compressed(folder/'predictions.npz', **arrays)
    print('EVAL COMPLETE '+json.dumps(report['summary']['macro']), flush=True)
    return report


def worker(path, output):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    c, m, data, prepared = ready(path)
    e.runtime(c, 'cuda:0')
    if torch.cuda.get_device_name(0) != e.read(e.BASE/'timing_dense/report.json')['device']:
        raise ValueError('Device differs from existing timing')
    output.mkdir(parents=True, exist_ok=False)
    model = e.model_factory(c, 'dense', 'cuda:0'); optimizer = common.make_optimizer(model, c)
    records = e.train_records(c, data, prepared, m['sixteen'], 'cuda:0')
    bindings = {'maximum_updates': 4000, 'scientific_config_fingerprint': fingerprint(c),
                'posthoc_manifest_sha256': digest(path), 'policy': m['consolidation'], 'rung': '16', 'arm': 'dense'}
    state.save(output/'checkpoint_step000000.pt', model, optimizer, bindings, 0)
    initial_encoder = common.state_digest(model.encoder.state_dict()); began = time.perf_counter()
    training = evaluate(c, model, data, prepared, m['sixteen'], output/'eval_step000000_training', 0)
    development = evaluate(c, model, data, prepared, m['development'], output/'eval_step000000_development', 0)
    with (output/'updates.jsonl').open('x', encoding='utf-8') as stream:
        for step in range(1, 4001):
            metrics = e.update(c, model, optimizer, records[(step-1) % 16][2], 'dense')
            stream.write(json.dumps({'step': step, **metrics})+'\n')
            if step % 50 == 0:
                stream.flush(); print(f'TRAIN sixteen supported-r2-min3 {step}/4000 loss={metrics["loss"]:.5f}', flush=True)
            if step % 500 == 0:
                state.save(output/f'checkpoint_step{step:06d}.pt', model, optimizer, bindings, step)
            if step in (2000, 4000):
                training = evaluate(c, model, data, prepared, m['sixteen'], output/f'eval_step{step:06d}_training', step)
                development = evaluate(c, model, data, prepared, m['development'], output/f'eval_step{step:06d}_development', step)
    absolute = e.absolute_sixteen_gate(c, training, development)
    report = {'status': 'completed', 'updates': 4000, 'exposures_per_box': 250, 'absolute_gate': absolute,
        'training': training['summary'], 'development': development['summary'],
        'seconds': time.perf_counter()-began, 'initial_encoder_sha256': initial_encoder,
        'posthoc_manifest_sha256': digest(path), 'policy': m['consolidation'],
        'paired_comparative_gate_evaluated': False, 'automatic_next_run': False,
        'interpretation': 'post-hoc consolidation candidate; paired Transformer/no-learning comparisons still required'}
    write_once(output/'report.json', report)
    print('SIXTEEN COMPLETE '+json.dumps({'report': str(output/'report.json'), 'absolute_gate': absolute,
                                        'automatic_next_run': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); c, m, _, _ = ready(args.manifest)
    if args.check:
        print('Conditional sixteen-box manifest verified; no model execution'); return
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = e.output_path(args.output or e.BASE/f'supported_vote_sixteen_{suffix}')
    if args.worker:
        worker(args.manifest, output); return
    import psutil
    if output.exists():
        raise FileExistsError('Preserve existing outputs')
    logs = output.with_name(output.name+'_wall_guard'); logs.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--manifest',
               str(args.manifest.resolve()), '--output', str(output)]
    print(f'OUTPUT {output}; RTX 3070 Ti; 4000 updates; wall cap 4h; consolidation r2/min3', flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        def relay():
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        reader = threading.Thread(target=relay, daemon=True); reader.start()
        try:
            receipt = wait_wall(process, m['wall_seconds'], lambda: terminate_owned_tree(owner))
        except BaseException:
            terminate_owned_tree(owner); process.wait(timeout=5); raise
        finally:
            reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError('Worker output pipe did not close')
    receipt.update(manifest_sha256=digest(args.manifest), output=str(output))
    write_once(logs/'report.json', receipt)
    if receipt['status'] != 'completed':
        raise SystemExit(f'Sixteen-box candidate {receipt["status"]}; no automatic continuation')


if __name__ == '__main__':
    main()
