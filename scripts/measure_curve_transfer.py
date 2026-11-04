"""User-local CPU diagnostics: matched no-learning forests or frozen-model transfer.

No training, architecture change, threshold selection or automatic next rung.
"""
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

import local_curve_experiment as e
import measure_saved_vote_scaffold as scaffold
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.local_curve import consolidate
from pn2.evaluation.formulation import graph_segments, clip_segments, segment_agreement, runs_graph
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.reference_shortcuts import ReferencePaths
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree

ROOT = e.ROOT
CONFIG = ROOT/'configs/curve_transfer_diagnostics.json'


def control():
    c = e.read(CONFIG)
    if (c['schema'] != 'pn2-curve-transfer-diagnostics-v1' or c['device'] != 'cpu'
            or c['policies'] != ['dense', 'raw_input', 'sato512'] or c['decoding_k'] != 6
            or c['shortcut_rule'] != 'D > 2*L + 4' or c['stage_wall_seconds'] != {'fit': 1800, 'development': 7200}):
        raise ValueError('Changed diagnostic scope')
    for name, sha in {**c['source_sha256'], **c['artifact_sha256']}.items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Diagnostic source/input changed: {name}')
    scaffold.control()
    scientific = e.control()
    data, manifest, selection = e.prepared(scientific)
    dev = [tuple(k) for k in selection['development'] if data.boxes[tuple(k)]['target']['count'] > 0]
    if len(dev) != 151 or len({cid for cid, _ in dev}) != 7:
        raise ValueError('Changed development reference-present domain')
    return c, scientific, data, manifest, selection, dev


def retained(points, scores, ids):
    ranked = consolidate(points, scores, radius=1., identities=ids)
    return ranked[scores[ranked] >= .25]


def check_checkpoint(c, scientific, manifest):
    import torch
    checkpoint = ROOT/c['checkpoint']
    marker = e.read(checkpoint.with_suffix('.complete.json'))
    if marker['step'] != 2000 or marker['checkpoint'] != checkpoint.name or marker['sha256'] != digest(checkpoint):
        raise ValueError('Incomplete terminal checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = {'maximum_updates': 2000, 'config_fingerprint': fingerprint(scientific), 'arm': 'dense',
                'rung': '4', 'data_sha256': manifest['data_sha256'], 'selection_sha256': manifest['selection_sha256']}
    if payload['schema'] != 'pn2-paired-node-fit-state-v1' or payload['step'] != 2000 or payload['bindings'] != expected:
        raise ValueError('Checkpoint identity/cursor mismatch')
    return payload


def inference_worker(output):
    import torch
    import sampler_node_common as common
    from pn2.training.node_runtime import image_batch
    c, scientific, data, manifest, selection, dev = control()
    e.runtime(scientific, 'cpu')
    payload = check_checkpoint(c, scientific, manifest)
    model = e.model_factory(scientific, 'dense', 'cpu')
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.eval()
    before = common.state_digest(model.state_dict())
    output.mkdir(exist_ok=False)
    parity = []
    with np.load(ROOT/c['fit_predictions'], allow_pickle=False) as saved, torch.inference_mode():
        for i, key in enumerate(selection['four']):
            image = data.image(*key, 2048)
            points, scores = e.unpack(model(image_batch(image, 'cpu')), 'dense')
            old_p, old_s = saved[f'box{i}_points'], saved[f'box{i}_scores']
            ids = image['packet']['flat_indices']
            a, b = retained(points, scores, ids), retained(old_p, old_s, ids)
            row = {'key': key, 'maximum_coordinate_error': float(np.max(np.abs(points-old_p))),
                   'maximum_score_error': float(np.max(np.abs(scores-old_s))),
                   'retained_source_indices_equal': bool(np.array_equal(np.sort(a), np.sort(b)))}
            parity.append(row)
        write_once(output/'cpu_parity.json', {'rows': parity})
        if any(r['maximum_coordinate_error'] > .001 or r['maximum_score_error'] > .0001
               or not r['retained_source_indices_equal'] for r in parity):
            raise ValueError('CPU/GPU numerical parity failed; no development continuation')
        rows = []
        for i, key in enumerate(dev):
            began = time.perf_counter()
            image = data.image(*key, 2048)
            points, scores = e.unpack(model(image_batch(image, 'cpu')), 'dense')
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite frozen inference')
            path = output/f'box{i:04d}.npz'
            np.savez_compressed(path, points=points, scores=scores)
            rows.append({'key': key, 'packet': path.name, 'sha256': digest(path)})
            print(f'CPU INFERENCE {i+1}/{len(dev)} seconds={time.perf_counter()-began:.2f}', flush=True)
    if before != common.state_digest(model.state_dict()) or torch.cuda.is_initialized():
        raise ValueError('Inference mutated state or initialized CUDA')
    write_once(output/'complete.json', {'rows': rows, 'device': 'cpu', 'updates': 0, 'optimizer_created': False,
               'model_state_unchanged': True, 'CUDA_initialized': False, 'config_fingerprint': fingerprint(c),
               'checkpoint_sha256': digest(ROOT/c['checkpoint']), 'torch_version': torch.__version__,
               'numpy_version': np.__version__, 'runtime': sys.executable})


def existing_input_evidence(c):
    report = e.read(ROOT/c['input_audit_report'])
    frog = None
    # Identity from the frozen fit selection, never fuzzy source-group matching.
    cid = next(k[0] for k in e.read(e.BASE/'prepare/selection.json')['four'] if k[1] == 5)
    with (ROOT/c['input_audit_rows']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (row['case_id'], row['tile_id']) == (cid, 5):
                frog = row['input']
    if frog is None:
        raise ValueError('Missing existing exact frog box evidence')
    return {'provenance': 'read existing completed audit only; no recomputation',
            'partitions': {part: {b: {'all_2': s['input'][b]['2.0'],
                'strata_2': {name: values['2.0'] for name, values in s['input_profile_strata'][b].items()}}
                for b in ('2048', '8192')} for part, s in report['summary']['partitions'].items()},
            'frog_fit_box5': {'case_id': cid, 'input': frog}}


def graphs(c, data, keys, output, dense_packets):
    records, guides = [], {}
    counts = []
    for i, key in enumerate(keys):
        image = data.image(*key, 2048)
        source = image['packet']['native_xyz']
        ids = image['packet']['flat_indices']
        sato = e.sato_scores(data, key, image['packet'], guides)
        raw = consolidate(source, np.ones(len(source)), radius=1., identities=ids)
        ranked = consolidate(source, sato, radius=1., identities=ids)[:512]
        records.extend([{'policy': 'raw_input', 'key': key, 'points': source[raw]},
                        {'policy': 'sato512', 'key': key, 'points': source[ranked]}])
        if dense_packets is not None:
            path = dense_packets[i]
            with np.load(path, allow_pickle=False) as a:
                points, scores = a['points'], a['scores']
                keep = retained(points, scores, ids)
                records.append({'policy': 'dense', 'key': key, 'points': points[keep]})
        counts.append({'key': key, 'raw_source': len(source), 'raw_retained': len(raw), 'sato512_retained': len(ranked)})
    packets = scaffold.build_graph_artifacts(records, guides, output/'graphs')
    if dense_packets is None:
        old = e.read(ROOT/c['saved_scaffold_report'])
        for i, row in enumerate(old['C']['rows']):
            if row['policy'] != 'original':
                continue
            folder = (ROOT/c['saved_scaffold_report']).parent/'C_graphs'/f'original_box{i}'
            for name, hash_key in [('graph.npz', 'graph_sha256'), ('scaffold.swc', 'swc_sha256')]:
                if digest(folder/name) != row['inference'][hash_key]:
                    raise ValueError('Saved fit graph changed')
            packets.append({'policy': 'dense', 'key': (row['case_id'], row['tile_id']),
                            'folder': str(folder), 'inference': row['inference']})
    write_once(output/'all_inference_complete.json', {'packets': packets,
               'reference_scoring_started': False, 'counts': counts})
    return packets, counts


def score(c, scientific, data, packets, output):
    # No reference geometry opens until all graph artifacts exist.
    rows = []
    for packet in packets:
        cid, tile = packet['key']
        reference = graph_segments(runs_graph(data.reference(cid, tile)['reference_runs']))
        box = data.boxes[(cid, tile)]['box']
        low, high = np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5
        swc = parse_swc(Path(packet['folder'])/'scaffold.swc')
        decoded = segment_agreement(reference, clip_segments(graph_segments(swc), low, high))
        candidates = {}
        with np.load(Path(packet['folder'])/'graph.npz', allow_pickle=False) as a:
            from types import SimpleNamespace
            for k in (4, 6, 10):
                graph = SimpleNamespace(xyz=a['xyz'], edges=a[f'candidate{k}'])
                candidates[str(k)] = segment_agreement(reference, clip_segments(graph_segments(graph), low, high))
        if decoded['tolerances']['2.0']['recall'] > candidates['6']['tolerances']['2.0']['recall']+1e-12:
            raise ValueError('Decoded recall exceeded fixed candidate union')
        rows.append({'policy': packet['policy'], 'case_id': cid, 'tile_id': tile,
                     'segments': decoded, 'candidate_unions': candidates, 'inference': packet['inference']})
    summaries = {}
    for policy in c['policies']:
        group = [r for r in rows if r['policy'] == policy]
        summaries[policy] = {'decoded': scaffold.segment_summary(group), 'candidate_unions': {
            str(k): scaffold.segment_summary([{'case_id': r['case_id'], 'segments': r['candidate_unions'][str(k)]}
                                             for r in group]) for k in (4, 6, 10)}}
    result = {'rows': rows, 'summaries': summaries, 'topology_certified': False, 'whole_image_merge': False}
    refs = e.References(scientific, data)
    paths = {}
    shortcut_rows = []
    for i, packet in enumerate(packets):
        cid, tile = packet['key']
        if cid not in paths:
            paths[cid] = ReferencePaths(refs.get(cid)['morphology'])
        box = data.boxes[(cid, tile)]['box']
        with np.load(Path(packet['folder'])/'graph.npz', allow_pickle=False) as a:
            diagnostic = paths[cid].score(a['xyz'], a['forest_edges'],
                np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5)
        path = output/f'shortcuts_{i:04d}.json'
        write_once(path, diagnostic)
        shortcut_rows.append({'policy': packet['policy'], 'case_id': cid, 'tile_id': tile,
            'categories': diagnostic['categories'], 'flagged': diagnostic['flagged'],
            'flagged_length': diagnostic['flagged_length'], 'details': path.name, 'sha256': digest(path)})
        if (i+1) % 25 == 0:
            print(f'CPU GRAPH SCORE {i+1}/{len(packets)}', flush=True)
    result['reference_shortcuts'] = shortcut_rows
    # The reused utility's historical four-box scope does not describe this run.
    result['scope'] = 'matched per-box forests; no whole-image merge or topology certification'
    return result


def readable(report):
    lines = ['# Frozen curve-head transfer diagnostic', '',
        f"Stage: {report['stage']}; completed CPU-only; training updates: 0.",
        'Original gate unchanged; no automatic next rung. Counts differ across policies.', '',
        '| Policy | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |', '|---|---:|---:|---:|---:|---:|---:|']
    for policy, s in report['measurement']['summaries'].items():
        v = s['decoded']['macro']
        values = [v[t][m] for t in ('2.0', '6.0') for m in ('precision', 'recall', 'f1')]
        lines.append('| '+policy+' | '+' | '.join(f'{100*x:.2f}%' for x in values)+' |')
    lines += ['', 'Reference-path shortcut flags (not exhaustive biological wrong-join rates):', '']
    for policy in ('dense', 'raw_input', 'sato512'):
        rows = [r for r in report['measurement']['reference_shortcuts'] if r['policy'] == policy]
        total = sum(r['categories']['scorable']['count'] for r in rows)
        flags = sum(r['flagged'] for r in rows)
        unknown = {name: sum(r['categories'][name]['count'] for r in rows)
                   for name in rows[0]['categories'] if name != 'scorable'}
        lines.append(f'- {policy}: {flags}/{total} scorable edges flagged; other categories {unknown}.')
    lines += ['', 'See report.json for every neuron/box, candidate-union diagnostics, counts, and source bindings.',
              'Fit stage is training fit only. Development stage is transfer to untrained neurons on repeatedly inspected',
              'reference-present boxes; unknown boxes excluded. Neither is a final untouched test or whole-neuron result.',
              'No model or threshold was selected; no edge was removed using reference scoring.']
    return '\n'.join(lines)+'\n'


def worker(stage, output):
    c, scientific, data, _, selection, dev = control()
    output.mkdir(exist_ok=False)
    began = time.perf_counter()
    inputs = existing_input_evidence(c)
    if stage == 'development':
        runtime = (ROOT/c['torch_runtime']).resolve()
        command = [str(runtime), '-B', str(Path(__file__).resolve()), '--inference-worker', '--output', str(output/'predictions')]
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        complete = e.read(output/'predictions/complete.json')
        if [tuple(r['key']) for r in complete['rows']] != dev or complete['config_fingerprint'] != fingerprint(c):
            raise ValueError('Incomplete or mismatched development inference')
        dense_packets = []
        for r in complete['rows']:
            path = output/'predictions'/r['packet']
            if digest(path) != r['sha256']:
                raise ValueError('Changed inference packet')
            dense_packets.append(path)
        keys = dev
    else:
        keys, dense_packets = [tuple(k) for k in selection['four']], None
    packets, counts = graphs(c, data, keys, output, dense_packets)
    measured = score(c, scientific, data, packets, output)
    report = {'status': 'completed', 'stage': stage, 'config_fingerprint': fingerprint(c),
        'measurement': measured, 'input_support_existing_evidence': inputs, 'counts': counts,
        'boxes': len(keys), 'neurons': len({cid for cid, _ in keys}), 'GPU_executed': False,
        'training_updates': 0, 'model_executed': stage == 'development', 'gate_changed': False,
        'sixteen_prepared': False, 'seconds': time.perf_counter()-began,
        'runtime': sys.executable, 'numpy_version': np.__version__,
        'scipy_version': importlib.metadata.version('scipy')}
    write_once(output/'report.json', report)
    (output/'report.md').write_text(readable(report), encoding='utf-8')
    print('DIAGNOSTIC COMPLETE '+json.dumps({'report': str(output/'report.json'), 'seconds': report['seconds'],
                                           'automatic_next_rung': False}), flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'development'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--inference-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inference_worker:
        inference_worker(e.output_path(args.output)); return
    c, _, _, _, _, _ = control()
    if args.check:
        print('Bindings verified; four fit boxes and 151 development boxes; no model/measurement executed'); return
    if args.stage is None:
        parser.error('--stage is required; no timed run is implicit')
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = e.output_path(args.output or ROOT/'tmp'/f'curve_transfer_{args.stage}_{suffix}')
    if args.worker:
        worker(args.stage, output); return
    import psutil
    if output.exists():
        raise FileExistsError('Preserve prior diagnostic output')
    logs = output.with_name(output.name+'_wall_guard')
    logs.mkdir(exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--stage', args.stage, '--output', str(output)]
    print(f'CPU ONLY; stage={args.stage}; OUTPUT {output}; wall cap {c["stage_wall_seconds"][args.stage]} seconds', flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        def relay():
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        thread = threading.Thread(target=relay, daemon=True); thread.start()
        try:
            receipt = wait_wall(process, c['stage_wall_seconds'][args.stage], lambda: terminate_owned_tree(owner))
        except BaseException:
            terminate_owned_tree(owner); process.wait(timeout=5); raise
        finally:
            thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError('Worker output pipe did not close')
    receipt.update(config_fingerprint=fingerprint(c), output=str(output))
    write_once(logs/'report.json', receipt)
    if receipt['status'] != 'completed':
        raise SystemExit(f'Diagnostic {receipt["status"]}; preserve partial artifacts; no automatic retry')


if __name__ == '__main__':
    main()
