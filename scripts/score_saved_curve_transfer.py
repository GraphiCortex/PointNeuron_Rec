"""User-local scoring-only continuation of the interrupted development diagnostic.

Reuses completed prediction/graph/SWC artifacts. No model, inference, graph
construction, filtering, training, threshold changes or automatic next rung.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np

import measure_curve_transfer as original
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.bounded_agreement import BoundedAgreement
from pn2.evaluation.formulation import graph_segments, clip_segments, runs_graph
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.reference_shortcuts import ReferencePaths
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree

ROOT = original.ROOT
CONFIG = ROOT/'configs/curve_transfer_scoring_resume.json'


def control():
    c = original.e.read(CONFIG)
    if (c['schema'] != 'pn2-curve-transfer-scoring-resume-v1' or c['wall_seconds'] != 1800
            or c['model_execution'] or c['graph_construction'] or c['tolerances'] != [2., 6.]
            or c['quadrature_step'] != .5):
        raise ValueError('Changed scoring-only scope')
    for name, sha in {**c['source_sha256'], **c['artifact_sha256']}.items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Changed continuation source/receipt: {name}')
    for package, version in c['runtime_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Changed scoring runtime: {package}')
    old, scientific, data, _, _, dev = original.control()
    source = ROOT/c['interrupted_root']
    predictions = original.e.read(source/'predictions/complete.json')
    packets = original.e.read(source/'all_inference_complete.json')['packets']
    if (predictions['config_fingerprint'] != fingerprint(old)
            or [tuple(r['key']) for r in predictions['rows']] != dev
            or predictions['updates'] != 0 or not predictions['model_state_unchanged']
            or predictions['CUDA_initialized']):
        raise ValueError('Incomplete or changed CPU inference prefix')
    for row in predictions['rows']:
        if digest(source/'predictions'/row['packet']) != row['sha256']:
            raise ValueError('Saved prediction packet changed')
    expected = {(policy, key) for policy in old['policies'] for key in dev}
    observed = {(p['policy'], tuple(p['key'])) for p in packets}
    if len(packets) != 453 or observed != expected:
        raise ValueError('Missing/duplicate development graphs')
    for packet in packets:
        folder = Path(packet['folder']).resolve()
        if not folder.is_relative_to((source/'graphs').resolve()):
            raise ValueError('Graph receipt escaped saved source')
        check_packet(packet)
    return c, old, scientific, data, packets


def check_packet(packet):
    folder = Path(packet['folder'])
    receipt = original.e.read(folder/'inference.json')
    if receipt != packet['inference'] or receipt['reference_edges_used'] or receipt['reference_roots_used']:
        raise ValueError('Graph inference receipt changed or used reference data')
    for name, key in [('graph.npz', 'graph_sha256'), ('scaffold.swc', 'swc_sha256')]:
        if digest(folder/name) != receipt[key]:
            raise ValueError(f'Saved graph/SWC changed: {folder/name}')


def metric_row(packet, data, agreement=None):
    cid, tile = packet['key']
    box = data.boxes[(cid, tile)]['box']
    low, high = np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5
    if agreement is None:
        reference = graph_segments(runs_graph(data.reference(cid, tile)['reference_runs']))
        agreement = BoundedAgreement(reference)
    folder = Path(packet['folder'])
    decoded = agreement.score(clip_segments(graph_segments(parse_swc(folder/'scaffold.swc')), low, high))
    candidate = {}
    with np.load(folder/'graph.npz', allow_pickle=False) as a:
        for k in (4, 6, 10):
            graph = SimpleNamespace(xyz=a['xyz'], edges=a[f'candidate{k}'])
            candidate[str(k)] = agreement.score(clip_segments(graph_segments(graph), low, high))
    if decoded['tolerances']['2.0']['recall'] > candidate['6']['tolerances']['2.0']['recall']+1e-12:
        raise ValueError('Decoded recall exceeded its candidate union')
    return {'policy': packet['policy'], 'case_id': cid, 'tile_id': tile, 'segments': decoded,
            'candidate_unions': candidate, 'inference': packet['inference']}


def cached_record(path, binding, compute):
    """Atomic complete JSON, durable before proceeding to another graph."""
    if path.exists():
        record = original.e.read(path)
        if record['binding'] != binding or record['status'] != 'completed':
            raise ValueError('Scoring cache belongs to different source/config')
        return record, True
    record = {'binding': binding, 'status': 'completed', **compute()}
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        # A partial temporary file is evidence only, never treated as a result.
        raise
    return record, False


def fit_parity(c, data, output):
    def compute():
        root = ROOT/c['fit_root']
        packets = original.e.read(root/'all_inference_complete.json')['packets']
        fit = original.e.read(root/'report.json')
        previous = {(r['policy'], r['case_id'], r['tile_id']): r for r in fit['measurement']['rows']}
        results = []
        for i, packet in enumerate(packets):
            check_packet(packet)
            row = metric_row(packet, data)
            old = previous[(row['policy'], row['case_id'], row['tile_id'])]
            if row['segments'] != old['segments'] or row['candidate_unions'] != old['candidate_unions']:
                raise ValueError('Bounded scorer differs from completed fit metric; stop before development')
            results.append({'policy': row['policy'], 'case_id': row['case_id'], 'tile_id': row['tile_id'],
                            'all_four_metric_dictionaries_exactly_equal': True})
            print(f'FIT SCORER PARITY {i+1}/{len(packets)} exact', flush=True)
        if len(results) != 12:
            raise ValueError('Incomplete real fit scorer parity')
        return {'rows': results, 'exact': True}
    binding = fingerprint({'config': fingerprint(c), 'kind': 'fit-metric-parity'})
    return cached_record(output/'fit_metric_parity.json', binding, compute)[0]


def worker(output):
    c, old, scientific, data, packets = control()
    output.mkdir(exist_ok=True)
    (output/'records').mkdir(exist_ok=True)
    binding = {'config_fingerprint': fingerprint(c), 'source': c['interrupted_root'],
               'source_config_fingerprint': fingerprint(old), 'kind': 'scoring-only; no model or graph execution'}
    marker = output/'binding.json'
    if marker.exists():
        if original.e.read(marker) != binding:
            raise ValueError('Continuation output belongs to different inputs/settings')
    else:
        write_once(marker, binding)
    began = time.perf_counter()
    fit_parity(c, data, output)
    refs = original.e.References(scientific, data)
    paths, agreements, records = {}, {}, []
    for i, packet in enumerate(packets):
        cid, tile = packet['key']
        path = output/'records'/f'graph{i:04d}.json'
        row_binding = fingerprint({'run': binding, 'packet': packet})
        print(f'SCORE {i+1}/{len(packets)} {packet["policy"]} tile={tile}', flush=True)
        started = time.perf_counter()
        def compute():
            key = (cid, tile)
            if key not in agreements:
                reference = graph_segments(runs_graph(data.reference(cid, tile)['reference_runs']))
                agreements[key] = BoundedAgreement(reference)
            if cid not in paths:
                paths[cid] = ReferencePaths(refs.get(cid)['morphology'])
            row = metric_row(packet, data, agreements[key])
            box = data.boxes[key]['box']
            with np.load(Path(packet['folder'])/'graph.npz', allow_pickle=False) as a:
                shortcuts = paths[cid].score(a['xyz'], a['forest_edges'],
                    np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5)
            return {'row': row, 'shortcuts': shortcuts}
        record, reused = cached_record(path, row_binding, compute)
        records.append((record, path))
        print(f'SAVED {i+1}/{len(packets)} reused={reused} seconds={time.perf_counter()-started:.2f} '
              f'elapsed={time.perf_counter()-began:.1f}', flush=True)
    rows = [r['row'] for r, _ in records]
    shortcut_rows = []
    for record, path in records:
        row, diagnostic = record['row'], record['shortcuts']
        shortcut_rows.append({'policy': row['policy'], 'case_id': row['case_id'], 'tile_id': row['tile_id'],
            'categories': diagnostic['categories'], 'flagged': diagnostic['flagged'],
            'flagged_length': diagnostic['flagged_length'], 'details': str(path.relative_to(output)), 'sha256': digest(path)})
    summaries = {}
    for policy in old['policies']:
        group = [r for r in rows if r['policy'] == policy]
        summaries[policy] = {'decoded': original.scaffold.segment_summary(group), 'candidate_unions': {
            str(k): original.scaffold.segment_summary([{'case_id': r['case_id'], 'segments': r['candidate_unions'][str(k)]}
                                                       for r in group]) for k in (4, 6, 10)}}
    measurement = {'rows': rows, 'summaries': summaries, 'reference_shortcuts': shortcut_rows,
        'topology_certified': False, 'whole_image_merge': False,
        'scope': 'all 151 reference-present development boxes, seven neurons; per-box only'}
    report = {'status': 'completed', 'stage': 'development', 'config_fingerprint': fingerprint(c),
        'source_config_fingerprint': fingerprint(old), 'source_interrupted_root': c['interrupted_root'],
        'measurement': measurement, 'input_support_existing_evidence': original.existing_input_evidence(old),
        'counts': original.e.read(ROOT/c['interrupted_root']/'all_inference_complete.json')['counts'],
        'boxes': 151, 'neurons': 7, 'GPU_executed': False, 'training_updates': 0, 'model_executed': False,
        'model_predictions_reused': True, 'graph_construction_executed': False, 'gate_changed': False,
        'sixteen_prepared': False, 'fit_metric_parity_exact': True, 'seconds_this_scoring_invocation': time.perf_counter()-began}
    if 'torch' in sys.modules:
        raise RuntimeError('Scoring-only continuation imported Torch')
    if not (output/'report.json').exists():
        write_once(output/'report.json', report)
        (output/'report.md').write_text(original.readable(report), encoding='utf-8')
    print('SCORING COMPLETE '+json.dumps({'report': str(output/'report.json'), 'automatic_next_rung': False}), flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Explicitly reuse this continuation output and completed score records')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    c, _, _, _, _ = control()
    if args.check:
        print('453 saved graph/SWC pairs and 151 predictions verified; no scoring/model/graph execution'); return
    if args.output and args.resume:
        parser.error('Use only one of --output or --resume')
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = original.e.output_path(args.resume or args.output or ROOT/'tmp'/f'curve_transfer_scoring_{suffix}')
    if args.worker:
        worker(output); return
    if output.exists() and not args.resume:
        raise FileExistsError('Use --resume explicitly to reuse completed scoring records')
    if args.resume and not (output/'binding.json').exists():
        raise ValueError('No bound scoring continuation to resume')
    import psutil
    logs = output.with_name(output.name+'_guard_'+suffix)
    logs.mkdir(exist_ok=False)
    print(f'CPU SCORING ONLY; OUTPUT {output}; cap {c["wall_seconds"]} seconds; no model/graph rebuild', flush=True)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--output', str(output)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        def relay():
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        thread = threading.Thread(target=relay, daemon=True); thread.start()
        start_wall, start_mono = time.time(), time.perf_counter()
        interrupted = False
        try:
            receipt = wait_wall(process, c['wall_seconds'], lambda: terminate_owned_tree(owner))
        except BaseException as error:
            terminate_owned_tree(owner); process.wait(timeout=5)
            receipt = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'guard_error',
                       'exit_code': process.returncode, 'error': type(error).__name__,
                       'elapsed_seconds': max(time.time()-start_wall, time.perf_counter()-start_mono)}
            interrupted = True
        finally:
            thread.join(timeout=10)
        receipt.update(output=str(output), config_fingerprint=fingerprint(c), maximum_seconds=c['wall_seconds'])
        write_once(logs/'report.json', receipt)
    if thread.is_alive() or interrupted or receipt['status'] != 'completed':
        raise SystemExit(f'Scoring {receipt["status"]}; complete score records preserved; no automatic retry')


if __name__ == '__main__':
    main()
