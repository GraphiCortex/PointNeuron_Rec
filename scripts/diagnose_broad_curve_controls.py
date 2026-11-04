"""Authorized CPU-only matched classical comparison and frozen-vote ablations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import numpy as np

import measure_curve_transfer as old
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.local_curve import consolidate
from pn2.evaluation.formulation import graph_segments, runs_graph
from pn2.evaluation.fixed_curve_graph import build, score, summarize
from pn2.training.guarded_run import run

ROOT = old.ROOT
CONFIG = ROOT/'configs/broad_curve_diagnostics_20261105.json'


def control():
    c = old.e.read(CONFIG)
    if c['schema'] != 'pn2-broad-curve-diagnostics-v1' or c['wall_seconds'] != 1800:
        raise ValueError('Changed CPU diagnostic scope')
    for name, sha in {**c['source_sha256'], **c['artifact_sha256']}.items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Changed diagnostic source/evidence: {name}')
    original, scientific, data, manifest, selection, dev = old.control()
    return c, original, scientific, data, dev


def matched_controls(c, data, dev):
    classical = old.e.read(ROOT/c['classical_report'])
    audit = old.e.read(ROOT/'configs/curve_formulation_audit.json')
    if audit['data_manifest_sha256'] != digest(data.folder/'manifest.json'):
        raise ValueError('Classical grid differs from current data')
    if (audit['support_radii_voxel'] != [2., 6., 8.] or audit['quadrature_step_voxel'] != .5
            or classical['config_fingerprint'] != fingerprint(audit)):
        raise ValueError('Classical scorer/protocol mismatch')
    for name in ['src/pn2/evaluation/formulation.py', 'src/pn2/evaluation/geometry.py']:
        if name in audit['source_sha256'] and digest(ROOT/name) != audit['source_sha256'][name]:
            raise ValueError('Classical geometric scorer changed')
    transfer = old.e.read(ROOT/c['transfer_report'])
    controls = {k: transfer['measurement']['summaries'][k]['decoded'] for k in ['raw_input', 'sato512']}
    rows = []
    for variant in ['raw', 'pruned']:
        group = [{'policy': 'classical_'+variant, 'case_id': arm['case_id'], 'tile_id': b['tile_id'],
                  'segments': b['variants'][variant]['segments']}
                 for arm in classical['arms'] if arm['partition'] == 'development'
                 for b in arm['boxes'] if b['reference_present']]
        if len(group) != 151 or {(r['case_id'], r['tile_id']) for r in group} != set(dev):
            raise ValueError('Classical reference-present scoring boxes differ')
        rows.extend(group); controls['classical_'+variant] = summarize(group)
    return {'controls': controls, 'classical_rows': rows,
        'historical_full_grid': classical['summary']['partitions']['development'],
        'domain': 'same151 reference-present; cached identical per-box scores, reaggregated only',
        'classical_extraction_or_metric_audit_repeated': False}


def ablation_sets(source, votes, support, sato, ids):
    if np.asarray(source).shape != np.asarray(votes).shape or np.asarray(support).shape != (len(source),):
        raise ValueError('Unaligned saved votes/source points')
    support_ids = consolidate(source, support, radius=1., identities=ids)[:512]
    sato_ids = consolidate(source, sato, radius=1., identities=ids)[:512]
    return {'raw_votes_all': (votes, np.arange(len(votes))),
            'support512_no_offset': (source[support_ids], support_ids),
            'sato512_with_offset': (votes[sato_ids], sato_ids)}, sato_ids


def worker(output):
    c, original, scientific, data, dev = control()
    output.mkdir(exist_ok=False)
    comparison = matched_controls(c, data, dev)
    write_once(output/'classical_matched.json', comparison)
    print('MATCHED CLASSICAL '+json.dumps(comparison['controls']['classical_pruned']['macro']), flush=True)
    source = ROOT/c['prediction_root']
    receipt = old.e.read(source/'predictions/complete.json')
    if [tuple(r['key']) for r in receipt['rows']] != dev or receipt['config_fingerprint'] != fingerprint(original):
        raise ValueError('Saved inference domain/source mismatch')
    guides, packets = {}, []
    for i, key in enumerate(dev):
        image = data.image(*key, 2048)['packet']
        path = source/'predictions'/receipt['rows'][i]['packet']
        if digest(path) != receipt['rows'][i]['sha256']:
            raise ValueError('Saved CPU prediction changed')
        with np.load(path, allow_pickle=False) as a:
            votes, support = a['points'], a['scores']
        sato = old.e.sato_scores(data, key, image, guides)
        arms, selected = ablation_sets(image['native_xyz'], votes, support, sato, image['flat_indices'])
        # The offset contrast must preserve the original source set exactly.
        with np.load(source/'graphs'/f'sato512_box{i}'/'graph.npz', allow_pickle=False) as a:
            if not np.array_equal(a['xyz'], image['native_xyz'][selected]):
                raise ValueError('Sato source identities differ from existing control')
        maximum = float(guides[key[0]].max())
        for policy, (points, indices) in arms.items():
            packets.append(build(points, guides[key[0]], maximum, output/'graphs'/f'{policy}_{i:04d}', key, policy, indices))
        print(f'CPU ABLATION GRAPH {i+1}/151', flush=True)
    write_once(output/'inference_complete.json', {'packets': packets, 'references_used_in_inference': False})
    (output/'records').mkdir()
    rows = []
    for i, packet in enumerate(packets):
        key = tuple(packet['key'])
        reference = graph_segments(runs_graph(data.reference(*key)['reference_runs']))
        row = score(packet, reference, data.boxes[key]['box'])
        write_once(output/'records'/f'graph{i:04d}.json', row)
        rows.append(row)
        if (i+1) % 3 == 0:
            print(f'CPU ABLATION SCORE {i+1}/453', flush=True)
    summaries = {policy: summarize([r for r in rows if r['policy'] == policy]) for policy in sorted({r['policy'] for r in rows})}
    report = {'status': 'completed', 'config_fingerprint': fingerprint(c), 'comparison': comparison,
        'ablation_rows': rows, 'ablations': summaries, 'model_executed': False, 'GPU_executed': False,
        'parameters_selected': False, 'training_recipe_changed': False, 'training_updates': 0}
    write_once(output/'report.json', report)
    lines = ['# Broad-run CPU diagnostics (select nothing)', '', '| Policy | F1 at 2 | F1 at 6 |', '|---|---:|---:|']
    for policy, s in {**comparison['controls'], **summaries}.items():
        lines.append(f'| {policy} | {100*s["macro"]["2.0"]["f1"]:.2f}% | {100*s["macro"]["6.0"]["f1"]:.2f}% |')
    lines += ['', 'Classical scores reuse the identical cached per-box scorer/domain; unknown boxes excluded.',
              'Raw votes have no cutoff/suppression. Support-only source points have no offset/cutoff.',
              'Sato+offset retains exactly the original Sato source IDs, no second suppression.',
              'No diagnostic arm is adopted and none is a no-learning comparator.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('CPU DIAGNOSTICS COMPLETE '+json.dumps({'output': str(output), 'ablations': {k:v['macro'] for k,v in summaries.items()}}), flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output', type=Path);p.add_argument('--worker', action='store_true');p.add_argument('--check', action='store_true')
    a = p.parse_args();c, _, _, _, _ = control()
    if a.check:
        print('CPU diagnostic source/input bindings verified; no execution');return
    output = old.e.output_path(a.output or ROOT/'tmp'/('broad_curve_diagnostics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    if a.worker:
        worker(output);return
    if output.exists():raise FileExistsError('Preserve diagnostic evidence')
    print(f'CPU DIAGNOSTICS OUTPUT {output}; cap1800s', flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(output)],
        output.with_name(output.name+'_guard'),1800,{'output':str(output),'config_fingerprint':fingerprint(c)})


if __name__ == '__main__':main()
