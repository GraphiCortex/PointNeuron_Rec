"""User-launched CPU-only A/B/C measurement on immutable update-2000 votes.

No Torch import, model forward, training, filtering, new classical extraction,
GPU run, or automatic sixteen-box launch. D/E are separate read-only audits.
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
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import local_curve_experiment as e
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.local_curve import consolidate, summarize, reference_geometry
from pn2.evaluation.formulation import (graph_segments, clip_segments, segment_agreement,
                                       curve_samples, runs_graph)
from pn2.evaluation.vote_scaffold import (neighbor_counts, support_histogram, isolation_decision,
    supported_consolidation, knn_edges, image_edge_costs, minimum_spanning_forest, guide_at, export_forest)
from pn2.evaluation.swc import parse_swc
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree

ROOT = e.ROOT
CONFIG = ROOT/'configs/saved_vote_scaffold.json'
SOURCES = ('scripts/measure_saved_vote_scaffold.py', 'src/pn2/evaluation/vote_scaffold.py',
           'tests/test_vote_scaffold.py', 'docs/saved_vote_scaffold_protocol_20261104.md',
           'scripts/run_supported_vote_sixteen.py')


def control():
    c = e.read(CONFIG)
    expected = {'schema': 'pn2-saved-vote-scaffold-v1', 'support_cutoff': .25,
        'support_radius_voxel': 2., 'isolated_support_below': 3,
        'isolated_bad_fraction_minimum': .5, 'far_source_distance_voxel': 6.,
        'far_bad_fraction_stop': .5, 'variant_radius_voxel': 2., 'variant_minimum_support': 3,
        'candidate_k_diagnostics': [4, 6, 10], 'decoding_k': 6,
        'line_sample_step_voxel': .5, 'image_cost_penalty': 4.,
        'quadrature_step_voxel': .5, 'tolerances_voxel': [2., 6.], 'wall_seconds': 1800}
    if any(c.get(k) != v for k, v in expected.items()) or set(c['source_sha256']) != set(SOURCES):
        raise ValueError('Unsupported/unbound scaffold recipe')
    for name, sha in {**c['source_sha256'], **c['artifact_sha256']}.items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Frozen measurement source/evidence changed: {name}')
    return c


def combine_histograms(rows, name):
    result = {}
    for kind in ('bins', 'exact'):
        labels = sorted({k for row in rows for k in row[name][kind]}, key=lambda s: int(s.split('-')[0].rstrip('+')))
        result[kind] = {}
        for label in labels:
            n = good = 0; neurons = []
            for cid in sorted({r['case_id'] for r in rows}):
                group = [r[name][kind].get(label, {}) for r in rows if r['case_id'] == cid]
                nn = sum(r.get('retained', 0) for r in group)
                gg = sum(r.get('within_2', 0) for r in group)
                n += nn; good += gg
                if nn:
                    neurons.append(gg/nn)
            result[kind][label] = {'retained': n, 'within_2': good, 'beyond_2': n-good,
                'pooled_precision': good/n if n else None,
                'neuron_macro_precision': float(np.mean(neurons)) if neurons else None,
                'neurons_with_bin': len(neurons)}
    return result


def branch_change(old_points, new_points, reference):
    rows = []; lost = gained = total = 0.
    for i, run in enumerate(reference['reference_runs']):
        p, w = curve_samples([run])
        old = cKDTree(old_points).query(p)[0] if len(old_points) else np.full(len(p), np.inf)
        new = cKDTree(new_points).query(p)[0] if len(new_points) else np.full(len(p), np.inf)
        a, b = old <= 2., new <= 2.
        length = float(w.sum()); ll = float(w[a & ~b].sum()); gg = float(w[~a & b].sum())
        rows.append({'clipped_reference_run': i, 'length': length, 'lost_covered_length': ll,
                     'gained_covered_length': gg, 'original_covered': float(w[a].sum()),
                     'variant_covered': float(w[b].sum())})
        lost += ll; gained += gg; total += length
    return {'curve_length': total, 'lost_covered_length': lost, 'gained_covered_length': gained,
            'per_clipped_run': rows, 'meaning': 'spatial support loss, not certified missing biological branches'}


def baseline_graphs(data, keys):
    expected = e.read(e.BASE/'prepare/report.json')['classical_packet_hashes']
    result = {}
    wanted = {cid for cid, _ in keys}
    for path in sorted((ROOT/'tmp/curve_classical_baseline_20261103').glob('case*/report.json')):
        receipt = e.read(path); cid = receipt['case_id']
        if cid not in wanted:
            continue
        packet = path.parent/'graphs.npz'
        if digest(packet) != expected[cid]:
            raise ValueError('Preserved classical graph changed')
        with np.load(packet, allow_pickle=False) as a:
            result[cid] = SimpleNamespace(xyz=a['pruned_xyz'].copy(), edges=a['pruned_edges'].copy())
    if set(result) != wanted:
        raise ValueError('Missing previously extracted training classical control')
    return result


def score_point_rows(scientific, data, refs, records, policy):
    from pn2.evaluation.node_profile_metrics import ProfileCache
    profiles = ProfileCache(ROOT, ROOT/scientific['profile_root'], data)
    result = []
    for record in records:
        cid, tile = record['key']; keep = record[policy]; points = record['points'][keep]
        reference = data.reference(cid, tile)
        row = e.score(data, refs, (cid, tile), points)
        row['thin_dim'] = e.thin_metrics(points, profiles.load(cid, tile), profiles.manifest['cutoffs'])
        row['incident'] = e.incident_metrics(points, e.incident_probes(refs.get(cid)['morphology']),
                                             data.boxes[(cid, tile)]['box'])
        row['longest_uncovered_length_2'] = e.gap_metrics(points, reference)
        row['branch_change'] = branch_change(record['points'][record['original']], points, reference)
        result.append(row)
    return {'rows': result, 'summary': summarize(result)}


def segment_summary(rows):
    neurons = []
    for cid in sorted({r['case_id'] for r in rows}):
        group = [r['segments'] for r in rows if r['case_id'] == cid]
        rl = sum(r['reference_length'] for r in group); pl = sum(r['predicted_length'] for r in group)
        values = {}
        for t in ('2.0', '6.0'):
            pc = sum(r['tolerances'][t]['supported_prediction_length'] for r in group)
            rc = sum(r['tolerances'][t]['supported_reference_length'] for r in group)
            precision, recall = pc/pl if pl else 0., rc/rl if rl else 0.
            values[t] = {'precision': precision, 'recall': recall,
                         'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.}
        neurons.append({'case_id': cid, 'reference_length': rl, 'predicted_length': pl, 'tolerances': values})
    return {'neurons': neurons, 'macro': {t: {m: float(np.mean([r['tolerances'][t][m] for r in neurons]))
            for m in ('precision', 'recall', 'f1')} for t in ('2.0', '6.0')}}


def build_graph_artifacts(image_records, guides, folder):
    """Image-only C inference phase: no references/labels accepted by this API."""
    folder.mkdir()
    packets = []
    maxima = {cid: float(guide.max()) for cid, guide in guides.items()}
    if any(not np.isfinite(v) for v in maxima.values()):
        raise ValueError('Nonfinite cached guide')
    for policy in sorted({r['policy'] for r in image_records}):
        group = [r for r in image_records if r['policy'] == policy]
        for i, r in enumerate(group):
            cid, tile = r['key']; points = r['points']
            out = folder/f'{policy}_box{i}'
            out.mkdir()
            candidates = {k: knn_edges(points, k) for k in (4, 6, 10)}
            costs, quality = image_edge_costs(points, candidates[6], guides[cid], maxima[cid])
            forest, components = minimum_spanning_forest(points, candidates[6], costs)
            node_guide = guide_at(guides[cid], points, maxima[cid])
            receipt = export_forest(out/'scaffold.swc', points, forest, node_guide)
            reparsed = parse_swc(out/'scaffold.swc')
            if (not np.array_equal(reparsed.xyz, points)
                    or set(map(tuple, np.sort(reparsed.edges, axis=1))) != set(map(tuple, forest))):
                raise ValueError('SWC round-trip changed proposal geometry/edges')
            np.savez_compressed(out/'graph.npz', xyz=points, forest_edges=forest,
                candidate4=candidates[4], candidate6=candidates[6], candidate10=candidates[10],
                candidate6_cost=costs, candidate6_mean_guide=quality, node_guide=node_guide)
            receipt.update(policy=policy, case_id=cid, tile_id=tile, decoding_k=6,
                candidate_components=components, guide_normalization_max=maxima[cid],
                graph_sha256=digest(out/'graph.npz'), swc_sha256=digest(out/'scaffold.swc'),
                reference_edges_used=False, reference_roots_used=False,
                proposal_coordinates_moved=False, image_cost='length*(1+4*(1-mean_normalized_Sato))')
            write_once(out/'inference.json', receipt)
            packets.append({'policy': policy, 'key': (cid, tile), 'folder': str(out),
                            'inference': receipt})
    write_once(folder/'inference_complete.json', {'packets': packets, 'C_reference_scoring_started': False,
               'references_passed_to_graph_builder': False, 'model_executed': False})
    return packets


def score_graph_artifacts(packets, data, classical):
    # C reference access starts only after all graph/SWC packets have been saved.
    rows = []; classical_rows = []
    for packet in packets:
        cid, tile = packet['key']; reference = data.reference(cid, tile)
        rs = graph_segments(runs_graph(reference['reference_runs']))
        box = data.boxes[(cid, tile)]['box']; lo, hi = np.array(box['start_xyz'])-.5, np.array(box['stop_xyz'])-.5
        with np.load(Path(packet['folder'])/'graph.npz', allow_pickle=False) as a:
            xyz = a['xyz']; candidate_scores = {}
            for k in (4, 6, 10):
                seg = graph_segments(SimpleNamespace(xyz=xyz, edges=a[f'candidate{k}']))
                candidate_scores[str(k)] = segment_agreement(rs, clip_segments(seg, lo, hi))
            exported = parse_swc(Path(packet['folder'])/'scaffold.swc')
            decoded = segment_agreement(rs, clip_segments(graph_segments(exported), lo, hi))
            if decoded['tolerances']['2.0']['recall'] > candidate_scores['6']['tolerances']['2.0']['recall']+1e-12:
                raise ValueError('Decoded geometry exceeded its candidate-union ceiling')
        rows.append({'policy': packet['policy'], 'case_id': cid, 'tile_id': tile,
                     'segments': decoded, 'candidate_unions': candidate_scores, 'inference': packet['inference']})
        if packet['policy'] == 'original':
            clipped = clip_segments(graph_segments(classical[cid]), lo, hi)
            classical_rows.append({'case_id': cid, 'tile_id': tile, 'segments': segment_agreement(rs, clipped)})
    summaries = {}
    for policy in sorted({r['policy'] for r in rows}):
        group = [r for r in rows if r['policy'] == policy]
        summaries[policy] = {'decoded': segment_summary(group), 'candidate_unions': {
            str(k): segment_summary([{'case_id': r['case_id'], 'segments': r['candidate_unions'][str(k)]}
                                    for r in group]) for k in (4, 6, 10)}}
    return {'rows': rows, 'summaries': summaries, 'classical_rows': classical_rows,
            'classical_summary': segment_summary(classical_rows), 'scope': 'four-box fit-level, per-box reconstruction only',
            'candidate_union_selects_k': False, 'product_GATv2_replaced': False,
            'topology_certified': False, 'whole_image_merge': False}


def render_report(folder, a, b, c):
    lines = ['# Saved update-2000 vote diagnostics and graph scaffold', '',
        'Post-failure CPU measurement; frozen original gate remains failed. Four TRAINING boxes, two neurons.', '',
        '## A. Retained-point support', '',
        '| Accepted raw votes within 2, including self | Retained | Beyond 2 | Pooled precision | Neuron-macro precision |',
        '|---|---:|---:|---:|---:|']
    pct = lambda v: f'{100*v:.2f}%' if v is not None else 'not present'
    for name, r in a['accepted_raw_support']['bins'].items():
        lines.append(f'| {name} | {r["retained"]} | {r["beyond_2"]} | {pct(r["pooled_precision"])} | {pct(r["neuron_macro_precision"])} |')
    lines += ['', 'Source-distance bands of off-reference retained positions (pooled counts):', '']
    for name, r in a['source_band_pooled'].items():
        lines.append(f'- {name}: {r["beyond_2"]}/{r["accepted"]} accepted positions beyond 2.')
    lines += ['', 'Sato and classical-node point controls (same point scorer; policies/counts differ):', '',
              '| Control | Macro point agreement | Macro curve-to-point coverage |', '|---|---:|---:|']
    for name, r in a['cached_point_controls'].items():
        lines.append(f'| {name} | {pct(r["macro"]["precision_2"])} | {pct(r["macro"]["coverage_2"])} |')
    lines += ['', 'Raw classical graph nodes (no consolidation): '+json.dumps(a['raw_classical_points']['summary']['macro']),
              '', '## B. Fixed conditional variant', '', 'Trigger: '+json.dumps(b['decision']), '',
              'Variant status: '+b['status']+'. Original gate and oracle denominators unchanged.', '']
    if b.get('variant'):
        lines += ['Post-failure variant gate: '+json.dumps(b['variant']['gate']), '',
                  'See report.json for per-run lost/gained support, roles, incident probes, thin/dim support and gaps.', '']
    lines += ['## C. Non-learned graph measurement', '',
        'k=6 was fixed before results. k=4/6/10 union coverage is diagnostic only. Root/radius are placeholders.', '',
        '| Policy | Segment P2 | Segment R2 | Segment F1 at 2 | Candidate R2: k4 / k6 / k10 |',
        '|---|---:|---:|---:|---|']
    for name, r in c['summaries'].items():
        m = r['decoded']['macro']['2.0']
        union = ' / '.join(pct(r['candidate_unions'][str(k)]['macro']['2.0']['recall']) for k in (4, 6, 10))
        lines.append(f'| {name} | {pct(m["precision"])} | {pct(m["recall"])} | {pct(m["f1"])} | {union} |')
    m = c['classical_summary']['macro']['2.0']
    lines.append(f'| preserved classical graph | {pct(m["precision"])} | {pct(m["recall"])} | {pct(m["f1"])} | not this candidate graph |')
    lines += ['', 'This scaffold does not replace GATv2. SWC validity/curve geometry do not certify connectivity.', '',
              '## D/E. Source audit and proposed criterion', '',
              f'See {ROOT / "docs/proposal_implementation_audit_20261104.md"}. No new node-stage criterion is adopted.', '',
              '## What executed', '', 'Saved-array reads, cached guides/graphs, one conditional consolidation at most, CPU kNN/MSF/SWC/segment scoring.',
              'No model execution, training, GPU, new filtering, classical extraction, reserved data or whole-image merging.', '']
    if b.get('next_manifest'):
        lines += ['The post-hoc variant passed the original numerical gate; sixteen-box manifest/command prepared, not launched.',
                  '', b['next_command'], '']
    else:
        lines += ['Stop this branch of the experiment. One formulation hypothesis for discussion: the dense source-affinity/offset task',
            'does not directly train a compact curve representation. A curve-supervised Transformer that selects/refines representative',
            'anchors is a candidate hypothesis, retaining a meaningful proposal role. It is not implemented or authorized by this report.', '']
    (folder/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def worker(c, folder):
    began = time.perf_counter(); folder.mkdir(parents=True, exist_ok=False)
    scientific = e.control(); data, manifest, selection = e.prepared(scientific)
    source = ROOT/c['terminal_root']; metrics = e.read(source/'eval_step002000_training/metrics.json')
    if metrics['step'] != 2000 or metrics['keys'] != selection['four']:
        raise ValueError('Unexpected terminal boxes/cursor')
    refs = e.References(scientific, data); records = []; diagnostic_rows = []
    classical = baseline_graphs(data, selection['four']); guides = {}; raw_classical = []
    with np.load(source/'eval_step002000_training/predictions.npz', allow_pickle=False) as a:
        for i, key in enumerate(selection['four']):
            key = tuple(key)  # JSON identities are lists; cache helpers use tuple dictionary keys.
            cid, tile = key; image = data.image(cid, tile, 2048); points = a[f'box{i}_points']; scores = a[f'box{i}_scores']
            original = a[f'box{i}_ranked_indices']; original = original[scores[original] >= .25]
            raw = np.flatnonzero(scores >= .25); ids = image['packet']['flat_indices']
            assert np.array_equal(original, consolidate(points, scores, cutoff=.25, identities=ids))
            counts = neighbor_counts(points[raw], points[original]); all_counts = neighbor_counts(points, points[original])
            distance = refs.get(cid)['distances'](points[original]); near = distance <= 2.
            labels = e.label_arrays(manifest, key); source_distance = labels['source_distance'][original]
            bands = {'<2': source_distance < 2, '2-4': (source_distance >= 2) & (source_distance < 4),
                     '4-6': (source_distance >= 4) & (source_distance <= 6), '>6': source_distance > 6}
            source_bands = {name: {'accepted': int(mask.sum()), 'beyond_2': int((mask & ~near).sum())}
                            for name, mask in bands.items()}
            diagnostic_rows.append({'case_id': cid, 'tile_id': tile,
                'accepted_support': support_histogram(counts, near), 'all_support': support_histogram(all_counts, near),
                'source_bands': source_bands})
            records.append({'key': key, 'points': points.copy(), 'scores': scores.copy(), 'ids': ids,
                'original': original, 'support_counts': counts, 'near': near, 'source_distance': source_distance})
            np.savez_compressed(folder/f'A_box{i}.npz', retained_indices=original, accepted_raw_counts=counts,
                                all_raw_counts=all_counts, within_2=near, source_distance=source_distance)
            e.sato_scores(data, key, image['packet'], guides)
            box = data.boxes[tuple(key)]['box']; p = classical[cid].xyz
            take = np.all((p >= np.array(box['start_xyz'])-.5) & (p <= np.array(box['stop_xyz'])-.5), axis=1)
            raw_classical.append(e.score(data, refs, key, p[take]))
    counts = np.concatenate([r['support_counts'] for r in records]); near = np.concatenate([r['near'] for r in records])
    source_distance = np.concatenate([r['source_distance'] for r in records])
    if int((~near).sum()) != 369 or sum(len(r['original']) for r in records) != 1578:
        raise ValueError('Terminal retained/error denominators differ from preserved evidence')
    cached = e.read(e.BASE/'prepare/report.json')['summaries']['four']
    a_report = {'scope': 'four-box fit-level saved update-2000', 'rows': diagnostic_rows,
        'accepted_raw_support': combine_histograms(diagnostic_rows, 'accepted_support'),
        'unthresholded_raw_support_diagnostic_only': combine_histograms(diagnostic_rows, 'all_support'),
        'source_band_pooled': {name: {v: sum(r['source_bands'][name][v] for r in diagnostic_rows)
                                    for v in ('accepted', 'beyond_2')} for name in ('<2', '2-4', '4-6', '>6')},
        'cached_point_controls': {name: cached[name] for name in ('sato', 'sato512', 'classical', 'classical512')},
        'raw_classical_points': {'rows': raw_classical, 'summary': summarize(raw_classical)}}
    write_once(folder/'A_diagnostics.json', a_report)
    print('A COMPLETE '+json.dumps(a_report['accepted_raw_support']['bins']), flush=True)
    decision = isolation_decision(counts, near, source_distance)
    write_once(folder/'B_predeclared_decision.json', decision)
    b_report = {'decision': decision, 'status': 'not_eligible', 'variant': None,
                'post_failure_variant': True, 'original_frozen_gate_unchanged': True}
    policies = ['original']
    if decision['eligible']:
        for record in records:
            record['supported_r2_min3'] = supported_consolidation(record['points'], record['scores'], record['ids'])
        variant = score_point_rows(scientific, data, refs, records, 'supported_r2_min3')
        variant['gate'] = e.four_gate(scientific, variant)
        b_report.update(status='variant_passed' if variant['gate']['passed'] else 'variant_failed', variant=variant)
        policies.append('supported_r2_min3')
        if variant['gate']['passed']:
            next_manifest = {'schema': 'pn2-supported-vote-sixteen-v1', 'scientific_config_fingerprint': fingerprint(scientific),
                'scaffold_config_fingerprint': fingerprint(c), 'four_box_posthoc_gate': variant['gate'],
                'consolidation': {'support_cutoff': .25, 'support_radius': 2., 'minimum_support': 3, 'suppression_radius': 2.},
                'sixteen': selection['sixteen'], 'development': selection['development'],
                'updates': 4000, 'wall_seconds': 14400, 'automatic_launch': False,
                'comparison_scope': 'dense candidate first; paired Transformer control still required before a comparative claim',
                'preparation_manifest_sha256': digest(e.BASE/'prepare/manifest.json')}
            write_once(folder/'sixteen_manifest.json', next_manifest)
            command = f"& '..\\pointneuron2\\tmp\\pointneuron2\\sampling_env\\Scripts\\python.exe' -B '.\\scripts\\run_supported_vote_sixteen.py' --manifest '{folder/'sixteen_manifest.json'}'"
            (folder/'run_sixteen.ps1').write_text(f"Set-Location '{ROOT}'\n{command}\n", encoding='utf-8')
            b_report.update(next_manifest=str(folder/'sixteen_manifest.json'), next_command=command)
    write_once(folder/'B_result.json', b_report)
    print('B COMPLETE '+json.dumps({'status': b_report['status'], 'decision': decision,
                                   'gate': b_report['variant']['gate'] if b_report['variant'] else None}), flush=True)
    # Remove A/B labels and all reference-bearing metadata at the C inference
    # boundary. Only selected saved coordinates, identity tags and guides enter.
    image_records = [{'policy': policy, 'key': r['key'], 'points': r['points'][r[policy]].copy()}
                     for policy in policies for r in records]
    packets = build_graph_artifacts(image_records, guides, folder/'C_graphs')
    c_report = score_graph_artifacts(packets, data, classical)
    write_once(folder/'C_measurement.json', c_report)
    report = {'status': 'completed', 'config_fingerprint': fingerprint(c), 'A': a_report, 'B': b_report,
        'C': c_report, 'seconds': time.perf_counter()-began, 'GPU_executed': False,
        'model_executed': False, 'training_executed': False, 'reference_edges_used_in_inference': False,
        'frozen_scientific_config_fingerprint': fingerprint(scientific)}
    write_once(folder/'report.json', report)
    render_report(folder, a_report, b_report, c_report)
    if 'torch' in sys.modules:
        raise RuntimeError('Unexpected Torch import in CPU-only saved-artifact measurement')
    print('CPU MEASUREMENT COMPLETE '+json.dumps({'report': str(folder/'report.json'),
        'readable_report': str(folder/'report.md'), 'B_status': b_report['status'],
        'seconds': report['seconds'], 'automatic_training_launch': False}), flush=True)


def main():
    # The module's control validator is also used by the separately authorized
    # future trainer. Hide CUDA only when this CPU measurement entrypoint runs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--check', action='store_true', help='Read-only source/artifact binding check')
    args = parser.parse_args(); c = control()
    if args.check:
        print('CPU measurement sources and saved-artifact bindings verified; no model or measurement executed')
        return
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = e.output_path(args.output or ROOT/'tmp'/f'saved_vote_scaffold_{suffix}')
    if args.worker:
        worker(c, folder); return
    import psutil
    if folder.exists():
        raise FileExistsError('Preserve existing measurement artifacts')
    logs = folder.with_name(folder.name+'_wall_guard'); logs.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--output', str(folder)]
    print(f'CPU ONLY; OUTPUT {folder}; wall cap 1800 seconds', flush=True)
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
            receipt = wait_wall(process, c['wall_seconds'], lambda: terminate_owned_tree(owner))
        except BaseException:
            terminate_owned_tree(owner); process.wait(timeout=5); raise
        finally:
            reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError('Worker pipe did not close')
    receipt.update(config_fingerprint=fingerprint(c), output=str(folder))
    write_once(logs/'report.json', receipt)
    if receipt['status'] != 'completed':
        raise SystemExit(f'CPU measurement {receipt["status"]}; no automatic retry')


if __name__ == '__main__':
    main()
