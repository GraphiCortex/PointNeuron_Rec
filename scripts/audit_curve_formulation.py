"""CPU-only cached-cloud/prediction audit and automatic classical reference.

No model imports, training, guide filtering, target rewriting or GT-derived
predicted edges. Classical graphs are mechanical image-only voxel graphs.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.evaluation.node_profile_metrics import ProfileCache, STRATA
from pn2.evaluation.node_profiles import classify
from pn2.evaluation.formulation import (curve_samples, runs_graph, graph_segments, support,
    point_diagnostics, segment_agreement, resample_runs, clip_segments)
from pn2.training.process_control import launch_worker

CONFIG = ROOT/'configs/curve_formulation_audit.json'


def load_control(path):
    c = json.loads(Path(path).read_text())
    if (c['schema'] != 'pn2-curve-formulation-audit-v1' or c['budgets'] != [2048, 8192]
            or c['support_radii_voxel'] != [2., 6., 8.] or c['quadrature_step_voxel'] != .5
            or c['prediction_cutoff'] != .5 or c['baseline_training_aliases'] != [22, 32, 75, 85, 126]
            or c['baseline_timing_alias'] != 41 or c['baseline_recipe'] != {
                'threshold': 'whole_guide_otsu_including_zeros', 'histogram_bins': 256,
                'skeleton': 'lee_3d', 'adjacency': 26, 'prune_terminal_below_voxel': 2., 'prune_passes': 1}):
        raise ValueError('Only the declared read-only audit and fixed classical recipe are supported')
    for name, sha in c['source_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError(f'Audit source changed: {name}')
    for name, version in c['runtime_versions'].items():
        if importlib.metadata.version(name) != version: raise ValueError(f'CPU runtime changed: {name}')
    data = NodeBudgetDataset(ROOT, ROOT/c['data_root'])
    if digest(data.folder/'manifest.json') != c['data_manifest_sha256']:
        raise ValueError('Frozen cohort changed')
    profiles = ProfileCache(ROOT, ROOT/c['profile_root'], data)
    if digest(profiles.folder/'manifest.json') != c['profile_manifest_sha256']:
        raise ValueError('Fixed profile labels changed')
    return c, data, profiles


def saved_sources(c, data):
    result = []
    for spec in c['predictions']:
        folder = ROOT/spec['folder']
        if digest(folder/'metrics.json') != spec['metrics_sha256'] or digest(folder/'predictions.npz') != spec['predictions_sha256']:
            raise ValueError('Saved model outputs changed')
        raw = json.loads((folder/'metrics.json').read_text())
        ids = raw['case_order']; rows = raw['boxes']
        expected = {key for key in data.boxes if data.cases[key[0]]['partition'] == spec['partition']}
        observed = [(r['case_id'], r['tile_id']) for r in rows]
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError('Saved inference did not cover the declared full image grid')
        if any(row['target_count'] != data.boxes[key]['target']['count'] for row, key in zip(rows, observed)):
            raise ValueError('Saved target denominators differ from shared references')
        packets = np.load(folder/'predictions.npz', allow_pickle=False)
        result.append({'spec': spec, 'rows': rows, 'indices': {key: i for i, key in enumerate(observed)},
                       'case_order': ids, 'packets': packets})
    return result


def summarize_audit(rows, data):
    def ratio(a, b):return a/b if b else None
    def mean(values):
        values = [x for x in values if x is not None]
        return float(np.mean(values)) if values else None
    neurons = []
    for cid in sorted(data.cases):
        group = [r for r in rows if r['case_id'] == cid]
        row = {'case_id': cid, 'partition': data.cases[cid]['partition'],
               'source_group': data.cases[cid]['source_group'], 'alias': data.cases[cid]['alias'],
               'reference_present_boxes': len(group), 'input': {}, 'predictions': {}}
        for budget in ('2048', '8192'):
            values = [r['input'][budget] for r in group]
            length = sum(v['curve_length'] for v in values)
            row['input'][budget] = {'curve_length': length,
                'support': {str(t): ratio(sum(v['supported_length'][str(t)] for v in values), length) for t in (2., 6., 8.)},
                'profile_strata': {name: {str(t): ratio(sum(v['strata'][name]['covered'][str(t)] for v in values),
                    sum(v['strata'][name]['length'] for v in values)) for t in (2., 6., 8.)} for name in STRATA},
                'nearest_sample_intensity_mean': ratio(sum((v['nearest_point_normalized_intensity_weighted_mean'] or 0.)*v['curve_length'] for v in values), length)}
        methods = sorted({m for r in group for m in r['predictions']})
        for method in methods:
            row['predictions'][method] = {}
            for selection in ('all', 'accepted'):
                values = [r['predictions'][method][selection] for r in group]
                n = sum(v['points'] for v in values);length = sum(v['curve_length'] for v in values)
                row['predictions'][method][selection] = {
                    'points_in_reference_present_boxes': n,
                    'on_reference_geometry_fraction_2': ratio(sum(v['point_near_curve']['2.0'] for v in values), n),
                    'near_reference_grid_fraction_2': ratio(sum(v['point_near_grid_2'] for v in values), n),
                    'one_to_one_node_precision_2': ratio(sum(v['unique_node_matches_2'] for v in values), n),
                    'one_to_one_node_recall_2': ratio(sum(v['unique_node_matches_2'] for v in values), sum(v['reference_nodes'] for v in values)),
                    'outside_box_fraction': ratio(sum(v['outside_box_points'] for v in values), n),
                    'on_curve_beyond_grid_fraction_2': ratio(sum(v['on_curve_beyond_grid_2'] for v in values), n),
                    'curve_to_nodes_support_2': ratio(sum(v['supported_length']['2.0'] for v in values), length),
                    'curve_to_nodes_support_6': ratio(sum(v['supported_length']['6.0'] for v in values), length)}
        for key in ('oracle_target_graph',):
            values = [r[key] for r in group];rl = sum(v['reference_length'] for v in values);pl = sum(v['predicted_length'] for v in values)
            p = ratio(sum(v['tolerances']['2.0']['supported_prediction_length'] for v in values), pl)
            q = ratio(sum(v['tolerances']['2.0']['supported_reference_length'] for v in values), rl)
            row[key] = {'precision_2': p, 'recall_2': q, 'f1_2': 2*p*q/(p+q) if p is not None and q is not None and p+q else 0.}
        row['target_grid_point_support_2'] = ratio(sum(r['target_grid_point_support']['supported_length']['2.0'] for r in group),
                                                   sum(r['target_grid_point_support']['curve_length'] for r in group))
        neurons.append(row)
    result = {'neurons': neurons, 'partitions': {}}
    for partition in ('training', 'development'):
        ns = [n for n in neurons if n['partition'] == partition]
        methods = sorted({m for n in ns for m in n['predictions']})
        result['partitions'][partition] = {'neurons': len(ns),
            'input': {b: {str(t): mean([n['input'][b]['support'][str(t)] for n in ns]) for t in (2., 6., 8.)} for b in ('2048', '8192')},
            'input_profile_strata': {b: {s: {str(t): mean([n['input'][b]['profile_strata'][s][str(t)] for n in ns])
                for t in (2., 6., 8.)} for s in STRATA} for b in ('2048', '8192')},
            'predictions': {m: {v: {k: mean([n['predictions'][m][v][k] for n in ns if m in n['predictions']])
                for k in next(n['predictions'][m][v] for n in ns if m in n['predictions'])
                if k != 'points_in_reference_present_boxes'} for v in ('all', 'accepted')} for m in methods},
            'reference_target_graph_oracle_f1_2': mean([n['oracle_target_graph']['f1_2'] for n in ns]),
            'reference_target_grid_point_support_2': mean([n['target_grid_point_support_2'] for n in ns])}
    result['contexts'] = {}
    for partition, group in sorted({(n['partition'], n['source_group']) for n in neurons}):
        ns = [n for n in neurons if n['partition'] == partition and n['source_group'] == group]
        methods = sorted({m for n in ns for m in n['predictions']})
        result['contexts'][partition+'/'+group] = {'neurons': len(ns),
            'input': {b: {str(t): mean([n['input'][b]['support'][str(t)] for n in ns]) for t in (2., 6., 8.)} for b in ('2048', '8192')},
            'accepted_predictions': {m: {k: mean([n['predictions'][m]['accepted'][k] for n in ns if m in n['predictions']])
                for k in ('on_reference_geometry_fraction_2', 'curve_to_nodes_support_2', 'on_curve_beyond_grid_fraction_2')}
                for m in methods}}
    return result


def audit(c, data, profiles, folder):
    began = time.perf_counter(); sources = saved_sources(c, data);rows = [];controls = []
    try:
        with (folder/'boxes.jsonl').open('x', encoding='utf-8') as journal:
            for cid in sorted(data.cases):
                case = data.cases[cid]
                for box in case['boxes']:
                    tile = box['tile_id']
                    if not box['target']['count']: continue
                    ref = data.reference(cid, tile);samples, weights = curve_samples(ref['reference_runs'])
                    reference_segments = graph_segments(runs_graph(ref['reference_runs']))
                    profile = profiles.load(cid, tile)
                    labels = classify(profile['window_width'], profile['window_width_valid'], profile['window_contrast'],
                                      profile['window_contrast_valid'], profiles.manifest['cutoffs'])[profile['curve_window_index']]
                    row = {'case_id': cid, 'tile_id': tile, 'partition': case['partition'], 'input': {}, 'predictions': {}}
                    for budget in c['budgets']:
                        image = data.image(cid, tile, budget)['packet']
                        value, distance, indices = support(image['native_xyz'], samples, weights)
                        value['actual_points'] = len(image['native_xyz'])
                        value['nearest_point_normalized_intensity_weighted_mean'] = float(np.average(image['attributes'][indices, 3], weights=weights)) if weights.sum() else None
                        _, pd, _ = support(image['native_xyz'], profile['curve_xyz'], profile['curve_weights'])
                        value['strata'] = {name: {'length': float(profile['curve_weights'][labels == k].sum()),
                            'covered': {str(t): float(profile['curve_weights'][(labels == k) & (pd <= t)].sum()) for t in (2., 6., 8.)}}
                            for k, name in enumerate(STRATA)}
                        row['input'][str(budget)] = value
                    row['target_grid_point_support'] = support(ref['target_xyz'], samples, weights)[0]
                    target_graph = SimpleNamespace(xyz=ref['target_xyz'], edges=ref['target_packet']['edges'])
                    row['oracle_target_graph'] = segment_agreement(reference_segments, graph_segments(target_graph))
                    if len(reference_segments) and not any(x['case_id'] == cid for x in controls):
                        shifted = resample_runs(ref['reference_runs'], spacing=8., phase=4., preserve_vertices=True)
                        shifted_graph = graph_segments(runs_graph(shifted))
                        agreement = segment_agreement(reference_segments, shifted_graph, tolerances=(.001, 2., 6.))
                        if agreement['tolerances']['0.001']['f1'] != 1.:
                            raise ValueError('Reference subdivision/phase control changed geometry')
                        shifted_points = np.concatenate(shifted)
                        controls.append({'case_id': cid, 'tile_id': tile,
                            'reference_geometry_with_new_subdivisions': agreement,
                            'shifted_reference_points_only': point_diagnostics(shifted_points, ref['target_xyz'], reference_segments, samples, weights),
                            'active_target_packets_changed': False})
                    for source in sources:
                        if source['spec']['partition'] != case['partition']: continue
                        index = source['indices'][(cid, tile)];packet = source['packets']
                        identity = packet[f'box{index}_identity']
                        if source['case_order'][int(identity[0])] != cid or int(identity[1]) != tile:
                            raise ValueError('Prediction identity mismatch')
                        centers, scores = packet[f'box{index}_centers'], packet[f'box{index}_scores']
                        if (centers.shape != (512, 3) or scores.shape != (512,)
                                or not np.isfinite(centers).all() or not np.isfinite(scores).all()
                                or np.any((scores < 0) | (scores > 1))): raise ValueError('Invalid proposals/scores')
                        values = {}
                        for name, points in [('all', centers), ('accepted', centers[scores >= .5])]:
                            value = point_diagnostics(points, ref['target_xyz'], reference_segments, samples, weights)
                            old = source['rows'][index]['tolerances']['2.0']
                            value['unique_node_matches_2'] = old['all_query_matched'] if name == 'all' else old['nodes']['matched']
                            value['reference_nodes'] = len(ref['target_xyz'])
                            low, high = np.array(box['box']['start_xyz'])-.5, np.array(box['box']['stop_xyz'])-.5
                            value['outside_box_points'] = int((~np.all((points >= low) & (points <= high), axis=1)).sum())
                            values[name] = value
                        row['predictions'][source['spec']['method']] = values
                    rows.append(row);journal.write(json.dumps(row, allow_nan=False)+'\n');journal.flush()
                    if len(rows) % 32 == 0:print(f'AUDIT reference-present boxes={len(rows)}/611 seconds={time.perf_counter()-began:.1f}', flush=True)
                print(f'AUDIT complete case={case["alias"]} partition={case["partition"]}', flush=True)
    finally:
        for source in sources:source['packets'].close()
    if len(rows) != 611: raise ValueError('Missing reference-present box')
    unknown = {s['spec']['method']+'_'+s['spec']['partition']: {
        'unknown_boxes': sum(not r['reference_present'] for r in s['rows']),
        'accepted_in_unknown_boxes': sum(r['unknown_box_accepted'] for r in s['rows'])} for s in sources}
    result = {'status': 'completed', 'config_fingerprint': fingerprint(c), 'source_sha256': c['source_sha256'],
        'reference_present_boxes': len(rows), 'unknown_outputs': unknown,
        'summary': summarize_audit(rows, data), 'metric_controls': controls, 'seconds': time.perf_counter()-began,
        'model_executed': False, 'training_executed': False, 'filter_executed': False,
        'point_support_is_architecture_ceiling': False, 'oracle_edges_are_predictions': False,
        'biological_background_certified': False, 'node_support_is_edge_reconstruction_metric': False}
    write_once(folder/'report.json', result);return result


def baseline_case(c, data, cid, folder):
    from pn2.evaluation.classical_skeleton import extract
    began = time.perf_counter();case = data.cases[cid]
    original = ROOT/case['boxes'][0]['clouds']['2048']['path'];cache = original.parent
    metadata_path = cache/'case.json'
    if digest(metadata_path) != case['baseline_case_sha256']: raise ValueError('Original guide receipt changed')
    metadata = json.loads(metadata_path.read_text());guide = cache/metadata['guide']['path']
    if digest(guide) != metadata['guide']['sha256']: raise ValueError('Guide changed')
    raw, pruned, extraction = extract(guide)
    # Extract both image-only graphs BEFORE opening reference packets.
    np.savez_compressed(folder/'graphs.npz', raw_xyz=raw.xyz, raw_edges=raw.edges,
                        pruned_xyz=pruned.xyz, pruned_edges=pruned.edges)
    segments = {'raw': graph_segments(raw), 'pruned': graph_segments(pruned)}
    graphs = {'raw': raw, 'pruned': pruned};rows = []
    for box in case['boxes']:
        tile = box['tile_id'];ref = data.reference(cid, tile)
        rs = graph_segments(runs_graph(ref['reference_runs']));lo = np.array(box['box']['start_xyz'])-.5;hi = np.array(box['box']['stop_xyz'])-.5
        samples, weights = curve_samples(ref['reference_runs']);row = {'tile_id': tile, 'reference_present': bool(box['target']['count']), 'variants': {}}
        for name, ps in segments.items():
            clipped = clip_segments(ps, lo, hi)
            xyz = graphs[name].xyz;nodes = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
            row['variants'][name] = {'segments': segment_agreement(rs, clipped),
                'nodes': point_diagnostics(nodes, ref['target_xyz'], rs, samples, weights)}
        rows.append(row)
    result = {'status': 'completed', 'case_id': cid, 'alias': case['alias'], 'partition': case['partition'],
        'source_group': case['source_group'], 'config_fingerprint': fingerprint(c),
        'guide_sha256': metadata['guide']['sha256'], 'extraction': extraction,
        'boxes': rows, 'seconds': time.perf_counter()-began, 'image_only_extraction': True,
        'model_or_filter_executed': False, 'learned_nodes_given_reference_edges': False}
    write_once(folder/'report.json', result)
    print(f'CLASSICAL alias={case["alias"]} raw_nodes={len(raw.xyz)} pruned_nodes={len(pruned.xyz)} seconds={result["seconds"]:.1f}', flush=True)
    return result


def summarize_baseline(arms):
    neurons = []
    for arm in arms:
        n = {k: arm[k] for k in ('case_id', 'alias', 'partition', 'source_group')};n['variants'] = {}
        for name in ('raw', 'pruned'):
            rows = [b['variants'][name] for b in arm['boxes']]
            present = [b['variants'][name] for b in arm['boxes'] if b['reference_present']]
            rl = sum(r['segments']['reference_length'] for r in rows);pl = sum(r['segments']['predicted_length'] for r in rows)
            ppl = sum(r['segments']['predicted_length'] for r in present)
            values = {'reference_length': rl, 'predicted_length': pl,
                'unknown_box_prediction_length': pl-ppl, 'tolerances': {}}
            for t in ('2.0', '6.0'):
                rc = sum(r['segments']['tolerances'][t]['supported_reference_length'] for r in rows)
                pc = sum(r['segments']['tolerances'][t]['supported_prediction_length'] for r in rows)
                p = pc/pl if pl else 0.;q = rc/rl if rl else 0.
                values['tolerances'][t] = {'full_grid_precision': p, 'recall': q,
                    'full_grid_f1': 2*p*q/(p+q) if p+q else 0.,
                    'reference_present_precision': pc/ppl if ppl else 0.}
            n['variants'][name] = values
        neurons.append(n)
    return {'neurons': neurons, 'partitions': {partition: {name: {t: {k: float(np.mean([
        n['variants'][name]['tolerances'][t][k] for n in neurons if n['partition'] == partition]))
        for k in ('full_grid_precision', 'recall', 'full_grid_f1', 'reference_present_precision')}
        for t in ('2.0', '6.0')} for name in ('raw', 'pruned')}
        for partition in sorted({n['partition'] for n in neurons})}}


def controller(c, data, args, folder):
    if args.action == 'audit':
        command = [sys.executable, '-B', __file__, '--action', 'audit', '--worker', '--config', str(args.config), '--output', str(folder)]
        code, status = launch_worker(command, folder, c['audit_timeout_seconds'])
        if code or status != 'completed': raise RuntimeError(f'Audit {status}; preserve partial journal; no retry')
        return
    ids = [cid for cid, case in data.cases.items() if case['alias'] == c['baseline_timing_alias']] if args.action == 'baseline-timing' else [
        cid for cid, case in data.cases.items() if case['partition'] == 'development' or case['alias'] in c['baseline_training_aliases']]
    began = time.perf_counter();arms = []
    for i, cid in enumerate(sorted(ids)):
        remaining = c['baseline_total_timeout_seconds']-(time.perf_counter()-began)
        if remaining <= 0: raise RuntimeError('Classical total wall cap reached; no retry')
        destination = folder/f'case{i:02d}';destination.mkdir()
        command = [sys.executable, '-B', __file__, '--action', args.action, '--worker', '--case', cid,
                   '--config', str(args.config), '--output', str(destination)]
        code, status = launch_worker(command, destination, min(remaining, c['baseline_case_timeout_seconds']))
        if code or status != 'completed': raise RuntimeError(f'Classical {status}; preserve partial cases; no retry')
        arms.append(json.loads((destination/'report.json').read_text()))
    report = {'status': 'completed', 'config_fingerprint': fingerprint(c), 'arms': arms,
        'summary': summarize_baseline(arms), 'seconds': time.perf_counter()-began,
        'parameters_selected_using_reference': False, 'geometry_is_topology_metric': False,
        'training_or_filter_executed': False, 'learned_nodes_given_reference_edges': False}
    write_once(folder/'report.json', report)


def main(args):
    c, data, profiles = load_control(args.config)
    folder = args.output.resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Outputs belong under PN2/tmp')
    if not args.worker:folder.mkdir(parents=True, exist_ok=False)
    if args.worker:
        if args.action == 'audit':audit(c, data, profiles, folder)
        else:
            allowed = [cid for cid, x in data.cases.items() if x['partition'] == 'development'
                       or x['alias'] in c['baseline_training_aliases']+[c['baseline_timing_alias']]]
            if args.case not in allowed: raise ValueError('Case outside the declared baseline set')
            baseline_case(c, data, args.case, folder)
    else:controller(c, data, args, folder)
    print(f'FINISHED {args.action}; output={folder}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['audit', 'baseline-timing', 'baseline'], required=True)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true');p.add_argument('--case')
    main(p.parse_args())
