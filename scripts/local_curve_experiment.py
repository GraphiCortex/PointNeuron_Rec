"""Bounded experimental curve head. Explicit prepare/cpucheck/timing/train actions.

No action trains by default; 64 is always rejected. All artifacts are write-once
under PN2/tmp. The historical experiments and their frozen sources are untouched.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.evaluation.local_curve import (consolidate, geometric_fps, reference_geometry,
                                        score_points, summarize)
from pn2.supervision.local_curve import ProjectionIndex, distance_bands

CONFIG = ROOT/'configs/local_curve_head.json'
BASE = ROOT/'tmp/local_curve_head_20261104'
SOURCES = ('scripts/local_curve_experiment.py', 'src/pn2/supervision/local_curve.py',
           'src/pn2/models/local_curve_head.py', 'src/pn2/evaluation/local_curve.py',
           'tests/test_local_curve_head.py', 'docs/local_curve_head_protocol.md')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def control():
    c = read(CONFIG)
    if (c['schema'] != 'pn2-local-curve-head-v1' or c['budget'] != 2048
            or c['rungs']['64']['authorized'] or set(c['source_sha256']) != set(SOURCES)):
        raise ValueError('Unbound/changed experimental scope')
    for name, sha in c['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Experimental source changed: {name}; rebind before preflight')
    return c


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'tmp').resolve()):
        raise ValueError('Generated outputs must stay inside PN2/tmp')
    return path


class References:
    """Explicit manifest only; full curves for labels and agreement, not inference."""
    def __init__(self, c, data):
        self.path = ROOT/c['reference_manifest']
        manifest = read(self.path)
        self.specs = {s['case_id']: s for s in manifest['cases']}
        if set(self.specs) != set(data.cases):
            raise ValueError('Reference/image identities differ; reserved access prohibited')
        self.data = data
        self.cache = {}

    def get(self, cid):
        from pn2.evaluation.swc import parse_swc
        from pn2.evaluation.geometry import _CurveIndex
        from scipy.spatial import cKDTree
        if cid not in self.cache:
            spec = self.specs[cid]
            if spec['partition'] != self.data.cases[cid]['partition']:
                raise ValueError('Reference partition mismatch')
            path = self.path.parent/spec['reference_swc']
            if digest(path) != spec['reference_sha256']:
                raise ValueError('Supplied trace changed')
            m = parse_swc(path, type_policy='opaque_integer')
            seg = m.segments
            positive = seg[np.linalg.norm(seg[:, 1]-seg[:, 0], axis=1) > 0]
            curve = _CurveIndex(positive, 4.) if len(positive) else None
            # Zero-length incident geometry is already covered by segments;
            # genuinely isolated vertices need an additional point index.
            isolated = cKDTree(m.isolated_points) if len(m.isolated_points) else None
            def distances(points):
                p = np.asarray(points).reshape(-1, 3)
                d = curve.distances(p) if curve is not None and len(p) else np.full(len(p), np.inf)
                if isolated is not None and len(p):
                    d = np.minimum(d, isolated.query(p)[0])
                return d
            self.cache[cid] = {'morphology': m, 'projection': ProjectionIndex(seg, m.isolated_points),
                               'distances': distances}
        return self.cache[cid]


def make_labels(c, refs, cid, points):
    return refs.get(cid)['projection'].labels(points, support_radius=c['support_radius'],
        offset_radius=c['offset_supervision_radius'], tie_distance=c['ambiguity_distance'],
        tie_separation=c['ambiguity_projection_separation'])


def choose_training(data):
    """Frozen metadata selection before oracle computation; never dev outcomes."""
    by_alias = {c['alias']: cid for cid, c in data.cases.items() if c['partition'] == 'training'}
    four = [(by_alias[a], t) for a, t in ((32, 27), (32, 20), (22, 2), (22, 5))]
    selected = list(four)
    # Mandatory known fragmented identities, then round-robin source groups.
    aliases = [23, 66, 75]
    groups = {}
    for alias, cid in sorted(by_alias.items()):
        if alias not in (22, 32, 23, 66, 75):
            groups.setdefault(data.cases[cid]['source_group'], []).append(alias)
    while len(aliases) < 12:
        for group in sorted(groups):
            if groups[group] and len(aliases) < 12:
                aliases.append(groups[group].pop(0))
    for i, alias in enumerate(aliases):
        cid = by_alias[alias]
        boxes = sorted((b for b in data.cases[cid]['boxes'] if b['target']['gradient_training_allowed']),
                       key=lambda b: (b['target']['count'], b['tile_id']))
        # Alternate low, median and high density among distinct neurons.
        rank = (0, (len(boxes)-1)//2, len(boxes)-1)[i % 3]
        selected.append((cid, boxes[rank]['tile_id']))
    if len(set(selected)) != 16 or any(not data.boxes[k]['target']['gradient_training_allowed'] for k in selected):
        raise ValueError('Invalid training-only diversity manifest')
    return four, selected


def score(data, refs, key, points):
    cid, tile = key
    reference = data.reference(cid, tile)
    function = refs.get(cid)['distances'] if reference['supervision'] == 'reference_present' else None
    result = score_points(points, reference, distance_function=function)
    box = data.boxes[key]['box']
    low, high = np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5
    p = np.asarray(points).reshape(-1, 3)
    result['outside_box_points'] = int((~np.all((p >= low) & (p <= high), axis=1)).sum())
    return {'case_id': cid, 'tile_id': tile, 'metrics': result}


def sato_scores(data, key, packet, guides):
    cid, _ = key
    folder = (ROOT/data.boxes[key]['clouds']['2048']['path']).parent
    if cid not in guides:
        receipt = read(folder/'case.json')
        marker = read(folder/'complete.json')
        if (digest(folder/'case.json') != marker['case_json_sha256']
                or digest(folder/'case.json') != data.cases[cid]['baseline_case_sha256']):
            raise ValueError('Guide receipt binding changed')
        path = folder/receipt['guide']['path']
        if digest(path) != receipt['guide']['sha256']:
            raise ValueError('Cached guide changed; no automatic rebuild')
        guides[cid] = np.load(path, mmap_mode='r')
    return np.asarray(guides[cid].reshape(-1)[packet['flat_indices']], dtype=float)


def classical_packets():
    root = ROOT/'tmp/curve_classical_baseline_20261103'
    result = {}
    for path in sorted(root.glob('case*/report.json')):
        report = read(path)
        packet = path.parent/'graphs.npz'
        with np.load(packet, allow_pickle=False) as arrays:
            result[report['case_id']] = (arrays['pruned_xyz'].copy(), digest(packet))
    if len(result) != 12:
        raise ValueError('Expected the preserved 12-image classical extraction')
    return result


def prepare(c, folder):
    from pn2.evaluation.node_profile_metrics import ProfileCache
    from pn2.evaluation.node_profiles import classify
    start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'labels').mkdir()
    data = NodeBudgetDataset(ROOT, ROOT/c['data_root'])
    profiles = ProfileCache(ROOT, ROOT/c['profile_root'], data)
    refs = References(c, data)
    four, sixteen = choose_training(data)
    development = [(cid, b['tile_id']) for cid, case in data.cases.items() if case['partition'] == 'development'
                   for b in case['boxes']]
    # Freeze identities before looking at new label or oracle outcomes.
    write_once(folder/'selection.json', {'four': four, 'sixteen': sixteen, 'development': development,
        'selection': 'existing four; mandatory 23/66/75; round-robin sources; low/median/high target counts',
        'development_outcomes_used_for_selection': False})
    rows = {name: [] for name in ('oracle', 'oracle512', 'sato', 'sato512', 'classical', 'classical512')}
    label_specs = []
    oracle_counts = []
    guides = {}
    classical = classical_packets()
    all_keys = sixteen + development
    controls = {}
    for i, key in enumerate(all_keys):
        cid, tile = key
        record = data.image(cid, tile, c['budget'])
        p = record['packet']['native_xyz']
        reference = data.reference(cid, tile)
        sato = sato_scores(data, key, record['packet'], guides)
        si = consolidate(p, sato, identities=record['packet']['flat_indices'])
        rows['sato'].append(score(data, refs, key, p[si]))
        rows['sato512'].append(score(data, refs, key, p[si[:512]]))
        controls[f'box{i}_sato'] = p[si]
        if cid in classical:
            cp = classical[cid][0]
            bounds = data.boxes[key]['box']
            cp = cp[np.all((cp >= np.asarray(bounds['start_xyz'])-.5)
                           & (cp <= np.asarray(bounds['stop_xyz'])-.5), axis=1)]
            ci = consolidate(cp, np.ones(len(cp)))
            cp = cp[ci]
            cp512 = cp[geometric_fps(cp, 512)]
            rows['classical'].append(score(data, refs, key, cp))
            rows['classical512'].append(score(data, refs, key, cp512))
            controls[f'box{i}_classical'] = cp
        if reference['supervision'] == 'reference_present':
            labels = make_labels(c, refs, cid, p)
            valid = labels['offset_mask']
            oi = consolidate(labels['projection'][valid], labels['support'][valid],
                             identities=record['packet']['flat_indices'][valid], cutoff=c['cutoff'])
            oracle = labels['projection'][valid][oi]
            rows['oracle'].append(score(data, refs, key, oracle))
            rows['oracle512'].append(score(data, refs, key, oracle[:512]))
            oracle_counts.append({'case_id': cid, 'tile_id': tile, 'eligible': int(valid.sum()),
                                  'ambiguous': int(labels['ambiguous'].sum()), 'source_points': len(p)})
            label_path = folder/'labels'/f'box{i:04d}.npz'
            np.savez_compressed(label_path, **labels, bands=distance_bands(labels['source_distance']))
            label_specs.append({'case_id': cid, 'tile_id': tile, 'partition': data.cases[cid]['partition'],
                                'path': label_path.relative_to(ROOT).as_posix(), 'sha256': digest(label_path)})
        profile = profiles.load(cid, tile)
        if key in sixteen:
            labels_profile = classify(profile['window_width'], profile['window_width_valid'],
                profile['window_contrast'], profile['window_contrast_valid'], profiles.manifest['cutoffs'])
            print(f'SELECT alias={data.cases[cid]["alias"]} tile={tile} targets={len(reference["target_xyz"])} '
                  f'thin_dim_windows={int((labels_profile==1).sum())} '
                  f'junctions={sum("junction" in x for x in reference["target_roles"])}', flush=True)
        if (i+1) % 16 == 0:
            print(f'PREPARE {i+1}/{len(all_keys)} cached boxes; elapsed={time.perf_counter()-start:.1f}s', flush=True)
    np.savez_compressed(folder/'controls.npz', **controls)
    groups = {'four': set(four), 'sixteen': set(sixteen), 'development': set(development)}
    summaries = {group: {name: summarize([r for r in values if (r['case_id'], r['tile_id']) in keys])
                          for name, values in rows.items()} for group, keys in groups.items()}
    feasible = {}
    for group, size in (('four', '4'), ('sixteen', '16'), ('development', '16')):
        g = c['rungs'][size]
        value = summaries[group]['oracle']['macro']
        p_floor = g['development_precision'] if group == 'development' else g['train_precision']
        c_floor = g['development_coverage'] if group == 'development' else g['train_coverage']
        feasible[group] = value['precision_2'] >= p_floor and value['coverage_2'] >= c_floor
    # Existing audit supplies the known/unknown domain explanation; don't rerun it.
    old = read(ROOT/'docs/evidence/curve_formulation_audit_20261103.json')
    report = {'status': 'passed' if all(feasible.values()) else 'failed_feasibility',
        'config_fingerprint': fingerprint(c), 'feasible': feasible, 'summaries': summaries,
        'rows': rows, 'oracle_counts': oracle_counts, 'seconds': time.perf_counter()-start,
        'misplaced_position_domain': '51.82% is macro over reference-present development boxes only; unknown boxes excluded; incompleteness within present boxes remains',
        'existing_audit_evidence_sha256': digest(ROOT/'docs/evidence/curve_formulation_audit_20261103.json'),
        'classical_packet_hashes': {cid: sha for cid, (_, sha) in classical.items()},
        'filter_reruns': 0, 'classical_extractions': 0, 'reserved_files_opened': 0, 'optimizer_updates': 0}
    write_once(folder/'report.json', report)
    write_once(folder/'manifest.json', {'config_fingerprint': fingerprint(c), 'labels': label_specs,
        'data_sha256': digest(data.folder/'manifest.json'), 'selection_sha256': digest(folder/'selection.json'),
        'report_sha256': digest(folder/'report.json'), 'controls_sha256': digest(folder/'controls.npz'),
        'profile_sha256': digest(profiles.folder/'manifest.json'),
        'reference_manifest_sha256': digest(ROOT/c['reference_manifest'])})
    print(json.dumps({'status': report['status'], 'feasible': feasible,
                      'oracle': {g: v['oracle']['macro'] for g, v in summaries.items()},
                      'seconds': report['seconds']}), flush=True)


def prepared(c):
    folder = BASE/'prepare'
    m = read(folder/'manifest.json')
    data = NodeBudgetDataset(ROOT, ROOT/c['data_root'])
    checks = ((folder/'selection.json', m['selection_sha256']), (folder/'report.json', m['report_sha256']),
              (folder/'controls.npz', m['controls_sha256']), (data.folder/'manifest.json', m['data_sha256']),
              (ROOT/c['reference_manifest'], m['reference_manifest_sha256']))
    if m['config_fingerprint'] != fingerprint(c) or any(digest(p) != sha for p, sha in checks):
        raise ValueError('CPU preparation is stale or changed')
    if read(folder/'report.json')['status'] != 'passed':
        raise ValueError('CPU feasibility failed; no automatic fit')
    return data, m, read(folder/'selection.json')


def model_factory(c, arm, device):
    import torch
    import geometric_query_experiment as geometric
    model = geometric.make_model(geometric.load_recipe(ROOT/c['geometric_config']), 'cpu')
    if arm == 'dense':
        from pn2.models.local_curve_head import LocalCurveHead
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c['seed']+202)
            model.head = LocalCurveHead(c['offset_bound'])
    return model.to(device)


def runtime(c, device):
    import sampler_node_common as common
    common.runtime(c, device)


def train_records(c, data, manifest, keys, device):
    import torch
    from pn2.training.node_runtime import training_batch
    result = []
    for key in keys:
        cid, tile = key
        record = data.training(cid, tile, c['budget'])
        batch = training_batch(record, device)
        spec = next(s for s in manifest['labels'] if (s['case_id'], s['tile_id']) == tuple(key))
        path = ROOT/spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError('Curve target changed')
        with np.load(path, allow_pickle=False) as a:
            labels = {k: a[k].copy() for k in a.files}
        batch['curve_labels'] = {k: torch.from_numpy(labels[k]).to(device)
                                 for k in ('support', 'offset', 'offset_mask', 'bands')}
        result.append((tuple(key), record, batch, labels))
    return result


def loss_for(output, batch, arm):
    if arm == 'dense':
        from pn2.models.local_curve_head import curve_loss
        labels = batch['curve_labels']
        total, support_loss, offset_loss = curve_loss(output, labels['support'], labels['offset'],
                                                     labels['offset_mask'], labels['bands'])
        return total, {'support_loss': float(support_loss.detach()), 'offset_loss': float(offset_loss.detach())}
    from pn2.models.node_objective import node_set_loss
    loss = node_set_loss(output, batch['target'])
    return loss.total, {'classification': float(loss.classification), 'center': float(loss.center)}


def update(c, model, optimizer, batch, arm):
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    total, parts = loss_for(output, batch, arm)
    total.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradient')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameters')
    return {'loss': float(total.detach()), 'gradient_norm': float(norm), **parts}


def unpack(output, arm):
    if arm == 'dense':
        return (output.centers[0].detach().cpu().numpy(),
                output.support_logits[0].sigmoid().detach().cpu().numpy())
    return (output.center_proposals[0].detach().cpu().numpy(),
            output.objectness_logits[0].softmax(-1)[:, 1].detach().cpu().numpy())


def cpucheck(c, folder):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import image_batch
    runtime(c, 'cpu')
    data, manifest, selection = prepared(c)
    folder.mkdir(parents=True, exist_ok=False)
    model = model_factory(c, 'dense', 'cpu')
    baseline = model_factory(c, 'transformer', 'cpu')
    encoder_hash = common.state_digest(model.encoder.state_dict())
    assert encoder_hash == common.state_digest(baseline.encoder.state_dict())
    del baseline
    records = train_records(c, data, manifest, selection['four'], 'cpu')
    checks = []
    for key, record, batch, _ in records:
        # Only image-only tensors reach inference; label dict is never passed.
        image = image_batch(data.image(*key, c['budget']), 'cpu')
        model.train()
        training = model(image)
        model.eval()
        with torch.no_grad():
            inference = model(image)
        assert torch.equal(training.centers, inference.centers)
        assert torch.equal(training.support_logits, inference.support_logits)
        total, _ = loss_for(training, batch, 'dense')
        model.zero_grad(set_to_none=True)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append({'case_id': key[0], 'tile_id': key[1], 'finite_gradient': True,
                       'exact_train_eval': True, 'image_only_forward': True})
    # Synthetic labels/inputs only for the optimizer/reload execution fixture.
    synthetic = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in records[0][2].items()}
    synthetic['attributes'] = torch.linspace(0, 1, synthetic['attributes'].numel()).reshape_as(synthetic['attributes'])
    synthetic['local_xyz'] = synthetic['attributes'][..., :3]
    synthetic['points'] = torch.cat((synthetic['local_xyz']*64, synthetic['attributes'][..., 3:4]), -1)
    synthetic['curve_labels'] = {'support': torch.ones(2048)*.5, 'offset': torch.ones(2048, 3),
                                'offset_mask': torch.ones(2048, dtype=torch.bool), 'bands': torch.zeros(2048, dtype=torch.int64)}
    optimizer = common.make_optimizer(model, c)
    update(c, model, optimizer, synthetic, 'dense')
    bindings = {'maximum_updates': 1, 'config': fingerprint(c), 'synthetic_only': True}
    payload = copy.deepcopy(state.capture(model, optimizer, bindings, 1))
    expected = common.nested_digest(payload)
    restored = model_factory(c, 'dense', 'cpu')
    restored_optimizer = common.make_optimizer(restored, c)
    assert state.restore(payload, restored, restored_optimizer, bindings) == 1
    assert common.nested_digest(state.capture(restored, restored_optimizer, bindings, 1)) == expected
    original_next = update(c, model, optimizer, synthetic, 'dense')
    restored_next = update(c, restored, restored_optimizer, synthetic, 'dense')
    assert original_next == restored_next
    assert common.state_digest(model.state_dict()) == common.state_digest(restored.state_dict())
    report = {'status': 'passed', 'config_fingerprint': fingerprint(c), 'checks': checks,
        'same_initial_encoder_sha256': encoder_hash, 'synthetic_checkpoint_next_update_exact': True,
        'parameters': sum(p.numel() for p in model.parameters()), 'real_optimizer_updates': 0,
        'reserved_files_opened': 0, 'device': 'cpu'}
    write_once(folder/'report.json', report)
    print(json.dumps(report), flush=True)


def require_cpu(c):
    report = read(BASE/'cpucheck/report.json')
    if report['status'] != 'passed' or report['config_fingerprint'] != fingerprint(c):
        raise ValueError('Missing/stale CPU head/gradient/reload check')


def label_arrays(manifest, key):
    spec = next(s for s in manifest['labels'] if (s['case_id'], s['tile_id']) == tuple(key))
    path = ROOT/spec['path']
    if digest(path) != spec['sha256']:
        raise ValueError('Diagnostic labels changed')
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in arrays.files}


def thin_metrics(points, profile, cutoffs):
    from scipy.spatial import cKDTree
    from pn2.evaluation.node_profiles import classify
    labels = classify(profile['window_width'], profile['window_width_valid'],
                      profile['window_contrast'], profile['window_contrast_valid'], cutoffs)
    thin = labels[profile['curve_window_index']] == 1
    w = profile['curve_weights']
    d = cKDTree(points).query(profile['curve_xyz'])[0] if len(points) else np.full(len(w), np.inf)
    return {'length': float(w[thin].sum()), 'covered': float(w[thin & (d <= 2)].sum())}


def offset_metrics(output, labels, distance_function):
    votes = output.centers[0].detach().cpu().numpy()
    offsets = output.offsets[0].detach().cpu().numpy()
    error = np.linalg.norm(offsets-labels['offset'], axis=1)
    distance = distance_function(votes)
    rows = []
    for band in range(3):
        selected = (labels['bands'] == band) & labels['offset_mask']
        rows.append({'band': band, 'eligible': int(selected.sum()),
            'vector_error_mean': float(error[selected].mean()) if selected.any() else None,
            'vector_error_p90': float(np.quantile(error[selected], .9)) if selected.any() else None,
            'vote_curve_distance_mean': float(distance[selected].mean()) if selected.any() else None,
            'votes_within_2': int((distance[selected] <= 2).sum())})
    return rows


def incident_probes(morphology):
    """Spatial support along incident chains; never a predicted connectivity score."""
    from pn2.supervision.curve_nodes import _chain_geometry
    _, _, chains = _chain_geometry(morphology)
    coordinates, multiplicity = np.unique(morphology.xyz, axis=0, return_counts=True)
    counts = {tuple(p): int(n) for p, n in zip(coordinates, multiplicity)}
    probes = []
    for ids, points, _, arc in chains:
        if arc[-1] <= 0:
            continue
        for index, position in ((int(ids[0]), min(4., arc[-1]/2)),
                                (int(ids[-1]), max(arc[-1]-4., arc[-1]/2))):
            center = morphology.xyz[index]
            probe = np.array([np.interp(position, arc, points[:, j]) for j in range(3)])
            probes.append((center, probe, counts[tuple(center)] > 1))
    return probes


def incident_metrics(points, probes, box):
    from scipy.spatial import cKDTree
    low, high = np.asarray(box['start_xyz'])-.5, np.asarray(box['stop_xyz'])-.5
    valid, ambiguous, outside = [], 0, 0
    for center, probe, tie in probes:
        if not np.all((center >= low) & (center <= high)):
            continue
        if tie:
            ambiguous += 1
        elif np.all((probe >= low) & (probe <= high)):
            valid.append(probe)
        else:
            outside += 1
    distance = (cKDTree(points).query(np.asarray(valid))[0] if len(points) and valid
                else np.full(len(valid), np.inf))
    return {'count': len(valid), 'covered': int((distance <= 2).sum()),
            'ambiguous_coincident': ambiguous, 'probe_outside_crop': outside,
            'meaning': 'spatial incident-chain support only; connectivity unmeasured'}


def gap_metrics(points, reference):
    from scipy.spatial import cKDTree
    from pn2.evaluation.cloud_support import polyline_samples
    tree = cKDTree(points) if len(points) else None
    largest = 0.
    for run in reference['reference_runs']:
        p, weights = polyline_samples(run, .5)
        distance = tree.query(p)[0] if tree is not None else np.full(len(p), np.inf)
        current = 0.
        for missing, weight in zip(distance > 2., weights):
            current = current+float(weight) if missing else 0.
            largest = max(largest, current)
    return largest


def evaluate(c, model, arm, data, manifest, keys, folder, step):
    import torch
    from pn2.training.node_runtime import image_batch
    from pn2.evaluation.node_profile_metrics import ProfileCache
    import sampler_node_common as common
    began = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=False)
    before = common.state_digest(model.state_dict())
    before_rng = common.nested_digest(common.rng_state())
    mode = model.training
    model.eval()
    refs = References(c, data)
    profiles = ProfileCache(ROOT, ROOT/c['profile_root'], data)
    rows, ranked_rows, cap_rows, raw_rows, raw_all_rows, arrays, diagnostics = [], [], [], [], [], {}, []
    sensitivities = {str(v): [] for v in c['diagnostic_cutoffs']} if arm == 'dense' else {}
    device = next(model.parameters()).device
    probe_cache = {}
    with torch.inference_mode():
        for i, key in enumerate(keys):
            cid, tile = key
            image = data.image(cid, tile, c['budget'])
            output = model(image_batch(image, device))
            # References are opened after image-only inference.
            points, scores = unpack(output, arm)
            ref = data.reference(cid, tile)
            geometry = reference_geometry(ref)
            known = ref['supervision'] == 'reference_present'
            context = refs.get(cid) if known else None
            function = context['distances'] if known else None
            scorer = lambda p: score_points(p, ref, geometry, function)
            ids = image['packet']['flat_indices'] if arm == 'dense' else None
            ranked = consolidate(points, scores, identities=ids)
            cutoff = c['cutoff'] if arm == 'dense' else .5
            accepted = ranked[scores[ranked] >= cutoff]
            raw_rows.append({'case_id': cid, 'tile_id': tile, 'metrics': scorer(points[scores >= cutoff])})
            raw_all_rows.append({'case_id': cid, 'tile_id': tile, 'metrics': scorer(points)})
            row = {'case_id': cid, 'tile_id': tile, 'metrics': scorer(points[accepted])}
            ranked_rows.append({'case_id': cid, 'tile_id': tile, 'metrics': scorer(points[ranked[:512]])})
            cap_rows.append({'case_id': cid, 'tile_id': tile, 'metrics': scorer(points[accepted[:512]])})
            b = data.boxes[(cid, tile)]['box']
            low, high = np.asarray(b['start_xyz'])-.5, np.asarray(b['stop_xyz'])-.5
            row['metrics']['outside_box_points'] = int((~np.all((points[accepted] >= low)
                                                                          & (points[accepted] <= high), axis=1)).sum())
            if known:
                profile = profiles.load(cid, tile)
                row['thin_dim'] = thin_metrics(points[accepted], profile, profiles.manifest['cutoffs'])
                if cid not in probe_cache:
                    probe_cache[cid] = incident_probes(context['morphology'])
                row['incident'] = incident_metrics(points[accepted], probe_cache[cid], b)
                row['longest_uncovered_length_2'] = gap_metrics(points[accepted], ref)
                if arm == 'dense':
                    labels = label_arrays(manifest, key)
                    diagnostics.append({'case_id': cid, 'tile_id': tile,
                                        'offset_bands': offset_metrics(output, labels, function)})
            rows.append(row)
            for threshold, values in sensitivities.items():
                selected = ranked[scores[ranked] >= float(threshold)]
                values.append({'case_id': cid, 'tile_id': tile, 'metrics': scorer(points[selected])})
            arrays[f'box{i}_points'] = points
            arrays[f'box{i}_scores'] = scores
            arrays[f'box{i}_ranked_indices'] = ranked
            if arm == 'dense':
                arrays[f'box{i}_offsets'] = output.offsets[0].cpu().numpy()
            if (i+1) % 32 == 0:
                print(f'EVAL {arm} step={step} {i+1}/{len(keys)} elapsed={time.perf_counter()-began:.1f}s', flush=True)
    model.train(mode)
    if common.state_digest(model.state_dict()) != before or common.nested_digest(common.rng_state()) != before_rng:
        raise ValueError('Evaluation changed model/RNG state')
    report = {'step': step, 'arm': arm, 'keys': keys, 'rows': rows, 'summary': summarize(rows),
        'top512_summary': summarize(ranked_rows), 'accepted512_summary': summarize(cap_rows),
        'raw_accepted_summary': summarize(raw_rows), 'raw_all_summary': summarize(raw_all_rows),
        'diagnostic_cutoffs': {k: summarize(v) for k, v in sensitivities.items()},
        'offset_diagnostics': diagnostics, 'seconds': time.perf_counter()-began,
        'precision_domain': 'full supplied trace; coverage domain is clipped box curve',
        'false_joins': 'unmeasured: no predicted edges', 'selection': 'fixed terminal checkpoint; no threshold selection'}
    write_once(folder/'metrics.json', report)
    np.savez_compressed(folder/'predictions.npz', **arrays)
    print(f'EVAL COMPLETE {arm} step={step} {json.dumps(report["summary"]["macro"])} '
          f'unknown_points={report["summary"]["unknown_points"]}', flush=True)
    return report


def timing(c, arm, folder):
    import torch
    import sampler_node_common as common
    require_cpu(c)
    runtime(c, 'cuda:0')
    data, manifest, selection = prepared(c)
    folder.mkdir(parents=True, exist_ok=False)
    model = model_factory(c, arm, 'cuda:0')
    optimizer = common.make_optimizer(model, c)
    keys = selection['four'] if arm == 'dense' else selection['sixteen']
    records = train_records(c, data, manifest, keys, 'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for i in range(c['timing_updates']):
        torch.cuda.synchronize()
        began = time.perf_counter()
        losses.append(update(c, model, optimizer, records[i % len(records)][2], arm))
        torch.cuda.synchronize()
        times.append(time.perf_counter()-began)
        print(f'TIMING {arm} {i+1}/8 seconds={times[-1]:.3f}', flush=True)
    train_eval = evaluate(c, model, arm, data, manifest, keys, folder/'training_eval', 8)
    dev_eval = evaluate(c, model, arm, data, manifest, selection['development'], folder/'development_eval', 8)
    per_update = float(np.mean(times[1:]))
    estimates = {}
    for rung in ('4', '16'):
        cap = c['rungs'][rung]
        # Training scoring at initialization/midpoint/terminal; dev at sixteen.
        evaluation_cost = train_eval['seconds']*(int(rung)/len(keys))*3
        if rung == '16':
            evaluation_cost += dev_eval['seconds']*3
        seconds = 1.5*(cap['updates']*per_update+evaluation_cost+60.)
        estimates[rung] = {'seconds_with_50_percent_margin': seconds,
                          'within_wall_limit': seconds < cap['wall_seconds_per_arm']}
    report = {'status': 'passed' if all(v['within_wall_limit'] for v in estimates.values()) else 'failed_timing',
        'arm': arm, 'config_fingerprint': fingerprint(c), 'device': torch.cuda.get_device_name(0),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'update_seconds': times,
        'projected': estimates, 'throwaway_updates': 8, 'losses': losses,
        'training_evaluation_seconds': train_eval['seconds'], 'development_evaluation_seconds': dev_eval['seconds']}
    write_once(folder/'report.json', report)
    print(json.dumps(report), flush=True)


def four_gate(c, terminal):
    prepared_report = read(BASE/'prepare/report.json')
    oracle = prepared_report['summaries']['four']['oracle']['macro']['coverage_2']
    g = c['rungs']['4']
    actual = terminal['summary']['macro']
    oracle_boxes = {(r['case_id'], r['tile_id']): r['metrics']['coverage_2']
                    for r in prepared_report['rows']['oracle']}
    ratios = [r['metrics']['coverage_2']/oracle_boxes[(r['case_id'], r['tile_id'])]
              if oracle_boxes[(r['case_id'], r['tile_id'])] > 0 else 0 for r in terminal['rows']]
    checks = {'precision': actual['precision_2'] >= g['train_precision'],
              'absolute_coverage': actual['coverage_2'] >= g['train_coverage'],
              'macro_oracle_ratio': actual['coverage_2'] >= oracle*g['oracle_ratio'],
              'minimum_box_oracle_ratio': min(ratios) >= g['minimum_box_oracle_ratio']}
    return {'passed': all(checks.values()), 'checks': checks, 'oracle_coverage': oracle,
            'box_oracle_ratios': ratios, 'terminal_macro': actual}


def absolute_sixteen_gate(c, training, development):
    g = c['rungs']['16']
    t, d = training['summary']['macro'], development['summary']['macro']
    checks = {'training_precision': t['precision_2'] >= g['train_precision'],
              'training_coverage': t['coverage_2'] >= g['train_coverage'],
              'development_precision': d['precision_2'] >= g['development_precision'],
              'development_coverage': d['coverage_2'] >= g['development_coverage'],
              'worst_development_coverage': min(n['coverage_2'] for n in development['summary']['neurons'])
                                           >= g['minimum_development_neuron_coverage']}
    return {'passed': all(checks.values()), 'checks': checks}


def role_macro(report, role):
    values = [n['roles'][role]['covered']/n['roles'][role]['count']
              for n in report['summary']['neurons'] if n['roles'][role]['count']]
    return float(np.mean(values)) if values else None


def thin_macro(report):
    values = []
    for cid in sorted({r['case_id'] for r in report['rows']}):
        group = [r['thin_dim'] for r in report['rows'] if r['case_id'] == cid and 'thin_dim' in r]
        length = sum(r['length'] for r in group)
        if length:
            values.append(sum(r['covered'] for r in group)/length)
    return float(np.mean(values)) if values else None


def compare_sixteen(c, folder):
    data, manifest, selection = prepared(c)
    dense_folder = BASE/'train16_dense'
    base_folder = BASE/'train16_transformer'
    dense_report, base_report = read(dense_folder/'report.json'), read(base_folder/'report.json')
    if (dense_report['config_fingerprint'] != fingerprint(c) or base_report['config_fingerprint'] != fingerprint(c)
            or dense_report['initial_encoder_sha256'] != base_report['initial_encoder_sha256']):
        raise ValueError('Paired initialization/bindings differ')
    folder.mkdir(parents=True, exist_ok=False)
    dense_eval = dense_folder/'eval_step004000_development'
    base_eval = base_folder/'eval_step004000_development'
    de, be = read(dense_eval/'metrics.json'), read(base_eval/'metrics.json')
    if de['keys'] != be['keys'] or de['keys'] != selection['development']:
        raise ValueError('Paired grids differ')
    prepared_keys = selection['sixteen']+selection['development']
    refs = References(c, data)
    rows = {name: {'dense': [], name: []} for name in ('transformer', 'sato', 'classical')}
    counts = {name: [] for name in rows}
    with np.load(dense_eval/'predictions.npz', allow_pickle=False) as dp, \
         np.load(base_eval/'predictions.npz', allow_pickle=False) as bp, \
         np.load(BASE/'prepare/controls.npz', allow_pickle=False) as cp:
        for i, key in enumerate(selection['development']):
            dense = dp[f'box{i}_points'][dp[f'box{i}_ranked_indices']]
            transformer = bp[f'box{i}_points'][bp[f'box{i}_ranked_indices']]
            j = prepared_keys.index(key)
            for name, other in (('transformer', transformer), ('sato', cp[f'box{j}_sato']),
                                ('classical', cp[f'box{j}_classical'])):
                k = min(512, len(dense), len(other))
                reduced = other[geometric_fps(other, k)] if name == 'classical' else other[:k]
                rows[name]['dense'].append(score(data, refs, key, dense[:k]))
                rows[name][name].append(score(data, refs, key, reduced))
                counts[name].append(k)
    comparisons = {}
    g = c['rungs']['16']
    for name, values in rows.items():
        summaries = {arm: summarize(r) for arm, r in values.items()}
        candidate, comparator = summaries['dense']['macro'], summaries[name]['macro']
        margin = g['transformer_coverage_gain'] if name == 'transformer' else g['no_learning_coverage_gain']
        comparisons[name] = {'summaries': summaries, 'K': counts[name],
            'coverage_gain': candidate['coverage_2']-comparator['coverage_2'],
            'precision_difference': candidate['precision_2']-comparator['precision_2'],
            'passed': candidate['coverage_2'] >= comparator['coverage_2']+margin
                      and candidate['precision_2'] >= comparator['precision_2']-g['maximum_precision_loss']}
    structural = {}
    for role in ('root', 'leaf', 'junction', 'crop_boundary'):
        a, b = role_macro(de, role), role_macro(be, role)
        structural[role] = {'dense': a, 'transformer': b,
                            'passed': a is not None and a >= b-.05 if b is not None else None}
    dt, bt = thin_macro(de), thin_macro(be)
    structural['thin_dim'] = {'dense': dt, 'transformer': bt,
                              'passed': dt is not None and dt >= bt-.05 if bt is not None else None}
    absolute = dense_report['absolute_gate']
    passed = (absolute['passed'] and all(r['passed'] for r in comparisons.values())
              and all(r['passed'] is not False for r in structural.values()))
    report = {'status': 'completed', 'gate_passed': passed, 'config_fingerprint': fingerprint(c),
        'absolute_gate': absolute, 'equal_count_comparisons': comparisons, 'structural_diagnostics': structural,
        'rows': rows, 'next_action': 'discuss before 64 or GATv2' if passed else 'stop this recipe; discuss findings',
        'interpretation': 'one-seed budgeted development proposal comparison, not converged capacity or graph quality'}
    write_once(folder/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


def train(c, arm, rung, folder):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    require_cpu(c)
    if rung == '4' and arm != 'dense':
        raise ValueError('Four-box Transformer is saved historical context, not another fit')
    if rung == '16' and not read(BASE/'train4_dense/report.json')['gate']['passed']:
        raise ValueError('Four-box gate failed; expansion is prohibited')
    proof = read(BASE/f'timing_{arm}/report.json')
    if (proof['status'] != 'passed' or proof['config_fingerprint'] != fingerprint(c)
            or not proof['projected'][rung]['within_wall_limit']):
        raise ValueError('Missing/stale/failed timing')
    runtime(c, 'cuda:0')
    data, manifest, selection = prepared(c)
    folder.mkdir(parents=True, exist_ok=False)
    keys = selection['four' if rung == '4' else 'sixteen']
    records = train_records(c, data, manifest, keys, 'cuda:0')
    model = model_factory(c, arm, 'cuda:0')
    optimizer = common.make_optimizer(model, c)
    maximum = c['rungs'][rung]['updates']
    bindings = {'maximum_updates': maximum, 'config_fingerprint': fingerprint(c), 'arm': arm,
                'rung': rung, 'data_sha256': manifest['data_sha256'],
                'selection_sha256': manifest['selection_sha256']}
    initial_encoder_hash = common.state_digest(model.encoder.state_dict())
    state.save(folder/'checkpoint_step000000.pt', model, optimizer, bindings, 0)
    began = time.perf_counter()
    evaluation = evaluate(c, model, arm, data, manifest, keys, folder/'eval_step000000_training', 0)
    if rung == '16':
        evaluate(c, model, arm, data, manifest, selection['development'], folder/'eval_step000000_development', 0)
    terminal_dev = None
    torch.cuda.reset_peak_memory_stats()
    with (folder/'updates.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, maximum+1):
            metrics = update(c, model, optimizer, records[(step-1) % len(records)][2], arm)
            log.write(json.dumps({'step': step, **metrics})+'\n')
            if step % 50 == 0:
                log.flush()
                print(f'TRAIN {rung} {arm} step={step}/{maximum} loss={metrics["loss"]:.5f} '
                      f'elapsed={time.perf_counter()-began:.1f}s', flush=True)
            if step in (8, maximum//2, maximum):
                state.save(folder/f'checkpoint_step{step:06d}.pt', model, optimizer, bindings, step)
            if step in (maximum//2, maximum):
                evaluation = evaluate(c, model, arm, data, manifest, keys,
                                      folder/f'eval_step{step:06d}_training', step)
                if rung == '16':
                    terminal_dev = evaluate(c, model, arm, data, manifest, selection['development'],
                                           folder/f'eval_step{step:06d}_development', step)
    absolute = absolute_sixteen_gate(c, evaluation, terminal_dev) if rung == '16' else None
    gate = four_gate(c, evaluation) if rung == '4' else {
        'passed': False, 'status': 'paired comparison required' if absolute['passed'] else 'failed_absolute_floors'}
    report = {'status': 'completed', 'rung': rung, 'arm': arm, 'config_fingerprint': fingerprint(c),
        'updates': maximum, 'exposures_per_box': maximum//len(keys), 'gate': gate, 'absolute_gate': absolute,
        'initial_encoder_sha256': initial_encoder_hash, 'seconds': time.perf_counter()-began,
        'training': evaluation['summary'], 'development': terminal_dev['summary'] if terminal_dev else None,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'device': torch.cuda.get_device_name(0),
        'interpretation': 'budgeted proposal fitting; no converged capacity or reconstruction verdict'}
    write_once(folder/'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'cpucheck', 'timing', 'train', 'compare16'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=['dense', 'transformer'], default='dense')
    parser.add_argument('--rung', choices=['4', '16'], default='4')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    c = control()
    suffix = f'timing_{args.arm}' if args.action == 'timing' else f'train{args.rung}_{args.arm}'
    folder = output_path(args.output or BASE/(suffix if args.action in ('train', 'timing') else args.action))
    if args.action in ('prepare', 'timing', 'train') and not args.worker:
        from pn2.training.process_control import launch_worker
        # Worker owns actual output; controller has a distinct immutable log dir.
        logs = folder.with_name(folder.name+'_controller')
        logs.mkdir(parents=True, exist_ok=False)
        timeout = c['cpu_wall_seconds'] if args.action == 'prepare' else (
            c['timing_wall_seconds'] if args.action == 'timing' else c['rungs'][args.rung]['wall_seconds_per_arm'])
        command = [sys.executable, '-B', str(Path(__file__).resolve()), args.action, '--worker',
                   '--arm', args.arm, '--rung', args.rung, '--output', str(folder)]
        code, status = launch_worker(command, logs, timeout)
        write_once(logs/'report.json', {'status': status, 'exit_code': code, 'timeout_seconds': timeout,
                                      'output': str(folder), 'config_fingerprint': fingerprint(c)})
        if status != 'completed':
            raise SystemExit(f'Bounded worker {status}; no automatic retry or extension')
        return
    if args.action == 'prepare':
        prepare(c, folder)
    elif args.action == 'cpucheck':
        cpucheck(c, folder)
    elif args.action == 'timing':
        timing(c, args.arm, folder)
    elif args.action == 'train':
        train(c, args.arm, args.rung, folder)
    elif args.action == 'compare16':
        compare_sixteen(c, folder)


if __name__ == '__main__':
    main()
