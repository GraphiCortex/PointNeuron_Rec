"""Audit saved stable-LR fit on CPU: no model forwards, optimizer steps or CUDA."""
from collections import Counter, deque
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.pointneuron2 import stable_joint_learnability as fit
from scripts.pointneuron2 import review_joint_saved_adam_timing as reviewer
from scripts.pointneuron2.review_joint_partial_stability import confidence_summary
from pointneuron.data.connectivity_targets import assign_edge_labels
from pointneuron.graph.proposal_graph import CandidateGraph


def checkpoint_audit(saved, report, recipe, q):
    assert saved['schema'] == fit.SCHEMA and saved['purpose'] == 'fit'
    assert saved['config'] == report['config'] and saved['training_recipe'] == recipe
    assert saved['provenance'] == report['provenance']
    assert saved['proposal_updates'] == saved['graph_updates'] == q
    assert saved['historical_stage'] == 1000 + q and saved['next_crop'] == q % 4
    assert saved['resumable'] == (q < 3000)
    assert saved['prefix_verified'] == (q > 64)  # Stage 1064 saved before replay proof.
    group = saved['optimizer']['param_groups'][0]
    assert group['lr'] == 1e-5 and len(group['params']) == len(saved['parameter_names'])
    for name, index in zip(saved['parameter_names'], group['params']):
        parameter = saved['model'][name]
        state = saved['optimizer']['state'][index]
        assert float(state['step']) == (5000 if name.startswith('proposal_model.') else 1000) + q
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
        for key in ('exp_avg', 'exp_avg_sq'):
            assert state[key].shape == parameter.shape and state[key].dtype == parameter.dtype
            assert bool(torch.isfinite(state[key]).all())
        assert bool((state['exp_avg_sq'] >= 0).all())
    assert all(bool(torch.isfinite(v).all()) for v in saved['model'].values())


def residual_details(saved, packet, decoded, record):
    """Explain exact-adjacency errors without altering selection or scores."""
    audit = fit.runner.audit
    q2t = dict(zip(saved['pred_indices'].tolist(), saved['target_indices'].tolist()))
    local2target = {j: q2t[int(q)] for j, q in enumerate(packet.node_ids) if int(q) in q2t}
    target2local = {t: j for j, t in local2target.items()}
    required = audit.sets(record['target_edges'])
    selected = audit.sets(decoded.selected_edges)
    scores = {tuple(sorted(map(int, e))): float(v) for e, v in zip(saved['retained_edges'], saved['retained_edge_logits'])}
    labels = {tuple(sorted(map(int, e))): float(v) for e, v in zip(saved['candidate_edges'], saved['edge_labels'])}

    def describe(pair):
        u, v = pair
        query_pair = sorted([int(packet.node_ids[u]), int(packet.node_ids[v])])
        targets = [local2target.get(u), local2target.get(v)]
        return {'query_pair': query_pair, 'target_pair': targets, 'logit': scores[tuple(sorted(pair))],
                'training_label': labels[tuple(query_pair)],
                'length_voxels': float(torch.linalg.vector_norm(packet.positions[u] - packet.positions[v]))}

    false_edges = [e for e in sorted(selected) if e[0] not in local2target or e[1] not in local2target
                   or tuple(sorted((local2target[e[0]], local2target[e[1]]))) not in required]
    adjacency = [[] for _ in packet.node_ids]
    for u, v in selected:
        adjacency[u].append(v); adjacency[v].append(u)
    missing = []
    for a, b in sorted(required):
        if a not in target2local or b not in target2local:
            continue
        pair = tuple(sorted((target2local[a], target2local[b])))
        if pair in selected:
            continue
        detail = {'target_pair': [a, b], 'candidate': pair in scores}
        if pair in scores:
            detail.update(describe(pair))
        parents = {pair[0]: None}; pending = deque([pair[0]])
        while pending and pair[1] not in parents:
            u = pending.popleft()
            for v in adjacency[u]:
                if v not in parents:
                    parents[v] = u; pending.append(v)
        path = []
        if pair[1] in parents:
            v = pair[1]
            while parents[v] is not None:
                u = parents[v]; path.append(tuple(sorted((u, v)))); v = u
        detail['selected_path'] = [describe(e) for e in reversed(path)]
        detail['nonreference_edges_on_path'] = [describe(e) for e in path if e in false_edges]
        if path and pair in scores:
            detail['all_path_scores_at_least_missing_score'] = all(scores[e] >= scores[pair] for e in path)
        missing.append(detail)
    assigned = set(saved['pred_indices'].tolist())
    confidence = saved['logits'][0].astype(float)
    extras = [{'query': int(q), 'probability': float(torch.sigmoid(torch.tensor(confidence[int(q), 1] - confidence[int(q), 0]))),
               'position': packet.positions[j].tolist(),
               'nearest_target_distance': float(np.linalg.norm(record['nodes'][:, 1:4] - packet.positions[j].numpy(), axis=1).min())}
              for j, q in enumerate(packet.node_ids) if int(q) not in assigned]
    return {'extra_nodes': extras, 'nonreference_edges': [describe(e) for e in false_edges], 'missing_required_edges': missing}


def run():
    torch.set_num_threads(2)
    control = fit.control; base = fit.base; audit = fit.runner.audit
    config = base.read(fit.CONFIG); fit.validate_config(config)
    directory = ROOT / config['output'] / 'model'; path = directory / 'report.json'; report = base.read(path)
    assert report['status'] == 'completed' and report['config'] == config
    assert report['proposal_updates'] == report['graph_updates'] == 3000
    assert report['prefix_verified'] and report['prefix_model_optimizer_rng_exact']
    assert report['sources_and_parent_unchanged'] and report['final_reload_prediction_delta'] == 0.
    base.verify_hashes(report['provenance']['sources']); base.verify_hashes(report['protected_files_sha256'])
    recipe, original, records, oracle, parent, _, protected, reference = fit.load_inputs(config)
    ledger = [json.loads(line) for line in (directory / 'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert ledger == report['updates'] and len(ledger) == 3000
    label_cases = Counter()
    for q, update in enumerate(ledger, 1):
        assert update['fork_update'] == q and update['stage_update'] == 1000 + q
        assert update['crop'] == config['training_crops'][(q - 1) % 4] and update['optimizer_step_returned']
        assert update['cursors_after'] == {'proposal': [float(5000 + q)], 'gat': [float(1000 + q)]}
        s = update['incoming']['edge_supervision']
        assert update['edge_active'] and s['positive'] + s['negative'] > 0
        label_cases['both' if s['positive'] and s['negative'] else 'positive_only' if s['positive'] else 'negative_only'] += 1
    assert [p['new_updates'] for p in report['probes']] == [q for q in range(1, 3001) if fit.is_probe_step(q)]
    assert not any(fit.geometry_stop(report['probes'][:i]) for i in range(1, len(report['probes']) + 1))
    assert fit.stable_tail(report, config) == report['terminal_stable_tail']
    checkpoints = []
    for q in config['snapshots']:
        checkpoint = directory / f'stage_{1000 + q:04d}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        checkpoint_audit(saved, report, recipe, q)
        if q == 0:
            assert control.equal(saved['model'], parent['model']) and control.equal(saved['rng'], parent['rng'])
            assert control.equal(saved['optimizer']['state'], parent['optimizer']['state'])
            options = control.cpu_snapshot(parent['optimizer']['param_groups']); options[0]['lr'] = 1e-5
            assert saved['optimizer']['param_groups'] == options
        if q == 64:
            source = torch.load(reference / 'step_0064_terminal.pt', map_location='cpu', weights_only=True)
            for key in ('model', 'optimizer', 'rng'):
                assert control.equal(saved[key], source[key]), key
        digest = base.file_digest(checkpoint)
        checkpoints.append({'historical_stage': 1000 + q, 'sha256': digest})
        if q == 3000:
            assert digest == report['last_checkpoint_sha256']
    assert [e['fork_update'] for e in report['evaluations']] == config['snapshots']
    rows = []
    for evaluation in report['evaluations']:
        step = evaluation['fork_update']; assert evaluation['historical_stage'] == 1000 + step
        for i, (saved_row, record, reference_record) in enumerate(zip(evaluation['crops'], records, oracle)):
            prediction = directory / f'step_{step:04d}_crop_{i}.npz'; swc = prediction.with_suffix('.swc')
            assert base.file_digest(prediction) == saved_row['prediction_sha256']
            assert base.file_digest(swc) == saved_row['swc_sha256']
            saved = control.read_arrays(prediction); ids = reviewer.verify_selection(saved)
            packet = audit.make_packet(saved, ids, record['meta']['patch_center'], original['patch_radius'])
            candidates = CandidateGraph(torch.from_numpy(saved['retained_edges']), recipe['candidate_neighbors'], recipe['candidate_radius'])
            row, decoded, labels = audit.assess(packet, candidates, torch.from_numpy(saved['retained_edge_logits']), saved, record, audit.stage.reference_graph(reference_record))
            for key, value in row.items():
                assert value == saved_row[key], (step, i, key)
            assert np.array_equal(labels, saved['evaluation_labels'])
            row['swc_roundtrip'] = reviewer.verify_swc(swc, packet, decoded)
            q2t = dict(zip(saved['pred_indices'].tolist(), saved['target_indices'].tolist()))
            row['retained_trusted_junctions'] = int(sum(bool(record['junction_mask'][q2t[int(q)]])
                and np.linalg.norm(packet.positions[j].numpy().astype(float) - record['nodes'][q2t[int(q)], 1:4]) <= 2.
                for j, q in enumerate(packet.node_ids) if int(q) in q2t))
            assert fit.runner.automatic_gate(row, len(record['nodes'])) == saved_row['automatic_gate']
            full = audit.make_packet(saved, np.arange(512), record['meta']['patch_center'], original['patch_radius'])
            all_candidates = CandidateGraph(torch.from_numpy(saved['candidate_edges']), recipe['candidate_neighbors'], recipe['candidate_radius'])
            transferred = assign_edge_labels(full, all_candidates, record['nodes'][:, 1:4], record['target_edges'], record['target_components'],
                (saved['pred_indices'], saved['target_indices']), maximum_match_error=2.)
            assert transferred.statistics == saved_row['training_edge_supervision']
            assert np.array_equal(transferred.labels.numpy(), saved['edge_labels'])
            metrics = {k: row[k] for k in ('assigned_nodes', 'unmatched_nodes', 'trusted_assigned_nodes', 'decoded_required_edges',
                'required_reference_edges', 'decoded_nonreference_edges', 'missing_edge_decomposition', 'reference_components',
                'retained_reference_junctions', 'retained_trusted_junctions', 'retained_junction_incidence_errors')}
            metrics.update(components=row['decoding']['components'], geometry_at_2=row['geometry']['tolerances']['2.0'], full_gate=saved_row['automatic_gate'])
            rows.append({'historical_stage': 1000 + step, 'crop': config['training_crops'][i], 'metrics': metrics,
                'confidence': confidence_summary(saved), 'residual': residual_details(saved, packet, decoded, record) if step == 3000 else None,
                'prediction_sha256': saved_row['prediction_sha256'], 'swc_sha256': saved_row['swc_sha256']})
            print(f"VERIFIED stage={1000 + step} crop={config['training_crops'][i]} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']} gate={saved_row['automatic_gate']}", flush=True)
        assert evaluation['automatic_gate'] == all(r['automatic_gate'] for r in evaluation['crops'])
    assert len(rows) == 24 and report['automatic_gate'] == report['evaluations'][-1]['automatic_gate']
    assert report['learnability_gate'] == (report['automatic_gate'] and report['terminal_stable_tail'])
    history = []
    for i, crop in enumerate(config['training_crops']):
        probes = [p['crops'][i] for p in report['probes']]
        history.append({'crop': crop, 'minimum_probe_trusted_fraction': min(p['trusted_fraction'] for p in probes),
            'maximum_probe_mean_distance': max(p['matched_distance_mean'] for p in probes),
            'terminal_probe_mean_distance': probes[-1]['matched_distance_mean'],
            'minimum_final_eight_probe_trusted_fraction': min(p['trusted_fraction'] for p in probes[-8:])})
    base.verify_hashes(protected); base.verify_hashes(report['provenance']['sources'])
    assert not torch.cuda.is_initialized()
    result = {'schema': 'pointneuron2-stable-joint-learnability-review-v1', 'status': 'completed', 'date': '2026-10-20',
        'scope': 'read-only saved-output CPU audit; zero model forwards/updates/CUDA initialization',
        'source_report': str(path.relative_to(ROOT)), 'source_report_sha256': base.file_digest(path),
        'reviewer_sha256': base.file_digest(Path(__file__)), 'validated_output_pairs': 24,
        'proposal_updates': 3000, 'graph_updates': 3000, 'label_cases': dict(label_cases),
        'prefix_model_optimizer_rng_exact': True, 'runtime_final_reload_prediction_delta': report['final_reload_prediction_delta'],
        'elapsed_seconds': report['elapsed_seconds'], 'terminal_stable_tail': report['terminal_stable_tail'],
        'automatic_gate': report['automatic_gate'], 'learnability_gate': report['learnability_gate'],
        'checkpoints': checkpoints, 'geometry_history': history, 'rows': rows,
        'cuda_initialized': False, 'new_training_launched': False}
    output = ROOT / 'tmp/pointneuron2/local_runs/stable_joint_learnability_review_20261020/summary.json'
    if output.exists():
        raise FileExistsError('Preserve prior review evidence')
    control.write(output, result)
    print(json.dumps({'review': str(output), 'label_cases': dict(label_cases), 'geometry_history': history,
                      'terminal': rows[-4:]}, indent=2))


if __name__ == '__main__':
    run()
