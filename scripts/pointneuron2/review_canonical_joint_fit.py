"""CPU-only audit of the capped canonical-target fit and matched terminal outputs."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.pointneuron2 import canonical_joint_fit as experiment
from scripts.pointneuron2 import review_joint_saved_adam_timing as reviewer
from scripts.pointneuron2.review_joint_partial_stability import confidence_summary
from scripts.pointneuron2.review_stable_joint_learnability import residual_details
from pointneuron.data.connectivity_targets import assign_edge_labels
from pointneuron.graph.proposal_graph import CandidateGraph


def saved_fingerprint(saved):
    """Same numeric encoding as the runtime; never restore a CUDA RNG on CPU."""
    def encode(value):
        if isinstance(value, torch.Tensor):
            return ['tensor', experiment.base.tensor_digest(value)]
        if isinstance(value, dict):
            return ['dict', [(str(k), encode(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0]))]]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [encode(v) for v in value]]
        return value
    state = encode((saved['model'], saved['optimizer'], saved['rng']))
    return hashlib.sha256(json.dumps(state, allow_nan=False).encode()).hexdigest()


def check_checkpoint(saved, report, recipe, parent, q, groups):
    assert saved['schema'] == experiment.SCHEMA and saved['purpose'] == 'fit'
    assert saved['config'] == report['config'] and saved['recipe'] == recipe and saved['provenance'] == report['provenance']
    assert saved['proposal_updates'] == saved['graph_updates'] == q
    assert saved['historical_stage'] == 1000 + q and saved['next_crop'] == q % 4
    assert saved['resumable'] == (q < 3000) and saved['timing_prefix_verified'] == (q >= 8)
    assert saved['parameter_names'] == parent['parameter_names']
    assert saved['proposal_spec'] == parent['proposal_spec'] and saved['graph_spec'] == parent['graph_spec']
    assert saved['optimizer']['param_groups'] == groups
    for name, index in zip(saved['parameter_names'], groups[0]['params']):
        parameter = saved['model'][name]; state = saved['optimizer']['state'][index]
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
        assert float(state['step']) == (5000 if name.startswith('proposal_model.') else 1000) + q
        for key in ('exp_avg', 'exp_avg_sq'):
            assert state[key].shape == parameter.shape and state[key].dtype == parameter.dtype and bool(torch.isfinite(state[key]).all())
        assert bool((state['exp_avg_sq'] >= 0).all())
    assert all(bool(torch.isfinite(v).all()) for v in saved['model'].values())


def run():
    torch.set_num_threads(2)
    control, runner, base = experiment.control, experiment.runner, experiment.base
    audit = runner.audit; config = base.read(experiment.CONFIG); experiment.validate_config(config)
    directory = ROOT / config['output'] / 'model'; path = directory / 'report.json'; report = base.read(path)
    timing = base.read(directory.parent / 'timing/report.json')
    assert report['status'] == 'completed' and report['action'] == 'train' and report['config'] == config
    assert report['proposal_updates'] == report['graph_updates'] == 3000 and report['timing_prefix_verified']
    assert report['sources_unchanged'] and report['initial_gpu_predictions_identical_to_baseline'] and report['reload_prediction_delta'] == 0.
    assert timing['status'] == 'completed' and timing['timing_gate'] and timing['provenance'] == report['provenance']
    base.verify_hashes(report['provenance']['sources']); base.verify_hashes(report['protected_files_sha256'])
    recipe, original, records, oracle, parent, baseline, protected, eligible = experiment.load_inputs(config)
    assert eligible == report['eligibility'] and not report['historical_preparation_gate_passed']
    experiment.verify_initial_prediction(directory, config)
    ledger = [json.loads(line) for line in (directory / 'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert ledger == report['updates'] and len(ledger) == 3000
    cases = Counter()
    for q, update in enumerate(ledger, 1):
        assert update['fork_update'] == q and update['stage_update'] == 1000 + q and update['optimizer_step_returned']
        assert update['crop'] == config['training_crops'][(q - 1) % 4] and update['edge_active']
        assert update['cursors_after'] == {'proposal': [float(5000 + q)], 'gat': [float(1000 + q)]}
        s = update['incoming']['edge_supervision']; assert s['positive'] + s['negative'] > 0
        cases['both' if s['positive'] and s['negative'] else 'positive_only' if s['positive'] else 'negative_only'] += 1
        assert all(np.isfinite(update[k]) for k in ('proposal_gradient_l2', 'gat_gradient_l2', 'preclip_norm'))
        if q <= 8:
            control.compare_values(update, timing['updates'][q - 1])
    assert [p['new_updates'] for p in report['probes']] == [q for q in range(1, 3001) if experiment.fit.is_probe_step(q)]
    assert not any(experiment.fit.geometry_stop(report['probes'][:i]) for i in range(1, len(report['probes']) + 1))
    assert experiment.fit.stable_tail(report, base.read(experiment.fit.CONFIG)) == report['terminal_stable_tail']
    groups = control.cpu_snapshot(parent['optimizer']['param_groups']); groups[0]['lr'] = 1e-5
    checkpoints = []
    for q in config['snapshots']:
        file = directory / f'stage_{1000 + q:04d}.pt'; saved = torch.load(file, map_location='cpu', weights_only=True)
        check_checkpoint(saved, report, recipe, parent, q, groups)
        if q == 0:
            for key in ('model', 'rng'):
                assert control.equal(saved[key], parent[key])
            assert control.equal(saved['optimizer']['state'], parent['optimizer']['state'])
        fingerprint = saved_fingerprint(saved)
        if q == 8:
            assert fingerprint == timing['terminal_state_fingerprint'] == report['terminal_state_fingerprint']
            for i in range(4):
                a = control.read_arrays(directory / f'step_0008_crop_{i}.npz')
                b = control.read_arrays(directory.parent / 'timing' / f'step_0008_crop_{i}.npz')
                assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
        digest = base.file_digest(file)
        if q == 3000:
            assert digest == report['last_checkpoint_sha256']
        checkpoints.append({'new_updates': q, 'historical_stage': 1000 + q, 'sha256': digest, 'numeric_state_fingerprint': fingerprint})
    rows, predictions = [], {}
    assert [e['fork_update'] for e in report['evaluations']] == config['snapshots']
    for evaluation in report['evaluations']:
        q = evaluation['fork_update']; assert evaluation['historical_stage'] == 1000 + q
        for i, (saved_row, record, reference) in enumerate(zip(evaluation['crops'], records, oracle)):
            prediction = directory / f'step_{q:04d}_crop_{i}.npz'; swc = prediction.with_suffix('.swc')
            assert base.file_digest(prediction) == saved_row['prediction_sha256'] and base.file_digest(swc) == saved_row['swc_sha256']
            saved = control.read_arrays(prediction); ids = reviewer.verify_selection(saved)
            packet = audit.make_packet(saved, ids, record['meta']['patch_center'], original['patch_radius'])
            candidates = CandidateGraph(torch.from_numpy(saved['retained_edges']), recipe['candidate_neighbors'], recipe['candidate_radius'])
            row, decoded, labels = audit.assess(packet, candidates, torch.from_numpy(saved['retained_edge_logits']), saved, record, audit.stage.reference_graph(reference))
            for key, value in row.items():
                assert value == saved_row[key], (q, i, key)
            assert np.array_equal(labels, saved['evaluation_labels'])
            row['swc_roundtrip'] = reviewer.verify_swc(swc, packet, decoded)
            q2t = dict(zip(saved['pred_indices'].tolist(), saved['target_indices'].tolist()))
            row['retained_trusted_junctions'] = int(sum(bool(record['junction_mask'][q2t[int(query)]])
                and np.linalg.norm(packet.positions[j].numpy().astype(float) - record['nodes'][q2t[int(query)], 1:4]) <= 2.
                for j, query in enumerate(packet.node_ids) if int(query) in q2t))
            assert runner.automatic_gate(row, len(record['nodes'])) == saved_row['automatic_gate']
            all_nodes = audit.make_packet(saved, np.arange(512), record['meta']['patch_center'], original['patch_radius'])
            all_candidates = CandidateGraph(torch.from_numpy(saved['candidate_edges']), recipe['candidate_neighbors'], recipe['candidate_radius'])
            transferred = assign_edge_labels(all_nodes, all_candidates, record['nodes'][:, 1:4], record['target_edges'], record['target_components'],
                (saved['pred_indices'], saved['target_indices']), maximum_match_error=2.)
            assert transferred.statistics == saved_row['training_edge_supervision'] and np.array_equal(transferred.labels.numpy(), saved['edge_labels'])
            comparator = baseline['evaluations'][-1]['crops'][i]
            absolute, old_absolute = experiment.lengths(row['geometry']), experiment.lengths(comparator['geometry'])
            assert absolute == saved_row['absolute_bad_lengths'] and old_absolute == saved_row['baseline_terminal_absolute_bad_lengths']
            paired = experiment.geometry_noninferior(row['geometry'], comparator['geometry'], config)
            assert paired == saved_row['paired_geometry_noninferior']
            metrics = {k: row[k] for k in ('assigned_nodes', 'unmatched_nodes', 'trusted_assigned_nodes', 'decoded_required_edges',
                'required_reference_edges', 'decoded_nonreference_edges', 'missing_edge_decomposition', 'reference_components',
                'retained_trusted_junctions', 'retained_junction_incidence_errors')}
            metrics.update(geometry_at_2=row['geometry']['tolerances']['2.0'], absolute_bad_lengths=absolute,
                baseline_terminal_absolute_bad_lengths=old_absolute, components=row['decoding']['components'],
                full_gate=saved_row['automatic_gate'], paired_geometry_noninferior=paired,
                absolute_errors_no_worse=all(absolute[k] <= old_absolute[k] + 1e-9 for k in absolute))
            residual = residual_details(saved, packet, decoded, record) if q == 3000 else None
            rows.append({'new_updates': q, 'historical_stage': 1000 + q, 'crop': config['training_crops'][i], 'metrics': metrics,
                'confidence': confidence_summary(saved), 'residual': residual,
                'prediction_sha256': saved_row['prediction_sha256'], 'swc_sha256': saved_row['swc_sha256']})
            predictions[q, i] = saved
            print(f"VERIFIED stage={1000 + q} crop={config['training_crops'][i]} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']}/{row['required_reference_edges']} gate={saved_row['automatic_gate']}", flush=True)
        assert evaluation['automatic_gate'] == all(r['automatic_gate'] for r in evaluation['crops'])
        assert evaluation['paired_geometry_noninferior'] == all(r['paired_geometry_noninferior'] for r in evaluation['crops'])
    assert len(rows) == 28
    final = report['evaluations'][-1]
    assert report['automatic_gate'] == final['automatic_gate']
    assert report['learnability_gate'] == (report['proposal_updates'] == 3000 and final['automatic_gate'] and report['terminal_stable_tail'])
    assert report['paired_geometry_noninferior'] == final['paired_geometry_noninferior']
    improvement = bool(report['learnability_gate'] and final['paired_geometry_noninferior'] and any(
        r['absolute_bad_lengths'][k] < r['baseline_terminal_absolute_bad_lengths'][k] - 1e-9 for r in final['crops'] for k in ('extra', 'missed')))
    assert improvement == report['paired_improvement_gate']
    terminal = rows[-4:]; traces = []
    for i, row in enumerate(terminal):
        for extra in row['residual']['extra_nodes']:
            query = extra['query']; trace = []
            for q in config['snapshots']:
                saved = predictions[q, i]; margin = float(saved['logits'][0, query, 1] - saved['logits'][0, query, 0])
                trace.append({'historical_stage': 1000 + q, 'probability': float(torch.sigmoid(torch.tensor(margin, dtype=torch.float64))),
                    'assigned': query in saved['pred_indices'], 'accepted': margin >= 0.})
            traces.append({'crop': row['crop'], 'query': query, 'saved_snapshot_trajectory': trace})
    history = []
    for i, crop in enumerate(config['training_crops']):
        probes = [p['crops'][i] for p in report['probes']]
        incoming = [a['incoming'] for a in ledger if a['crop'] == crop]
        history.append({'crop': crop, 'minimum_probe_trusted_fraction': min(p['trusted_fraction'] for p in probes),
            'minimum_incoming_trusted_fraction': min(p['edge_supervision']['trusted_matches'] / p['edge_supervision']['targets'] for p in incoming),
            'maximum_probe_mean_distance': max(p['matched_distance_mean'] for p in probes),
            'terminal_probe_mean_distance': probes[-1]['matched_distance_mean']})
    separation = [max(r['confidence']['maximum_unmatched_probability'] for r in terminal),
                  min(r['confidence']['minimum_assigned_probability'] for r in terminal)]
    base.verify_hashes(protected); base.verify_hashes(report['provenance']['sources']); assert not torch.cuda.is_initialized()
    result = {'schema': 'pointneuron2-canonical-joint-fit-review-v1', 'date': '2026-10-21', 'status': 'completed',
        'scope': 'CPU saved-output and checkpoint audit; zero model forwards/optimizer updates/CUDA initialization',
        'source_report': str(path.relative_to(ROOT)), 'source_report_sha256': base.file_digest(path),
        'reviewer_sha256': base.file_digest(Path(__file__)), 'validated_output_pairs': 28, 'checkpoints': checkpoints,
        'proposal_updates': 3000, 'graph_updates': 3000, 'edge_label_cases': dict(cases),
        'exact_parent_model_adam_rng_verified': True, 'exact_eight_update_numeric_state_and_prediction_replay_verified': True,
        'runtime_terminal_reload_prediction_delta': report['reload_prediction_delta'], 'terminal_stable_tail': report['terminal_stable_tail'],
        'learnability_gate': report['learnability_gate'], 'paired_geometry_noninferior': report['paired_geometry_noninferior'],
        'paired_improvement_gate': improvement, 'elapsed_seconds': report['elapsed_seconds'], 'geometry_history': history,
        'terminal_four_crop_probability_separation_interval': separation,
        'terminal_interval_is_nonempty': separation[0] < separation[1], 'remaining_extra_trajectory': traces,
        'rows': rows, 'cuda_initialized': False, 'new_training_launched': False,
        'decision': 'Capped fit complete; original verdicts fail. Three full crop passes, exact target-to-target adjacency, one remaining extra spur. Discuss confidence selection; no automatic extension or threshold change.'}
    output = ROOT / 'tmp/pointneuron2/local_runs/canonical_joint_fit_review_20261021/summary.json'
    if output.exists():
        raise FileExistsError('Preserve review evidence')
    experiment.write(output, result)
    print(json.dumps({'review': str(output), 'terminal': terminal, 'probability_interval': separation,
        'remaining_extra_trajectory': traces, 'geometry_history': history}, indent=2))


if __name__ == '__main__':
    run()
