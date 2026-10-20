"""Read-only CPU verification of the canonical-target GPU timing evidence."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.pointneuron2 import canonical_joint_fit as experiment
from scripts.pointneuron2 import review_joint_saved_adam_timing as reviewer
from pointneuron.data.connectivity_targets import assign_edge_labels
from pointneuron.graph.proposal_graph import CandidateGraph


def run():
    torch.set_num_threads(2)
    control, runner, base = experiment.control, experiment.runner, experiment.base
    audit = runner.audit
    config = base.read(experiment.CONFIG); experiment.validate_config(config)
    root = ROOT / config['output']; directory = root / 'timing'; path = directory / 'report.json'
    report = base.read(path); cpu = base.read(root / 'check/report.json')
    assert report['status'] == 'completed' and report['action'] == 'timing' and report['config'] == config
    assert report['proposal_updates'] == report['graph_updates'] == 8 and report['timing_gate']
    assert report['initial_gpu_predictions_identical_to_baseline'] and report['reload_prediction_delta'] == 0.
    assert report['sources_unchanged'] and report['eligibility']['passed'] and not report['historical_preparation_gate_passed']
    assert len(report['terminal_state_fingerprint']) == 64
    assert cpu['status'] == 'completed' and cpu['cpu_check_passed'] and cpu['config'] == config
    assert cpu['provenance']['sources'] == report['provenance']['sources']
    assert not list(directory.glob('*.pt*'))  # Throwaway timing retains no checkpoint.
    base.verify_hashes(report['provenance']['sources']); base.verify_hashes(report['protected_files_sha256'])
    recipe, original, records, oracle, _, baseline, protected, eligible = experiment.load_inputs(config)
    assert report['eligibility'] == eligible and report['provenance']['backend'] == baseline['provenance']['backend']
    ledger = [json.loads(line) for line in (directory / 'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert ledger == report['updates'] and len(ledger) == 8
    for q, update in enumerate(ledger, 1):
        assert update['fork_update'] == q and update['stage_update'] == 1000 + q
        assert update['crop'] == config['training_crops'][(q - 1) % 4] and update['optimizer_step_returned'] and update['edge_active']
        assert update['cursors_after'] == {'proposal': [float(5000 + q)], 'gat': [float(1000 + q)]}
        stats = update['incoming']['edge_supervision']
        assert stats['positive'] > 0 and stats['negative'] > 0 and stats['trusted_matches'] == stats['targets']
        assert all(np.isfinite(update[k]) for k in ('proposal_gradient_l2', 'gat_gradient_l2', 'preclip_norm'))
    assert [p['new_updates'] for p in report['probes']] == [4, 8]
    for probe in report['probes']:
        assert len(probe['crops']) == 4
        for record, row in zip(records, probe['crops']):
            assert row['trusted_fraction'] == 1. and row['edge_supervision']['trusted_matches'] == len(record['nodes'])
    # Inference arrays must match the old baseline, independent of changed labels.
    experiment.verify_initial_prediction(directory, config)
    assert [e['fork_update'] for e in report['evaluations']] == [0, 8]
    rows = []
    for evaluation in report['evaluations']:
        q = evaluation['fork_update']
        for i, (saved_row, record, reference) in enumerate(zip(evaluation['crops'], records, oracle)):
            prediction = directory / f'step_{q:04d}_crop_{i}.npz'; swc = prediction.with_suffix('.swc')
            assert base.file_digest(prediction) == saved_row['prediction_sha256']
            assert base.file_digest(swc) == saved_row['swc_sha256']
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
            assert transferred.statistics == saved_row['training_edge_supervision']
            assert np.array_equal(transferred.labels.numpy(), saved['edge_labels'])
            absolute = experiment.lengths(row['geometry'])
            assert absolute == saved_row['absolute_bad_lengths']
            paired = experiment.geometry_noninferior(row['geometry'], baseline['evaluations'][-1]['crops'][i]['geometry'], config)
            assert paired == saved_row['paired_geometry_noninferior']
            metrics = {k: row[k] for k in ('assigned_nodes', 'unmatched_nodes', 'trusted_assigned_nodes', 'decoded_required_edges',
                'required_reference_edges', 'decoded_nonreference_edges', 'missing_edge_decomposition', 'retained_trusted_junctions')}
            metrics.update(geometry_at_2=row['geometry']['tolerances']['2.0'], absolute_bad_lengths=absolute,
                components=row['decoding']['components'], reference_components=row['reference_components'],
                full_gate=saved_row['automatic_gate'], paired_geometry_noninferior=paired)
            rows.append({'new_updates': q, 'crop': config['training_crops'][i], 'metrics': metrics,
                'prediction_sha256': saved_row['prediction_sha256'], 'swc_sha256': saved_row['swc_sha256']})
            print(f"VERIFIED timing step={q} crop={config['training_crops'][i]} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']} SWC=True", flush=True)
        assert evaluation['automatic_gate'] == all(r['automatic_gate'] for r in evaluation['crops'])
        assert evaluation['paired_geometry_noninferior'] == all(r['paired_geometry_noninferior'] for r in evaluation['crops'])
    assert len(rows) == 8
    base.verify_hashes(protected); base.verify_hashes(report['provenance']['sources'])
    assert not torch.cuda.is_initialized()
    result = {'schema': 'pointneuron2-canonical-joint-timing-review-v1', 'date': '2026-10-21', 'status': 'completed',
        'scope': 'CPU saved-output audit; zero model forwards/optimizer updates/CUDA initialization',
        'source_report': str(path.relative_to(ROOT)), 'source_report_sha256': base.file_digest(path),
        'reviewer_sha256': base.file_digest(Path(__file__)), 'validated_output_pairs': 8, 'timing_gate': True,
        'initial_inference_exactly_matches_saved_baseline': True, 'runtime_reload_prediction_delta': report['reload_prediction_delta'],
        'runtime_model_optimizer_rng_fingerprint': report['terminal_state_fingerprint'],
        'fingerprint_limitation': 'Runtime checks exact full-state reload; the throwaway state is not retained for independent recomputation.',
        'terminal_optimizer_clocks': ledger[-1]['cursors_after'], 'both_known_edge_classes_on_all_updates': True,
        'trusted_fraction_all_incoming_and_probes': 1., 'peak_allocated_gib': report['peak_allocated_gib'],
        'update_only_estimate_minutes': report['estimated_update_seconds'] / 60,
        'elapsed_action_seconds': report['elapsed_seconds'], 'rows': rows,
        'training_launched': False, 'cuda_initialized': False,
        'decision': 'Execution timing passes; proceed only with the existing capped matched-fit contract. No reconstruction improvement established.'}
    output = ROOT / 'tmp/pointneuron2/local_runs/canonical_joint_timing_review_20261021/summary.json'
    if output.exists():
        raise FileExistsError('Preserve prior review evidence')
    experiment.write(output, result)
    print(json.dumps({'review': str(output), 'timing_gate': True, 'peak_allocated_gib': result['peak_allocated_gib'],
        'update_only_estimate_minutes': result['update_only_estimate_minutes'], 'terminal_clocks': result['terminal_optimizer_clocks']}, indent=2))


if __name__ == '__main__':
    run()
