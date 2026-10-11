"""Check paired-run wiring without launching inference or geodesic tracing."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pointneuron2 import compare_graph_cutoffs as comparison


def recipe():
    return {'initializer': 'geodesic', 'graph_selection_mode': 'adaptive_connected_coverage_nms',
            'graph_min_proposal_score': .85, 'graph_nms_mode': 'distance',
            'graph_nms_distance': 18, 'graph_max_nodes': 128,
            'proposal_threshold_fraction': .2, 'proposal_score_threshold': .5,
            'proposal_top_per_patch': 512, 'proposal_global_top': 4096,
            'proposal_local_nms_radius': 8, 'proposal_global_nms_radius': 12,
            'point_sample_strategy': 'auto'}


def test_commands_vary_only_output_and_cutoff(tmp_path):
    plan = {'reconstruction': recipe(), 'data_root': str(tmp_path / 'data'),
            'sample_index': 21, 'checkpoint': str(tmp_path / 'model.pt')}
    # Parse flag/value pairs without treating the bare fail-fast switch as a value.
    def pairs(command):
        command = command[3:]; result = {}; index = 0
        while index < len(command):
            flag = command[index]
            if flag == '--fail-fast':
                result[flag] = True; index += 1
            else:
                result[flag] = command[index + 1]; index += 2
        return result
    first = pairs(comparison.reconstruction_command(plan, tmp_path / 'high', tmp_path / 'shared', .85, 'cuda'))
    second = pairs(comparison.reconstruction_command(plan, tmp_path / 'low', tmp_path / 'shared', .5, 'cuda'))
    differences = {key for key in first if first[key] != second[key]}
    assert differences == {'--output-root', '--graph-min-proposal-score'}
    assert first['--existing-proposal-dir'] == second['--existing-proposal-dir']
    assert first['--graph-max-nodes'] == second['--graph-max-nodes'] == '128'
    assert plan['reconstruction']['graph_min_proposal_score'] == .85


@pytest.mark.parametrize('tamper', [False, True])
def test_pair_reuses_actual_proposal_file_and_detects_tampering(tmp_path, monkeypatch, tamper):
    output = tmp_path / 'comparison'; output.mkdir()
    plan = {'status': 'prepared', 'reconstruction': recipe(), 'data_root': str(tmp_path / 'data'),
            'sample_id': 'validation/neuron21', 'sample_index': 21,
            'checkpoint': str(tmp_path / 'model.pt'), 'cutoffs': [.85, .5], 'results': []}
    comparison.write(output / 'comparison.json', plan)
    monkeypatch.setattr(comparison, 'verify', lambda plan: None)
    reused = []
    def fake_live(command, log):
        def arg(key): return command[command.index(key) + 1]
        script = Path(command[2]).name
        if script == 'aggregate_proposals.py':
            np.savez(arg('--output'), centers=np.zeros((2, 3)), radii=np.ones(2), scores=np.ones(2),
                     metadata=np.array(json.dumps({'sample_id': plan['sample_id'],
                                                   'checkpoint': plan['checkpoint'], 'patches': 2})))
        elif script == 'run_end_to_end.py':
            arm = Path(arg('--output-root')); arm.mkdir()
            shared = Path(arg('--existing-proposal-dir')) / 'sample_0021_proposals.npz'
            reused.append(shared)
            if tamper and len(reused) == 2:
                with shared.open('ab') as handle: handle.write(b'changed')
            graph = arm / 'graph.npz'; swc = arm / 'result.swc'
            swc.write_text('1 3 0 0 0 1 -1\n')
            np.savez(graph, metadata=np.array(json.dumps({
                'min_proposal_score': float(arg('--graph-min-proposal-score')),
                'max_nodes': 128, 'selection_mode': 'adaptive_connected_coverage_nms',
                'nms_distance': 18, 'nms_mode': 'distance'})))
            comparison.write(arm / 'summary.json', {'failures': [], 'samples': [{
                'sample_index': 21, 'swc_valid': True, 'proposal_path': str(shared),
                'graph_path': str(graph), 'swc_path': str(swc), 'bridge_edges': 0,
                'foreground_cap_satisfied': True}]})
        else:
            fields = ({'f1': .5, 'precision': .5, 'recall': .5} if script == 'evaluate_paper_style_swc.py'
                      else {'edge_f1': .5})
            comparison.write(Path(arg('--json-output')), {'samples': [{'sample_index': 21, **fields}]})
    monkeypatch.setattr(comparison, 'live', fake_live)
    if tamper:
        with pytest.raises(ValueError, match='unchanged shared proposals'):
            comparison.run(output, 'cpu')
        saved = comparison.read(output / 'comparison.json')
        assert saved['status'] == 'failed_or_interrupted'
        assert len(saved['results']) == 1
    else:
        comparison.run(output, 'cpu')
        saved = comparison.read(output / 'comparison.json')
        assert saved['status'] == 'completed' and saved['paired_checks'] == 'passed'
        assert len(saved['results']) == 2
        assert saved['results'][0]['shared_proposal_sha256'] == saved['results'][1]['shared_proposal_sha256']
    assert len(reused) == 2 and reused[0] == reused[1]
