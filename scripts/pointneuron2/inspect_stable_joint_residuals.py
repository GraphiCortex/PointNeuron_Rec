"""Reference-conditioned interpretation of saved errors; no change to gates/inference."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from pointneuron.data.query_cache import file_digest


def run():
    review_path = ROOT / 'tmp/pointneuron2/local_runs/stable_joint_learnability_review_20261020/summary.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    assert review['status'] == 'completed' and review['validated_output_pairs'] == 24
    report_path = ROOT / review['source_report']
    assert file_digest(report_path) == review['source_report_sha256']
    report = json.loads(report_path.read_text(encoding='utf-8'))
    cache = ROOT / 'tmp/pointneuron2/local_runs/balanced_query_20261018/cache'
    manifest_path = cache / 'manifest.json'
    assert file_digest(manifest_path) == report['protected_files_sha256'][str(manifest_path)]
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cases = []
    terminal = review['rows'][-4:]
    for i, row in enumerate(terminal):
        record = manifest['records'][i]; path = cache / record['path']
        assert file_digest(path) == record['sha256']
        with np.load(path, allow_pickle=False) as saved:
            xyz, edges, roles = (saved[k].copy() for k in ('nodes', 'target_edges', 'roles'))
        xyz = xyz[:, 1:4]; degree = np.bincount(edges.ravel(), minlength=len(xyz))
        required = {tuple(sorted(map(int, e))) for e in edges}
        selected = required.copy()
        for missing in row['residual']['missing_required_edges']:
            selected.remove(tuple(sorted(missing['target_pair'])))
        for extra in row['residual']['nonreference_edges']:
            if None not in extra['target_pair']:
                selected.add(tuple(sorted(extra['target_pair'])))
        predicted_degree = np.bincount(np.asarray(sorted(selected)).ravel(), minlength=len(xyz))
        contractions = {}
        for missing in row['residual']['missing_required_edges']:
            common = [e for e in missing['selected_path'] if e['training_label'] == 1.]
            alternative = missing['nonreference_edges_on_path']
            assert len(common) == len(alternative) == 1 and len(missing['selected_path']) == 2
            a, b = common[0]['target_pair']
            assert degree[a] == 2 and degree[b] == 1 and roles[a] == 8 and roles[b] == 2
            assert predicted_degree[a] == 1 and predicted_degree[b] == 2
            contractions[b] = a
            cases.append({'crop': row['crop'], 'reference_arc_sample': a, 'reference_leaf': b,
                'reference_separation_voxels': float(np.linalg.norm(xyz[a] - xyz[b])),
                'predicted_separation_voxels': common[0]['length_voxels'],
                'reference_degrees': [int(degree[a]), int(degree[b])],
                'predicted_degrees': [int(predicted_degree[a]), int(predicted_degree[b])],
                'missing_edge': missing, 'interpretation': 'Leaf and adjacent arc sample exchange their order along the same three-node path.'})
        if contractions:
            def quotient(edge_set):
                pairs = {(contractions.get(u, u), contractions.get(v, v)) for u, v in edge_set}
                return {tuple(sorted((u, v))) for u, v in pairs if u != v}
            assert quotient(required) == quotient(selected)
    assert len(cases) == 2
    interval = [max(r['confidence']['maximum_unmatched_probability'] for r in terminal),
                min(r['confidence']['minimum_assigned_probability'] for r in terminal)]
    assert interval[0] < interval[1]
    result = {'schema': 'pointneuron2-stable-joint-residual-interpretation-v1',
        'scope': 'Post-hoc reference-conditioned interpretation only; original gates and .5 decisions unchanged.',
        'review_sha256': file_digest(review_path), 'script_sha256': file_digest(Path(__file__)),
        'dense_residuals': cases, 'dense_graph_equal_after_contracting_only_two_identified_reference_leaf_edges': True,
        'terminal_four_crop_node_probability_separation_interval': interval,
        'threshold_changed': False, 'learnability_gate': False, 'new_training_launched': False}
    output = review_path.with_name('residual_interpretation.json')
    if output.exists():
        raise FileExistsError('Preserve prior evidence')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
