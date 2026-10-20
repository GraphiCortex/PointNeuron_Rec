"""Training-only CPU audit and separate four-crop canonical target cache. No fit."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from pointneuron.data.canonical_curve_targets import CanonicalTargetBuilder, TARGET_SCHEMA
from pointneuron.data.connectivity_targets import full_reference_components
from pointneuron.data.curve_targets import patch_targets, target_arrays
from pointneuron.data.swc import parse_swc
from pointneuron.graph.proposal_graph import build_candidate_graph
from pointneuron.graph.representation_oracle import candidate_report, clipped_reference_graph, geometry_report, target_graph
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compact(metric):
    return {k: metric[k] for k in ('reference_length', 'export_length', 'max_reference_distance', 'max_export_distance', 'tolerances')}


def run(args):
    started = time.perf_counter(); torch.set_num_threads(2)
    config_path = ROOT / 'configs/pointneuron2/canonical_targets.json'; config = read(config_path)
    directory = (args.output or ROOT / config['output']).resolve()
    if not directory.is_relative_to((ROOT / 'tmp/pointneuron2').resolve()) or directory.exists():
        raise ValueError('Choose a fresh output under ignored tmp/pointneuron2')
    protected = {}
    for key in ('source_audit', 'partition', 'four_crop_manifest'):
        path = ROOT / config[key]
        if digest(path) != config[key + '_sha256']:
            raise ValueError(f'Changed protected input: {key}')
        protected[str(path)] = digest(path)
    source = read(ROOT / config['source_audit']); partition = read(ROOT / config['partition'])
    assert source['status'] == 'completed' and source['verification'] == 'passed'
    assert source['partition_sha256'] == config['partition_sha256']
    allowed = {r['sample_index']: r['sample_id'] for r in partition['partitions']['train']}
    assert all(r['sample_index'] not in allowed and r['sample_id'] not in allowed.values()
               for role in ('val', 'test') for r in partition['partitions'][role])
    assert len(source['records']) == 262 and len(source['sources']) == 33
    directory.mkdir(parents=True)
    report = {'schema': config['schema'], 'status': 'running', 'config': config,
        'scope': '262 existing TRAINING crops only; target preparation, no model forwards or training',
        'records': [], 'sources': [], 'four_crop_cache': [], 'protected_files_sha256': protected}
    write(directory / 'summary.json', report)
    try:
        builders = {}
        for row in source['sources']:
            assert row['sample_index'] in allowed
            path = (args.data_root / row['path']).resolve()
            assert path.is_relative_to(args.data_root.resolve()) and digest(path) == row['sha256']
            assert Path(row['path']).parent.as_posix() == allowed[row['sample_index']]
            protected[str(path)] = row['sha256']
            builder = CanonicalTargetBuilder(parse_swc(path), **config['target_policy'])
            builders[row['sample_index']] = builder
            report['sources'].append({'sample_index': row['sample_index'], 'source_sha256': row['sha256'],
                'candidate_removals': len(builder.decisions), 'approved_global_removals': len(builder.removed), 'decisions': builder.decisions})
        selected = {tuple(k) for k in config['training_crops']}; patches = {}
        for index, row in enumerate(source['records']):
            assert allowed[row['sample_index']] == row['sample_id']
            builder = builders[row['sample_index']]; center = row['patch_center']; radius = config['patch_radius']
            legacy = patch_targets(builder.geometry, center, radius, builder.step)
            assert len(legacy) == row['steps']['8.0']['targets']
            canonical = builder.patch(center, radius)
            legacy_graph = target_graph(builder.tree, [(center, radius)], builder.step)
            old_candidates = candidate_report(legacy_graph, config['candidate_neighbors'], config['candidate_radius'])['knn_union_radius']
            new_candidates = candidate_report(canonical.graph, config['candidate_neighbors'], config['candidate_radius'])['knn_union_radius']
            def missing_identities(graph, candidates):
                return {frozenset((graph.identities[u], graph.identities[v])) for u, v in candidates['missing_edge_indices']}
            added_missing = missing_identities(canonical.graph, new_candidates) - missing_identities(legacy_graph, old_candidates)
            counts = Counter(role for t in legacy for role in t['roles'])
            new_counts = Counter(role for t in canonical.targets for role in t['roles'])
            assert all(counts[role] == new_counts[role] for role in ('root', 'leaf', 'junction', 'crop_boundary'))
            assert (not canonical.targets) == (not legacy)  # No invented background supervision.
            record = {'sample_index': row['sample_index'], 'patch_index': row['patch_index'], 'sample_id': row['sample_id'],
                'legacy_targets': len(legacy), 'canonical_targets': len(canonical.targets), 'removed': canonical.removed,
                'certificates': canonical.certificates, 'protected_roles': {k: counts[k] for k in ('root', 'leaf', 'junction', 'crop_boundary')},
                'topology': canonical.topology, 'legacy_candidates': old_candidates, 'canonical_candidates': new_candidates,
                'newly_unrepresented_required_edges': len(added_missing)}
            key = (row['sample_index'], row['patch_index'])
            if key in selected:
                patches[key] = canonical
                reference = clipped_reference_graph(builder.tree, [(center, radius)])
                record['legacy_geometry'] = compact(geometry_report(reference, legacy_graph, config['quadrature_step']))
                record['canonical_geometry'] = compact(geometry_report(reference, canonical.graph, config['quadrature_step']))
                # Match the actual stable-ID candidate builder used by the model.
                actual = build_candidate_graph(torch.as_tensor(canonical.graph.xyz), torch.arange(len(canonical.targets)),
                    neighbors=config['candidate_neighbors'], radius=config['candidate_radius'])
                represented = {tuple(sorted(map(int, e))) for e in actual.edges}
                record['actual_builder_missing_edges'] = sum(tuple(sorted(e)) not in represented for e in canonical.graph.edges)
            report['records'].append(record)
            if (index + 1) % 25 == 0 or key in selected:
                print(f"CPU {index + 1}/262 crop={key} targets={len(legacy)}->{len(canonical.targets)} missing_candidates={new_candidates['missing_required_edges']}", flush=True)
                write(directory / 'summary.json', report)
        assert set(patches) == selected
        manifest_path = ROOT / config['four_crop_manifest']; manifest = read(manifest_path)
        cache_records = []
        for i, old in enumerate(manifest['records']):
            key = (old['sample_index'], old['patch_index']); assert list(key) == config['training_crops'][i]
            source_path = manifest_path.parent / old['path']; assert digest(source_path) == old['sha256']
            protected[str(source_path)] = old['sha256']
            with np.load(source_path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            patch = patches[key]; nodes, roles = target_arrays(patch.targets, config['query_capacity'])
            arrays.update(nodes=nodes, roles=roles, target_edges=np.array(patch.graph.edges, dtype=np.int64).reshape(-1, 2),
                target_components=full_reference_components(builders[key[0]].tree, patch.graph.identities),
                junction_mask=np.array([False] * len(nodes)), target_identities=np.array(json.dumps(patch.graph.identities)),
                target_policy=np.array(json.dumps(builders[key[0]].policy)))
            degree = np.bincount(arrays['target_edges'].ravel(), minlength=len(nodes)); arrays['junction_mask'] = degree >= 3
            destination = directory / 'four_crop_cache' / old['path']; destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **arrays)
            with np.load(destination, allow_pickle=False) as saved, np.load(source_path, allow_pickle=False) as original:
                for name in original.files:
                    if name not in ('nodes', 'roles', 'target_edges', 'target_components', 'junction_mask'):
                        assert np.array_equal(saved[name], original[name]), name
            cached = {'crop': list(key), 'path': str(destination.relative_to(directory)), 'sha256': digest(destination),
                'source_path': str(source_path.relative_to(ROOT)), 'source_sha256': old['sha256'], 'targets': len(nodes),
                'points_unchanged': True, 'source_geometry_and_metadata_unchanged': True}
            cache_records.append(cached)
        report['four_crop_cache'] = cache_records
        rows = report['records']; four = [r for r in rows if (r['sample_index'], r['patch_index']) in selected]
        geometry_ok = all(r['canonical_geometry']['tolerances']['2.0'][metric] >= r['legacy_geometry']['tolerances']['2.0'][metric]
                          - config['maximum_precision_recall_decline_at_2'] - 1e-12 for r in four for metric in ('precision', 'recall'))
        report['preparation_gate'] = {'structural_topology_all_crops': all(r['topology']['passed'] for r in rows),
            'no_new_candidate_losses': all(r['newly_unrepresented_required_edges'] == 0 for r in rows),
            'four_crop_candidate_recall_complete': all(r['actual_builder_missing_edges'] == 0 for r in four),
            'four_crop_no_precision_recall_decline_at_2': geometry_ok,
            'all_targets_fit_capacity': all(r['canonical_targets'] <= config['query_capacity'] for r in rows),
            'exact_input_clouds_preserved': all(r['points_unchanged'] for r in cache_records)}
        report['preparation_gate']['passed'] = all(report['preparation_gate'].values())
        report['totals'] = {'training_neurons': len(builders), 'training_crops': len(rows),
            'legacy_targets': sum(r['legacy_targets'] for r in rows), 'canonical_targets': sum(r['canonical_targets'] for r in rows),
            'changed_crops': sum(bool(r['removed']) for r in rows), 'empty_unknown_crops': sum(r['canonical_targets'] == 0 for r in rows),
            'guarded_global_candidates': sum(sum(not d['remove'] for d in s['decisions']) for s in report['sources'])}
        report['source_files_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), config_path, ROOT / 'src/pointneuron/data/canonical_curve_targets.py',
            ROOT / 'src/pointneuron/data/curve_targets.py', ROOT / 'src/pointneuron/graph/representation_oracle.py',
            ROOT / 'src/pointneuron/graph/proposal_graph.py', ROOT / 'src/pointneuron/data/connectivity_targets.py')}
        for path, expected in protected.items():
            assert digest(path) == expected
        assert not torch.cuda.is_initialized()
        report.update(status='completed', elapsed_seconds=time.perf_counter() - started, cuda_initialized=False, model_forwards=0,
            environment={'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
            optimizer_updates=0, next_step='Review preparation gates and agree a matched fit; no training launched or legacy cache changed.')
        write(directory / 'summary.json', report)
        write(directory / 'four_crop_cache/manifest.json', {'schema': TARGET_SCHEMA + '-experimental-cache', 'status': 'completed',
            'training_ready': report['preparation_gate']['passed'], 'config': config, 'records': cache_records})
        print(json.dumps({'totals': report['totals'], 'preparation_gate': report['preparation_gate'], 'four_crops': four, 'output': str(directory)}, indent=2))
    except BaseException as exc:
        report.update(status='failed', error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write(directory / 'summary.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'data/gold166')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())
