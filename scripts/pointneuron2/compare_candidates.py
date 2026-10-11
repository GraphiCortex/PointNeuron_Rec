"""Training-only candidate/selection comparison from saved image-mask artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.candidate_diagnostics import image_candidate_indices, select_candidate_rows, positive_integer
from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.training_cache import indices_to_point_array
from scripts.pointneuron2.audit_sampling import reference_support
from scripts.pointneuron2.reference_gaps import segment_probe, segment_support


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_config(config):
    positive_integer(config['selected_points'], 'selected_points')
    positive_integer(config['cell_size'], 'cell_size')
    for count in config['candidate_budgets']:
        positive_integer(count, 'candidate budget')
        if count < config['selected_points']:
            raise ValueError('Every candidate budget must cover selected_points')
    for key, allowed in (('candidate_methods', {'uniform', 'spatial'}),
                         ('selectors', {'random', 'fixed_medial', 'farthest'})):
        if not config[key] or set(config[key]) - allowed or len(set(config[key])) != len(config[key]):
            raise ValueError(f'Invalid {key}')
    for key in ('candidate_budgets', 'seed_offsets'):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f'Empty or repeated {key}')
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in config['seed_offsets']):
        raise ValueError('Seed offsets must be nonnegative integers')
    for key in ('reference_distance', 'reference_step'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')


def measurements(curve, roles, probe, xyz, spacing, distance):
    return {'reference_support': reference_support(curve, xyz, spacing),
            'reference_segment_support': segment_support(probe, xyz, distance),
            'role_support': {name: reference_support(coords, xyz, spacing) if coords is not None else None
                             for name, coords in roles.items()}}


def summaries(patches):
    # Seed repeats are averaged inside each patch; they are not independent neurons.
    groups = defaultdict(list)
    for patch in patches:
        if patch['reference_curve_points'] <= 1:
            continue
        local = defaultdict(list)
        for mask, output in patch['methods'].items():
            for variant in output.get('variants', []):
                for selector, metrics in variant['selectors'].items():
                    key = (mask, variant['candidate_method'], variant['candidate_budget'], selector)
                    local[key].append(metrics)
        for key, values in local.items():
            gaps = [v['reference_segment_support'] for v in values if v['reference_segment_support'] is not None]
            groups[key].append({'sample_index': patch['sample_index'], 'source_group': patch['source_group'],
                                'within_3': float(np.mean([v['reference_support']['within_3'] for v in values])),
                                'supported_length_fraction': float(np.mean([v['supported_length_fraction'] for v in gaps])) if gaps else None,
                                'max_unsupported_run_length': float(np.mean([v['max_unsupported_run_length'] for v in gaps])) if gaps else None})
    result = []
    for (mask, method, budget, selector), values in sorted(groups.items()):
        def means(rows):
            return {key: float(np.mean([v[key] for v in rows if v[key] is not None]))
                    if any(v[key] is not None for v in rows) else None
                    for key in ('within_3', 'supported_length_fraction', 'max_unsupported_run_length')}
        result.append({'mask_method': mask, 'candidate_method': method, 'candidate_budget': budget, 'selector': selector,
                       'reference_patches': len(values), 'reference_neurons': len({v['sample_index'] for v in values}),
                       'mean_over_patches_after_seed_average': means(values),
                       'by_source_group': {source: means([v for v in values if v['source_group'] == source])
                                           for source in sorted({v['source_group'] for v in values})}})
    return result


def render(path, image, lower, segments, method, views):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices = [('uniform', 1024, 'random'), ('uniform', 1024, 'fixed_medial'), ('spatial', 1024, 'fixed_medial'),
               ('uniform', 4096, 'fixed_medial'), ('uniform', 4096, 'farthest')]
    choices = [choice for choice in choices if choice in views]
    if not choices:
        return False
    depth, height, width = image.shape
    upper = lower + np.array([width, height, depth]) - 1
    fig, axes = plt.subplots(2, len(choices), figsize=(4 * len(choices), 7), squeeze=False, constrained_layout=True)
    vmax = max(1., float(np.quantile(image, .998)))
    for r, (a, b, projection) in enumerate(((0, 1, image.max(0)), (0, 2, image.max(1)))):
        for c, choice in enumerate(choices):
            xyz, metric = views[choice]
            ax = axes[r, c]
            ax.imshow(projection, origin='lower', cmap='gray', vmin=0, vmax=vmax,
                      extent=(lower[a] - .5, upper[a] + .5, lower[b] - .5, upper[b] + .5))
            ax.scatter(xyz[:, a], xyz[:, b], s=4, color='#49e39a', alpha=.7)
            for start, end in segments:
                ax.plot([start[a], end[a]], [start[b], end[b]], color='#ff4747', lw=1.2)
            support = metric['reference_support']
            text = 'no curve metric' if support is None else f"proximity at 3 voxels: {support['within_3']:.1%}"
            ax.set_title(f'{choice[0]} pool {choice[1]}, {choice[2]}\n{text}', fontsize=9)
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels)')
            ax.set_aspect('equal')
    fig.suptitle(f'{method} mask | green: selected inputs, red: clipped reference | first seed only\n'
                 'XY / XZ projections; no training or predicted reconstruction', fontsize=11)
    fig.savefig(path, dpi=125); plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/pointneuron2/candidate_comparison.json')
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--limit-patches', type=int, help='Image-order prefix for a preparation smoke only')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); validate_config(config)
    if args.limit_patches is not None:
        positive_integer(args.limit_patches, 'limit-patches')
    comparison = json.loads(args.comparison.read_text())
    if comparison['status'] != 'completed' or comparison['failures']:
        raise ValueError('Requires a completed foreground comparison with no failures')
    partition = json.loads(args.partition.read_text())
    train = {(r['sample_index'], r['sample_id']) for r in partition['partitions']['train']}
    if comparison['partition_sha256'] != digest(args.partition) or not comparison['patches']:
        raise ValueError('Comparison partition mismatch or empty comparison')
    if any(p['partition'] != 'train' or (p['sample_index'], p['sample_id']) not in train for p in comparison['patches']):
        raise ValueError('Candidate comparison must use training neurons only')
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    medial = MedialSamplingConfig(**comparison['medial_config'])
    if not np.array_equal(comparison['metric_spacing_xyz'], medial.spacing_xyz):
        raise ValueError('Comparison and candidate diagnostic rulers must agree')
    if config['reference_distance'] != 3.:
        raise ValueError('Current summaries are at three units; set reference_distance to 3')
    args.output_dir.mkdir(parents=True); (args.output_dir / 'artifacts').mkdir()
    paths = ['scripts/pointneuron2/compare_candidates.py', 'scripts/pointneuron2/reference_gaps.py',
             'src/pointneuron/data/candidate_diagnostics.py', 'src/pointneuron/data/manual_review.py',
             'src/pointneuron/data/medial_sampling.py', 'scripts/pointneuron2/audit_sampling.py',
             'src/pointneuron/data/training_cache.py']
    report = {'status': 'running', 'scope': 'Training-only CPU input diagnostic; no learned model or training run',
              'config': config, 'comparison_sha256': digest(args.comparison), 'partition_sha256': digest(args.partition),
              'config_sha256': digest(args.config), 'source_sha256': {p: digest(REPO_ROOT / p) for p in paths},
              'versions': {name: version(name) for name in ('numpy', 'scipy')}, 'medial_config': medial.to_dict(),
              'limit_patches': args.limit_patches, 'patches': [], 'failures': [], 'promoted_method': None,
              'selection_rule': 'Fixed NumPy Gumbel top-k probes: same exponential/uniform distribution, different RNG stream from Torch',
              'guide_rule': 'Every saved full_guide_xyz vertex; no guide cap or re-extraction',
              'metric_units': 'Saved isotropic voxel ruler; proximity and proximity gaps, not segmentation precision or graph connectivity',
              'summary_rule': 'Seed-average within each patch, then equal-weight patch means; exclude zero/singleton reference curves',
              'figure_rule': 'First patch with >1 clipped reference points per neuron, first seed, each mask; all patches enter report'}
    path = args.output_dir / 'comparison.json'; write(path, report)
    patches = comparison['patches'][:args.limit_patches] if args.limit_patches else comparison['patches']
    plotted = set(); started = time.perf_counter()
    try:
        for i, patch in enumerate(patches):
            result = {k: patch[k] for k in ('sample_index', 'sample_id', 'source_group', 'partition', 'patch_index', 'reference_curve_points')}
            result['methods'] = {}
            print(f"Patch {i + 1}/{len(patches)}: train {patch['sample_index']}/{patch['patch_index']}", flush=True)
            for method, metadata in patch['methods'].items():
                try:
                    source = (args.comparison.parent / metadata['artifact']).resolve()
                    if digest(source) != metadata['artifact_sha256']:
                        raise ValueError('Saved mask artifact changed')
                    with np.load(source, allow_pickle=False) as data:
                        image, mask = data['raw_image'].copy(), data['mask'].copy()
                        lower, curve, segments = data['lower_xyz'].copy(), data['reference_curve'].copy(), data['reference_segments'].copy()
                        guide = data['full_guide_xyz'].copy(); original_points = data['points'].copy()
                        roles = {name: data[f'reference_{name}'].copy() if f'reference_{name}' in data else None
                                 for name in ('terminals', 'junctions')}
                    if len(guide) != metadata['full_guide_vertices']:
                        raise ValueError('Incomplete saved full guide')
                    indices = np.flatnonzero(mask); _, height, width = image.shape
                    probe = segment_probe(segments, medial.spacing_xyz, config['reference_step'])
                    output = {'source_artifact_sha256': metadata['artifact_sha256'], 'foreground_voxels': len(indices),
                              'full_guide_vertices': len(guide), 'full_guide': measurements(curve, roles, probe, guide,
                                                                                        medial.spacing_xyz, config['reference_distance']),
                              'role_coordinates_available': all(v is not None for v in roles.values()), 'variants': []}
                    arrays, views = {}, {}
                    for candidate_method in config['candidate_methods']:
                        for budget in config['candidate_budgets']:
                            for offset in config['seed_offsets']:
                                seed = comparison['config']['seed'] + patch['sample_index'] * 10000 + patch['patch_index'] + offset * 1000000
                                chosen = image_candidate_indices(indices, image.ravel(), width, height, budget,
                                                                 candidate_method, seed, config['cell_size'], medial.spacing_xyz)
                                points = indices_to_point_array(chosen, image.ravel(), width, height); points[:, :3] += lower
                                if candidate_method == 'uniform' and offset == 0 and budget == comparison['config']['candidate_points']:
                                    np.testing.assert_array_equal(points, original_points)
                                prefix = f'{candidate_method}_{budget}_seed{offset}'
                                arrays[prefix + '_points'] = points; arrays[prefix + '_indices'] = chosen
                                variant = {'candidate_method': candidate_method, 'candidate_budget': budget, 'seed_offset': offset,
                                           'seed': seed, 'candidate_rows': len(points), 'unique_candidate_voxels': len(np.unique(chosen)),
                                           'pool': measurements(curve, roles, probe, points[:, :3], medial.spacing_xyz, config['reference_distance']),
                                           'array_prefix': prefix, 'selectors': {}}
                                for selector in config['selectors']:
                                    rows = select_candidate_rows(points, guide, config['selected_points'], selector, seed + 37, medial)
                                    arrays[prefix + '_' + selector + '_rows'] = rows
                                    xyz = points[rows, :3]
                                    metric = measurements(curve, roles, probe, xyz, medial.spacing_xyz, config['reference_distance'])
                                    metric.update(selected_rows=len(rows), unique_selected_voxels=len(np.unique(chosen[rows])))
                                    variant['selectors'][selector] = metric
                                    if offset == config['seed_offsets'][0]:
                                        views[(candidate_method, budget, selector)] = (xyz, metric)
                                output['variants'].append(variant)
                    stem = f"sample_{patch['sample_index']:04d}_patch_{patch['patch_index']:03d}_{method}"
                    asset = args.output_dir / 'artifacts' / (stem + '.npz')
                    np.savez_compressed(asset, **arrays)
                    output.update(artifact=asset.relative_to(args.output_dir).as_posix(), artifact_sha256=digest(asset))
                    if patch['sample_index'] not in plotted and len(curve) > 1:
                        figure = args.output_dir / (stem + '.png')
                        if render(figure, image, lower, segments, method, views):
                            output['figure'] = figure.name
                    result['methods'][method] = output
                    print(f"  {method}: full guide {len(guide)}, {len(output['variants'])} pool variants; elapsed={time.perf_counter() - started:.1f}s", flush=True)
                except Exception as exc:
                    result['methods'][method] = {'error': str(exc)}
                    report['failures'].append({'sample_index': patch['sample_index'], 'patch_index': patch['patch_index'],
                                               'method': method, 'error': str(exc)})
                    print(f'  FAILED {method}: {exc}', flush=True)
            if any('figure' in output for output in result['methods'].values()):
                plotted.add(patch['sample_index'])
            report['patches'].append(result); report['summary'] = summaries(report['patches']); write(path, report)
        report['status'] = 'failed' if report['failures'] else 'completed'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=str(exc)); raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started; write(path, report)
    print(f'Candidate report: {path.resolve()}', flush=True)
    return bool(report['failures'])


if __name__ == '__main__':
    raise SystemExit(main())
