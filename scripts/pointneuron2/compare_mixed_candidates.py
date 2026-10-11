"""Compare bright, permissive and mixed pools on corrected saved training crops."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.candidate_diagnostics import image_candidate_indices, positive_integer, select_candidate_rows
from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.mixed_candidates import mixed_candidate_indices
from pointneuron.data.training_cache import indices_to_point_array
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from scripts.pointneuron2.audit_sampling import reference_support
from scripts.pointneuron2.compare_foreground import digest, write_json
from scripts.pointneuron2.reference_gaps import segment_probe, segment_support


def distances_to_set(curve, xyz):
    return cKDTree(xyz).query(curve)[0] if len(xyz) else np.full(len(curve), np.inf)


def measure(curve, groups, probe, xyz, distance):
    return {'reference_support': reference_support(curve, xyz, np.ones(3)),
            'reference_groups': {name: reference_support(points, xyz, np.ones(3)) for name, points in groups.items()},
            'segment_support': segment_support(probe, xyz, distance)}


def summaries(rows, selectors):
    result = {}
    for method in ('bright', 'permissive', 'mixed'):
        result[method] = {}
        for stage in ('pool', *selectors):
            per_patch = []
            for row in rows:
                records = [v[stage] for v in row['methods'][method]['repeats']]
                def avg(fn):
                    values = [fn(r) for r in records]
                    values = [v for v in values if v is not None]
                    return float(np.mean(values)) if values else None
                per_patch.append({
                    'within_3': avg(lambda r: None if r['reference_support'] is None else r['reference_support']['within_3']),
                    'within_6': avg(lambda r: None if r['reference_support'] is None else r['reference_support']['within_6']),
                    **{name: avg(lambda r, name=name: None if r['reference_groups'][name] is None else r['reference_groups'][name]['within_3'])
                       for name in ('bright_available', 'fixed_missed_noise_available')},
                    'supported_length_fraction': avg(lambda r: None if r['segment_support'] is None else r['segment_support']['supported_length_fraction'])})
            result[method][stage] = {}
            for key in per_patch[0]:
                values = [p[key] for p in per_patch if p[key] is not None]
                result[method][stage][key] = {'mean': float(np.mean(values)) if values else None, 'patches': len(values)}
    return result


def render_chart(path, rows, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    annotated = [r for r in rows if r['reference_curve_points']]
    stages = ['pool', 'random'] + (['farthest'] if 'farthest' in config['selectors'] else [])
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 3.5 * len(stages)), sharex=True, constrained_layout=True)
    xs, width = np.arange(len(annotated)), .25
    for i, method in enumerate(('bright', 'permissive', 'mixed')):
        for ax, stage in zip(axes, stages):
            repeats = [[v[stage]['reference_support']['within_3'] for v in row['methods'][method]['repeats']] for row in annotated]
            ax.bar(xs + (i - 1) * width, [np.mean(v) for v in repeats], width, label=method,
                   yerr=[np.std(v) for v in repeats], capsize=2)
    titles = {'pool': f"{config['candidate_points']}-row candidate pools",
              'random': f"{config['selected_points']} randomly selected encoder inputs",
              'farthest': f"{config['selected_points']} spatially spread encoder inputs"}
    for ax, stage in zip(axes, stages):
        ax.set_ylim(0, 1.08); ax.set_ylabel('Reference coverage within 3 voxels')
        ax.set_title(titles[stage] + f": {len(config['seed_offsets'])} matched seeds; error bars show draw variability")
        ax.grid(axis='y', alpha=.2); ax.legend()
    axes[-1].set_xticks(xs, [f"{r['sample_index']}/{r['patch_index']}" for r in annotated])
    axes[-1].set_xlabel(f"TRAIN neuron / crop. {len(annotated)} annotated crops; unannotated crops retain null scores in JSON.")
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/pointneuron2/mixed_candidate_comparison.json')
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    config = json.loads(args.config.read_text())
    partition = json.loads(args.partition.read_text())
    train = {(r['sample_index'], r['sample_id']) for r in partition['partitions']['train']}
    if (comparison['status'] != 'completed' or comparison['failures'] or not comparison['patches']
            or comparison.get('volume_reader_version') != VOLUME_READER_VERSION
            or comparison['partition_sha256'] != digest(args.partition)
            or any(r['partition'] != 'train' or (r['sample_index'], r['sample_id']) not in train for r in comparison['patches'])):
        raise ValueError('Requires a completed corrected-reader comparison on the frozen training partition')
    if comparison['config']['methods'] != ['fixed', 'noise_hysteresis']:
        raise ValueError('Requires fixed and noise-hysteresis mask artifacts')
    for field in ('candidate_points', 'selected_points'):
        positive_integer(config[field], field)
    if config['selected_points'] > config['candidate_points']:
        raise ValueError('Selection exceeds candidate budget')
    if (not config['seed_offsets'] or len(set(config['seed_offsets'])) != len(config['seed_offsets'])
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in config['seed_offsets'])
            or not config['selectors'] or 'random' not in config['selectors']
            or len(set(config['selectors'])) != len(config['selectors'])
            or set(config['selectors']) - {'random', 'fixed_medial', 'farthest'}
            or not np.isfinite(config['bright_fraction']) or not 0 <= config['bright_fraction'] <= 1
            or config['reference_distance'] != 3. or config['reference_step'] != 1.):
        raise ValueError('Invalid frozen diagnostic settings')
    medial = MedialSamplingConfig(**comparison['medial_config'])
    if not np.array_equal(medial.spacing_xyz, [1, 1, 1]) or comparison['metric_spacing_xyz'] != [1, 1, 1]:
        raise ValueError('This diagnostic requires the saved isotropic voxel ruler')
    if args.output_dir.exists() or args.artifact_dir.exists():
        raise FileExistsError('Use fresh report and artifact directories')
    paths = ['scripts/pointneuron2/compare_mixed_candidates.py', 'src/pointneuron/data/mixed_candidates.py',
             'src/pointneuron/data/candidate_diagnostics.py', 'src/pointneuron/data/medial_sampling.py',
             'src/pointneuron/data/training_cache.py', 'src/pointneuron/data/manual_review.py',
             'scripts/pointneuron2/reference_gaps.py', 'scripts/pointneuron2/audit_sampling.py']
    report = {'status': 'running', 'scope': 'CPU saved-training-crop representation diagnostic; no training or reconstruction',
              'comparison_sha256': digest(args.comparison), 'partition_sha256': digest(args.partition),
              'config': config, 'config_sha256': digest(args.config), 'volume_reader_version': VOLUME_READER_VERSION,
              'source_sha256': {p: digest(REPO_ROOT / p) for p in paths},
              'versions': {n: version(n) for n in ('numpy', 'scipy')},
              'selection_rule': 'Image-only pools; no reference-selected voxels; seed repeats averaged within each patch',
              'guide_rule': 'Same saved full FIXED-mask guide for every pool; no re-extraction or guide cap',
              'reference_group_rule': 'Diagnostic only: available within 3 voxels of bright mask, or missed by bright but available in noise mask',
              'metric_units': 'isotropic voxel proximity and along-edge proximity gaps; not semantic precision or reconstruction F1',
              'patches': [], 'failures': [], 'promoted_method': None}
    args.output_dir.mkdir(parents=True); args.artifact_dir.mkdir(parents=True)
    report_path = args.output_dir / 'comparison.json'
    write_json(report_path, report)
    started = time.perf_counter()
    try:
        for patch in comparison['patches']:
            saved = {}
            for method, meta in patch['methods'].items():
                source = args.comparison.parent / meta['artifact']
                if digest(source) != meta['artifact_sha256']:
                    raise ValueError('Saved foreground artifact has changed')
                with np.load(source, allow_pickle=False) as data:
                    names = ['raw_image', 'mask', 'lower_xyz', 'reference_curve', 'reference_segments', 'points']
                    if method == 'fixed':
                        names.append('full_guide_xyz')
                    saved[method] = {name: data[name].copy() for name in names}
            a, b = saved['fixed'], saved['noise_hysteresis']
            for field in ('raw_image', 'lower_xyz', 'reference_curve', 'reference_segments'):
                if not np.array_equal(a[field], b[field]):
                    raise ValueError('Methods do not describe the same image/reference crop')
            image, lower, curve = a['raw_image'], a['lower_xyz'], a['reference_curve']
            _, height, width = image.shape
            bright, permissive = np.flatnonzero(a['mask']), np.flatnonzero(b['mask'])
            xyz = lambda indices: indices_to_point_array(indices, image.ravel(), width, height)[:, :3] + lower
            bright_dist, permissive_dist = distances_to_set(curve, xyz(bright)), distances_to_set(curve, xyz(permissive))
            groups = {'bright_available': curve[bright_dist <= 3],
                      'fixed_missed_noise_available': curve[(bright_dist > 3) & (permissive_dist <= 3)]}
            probe = segment_probe(a['reference_segments'], medial.spacing_xyz, config['reference_step'])
            row = {k: patch[k] for k in ('sample_index', 'sample_id', 'source_group', 'partition', 'patch_index', 'reference_curve_points')}
            row.update(reference_group_counts={k: len(v) for k, v in groups.items()},
                       foreground_counts={'bright': len(bright), 'permissive': len(permissive),
                                          'weak_only': len(np.setdiff1d(permissive, bright)),
                                          'bright_outside_permissive': len(np.setdiff1d(bright, permissive))},
                       full_guide_vertices=len(a['full_guide_xyz']), source_artifact_sha256={m: patch['methods'][m]['artifact_sha256'] for m in saved}, methods={})
            artifact_arrays = {}
            for method in ('bright', 'permissive', 'mixed'):
                records = []
                for offset in config['seed_offsets']:
                    seed = comparison['config']['seed'] + patch['sample_index'] * 10000 + patch['patch_index'] + offset * 1000000
                    if method == 'mixed':
                        chosen = mixed_candidate_indices(bright, permissive, config['candidate_points'], seed, config['bright_fraction'])
                    else:
                        indices = bright if method == 'bright' else permissive
                        chosen = image_candidate_indices(indices, image.ravel(), width, height, config['candidate_points'], 'uniform', seed)
                    points = indices_to_point_array(chosen, image.ravel(), width, height)
                    points[:, :3] += lower
                    if method != 'mixed' and offset == 0 and config['candidate_points'] == comparison['config']['candidate_points']:
                        np.testing.assert_array_equal(points, saved['fixed' if method == 'bright' else 'noise_hysteresis']['points'])
                    prefix = f'{method}_seed{offset}'
                    artifact_arrays[prefix + '_indices'] = chosen
                    artifact_arrays[prefix + '_points'] = points
                    record = {'seed': seed, 'seed_offset': offset, 'unique_pool_voxels': len(np.unique(chosen)),
                              'bright_pool_rows': int(np.isin(chosen, bright).sum()),
                              'pool': measure(curve, groups, probe, points[:, :3], config['reference_distance'])}
                    for selector in config['selectors']:
                        selection = select_candidate_rows(points, a['full_guide_xyz'], min(config['selected_points'], len(points)), selector, seed + 37, medial) if len(points) else np.empty(0, dtype=int)
                        metric = measure(curve, groups, probe, points[selection, :3], config['reference_distance'])
                        metric.update(unique_selected_voxels=len(np.unique(chosen[selection])), bright_selected_rows=int(np.isin(chosen[selection], bright).sum()))
                        record[selector] = metric
                        artifact_arrays[prefix + '_' + selector + '_rows'] = selection
                    records.append(record)
                row['methods'][method] = {'repeats': records}
            asset = args.artifact_dir / f"sample_{patch['sample_index']:04d}_patch_{patch['patch_index']:03d}.npz"
            np.savez_compressed(asset, **artifact_arrays)
            row.update(artifact=Path(os.path.relpath(asset, args.output_dir)).as_posix(), artifact_sha256=digest(asset))
            report['patches'].append(row)
            report['summary'] = summaries(report['patches'], config['selectors'])
            write_json(report_path, report)
            print(f"train {patch['sample_index']}/{patch['patch_index']} curve={len(curve)} recovered_group={len(groups['fixed_missed_noise_available'])} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        render_chart(args.output_dir / 'coverage.png', report['patches'], config)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(report_path, report)
    print(f'Report: {report_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
