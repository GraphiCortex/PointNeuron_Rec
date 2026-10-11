"""Training-only controlled foreground comparison; never changes training caches."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.foreground import foreground_mask
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.manual_review import spacing_metadata
from pointneuron.data.medial_sampling import image_medial_skeleton, load_medial_config
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import foreground_bounds, indices_to_point_array, skeleton_to_array, volume_data_array
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION, read_volume
from scripts.aggregate_proposals import effective_threshold, select_spatially_diverse_patches
from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def image_patch_boxes(image, threshold, config):
    """Same image-defined coverage rule as the existing cache, with compact metadata."""
    depth, height, width = image.shape
    bounds = foreground_bounds(image.ravel(), width, height, depth, threshold)
    if bounds is None:
        return []
    axes = [np.arange(bounds[0][i], bounds[1][i] + 1, config['patch_stride']) for i in range(3)]
    specs = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                center = np.array([x, y, z], dtype=float)
                lo = np.maximum(0, center.astype(int) - config['patch_radius'])
                hi = np.minimum([width, height, depth], center.astype(int) + config['patch_radius'] + 1)
                count = int(np.count_nonzero(image[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] > threshold))
                if count >= config['min_foreground_points']:
                    specs.append((len(specs), {'center': center, 'foreground': count, 'lower': lo, 'upper': hi - 1}))
    if len(specs) > config['max_patches_per_neuron']:
        specs = select_spatially_diverse_patches(specs, config['max_patches_per_neuron'],
                                                np.array([width, height, depth], dtype=float))
    return [spec for _, spec in specs]


def reference_roles(nodes, lower, upper):
    children = Counter(int(row[5]) for row in nodes if int(row[5]) >= 0)
    inside = np.all((nodes[:, 1:4] >= lower) & (nodes[:, 1:4] <= upper), axis=1)
    return {name: nodes[inside & np.array([rule(children[int(row[0])]) for row in nodes]), 1:4]
            for name, rule in (('terminals', lambda n: n == 0), ('junctions', lambda n: n > 1))}


def frozen_patch_boxes(comparison, sample_index, sample_id, dimensions):
    """Reuse exact image-defined boxes from a completed training-only experiment."""
    width, height, depth, _ = dimensions
    boxes = []
    for row in comparison['patches']:
        if row['sample_index'] != sample_index:
            continue
        if row['sample_id'] != sample_id or row['partition'] != 'train':
            raise ValueError('Frozen patch identity or partition mismatch')
        lo, hi = np.asarray(row['lower_xyz']), np.asarray(row['upper_xyz'])
        if (lo.shape != (3,) or hi.shape != (3,) or not np.issubdtype(lo.dtype, np.integer)
                or not np.issubdtype(hi.dtype, np.integer) or np.any(lo < 0) or np.any(hi < lo)
                or np.any(hi >= [width, height, depth])):
            raise ValueError('Invalid frozen patch bounds')
        boxes.append((row['patch_index'], {'lower': lo, 'upper': hi}))
    if not boxes or len({index for index, _ in boxes}) != len(boxes):
        raise ValueError('Missing or duplicate frozen patches')
    return boxes


def process_mask(image, mask, lower, medial, config, seed):
    """Uniform pools preserve raw intensity and use the same seed/budgets for each mask."""
    _, height, width = image.shape
    indices = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    budget = config['candidate_points']
    if not len(indices):
        return {'points': np.empty((0, 4)), 'full_guide_xyz': np.empty((0, 3)),
                'guide_edges': np.empty((0, 2), dtype=int), 'image_guide_xyz': np.empty((0, 3)),
                'foreground_xyz': np.empty((0, 3))}
    chosen = (rng.choice(indices, budget, replace=False) if len(indices) >= budget else
              np.r_[indices, rng.choice(indices, budget - len(indices))])
    points = indices_to_point_array(np.sort(chosen), image.ravel(), width, height)
    points[:, :3] += lower
    guide, edges = image_medial_skeleton(indices, width, height, medial)
    guide += lower
    cap = config['guide_points']
    bounded = (guide[np.sort(rng.choice(len(guide), cap, replace=False))]
               if cap is not None and len(guide) > cap else guide.copy())
    foreground = indices_to_point_array(indices, image.ravel(), width, height)[:, :3] + lower
    return {'points': points, 'full_guide_xyz': guide, 'guide_edges': edges,
            'image_guide_xyz': bounded, 'foreground_xyz': foreground}


def density_controls(curve, guide, candidates, lower, upper, seed, budget=1024, repeats=8):
    """Separate image association from coverage obtained simply by adding points.

    Null points are distinct voxel centers uniformly distributed in the SAME box.
    References only score already prepared sets. Reverse proximity is association
    with the available annotation, not precision against complete neuron labels.
    """
    if repeats < 2 or budget < 1:
        raise ValueError('Require at least two null repetitions and a positive budget')
    lower, upper = np.asarray(lower, dtype=int), np.asarray(upper, dtype=int)
    shape = upper - lower + 1
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError('Expected valid inclusive XYZ bounds')
    total = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    guide = np.unique(guide, axis=0)
    candidates = np.unique(candidates, axis=0)
    matched = guide[np.sort(rng.choice(len(guide), min(budget, len(guide)), replace=False))]
    result = {'rule': 'count-matched uniform voxel centers without replacement; same inclusive box',
              'seed': int(seed), 'null_repetitions': repeats, 'guide_budget': budget,
              'association_note': 'distance to available sampled SWC curve; not segmentation precision', 'sets': {}}
    tree = cKDTree(curve) if len(curve) else None
    for name, xyz in (('full_guide', guide), ('budget_guide', matched), ('candidate_pool', candidates)):
        count = len(xyz)
        if count > total:
            raise ValueError('Unique set exceeds the number of voxel centers in the box')
        observed = reference_support(curve, xyz, np.ones(3))
        nulls = []
        if tree is not None:
            for _ in range(repeats):
                flat = rng.choice(total, count, replace=False)
                null_xyz = np.column_stack(np.unravel_index(flat, tuple(shape))) + lower
                nulls.append(reference_support(curve, null_xyz, np.ones(3)))
        null_mean = ({f'within_{r}': float(np.mean([n[f'within_{r}'] for n in nulls]))
                      for r in (1, 3, 6)} if nulls else None)
        distances = tree.query(xyz)[0] if tree is not None and count else None
        result['sets'][name] = {
            'unique_points': count, 'reference_support': observed,
            'null_mean_support': null_mean,
            'null_std_support': {f'within_{r}': float(np.std([n[f'within_{r}'] for n in nulls]))
                                 for r in (1, 3, 6)} if nulls else None,
            'support_above_null': {f'within_{r}': observed[f'within_{r}'] - null_mean[f'within_{r}']
                                   for r in (1, 3, 6)} if nulls else None,
            'point_association': {f'within_{r}': float(np.mean(distances <= r)) for r in (1, 3, 6)}
                                 if distances is not None else None}
    return result


def render(path, image, lower, upper, segments, outputs, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(outputs)
    fig, axes = plt.subplots(len(methods), 3, figsize=(13, 3.5 * len(methods)), squeeze=False,
                              constrained_layout=True)
    vmax = max(1., float(np.quantile(image, .998)))
    for r, method in enumerate(methods):
        mask, arrays, meta = outputs[method]
        for col, (a, b, projection) in enumerate(((0, 1, image.max(0)), (0, 1, mask.max(0)),
                                                 (0, 2, image.max(1)))):
            ax = axes[r, col]
            ax.imshow(projection, cmap='gray', origin='lower', vmin=0,
                      vmax=1 if col == 1 else vmax,
                      extent=(lower[a] - .5, upper[a] + .5, lower[b] - .5, upper[b] + .5))
            if col != 1:
                g = arrays['full_guide_xyz']
                ax.scatter(g[:, a], g[:, b], color='#24b5f0', s=1.2, alpha=.5)
            for start, end in segments:
                ax.plot([start[a], end[a]], [start[b], end[b]], color='#ee4646', lw=.9)
            coverage = meta['reference_support']['full_guide']
            label_text = 'no local reference' if coverage is None else f"guide support {coverage['within_3']:.1%}"
            ax.set_title(f"{method}: {('raw XY + guide', 'mask XY', 'raw XZ + guide')[col]}\n"
                         f"mask {meta['foreground_fraction']:.1%}; {label_text}", fontsize=9)
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels)')
            ax.set_aspect('equal')
    fig.suptitle(f"TRAIN neuron {row['sample_index']}, patch {row['patch_index']} | {row['source_group']}\n"
                 'Red: reference diagnostic; cyan: full guide. Same raw display scale across methods; projections hide depth.', fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(rows, methods):
    result = {}
    for method in methods:
        available = [r['methods'][method] for r in rows if method in r['methods'] and 'error' not in r['methods'][method]]
        annotated = [v for v in available if v['reference_support']['full_guide'] is not None]
        result[method] = {'completed_patches': len(available), 'patches_with_reference_curve': len(annotated),
                          'empty_masks': sum(v['foreground_voxels'] == 0 for v in available),
                          'mean_foreground_fraction': float(np.mean([v['foreground_fraction'] for v in available])) if available else None,
                          'mean_support_within_3': {key: float(np.mean([v['reference_support'][key]['within_3'] for v in annotated])) if annotated else None
                                                   for key in ('all_foreground', 'full_guide', 'candidate_pool', 'bounded_guide')}}
        controlled = [v['density_controls']['sets'] for v in annotated if 'density_controls' in v]
        result[method]['mean_full_guide_vertices'] = float(np.mean([v['full_guide_vertices'] for v in available])) if available else None
        result[method]['density_controls'] = {
            key: {metric: float(np.mean([v[key][metric]['within_6'] for v in controlled])) if controlled else None
                  for metric in ('reference_support', 'null_mean_support', 'support_above_null')}
            for key in ('full_guide', 'budget_guide', 'candidate_pool')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT.parent / 'data/gold166')
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/pointneuron2/foreground_comparison.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, help='Fresh ignored directory for raw crop/mask arrays; report and figures stay in output-dir')
    parser.add_argument('--boxes-from', type=Path, help='Reuse exact boxes and verify source hashes from a completed comparison')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    partition = json.loads(args.partition.read_text(encoding='utf-8'))
    train = {r['sample_index']: r['sample_id'] for r in partition['partitions']['train']}
    if not config['sample_indices'] or any(i not in train for i in config['sample_indices']):
        raise ValueError('Foreground comparison is restricted to training neurons')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    artifact_dir = args.artifact_dir or args.output_dir / 'artifacts'
    if artifact_dir.exists():
        raise FileExistsError('Use a fresh artifact directory')
    frozen = None
    if args.boxes_from:
        frozen = json.loads(args.boxes_from.read_text(encoding='utf-8-sig'))
        if frozen['status'] != 'completed' or frozen['failures'] or frozen['partition_sha256'] != digest(args.partition):
            raise ValueError('Requires a completed comparison from the same partition')
        if any(row['partition'] != 'train' or train.get(row['sample_index']) != row['sample_id'] for row in frozen['patches']):
            raise ValueError('Frozen comparison must contain training patches only')
    medial = load_medial_config(REPO_ROOT / config['medial_config'])
    samples = scan_gold166(args.root)
    if any(samples[i].sample_id != train[i] for i in config['sample_indices']):
        raise ValueError('Dataset identity differs from frozen partition')
    args.output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    paths = [Path(__file__), REPO_ROOT / 'src/pointneuron/data/foreground.py',
             REPO_ROOT / 'src/pointneuron/data/medial_sampling.py', REPO_ROOT / 'scripts/aggregate_proposals.py',
             REPO_ROOT / 'scripts/pointneuron2/audit_sampling.py', REPO_ROOT / 'src/pointneuron/data/training_cache.py',
             REPO_ROOT / 'src/pointneuron/data/vaa3d_raw.py']
    report = {'status': 'running', 'scope': 'training-only mask comparison; no reconstruction or training run',
              'config': config, 'medial_config': medial.to_dict(), 'partition_sha256': digest(args.partition),
              'config_sha256': digest(args.config), 'source_sha256': {str(p.relative_to(REPO_ROOT)): digest(p) for p in paths},
              'versions': {n: version(n) for n in ('numpy', 'scipy', 'kimimaro')},
              'volume_reader_version': VOLUME_READER_VERSION, 'artifact_directory': str(artifact_dir.resolve()),
              'metric_spacing_xyz': [1, 1, 1], 'metric_units': 'voxel proximity; not segmentation precision or reconstruction F1',
              'patch_rule': 'image-defined coverage from fixed mask; identical boxes across methods; no reference filtering',
              'source_spacing_applied': False, 'sources': [], 'patches': [], 'failures': [], 'promoted_method': None,
              'inventory_note': 'Dataset inventory scans SWCs for valid source selection; only requested training sources enter metrics.'}
    if frozen is not None:
        report.update(boxes_from=str(args.boxes_from.resolve()), boxes_from_sha256=digest(args.boxes_from),
                      patch_rule='exact saved boxes from prior completed training-only mask comparison')
    report_path = args.output_dir / 'comparison.json'
    write_json(report_path, report)
    for sample_index in config['sample_indices']:
        sample = samples[sample_index]
        print(f'Loading training neuron {sample_index}: {sample.sample_id}', flush=True)
        try:
            volume = read_volume(sample.volume_path)
            width, height, depth, channels = volume.dimensions
            if channels != 1:
                raise ValueError('Expected a single-channel image')
            image = volume_data_array(volume).reshape(depth, height, width)
            threshold = effective_threshold(volume.header.datatype, 0, config['threshold_fraction'], data=image,
                                            mode=config.get('threshold_mode', 'dtype_fraction'))
            boxes = (list(enumerate(image_patch_boxes(image, threshold, config))) if frozen is None else
                     frozen_patch_boxes(frozen, sample_index, sample.sample_id, volume.dimensions))
            if frozen is not None:
                source = next(s for s in frozen['sources'] if s['sample_index'] == sample_index)
                if (source['sample_id'] != sample.sample_id or source['volume_sha256'] != digest(sample.volume_path)
                        or source['swc_sha256'] != digest(sample.swc_path)):
                    raise ValueError('Source image or diagnostic reference has changed since the frozen-box run')
            # Reference cannot affect boxes, masks, guides or pool selection.
            nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
            report['sources'].append({'sample_index': sample_index, 'sample_id': sample.sample_id,
                                      'dimensions': volume.dimensions, 'volume_sha256': digest(sample.volume_path),
                                      'swc_sha256': digest(sample.swc_path), 'spacing_metadata': spacing_metadata(sample.pixel_size_path),
                                      'datatype': volume.header.datatype, 'decoded_range': [int(image.min()), int(image.max())],
                                      'fixed_threshold': threshold,
                                      'selected_patch_count': len(boxes)})
            if not boxes:
                raise ValueError('No original foreground patches; retained as source failure')
            for patch_index, box in boxes:
                lower, upper = box['lower'], box['upper']
                halo = np.ceil(4 * config['background_sigma'] / np.asarray(medial.spacing_xyz)).astype(int)
                context_lo = np.maximum(0, lower - halo)
                context_hi = np.minimum([width, height, depth], upper + halo + 1)
                context = image[context_lo[2]:context_hi[2], context_lo[1]:context_hi[1], context_lo[0]:context_hi[0]]
                core_lo, core_hi = lower - context_lo, upper + 1 - context_lo
                core = tuple(slice(int(core_lo[i]), int(core_hi[i])) for i in (2, 1, 0))
                crop = context[core]
                curve, segments, crossings = clipped_reference(nodes, lower, upper)
                roles = reference_roles(nodes, lower, upper)
                row = {'sample_index': sample_index, 'sample_id': sample.sample_id, 'source_group': sample.sample_id.split('/')[0],
                       'partition': 'train', 'patch_index': patch_index, 'lower_xyz': lower.tolist(), 'upper_xyz': upper.tolist(),
                       'reference_curve_points': len(curve), 'boundary_crossing_edges': crossings,
                       'terminal_count': len(roles['terminals']), 'junction_count': len(roles['junctions']), 'methods': {}}
                outputs = {}
                for method in config['methods']:
                    start = time.perf_counter()
                    try:
                        mask, score, info = foreground_mask(context, method, threshold, config['background_sigma'],
                                                            medial.spacing_xyz, core_slices=core,
                                                            noise_high=config.get('noise_high', 4.), noise_low=config.get('noise_low', 2.))
                        arrays = process_mask(crop, mask, lower, medial, config, config['seed'] + sample_index * 10000 + patch_index)
                        _, components = label(mask, structure=np.ones((3, 3, 3)))
                        sets = {'all_foreground': arrays['foreground_xyz'], 'full_guide': arrays['full_guide_xyz'],
                                'candidate_pool': arrays['points'][:, :3], 'bounded_guide': arrays['image_guide_xyz']}
                        meta = {**info, 'foreground_voxels': int(mask.sum()), 'foreground_fraction': float(mask.mean()),
                                'connected_components_26': components, 'full_guide_vertices': len(arrays['full_guide_xyz']),
                                'guide_edges': len(arrays['guide_edges']), 'candidate_rows': len(arrays['points']),
                                'bounded_guide_vertices': len(arrays['image_guide_xyz']),
                                'reference_support': {name: reference_support(curve, xyz, np.ones(3)) for name, xyz in sets.items()},
                                'role_support': {role: {name: reference_support(points, xyz, np.ones(3)) for name, xyz in sets.items()}
                                                 for role, points in roles.items()}, 'seconds': time.perf_counter() - start}
                        if config.get('density_controls', False):
                            meta['density_controls'] = density_controls(
                                curve, arrays['full_guide_xyz'], arrays['points'][:, :3], lower, upper,
                                config['seed'] + sample_index * 10000 + patch_index,
                                budget=config['candidate_points'], repeats=config.get('null_repetitions', 8))
                        stem = f'sample_{sample_index:04d}_patch_{patch_index:03d}_{method}'
                        artifact = artifact_dir / f'{stem}.npz'
                        np.savez_compressed(artifact, raw_image=crop, mask=mask, score=score, lower_xyz=lower,
                                            reference_curve=curve, reference_segments=np.asarray(segments).reshape(-1, 2, 3),
                                            reference_terminals=roles['terminals'], reference_junctions=roles['junctions'],
                                            **{k: v for k, v in arrays.items() if k != 'foreground_xyz'})
                        meta.update(artifact=Path(os.path.relpath(artifact, args.output_dir)).as_posix(), artifact_sha256=digest(artifact))
                        row['methods'][method] = meta
                        outputs[method] = (mask, arrays, meta)
                        support = meta['reference_support']['full_guide']
                        print(f"  patch {patch_index} {method}: foreground={mask.sum()} guide={meta['full_guide_vertices']} "
                              f"support={None if support is None else round(support['within_3'], 3)}", flush=True)
                    except Exception as exc:
                        row['methods'][method] = {'error': str(exc), 'error_type': type(exc).__name__}
                        report['failures'].append({'sample_index': sample_index, 'patch_index': patch_index, 'method': method, 'error': str(exc)})
                if outputs:
                    figure = args.output_dir / f'sample_{sample_index:04d}_patch_{patch_index:03d}.png'
                    render(figure, crop, lower, upper, segments, outputs, row)
                    row['figure'] = figure.name
                report['patches'].append(row)
                report['summary'] = summarize(report['patches'], config['methods'])
                write_json(report_path, report)
        except Exception as exc:
            report['failures'].append({'sample_index': sample_index, 'error': str(exc), 'error_type': type(exc).__name__})
            print(f'FAILED source {sample_index}: {exc}', flush=True)
            write_json(report_path, report)
    report['status'] = 'failed' if report['failures'] else 'completed'
    report['limitations'] = [f"{len(config['sample_indices'])} training neurons, small image-defined patch cap; not a dataset-wide estimate.",
                             'Masks are threshold rules, not verified semantic segmentation; incomplete SWCs limit precision claims.',
                             'Original isotropic extraction spacing held fixed to isolate masks; source metadata recorded, not applied.',
                             'Background filtering uses a 4-sigma halo; automatic thresholds use the core; fixed cutoff uses the configured full-image range.',
                             'Uniform nulls assess point density, not anatomical topology or background truth; count-matched sets do not correct all spatial confounds.',
                             'Uniform candidate preparation held fixed; guide cap is explicit or null for full guides; unannotated patches have null reference metrics.',
                             'No method is promoted; no validation or reserved-test neuron evaluated.']
    write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)
    return bool(report['failures'])


if __name__ == '__main__':
    raise SystemExit(main())
