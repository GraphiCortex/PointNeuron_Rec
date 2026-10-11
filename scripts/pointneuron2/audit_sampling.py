"""Audit original image support and bounded-pool losses; references never select inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.foreground import fractional_intensity_threshold
from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_axis
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import (indices_to_point_array, patch_foreground_indices,
                                          skeleton_to_array, volume_data_array)
from pointneuron.data.vaa3d_raw import read_volume


def clipped_reference(nodes, lower, upper, step=1.):
    """Clip every full-reference edge to the closed patch box before resampling.

    Returns nodes/curve and edge segments including crossings with no endpoint
    inside. This is a diagnostic only; it does not alter training labels.
    """
    lower, upper = np.asarray(lower), np.asarray(upper)
    if step <= 0 or not np.isfinite(step) or np.any(lower > upper):
        raise ValueError('Invalid reference box or resampling step')
    xyz = nodes[:, 1:4]
    pieces = [xyz[np.all((xyz >= lower) & (xyz <= upper), axis=1)]]
    lookup = {int(n[0]): n[1:4] for n in nodes}
    segments = []
    crossings = 0
    for n in nodes:
        parent = lookup.get(int(n[5]))
        if parent is None:
            continue
        child = n[1:4]
        delta = child - parent
        entry, leave = 0., 1.
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if not lower[axis] <= parent[axis] <= upper[axis]:
                    entry, leave = 1., 0.
                    break
            else:
                bounds = sorted(((lower[axis] - parent[axis]) / delta[axis],
                                 (upper[axis] - parent[axis]) / delta[axis]))
                entry, leave = max(entry, bounds[0]), min(leave, bounds[1])
        if entry > leave:
            continue
        a, b = parent + entry * delta, parent + leave * delta
        segments.append((a, b))
        crossings += int(entry > 0 or leave < 1)
        intervals = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pieces.append(np.linspace(a, b, intervals + 1))
    curve = np.unique(np.concatenate(pieces), axis=0)
    return curve, segments, crossings


def reference_support(curve, xyz, spacing):
    if not len(curve):
        return None
    if not len(xyz):
        return {'within_1': 0., 'within_3': 0., 'within_6': 0., 'median_distance': None}
    distances = cKDTree(xyz * spacing).query(curve * spacing)[0]
    return {**{f'within_{r}': float(np.mean(distances <= r)) for r in (1, 3, 6)},
            'median_distance': float(np.median(distances))}


def draw_case(axes, case):
    crop, lower, upper, curve, segments, guide, row = case
    vmax = max(1., float(np.quantile(crop, .998)))
    for column, (a, b, projection, with_guide) in enumerate(
            ((0, 1, crop.max(axis=0), False), (0, 1, crop.max(axis=0), True),
             (0, 2, crop.max(axis=1), True))):
        ax = axes[column]
        ax.imshow(projection, origin='lower', cmap='gray', vmin=0, vmax=vmax,
                  extent=(lower[a] - .5, upper[a] + .5, lower[b] - .5, upper[b] + .5))
        if with_guide:
            ax.scatter(guide[:, a], guide[:, b], s=2, color='#38bdf8', alpha=.45,
                       label='Full image skeleton')
        for index, (start, end) in enumerate(segments):
            ax.plot([start[a], end[a]], [start[b], end[b]], color='#ff4444', lw=1.1,
                    label='Clipped SWC reference' if index == 0 else None)
        if len(curve) and not segments:
            ax.scatter(curve[:, a], curve[:, b], s=7, color='#ff4444', label='SWC reference')
        label = 'Raw image + reference' if not with_guide else 'Raw image + full guide + reference'
        ax.set_title(f"{row['partition']} neuron {row['sample_index']}, patch {row['patch_index']}\n{label} ({'XY' if b == 1 else 'XZ'})", fontsize=9)
        ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels)')
        ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier audits; choose a fresh output directory')
    manifest = load_sampling_manifest(args.manifest, args.partition)
    if manifest.get('status') != 'completed':
        raise ValueError('Audit requires a completed cache')
    samples = scan_gold166(args.root)
    medial = MedialSamplingConfig(**manifest['medial_config'])
    spacing = np.asarray(medial.spacing_xyz)
    rows, cases = [], {}
    for sample_index in sorted(set(row['sample_index'] for row in manifest['records'])):
        sample = samples[sample_index]
        records = [row for row in manifest['records'] if row['sample_index'] == sample_index]
        if any(row['sample_id'] != sample.sample_id for row in records):
            raise ValueError('Frozen cache and dataset identities differ')
        volume = read_volume(sample.volume_path)
        width, height, depth, channels = volume.dimensions
        if channels != 1:
            raise ValueError('Expected single-channel source')
        data = volume_data_array(volume)
        nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
        with sample.volume_path.open('rb') as handle:
            image_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        swc_sha = hashlib.sha256(sample.swc_path.read_bytes()).hexdigest()
        for row in records:
            with np.load(row['resolved_path'], allow_pickle=False) as cache:
                metadata = json.loads(cache['metadata'].item())
                points, bounded_guide = cache['points'].copy(), cache['image_guide_xyz'].copy()
                indices = cache['candidate_indices'].copy()
            if metadata['volume_sha256'] != image_sha or metadata['swc_sha256'] != swc_sha:
                raise ValueError('Source changed since cache creation')
            np.testing.assert_array_equal(points, indices_to_point_array(indices, data, width, height))
            center, radius = np.asarray(metadata['patch_center']), manifest['config']['patch_radius']
            lower = np.maximum(0, np.floor(center - radius)).astype(int)
            upper = np.minimum(np.array([width, height, depth]) - 1, np.ceil(center + radius)).astype(int)
            curve, segments, crossings = clipped_reference(nodes, lower, upper)
            foreground = patch_foreground_indices(data, width, height, depth, center, radius, metadata['threshold'])
            assert len(foreground) == metadata['foreground_count']
            full_guide = image_medial_axis(foreground, width, height, medial)
            assert len(full_guide) == metadata['full_guide_count']
            fg_xyz = indices_to_point_array(foreground, data, width, height)[:, :3]
            output = {key: row[key] for key in ('partition', 'sample_index', 'sample_id', 'patch_index', 'label_nodes')}
            output.update(reference_curve_points=len(curve), boundary_crossing_edges=crossings,
                          foreground_voxels=len(foreground), full_guide_vertices=len(full_guide),
                          candidate_rows=len(points), bounded_guide_vertices=len(bounded_guide),
                          cached_xyz_and_intensity_match_source=True,
                          source_volume_sha256=image_sha, source_swc_sha256=swc_sha,
                          reference_support={name: reference_support(curve, xyz, spacing)
                              for name, xyz in (('all_foreground', fg_xyz), ('full_guide', full_guide),
                                                ('candidate_pool', points[:, :3]), ('bounded_guide', bounded_guide))})
            # Threshold sensitivity measured on the same reference curve and box.
            output['threshold_sensitivity'] = {}
            for fraction in (.1, .2, .3):
                threshold = fractional_intensity_threshold(volume.header.datatype, fraction, data=data,
                                                            mode=manifest['config'].get('threshold_mode', 'dtype_fraction'))
                other = patch_foreground_indices(data, width, height, depth, center, radius, threshold)
                xyz = indices_to_point_array(other, data, width, height)[:, :3]
                output['threshold_sensitivity'][str(fraction)] = {
                    'threshold': threshold, 'foreground_voxels': len(other),
                    'reference_support': reference_support(curve, xyz, spacing)}
            rows.append(output)
            if len(curve) > 1 and row['partition'] not in cases:
                x0, y0, z0 = lower; x1, y1, z1 = upper + 1
                crop = data.reshape(depth, height, width)[z0:z1, y0:y1, x0:x1].copy()
                cases[row['partition']] = (crop, lower, upper, curve, segments, full_guide, row)
            print(f"audited {row['partition']} neuron={sample_index} patch={row['patch_index']} reference={len(curve)}", flush=True)
    args.output_dir.mkdir(parents=True)
    report = {'scope': 'input audit of existing train/validation cache, not a reconstruction benchmark',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_rule': 'all full-SWC edges clipped to cached box, including boundary crossings; step <=1 voxel',
              'distance_units': 'configured spacing; currently voxel units', 'spacing_xyz': spacing.tolist(),
              'threshold_sensitivity_mode': manifest['config'].get('threshold_mode', 'dtype_fraction'),
              'test_neurons_evaluated': False, 'patch_count': len(rows), 'patches': rows,
              'inventory_note': 'Dataset inventory scans SWCs for valid file selection; only cached train/val sources enter diagnostic metrics.',
              'limitations': ['small train/validation diagnostic; no alternate reference or coordinate transform chosen',
                             'reference proximity is not foreground precision with incomplete annotation',
                             'MIP projections can hide depth mismatch; matching linear-index coordinates is not full anatomical alignment QA']}
    (args.output_dir / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if cases:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = [cases[role] for role in ('train', 'val') if role in cases]
        fig, axes = plt.subplots(len(selected), 3, figsize=(14, 4 * len(selected)), squeeze=False, constrained_layout=True)
        for r, case in enumerate(selected):
            draw_case(axes[r], case)
        fig.suptitle('Original image and full image skeleton (cyan) vs clipped reference (red)\nDiagnostic projections; SWC never selects image inputs', fontsize=12)
        fig.savefig(args.output_dir / 'inspection.png', dpi=150)
        plt.close(fig)
    print(f'audit: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
