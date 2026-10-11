"""Inspect actual Z slices of saved mask artifacts; no extraction or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.pointneuron2.compare_foreground import digest, write_json


def signed_reference_contrast(raw, lower, curve):
    """Diagnostic only: sample traced XYZ and six neighbors three voxels away."""
    from scipy.ndimage import map_coordinates
    metadata = {'raw_max_fraction': float(np.mean(raw == raw.max())),
                'raw_uint8_saturation_fraction': float(np.mean(raw == 255)) if raw.dtype == np.uint8 else None,
                'rule': 'Trilinear intensity at reference curve versus median of six axial neighbors at +/- 3 voxels; boundary points excluded'}
    xyz = np.asarray(curve) - lower
    shape_xyz = np.asarray(raw.shape[::-1])
    xyz = xyz[np.all((xyz >= 3) & (xyz <= shape_xyz - 4), axis=1)]
    metadata['interior_reference_points'] = len(xyz)
    if not len(xyz):
        return {**metadata, 'median_signed_contrast': None, 'reference_uint8_saturation_fraction': None}
    zyx = xyz[:, ::-1].T
    image = raw.astype(float)
    center = map_coordinates(image, zyx, order=1)
    neighbors = []
    for axis in range(3):
        for sign in (-1, 1):
            coords = zyx.copy(); coords[axis] += sign * 3
            neighbors.append(map_coordinates(image, coords, order=1))
    delta = center - np.median(neighbors, axis=0)
    return {**metadata, 'median_signed_contrast': float(np.median(delta)),
            'fraction_contrast_above_2': float(np.mean(delta > 2)),
            'fraction_contrast_below_minus_2': float(np.mean(delta < -2)),
            'reference_uint8_saturation_fraction': float(np.mean(center >= 254.5)) if raw.dtype == np.uint8 else None}


def clip_segments_to_slab(segments, z, half_width=.5):
    """Clip XYZ reference edges to one slice, rather than projecting every edge."""
    result = []
    for start, end in np.asarray(segments).reshape(-1, 2, 3):
        dz = end[2] - start[2]
        if dz == 0:
            if abs(start[2] - z) <= half_width:
                result.append((start, end))
            continue
        bounds = sorted(((z - half_width - start[2]) / dz,
                         (z + half_width - start[2]) / dz))
        lo, hi = max(0., bounds[0]), min(1., bounds[1])
        if lo <= hi:
            result.append((start + lo * (end - start), start + hi * (end - start)))
    return result


def render_slices(path, artifacts, row, z_indices=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    first = next(iter(artifacts.values()))
    raw, lower = first['raw_image'], first['lower_xyz']
    depth, height, width = raw.shape
    zs = sorted(set(z_indices if z_indices is not None else
                    [int((depth - 1) * q) for q in (.25, .5, .75)]))
    if not zs or any(z < 0 or z >= depth for z in zs):
        raise ValueError('Slice indices must be inside the saved patch')
    methods = list(artifacts)
    fig, axes = plt.subplots(len(zs), len(methods) + 1,
                             figsize=(3.4 * (len(methods) + 1), 3.5 * len(zs)),
                             squeeze=False, constrained_layout=True)
    extent = (lower[0] - .5, lower[0] + width - .5,
              lower[1] - .5, lower[1] + height - .5)
    vmax = max(1., float(np.quantile(raw, .998)))
    summary = []
    for ri, zi in enumerate(zs):
        absolute_z = float(lower[2] + zi)
        segments = clip_segments_to_slab(first['reference_segments'], absolute_z)
        counts = {}
        for ci, method in enumerate(['raw'] + methods):
            ax = axes[ri, ci]
            if method == 'raw':
                ax.imshow(raw[zi], origin='lower', cmap='gray', vmin=0, vmax=vmax, extent=extent)
                label = f'Raw image; Z={absolute_z:g}'
            else:
                data = artifacts[method]
                mask = data['mask'][zi]
                counts[method] = int(mask.sum())
                ax.imshow(mask, origin='lower', cmap='gray', vmin=0, vmax=1, extent=extent)
                guide = data['full_guide_xyz']
                in_slice = guide[np.abs(guide[:, 2] - absolute_z) <= .5]
                ax.scatter(in_slice[:, 0], in_slice[:, 1], s=3, color='#24b5f0')
                label = f'{method}; foreground={mask.mean():.1%}'
            for start, end in segments:
                ax.plot([start[0], end[0]], [start[1], end[1]], color='#ee4646', lw=1.1)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)')
        summary.append({'local_z': zi, 'absolute_z': absolute_z,
                        'foreground_voxels_by_method': counts,
                        'reference_segments_in_slab': len(segments)})
    fig.suptitle(f"TRAIN neuron {row['sample_index']}, patch {row['patch_index']}: actual slices\n"
                 'Red edges clipped to Z +/- 0.5; cyan guide vertices in the same slab. No maximum projections.', fontsize=11)
    fig.savefig(path, dpi=130); plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--sample-index', type=int, required=True)
    parser.add_argument('--patch-index', type=int, required=True)
    parser.add_argument('--z', type=int, action='append', help='Local Z index; repeat to inspect multiple slices')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding='utf-8-sig'))
    row = next(x for x in report['patches'] if x['sample_index'] == args.sample_index and
               x['patch_index'] == args.patch_index)
    if row['partition'] != 'train':
        raise ValueError('This diagnostic is restricted to training patches')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    artifacts = {}
    hashes = {}
    for method, record in row['methods'].items():
        if 'artifact' not in record:
            continue
        source = args.comparison.parent / record['artifact']
        hashes[method] = digest(source)
        if hashes[method] != record['artifact_sha256']:
            raise ValueError('Saved artifact has changed')
        with np.load(source, allow_pickle=False) as data:
            artifacts[method] = {k: data[k] for k in
                                ('raw_image', 'mask', 'lower_xyz', 'reference_segments', 'reference_curve', 'full_guide_xyz')}
    first = next(iter(artifacts.values()))
    for data in artifacts.values():
        for name in ('raw_image', 'lower_xyz', 'reference_segments'):
            if not np.array_equal(first[name], data[name]):
                raise ValueError('Mask artifacts do not describe the same patch')
    args.output_dir.mkdir(parents=True)
    slices = render_slices(args.output_dir / 'slices.png', artifacts, row, args.z)
    write_json(args.output_dir / 'inspection.json', {
        'status': 'completed', 'scope': 'saved training-patch slice inspection; no new extraction',
        'comparison': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
        'artifact_sha256': hashes, 'script_sha256': digest(__file__),
        'sample_index': args.sample_index, 'patch_index': args.patch_index,
        'contrast_diagnostic': signed_reference_contrast(first['raw_image'], first['lower_xyz'], first['reference_curve']),
        'slice_rule': 'user-specified local Z or image-depth quartiles; no reference-based choice',
        'slices': slices, 'display_vmax': max(1., float(np.quantile(first['raw_image'], .998)))})
    print(f'Saved slices: {(args.output_dir / "slices.png").resolve()}', flush=True)


if __name__ == '__main__':
    main()
