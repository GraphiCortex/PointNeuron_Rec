"""Single corrected image-defined crop: mask -> guide -> candidate -> sampler.

Read-only validation diagnosis; no model training, threshold fitting or full
reconstruction. Reference geometry only scores the resulting representations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src')); sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.foreground import foreground_mask
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_to_array, volume_data_array
from pointneuron.data.vaa3d_raw import read_volume, VOLUME_READER_VERSION
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler
from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support
from scripts.pointneuron2.compare_foreground import process_mask
from scripts.pointneuron2.sampling_ablation import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--sample-index', type=int, default=94)
    parser.add_argument('--patch-index', type=int, default=1)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier probes')
    manifest = load_sampling_manifest(args.manifest, args.partition)
    row = next(r for r in manifest['records'] if (r['sample_index'], r['patch_index']) == (args.sample_index, args.patch_index))
    sample = scan_gold166(args.root)[args.sample_index]
    with np.load(row['resolved_path'], allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata']))
    if (sample.sample_id != row['sample_id'] or digest(sample.volume_path) != metadata['volume_sha256'] or
            digest(sample.swc_path) != metadata['swc_sha256']):
        raise ValueError('Frozen source changed')
    volume = read_volume(sample.volume_path)
    width, height, depth, _ = volume.dimensions
    data = volume_data_array(volume).reshape(depth, height, width)
    center = np.array(metadata['patch_center']); radius = manifest['config']['patch_radius']
    lower = np.maximum(0, np.floor(center - radius)).astype(int)
    upper = np.minimum(np.array([width, height, depth]) - 1, np.ceil(center + radius)).astype(int)
    halo_lower = np.maximum(0, lower - 24)
    halo_upper = np.minimum(np.array([width, height, depth]) - 1, upper + 24)
    a, b, c = halo_lower; d, e, f = halo_upper + 1
    halo = data[c:f, b:e, a:d]
    core_slices = tuple(slice(int(lo - hlo), int(hi - hlo + 1)) for lo, hi, hlo in zip(lower[::-1], upper[::-1], halo_lower[::-1]))
    image = halo[core_slices]
    nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
    curve, segments, _ = clipped_reference(nodes, lower, upper)
    medial = MedialSamplingConfig(**manifest['medial_config'])
    config = {'candidate_points': manifest['config']['candidate_points'], 'guide_points': None}
    seed = metadata['seed']; rows = []; arrays_by_method = {}
    for method in ('fixed', 'noise_hysteresis'):
        mask, _, details = foreground_mask(halo, method, fixed_threshold=metadata['threshold'], core_slices=core_slices)
        arrays = process_mask(image, mask, lower, medial, config, seed)
        points = torch.tensor(arrays['points'], dtype=torch.float32)[None]
        guide = torch.tensor(arrays['full_guide_xyz'], dtype=torch.float32)[None]
        generator = torch.Generator().manual_seed(seed)
        selected_count = manifest['config']['selected_points']
        random_points = points[:, torch.randperm(points.shape[1], generator=generator)[:selected_count]]
        with torch.inference_mode():
            guided = AdaptiveMedialSampler(initial_alpha=.25, uniform_fraction=.25, distance_chunk_size=256).eval()(points, guide, selected_count, seed=seed).points
        representations = {'foreground': arrays['foreground_xyz'], 'full_guide': arrays['full_guide_xyz'],
                           'candidate_pool_1024': arrays['points'][:, :3],
                           'random_input_512': random_points[0, :, :3].numpy(),
                           'fixed_guide_input_512': guided[0, :, :3].numpy()}
        result = {'method': method, 'mask_parameters': details,
                  'counts': {k: len(v) for k, v in representations.items()},
                  'reference_support': {k: reference_support(curve, v, np.ones(3)) for k, v in representations.items()}}
        rows.append(result); arrays_by_method[method] = arrays
        print(method, json.dumps(result['reference_support']), flush=True)
    args.output_dir.mkdir(parents=True)
    report = {'scope': __doc__, 'volume_reader_version': VOLUME_READER_VERSION, 'sample_index': args.sample_index,
              'patch_index': args.patch_index, 'sample_id': sample.sample_id, 'manifest_sha256': digest(args.manifest),
              'volume_sha256': metadata['volume_sha256'], 'swc_sha256': metadata['swc_sha256'],
              'script_sha256': digest(Path(__file__)), 'reference_curve_samples': len(curve),
              'lower_xyz': lower.tolist(), 'upper_xyz': upper.tolist(), 'methods': rows,
              'limitations': ['single validation crop; no foreground precision or topology measurement',
                              'fixed prior sampler, not a trained repaired proposal model',
                              'noise MAD can collapse to quantization floor; compare training domains before promotion']}
    write(args.output_dir / 'summary.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for r, method in enumerate(('fixed', 'noise_hysteresis')):
        arrays = arrays_by_method[method]
        for col, key in enumerate(('full_guide_xyz', 'points')):
            ax = axes[r, col]
            ax.imshow(image.max(axis=0), cmap='gray', origin='lower', vmin=0, vmax=255,
                      extent=(lower[0] - .5, upper[0] + .5, lower[1] - .5, upper[1] + .5))
            xyz = arrays[key]
            ax.scatter(xyz[:, 0], xyz[:, 1], s=2, alpha=.5, color='#38bdf8')
            for start, end in segments:
                ax.plot([start[0], end[0]], [start[1], end[1]], color='#ff4444', lw=.8)
            ax.set_title(f'{method}: {"full guide" if col == 0 else "1,024 candidates"}')
            ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)')
    fig.suptitle(f'Corrected neuron {args.sample_index}, image-defined patch {args.patch_index}\nRed = reference; cyan = extracted representation. XY projection can hide depth.')
    fig.savefig(args.output_dir / 'inspection.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
