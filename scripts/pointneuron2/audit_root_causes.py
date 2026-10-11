"""Bounded input-decoder and coordinate audit; no optimization or reconstruction.

The historical MSB-first decoder is retained here only to quantify the defect.
The independent decoder follows Vaa3D ImageLoaderBasic::decompressPBD8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.foreground import foreground_mask
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_to_array
from pointneuron.data.vaa3d_raw import PBD_HEADER_KEY, read_header, read_volume, VOLUME_READER_VERSION
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint
from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support
from scripts.pointneuron2.sampling_ablation import digest, read, write

VAA3D_SOURCE = 'https://github.com/Vaa3D/v3d_external/blob/master/v3d_main/neuron_annotator/utility/ImageLoaderBasic.cpp#L804-L879'


def audit_decode(encoded, expected_voxels, *, historical=False, datatype=1, byteorder='little'):
    """Independent run-wise decoder; historical=True reproduces the old bit order."""
    result = np.empty(expected_voxels, dtype=np.uint8 if datatype == 1 else np.uint16)
    source = np.frombuffer(encoded, dtype=np.uint8)
    cp = dp = prior = 0
    shifts = np.array([6, 4, 2, 0] if historical else [0, 2, 4, 6])
    while cp < len(source):
        code = int(source[cp]); cp += 1
        if code < (33 if datatype == 1 else 32):
            length = code + 1
            dtype = np.uint8 if datatype == 1 else ('<u2' if byteorder == 'little' else '>u2')
            result[dp:dp + length] = np.frombuffer(encoded, dtype=dtype, count=length, offset=cp)
            cp += length * datatype
        elif code < (128 if datatype == 1 else 80):
            if datatype == 1:
                length = code - 32
                packed = (length + 3) // 4
                differences = ((source[cp:cp + packed, None].astype(np.int16) >> shifts) & 3).ravel()[:length]
                differences[differences == 3] = -1
            else:
                length = code - 31
                packed = (length * 3 + 7) // 8
                bits = np.unpackbits(source[cp:cp + packed])[:length * 3].reshape(length, 3)
                differences = (bits @ np.array([4, 2, 1])).astype(np.int16)
                negative = differences >= 4 if historical else differences > 4
                differences[negative] = differences[negative] - 8 if historical else 4 - differences[negative]
            result[dp:dp + length] = (prior + np.cumsum(differences)) % (256 if datatype == 1 else 65536)
            cp += packed
        elif datatype == 2 and code < 223:
            raise NotImplementedError('Unverified PBD16 four/five-bit extension')
        else:
            length = code - (127 if datatype == 1 else 222)
            result[dp:dp + length] = int.from_bytes(encoded[cp:cp + datatype], byteorder); cp += datatype
        dp += length
        prior = int(result[dp - 1])
    if cp != len(source) or dp != expected_voxels:
        raise ValueError('Invalid compressed stream length')
    return result


def intensity_support(data, curve, width, height, depth, threshold):
    nearest = np.rint(curve).astype(int)
    nearest = np.clip(nearest, 0, np.array([width, height, depth]) - 1)
    x, y, z = nearest.T
    direct = data.reshape(depth, height, width)[z, y, x]
    lower = np.maximum(nearest.min(axis=0) - 4, 0)
    upper = np.minimum(nearest.max(axis=0) + 5, [width, height, depth])
    a, b, c = lower; d, e, f = upper
    local = data.reshape(depth, height, width)[c:f, b:e, a:d]
    maxima = maximum_filter(local, size=7)[z - c, y - b, x - a]
    return {'curve_points': len(curve), 'median_on_curve': float(np.median(direct)),
            'on_curve_gt_threshold': float(np.mean(direct > threshold)),
            'within_chebyshev_3_gt_threshold': float(np.mean(maxima > threshold))}


def decoder_audit(args, manifest, samples):
    rows = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for index in (21, 44, 94):
        sample = samples[index]
        header = read_header(sample.volume_path)
        if header.key != PBD_HEADER_KEY or header.datatype not in (1, 2):
            raise ValueError('Audit cases must be compressed uint8/uint16')
        records = [r for r in manifest['records'] if r['sample_index'] == index]
        with np.load(records[0]['resolved_path'], allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata']))
        if digest(sample.volume_path) != metadata['volume_sha256'] or digest(sample.swc_path) != metadata['swc_sha256']:
            raise ValueError('Frozen source mismatch')
        encoded = sample.volume_path.read_bytes()[header.data_offset:]
        options = {'datatype': header.datatype, 'byteorder': 'little' if header.endian == 'L' else 'big'}
        old = audit_decode(encoded, header.voxel_count, historical=True, **options)
        corrected = audit_decode(encoded, header.voxel_count, **options)
        threshold = metadata['threshold']
        width, height, depth, channels = header.dimensions
        if channels != 1:
            raise ValueError('Single channel expected')
        nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
        curve, segments, _ = clipped_reference(nodes, np.zeros(3), np.array([width, height, depth]) - 1)
        difference = old.astype(np.int32) - corrected.astype(np.int32)
        row = {'sample_index': index, 'sample_id': sample.sample_id,
               'volume_sha256': metadata['volume_sha256'], 'swc_sha256': metadata['swc_sha256'],
               'dimensions_xyzc': list(header.dimensions), 'changed_voxel_fraction': float(np.mean(difference != 0)),
               'mean_absolute_intensity_error': float(np.mean(np.abs(difference))),
               'threshold': threshold, 'old_intensity_max': int(old.max()), 'corrected_intensity_max': int(corrected.max()),
               'foreground_old': int(np.sum(old > threshold)), 'foreground_corrected': int(np.sum(corrected > threshold)),
               'old_reference_intensity': intensity_support(old, curve, width, height, depth, threshold),
               'corrected_reference_intensity': intensity_support(corrected, curve, width, height, depth, threshold),
               'old_decoded_sha256': hashlib.sha256(old.tobytes()).hexdigest(),
               'corrected_decoded_sha256': hashlib.sha256(corrected.tobytes()).hexdigest(), 'patches': []}
        for record in records:
            with np.load(record['resolved_path'], allow_pickle=False) as cached:
                indices = cached['candidate_indices']; points = cached['points']
            np.testing.assert_array_equal(points[:, 3], old[indices])
            row['patches'].append({'patch_index': record['patch_index'], 'cached_intensity_matches_historical_decoder': True,
                                  'cached_intensity_disagrees_with_corrected_fraction': float(np.mean(points[:, 3] != corrected[indices]))})
        # Same source coordinates and display scale, whole-image projection and actual reference-depth slice.
        z_slice = int(np.clip(round(np.median(nodes[:, 3])), 0, depth - 1))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        display_max = 255 if header.datatype == 1 else max(1, int(corrected.max()))
        for column, (name, data) in enumerate((('Historical reader', old), ('Corrected reader', corrected))):
            volume = data.reshape(depth, height, width)
            axes[0, column].imshow(volume.max(axis=0), origin='lower', cmap='gray', vmin=0, vmax=display_max)
            axes[1, column].imshow(volume[z_slice], origin='lower', cmap='gray', vmin=0, vmax=display_max)
            for start, end in segments:
                axes[0, column].plot([start[0], end[0]], [start[1], end[1]], color='#ff4444', lw=.5)
                if max(start[2], end[2]) >= z_slice - .5 and min(start[2], end[2]) <= z_slice + .5:
                    delta = end - start
                    t0, t1 = 0., 1.
                    if abs(delta[2]) > 1e-12:
                        bounds = sorted(((z_slice - .5 - start[2]) / delta[2], (z_slice + .5 - start[2]) / delta[2]))
                        t0, t1 = max(0., bounds[0]), min(1., bounds[1])
                    a, b = start + t0 * delta, start + t1 * delta
                    axes[1, column].plot([a[0], b[0]], [a[1], b[1]], color='#ff4444', lw=1.)
            axes[0, column].set_title(f'{name}: XY maximum projection')
            axes[1, column].set_title(f'{name}: actual Z={z_slice} slice; reference clipped to slab')
            for ax in axes[:, column]:
                ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)')
        fig.suptitle(f'Neuron {index}: same file, coordinates and intensity scale; red = SWC\nDecoder correction only; no threshold change or training')
        fig.savefig(args.output_dir / f'decoder_sample_{index:04d}.png', dpi=140); plt.close(fig)
        rows.append(row)
        print(f"decoder neuron={index} changed={row['changed_voxel_fraction']:.1%} foreground={row['foreground_old']} -> {row['foreground_corrected']} max={row['corrected_intensity_max']}", flush=True)
    return rows


def coordinate_audit(args, manifest):
    checkpoint_path = args.experiment / 'learned_no_aux/proposal.pt'
    plan = read(args.experiment / 'experiment.json')
    arm = next(a for a in plan['arms'] if a['name'] == 'learned_no_aux')
    if digest(checkpoint_path) != arm['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    encoder = DGCNNEncoder(k=checkpoint['args']['k']).to(args.device).eval()
    head = SkeletonProposalHead(in_channels=encoder.output_dim + 3,
                                coordinate_mode=checkpoint['args']['proposal_coordinate_mode']).to(args.device).eval()
    encoder.load_state_dict(checkpoint['encoder']); head.load_state_dict(checkpoint['proposal'])
    sampler = sampler_from_checkpoint(checkpoint, device=args.device)
    pipeline = checkpoint['sampling_pipeline']; rows = []
    with torch.inference_mode():
        for row in manifest['records']:
            if row['partition'] != 'val':
                continue
            with np.load(row['resolved_path'], allow_pickle=False) as cached:
                metadata = json.loads(str(cached['metadata']))
                points = torch.as_tensor(cached['points'], device=args.device)[None]
                guide = torch.as_tensor(cached['image_guide_xyz'], device=args.device)[None]
            selected = sampler(points, guide, pipeline['selected_points'], seed=metadata['seed'] + pipeline['evaluation_seed']).points
            features = encoder(selected); base = head(selected, features)
            scores = base.objectness_logits.softmax(-1)[0, :, 1]
            shifted = selected.clone(); shifted[..., :3] += torch.tensor([512., 512., 0.], device=args.device)
            # Hold geometric features constant to isolate the raw-XYZ head input.
            translated = head(shifted, features); other = translated.objectness_logits.softmax(-1)[0, :, 1]
            test_features = encoder(shifted) if row['sample_index'] == 21 and row['patch_index'] == 0 else None
            rows.append({'sample_index': row['sample_index'], 'patch_index': row['patch_index'],
                         'mean_score_original': float(scores.mean()), 'mean_score_translated': float(other.mean()),
                         'mean_abs_score_change': float((scores - other).abs().mean()),
                         'decision_flips_at_0_85': int(((scores >= .85) != (other >= .85)).sum()),
                         'offset_mean_abs_change': float((base.offsets - translated.offsets).abs().mean()),
                         'encoder_max_abs_feature_change': None if test_features is None else float((features - test_features).abs().max())})
    print(f"coordinate counterfactual: {sum(r['decision_flips_at_0_85'] for r in rows)} / {len(rows) * pipeline['selected_points']} decisions changed", flush=True)
    return {'checkpoint_sha256': arm['checkpoint_sha256'], 'translation_xyz': [512, 512, 0],
            'scope': 'historical cached candidates; head features held fixed; original input geometry unchanged', 'patches': rows}


def verify_repair(args, manifest, samples):
    """Compare production decoder byte-for-byte with independently audited source.

    Foreground comparisons reuse historical image boxes; SWC curves only score
    coverage, never choose the boxes, thresholds, seeds or foreground voxels.
    """
    report = read(args.output_dir / 'summary.json')
    cases = ({(r['sample_index'], r['patch_index']) for r in manifest['records']} if args.comparison_manifest
             else {(21, 4), (44, 2), (44, 5), (44, 6), (94, 3)})
    result = {'volume_reader_version': VOLUME_READER_VERSION, 'volumes': [], 'foreground_crops': [],
              'scope': 'decoder correctness and exploratory crop coverage; no training or reconstruction gain'}
    if args.comparison_manifest:
        result['comparison_manifest_sha256'] = digest(args.comparison_manifest)
    for audited in report['decoder_audit']:
        index = audited['sample_index']; sample = samples[index]
        volume = read_volume(sample.volume_path)
        assert digest(sample.volume_path) == audited['volume_sha256']
        assert digest(sample.swc_path) == audited['swc_sha256']
        checksum = hashlib.sha256(volume.data).hexdigest()
        if checksum != audited['corrected_decoded_sha256']:
            raise AssertionError('Production reader differs from independent reference decoder')
        result['volumes'].append({'sample_index': index, 'production_matches_reference_byte_for_byte': True,
                                  'decoded_sha256': checksum})
        width, height, depth, _ = volume.dimensions
        dtype = np.uint8 if volume.header.datatype == 1 else ('<u2' if volume.header.endian == 'L' else '>u2')
        data = np.frombuffer(volume.data, dtype=dtype).reshape(depth, height, width)
        nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
        for row in manifest['records']:
            if row['sample_index'] != index or (index, row['patch_index']) not in cases:
                continue
            with np.load(row['resolved_path'], allow_pickle=False) as cached:
                metadata = json.loads(str(cached['metadata']))
            center = np.array(metadata['patch_center']); radius = manifest['config']['patch_radius']
            lower = np.maximum(0, np.floor(center - radius)).astype(int)
            upper = np.minimum(np.array([width, height, depth]) - 1, np.ceil(center + radius)).astype(int)
            halo_lower = np.maximum(0, lower - 24); halo_upper = np.minimum(np.array([width, height, depth]) - 1, upper + 24)
            a, b, c = halo_lower; d, e, f = halo_upper + 1
            image = data[c:f, b:e, a:d]
            core = tuple(slice(int(lo - hlo), int(hi - hlo + 1)) for lo, hi, hlo in zip(lower[::-1], upper[::-1], halo_lower[::-1]))
            curve, _, _ = clipped_reference(nodes, lower, upper)
            item = {'sample_index': index, 'patch_index': row['patch_index'], 'reference_points': len(curve),
                    'image_box_xyz': [lower.tolist(), upper.tolist()], 'methods': {}}
            methods = [('historical_dtype_fraction', 'fixed', metadata['threshold']),
                       ('image_max_fraction', 'fixed', .2 * float(data.max())),
                       ('raw_otsu', 'raw_otsu', 0), ('background_otsu', 'background_otsu', 0),
                       ('noise_hysteresis', 'noise_hysteresis', 0), ('guarded_noise_hysteresis', 'guarded_noise_hysteresis', 0)]
            for name, method, threshold in methods:
                mask, _, details = foreground_mask(image, method, fixed_threshold=threshold, core_slices=core)
                z, y, x = np.nonzero(mask)
                xyz = np.column_stack((x, y, z)) + lower
                item['methods'][name] = {'foreground_voxels': len(xyz), 'foreground_fraction': float(mask.mean()),
                                         'parameters': details, 'reference_support': reference_support(curve, xyz, np.ones(3))}
            result['foreground_crops'].append(item)
        print(f'verified decoder neuron={index}: production == independent reference', flush=True)
    # Audit all training candidate pools for cropped-node supervision gaps.
    supervision = []
    for index in sorted({r['sample_index'] for r in manifest['records'] if r['partition'] == 'train'}):
        sample = samples[index]; header = read_header(sample.volume_path)
        nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
        for row in manifest['records']:
            if row['sample_index'] != index or row['partition'] != 'train':
                continue
            with np.load(row['resolved_path'], allow_pickle=False) as cached:
                metadata = json.loads(str(cached['metadata'])); points = cached['points']; local = cached['skeleton_nodes']
            center = np.array(metadata['patch_center']); radius = manifest['config']['patch_radius']
            lower = np.maximum(0, np.floor(center - radius))
            upper = np.minimum(np.array(header.dimensions[:3]) - 1, np.ceil(center + radius))
            curve, _, crossings = clipped_reference(nodes, lower, upper)
            close = cKDTree(curve).query(points[:, :3])[0] <= 6 if len(curve) else np.zeros(len(points), dtype=bool)
            if len(local):
                distance, match = cKDTree(local[:, 1:4]).query(points[:, :3])
                positive = distance <= np.maximum(6., np.maximum(3., local[match, 4]) * 1.5)
            else:
                positive = np.zeros(len(points), dtype=bool)
            supervision.append({'sample_index': index, 'patch_index': row['patch_index'], 'local_label_nodes': len(local),
                                'clipped_reference_points': len(curve), 'crossing_edges': crossings,
                                'candidate_rows_near_curve_6': int(close.sum()),
                                'near_curve_but_node_target_negative': int((close & ~positive).sum()) if len(local) else 0,
                                'near_curve_in_unlabelled_patch': int(close.sum()) if not len(local) else 0})
    result['training_supervision'] = supervision
    result['verification_source_sha256'] = digest(Path(__file__))
    write(args.output_dir / ('corrected_cache_masks.json' if args.comparison_manifest else 'verification.json'), result)
    print(f"training supervision gaps: {sum(r['near_curve_but_node_target_negative'] for r in supervision)} candidate rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--verify-repair', action='store_true', help='Verify production reader and inspect corrected crop masks; no GPU forward passes')
    parser.add_argument('--comparison-manifest', type=Path, help='With --verify-repair, inspect corrected image-defined cache boxes instead of historical boxes')
    args = parser.parse_args()
    if args.output_dir.exists() and not args.verify_repair and (not args.resume_incomplete or (args.output_dir / 'summary.json').exists()):
        raise FileExistsError('Choose a fresh diagnostic output directory')
    plan = read(args.experiment / 'experiment.json')
    manifest_path = args.experiment / 'cache/manifest.json'
    if digest(manifest_path) != plan['manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest = load_sampling_manifest(manifest_path, plan['partition'], allow_historical_reader=True)
    samples = scan_gold166(plan['data_root'])
    if args.verify_repair:
        output_name = 'corrected_cache_masks.json' if args.comparison_manifest else 'verification.json'
        if (args.output_dir / output_name).exists():
            raise FileExistsError('Preserve prior verification')
        if args.comparison_manifest:
            manifest = load_sampling_manifest(args.comparison_manifest, plan['partition'])
        verify_repair(args, manifest, samples)
        return
    inventory = [{'sample_index': i, 'sample_id': samples[i].sample_id,
                  'key': read_header(samples[i].volume_path).key, 'datatype': read_header(samples[i].volume_path).datatype}
                 for i in sorted({r['sample_index'] for r in manifest['records']})]
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_incomplete)
    report = {'scope': 'bounded decoder audit and translation counterfactual; no test neurons or training',
              'source_reference': VAA3D_SOURCE, 'diagnostic_sha256': digest(Path(__file__)),
              'manifest_sha256': digest(manifest_path), 'inventory': inventory,
              'coordinate_audit': coordinate_audit(args, manifest),
              'decoder_audit': decoder_audit(args, manifest, samples)}
    write(args.output_dir / 'summary.json', report)
    print(f'Report: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
