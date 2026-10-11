"""Inspect cached checkpoint inputs and proposals over source images; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_to_array, volume_data_array, indices_to_point_array
from pointneuron.data.vaa3d_raw import read_volume
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint
from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def draw(path, row, crop, lower, upper, curve, segments, guide, learned, random, centers, scores, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    vmax = max(1., float(np.quantile(crop, .998)))
    best = np.argsort(-scores, kind='stable')[:32]
    accepted = scores >= threshold
    for r, (a, b, projection) in enumerate(((0, 1, crop.max(axis=0)), (0, 2, crop.max(axis=1)))):
        for c, ax in enumerate(axes[r]):
            ax.imshow(projection, origin='lower', cmap='gray', vmin=0, vmax=vmax,
                      extent=(lower[a] - .5, upper[a] + .5, lower[b] - .5, upper[b] + .5))
            ax.set_xlim(lower[a] - .5, upper[a] + .5); ax.set_ylim(lower[b] - .5, upper[b] + .5)
            ax.set_aspect('equal'); ax.set_xlabel('X (voxels)'); ax.set_ylabel('Y (voxels)' if b == 1 else 'Z (voxels)')
            for i, (start, end) in enumerate(segments):
                ax.plot([start[a], end[a]], [start[b], end[b]], color='#ff4747', lw=1.2,
                        label='Clipped reference' if i == 0 else None, zorder=4)
            if len(curve) and not len(segments):
                ax.scatter(curve[:, a], curve[:, b], color='#ff4747', s=8, label='Reference', zorder=4)
            if c == 0:
                ax.set_title('Raw image + reference', fontsize=10)
            elif c == 1:
                ax.scatter(guide[:, a], guide[:, b], s=2, alpha=.55, color='#38bdf8', label='Full image guide')
                ax.set_title(f'Full guide: {len(guide):,} vertices', fontsize=10)
            elif c == 2:
                ax.scatter(random[:, a], random[:, b], s=4, alpha=.5, color='#d9ab55', label='Random inputs')
                ax.scatter(learned[:, a], learned[:, b], s=5, alpha=.7, color='#49e39a', label='Learned inputs')
                same = 'identical to fixed sampler' if row['learned_same_as_fixed'] else f"{row['learned_fixed_shared_candidates']} shared with fixed"
                ax.set_title(f'{len(learned)} encoder inputs\nLearned: {same}', fontsize=9)
            else:
                dots = ax.scatter(centers[best, a], centers[best, b], c=scores[best], s=15,
                                  cmap='viridis', vmin=0, vmax=1, label='Top 32 scored centers')
                if accepted.any():
                    ax.scatter(centers[accepted, a], centers[accepted, b], s=30, facecolors='none',
                               edgecolors='#49e39a', label=f'Score >= {threshold:g}')
                ax.set_title(f"Proposals: {int(accepted.sum())} / {len(scores)} above {threshold:g}\n"
                             f"max score={scores.max():.4f}; dots show top 32 regardless of score", fontsize=9)
                if r == 1:
                    fig.colorbar(dots, ax=ax, shrink=.7, label='Objectness score')
            ax.legend(fontsize=6, loc='upper right')
    fig.suptitle(f"{row['partition']} neuron {row['sample_index']}, patch {row['patch_index']} | two-epoch full-guide checkpoint\n"
                 'XY / XZ maximum projections: reference is diagnostic only; proposals are not a connected reconstruction', fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--score-threshold', type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        parser.error('Score threshold must be in [0,1]')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('Local CUDA device unavailable')
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    manifest = load_sampling_manifest(args.manifest, args.partition)
    if manifest.get('status') != 'completed' or manifest.get('failures'):
        raise ValueError('Requires a completed cache')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('training_manifest_sha256') != digest(args.manifest):
        raise ValueError('Checkpoint and manifest do not share provenance')
    pipeline = checkpoint['sampling_pipeline']
    if any(pipeline['pool_config'][key] != manifest['config'][key] for key in ('candidate_points', 'guide_points')):
        raise ValueError('Checkpoint and cached pool settings differ')
    sampler = sampler_from_checkpoint(checkpoint, args.device)
    if sampler is None:
        raise ValueError('Requires a learned sampling checkpoint')
    fixed = AdaptiveMedialSampler(**pipeline['sampler_config']).to(args.device).eval()
    recipe = checkpoint['args']
    encoder = DGCNNEncoder(k=recipe['k']).to(args.device).eval()
    proposal = SkeletonProposalHead(encoder.output_dim + 3, coordinate_mode=recipe['proposal_coordinate_mode']).to(args.device).eval()
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    proposal.load_state_dict(checkpoint['proposal'], strict=True)
    spacing = np.asarray(pipeline['medial_config']['spacing_xyz'])
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'artifacts').mkdir()
    source_paths = ['scripts/pointneuron2/inspect_sampling_checkpoint.py', 'scripts/pointneuron2/audit_sampling.py',
                    'src/pointneuron/models/sampling_pipeline.py', 'src/pointneuron/models/adaptive_sampling.py',
                    'src/pointneuron/models/guide_distances.py', 'src/pointneuron/models/dgcnn.py', 'src/pointneuron/models/proposal.py']
    report = {'status': 'running', 'scope': 'Cached patch input/proposal inspection; no training or full-neuron reconstruction evaluation',
              'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(args.manifest), 'checkpoint_epoch': checkpoint['epoch'],
              'source_sha256': {p: digest(REPO_ROOT / p) for p in source_paths},
              'device': args.device, 'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
              'learned_alpha': float(sampler.alpha.detach()), 'initial_alpha': float(fixed.alpha.detach()),
              'score_threshold': args.score_threshold, 'optimizer_updates': 0, 'patches': [],
              'reference_rule': 'Full SWC edges clipped to patch including boundary crossings; used for metrics/overlays only',
              'distance_units': 'Saved spacing; isotropic voxel units for this run', 'test_neurons_evaluated': False,
              'figure_selection_rule': 'First patch with >1 cached reference nodes in each present partition; all cached patches enter report',
              'limitations': ['Small checkpoint execution diagnostic, not a sampler ranking',
                             'Incomplete traces cannot establish image/guide false positives',
                             'Maximum projections may hide depth mismatch; no proposal connectivity is inferred']}
    report_path = args.output_dir / 'inspection.json'; write(report_path, report)
    plotted = set()
    samples = scan_gold166(args.root)
    try:
        with torch.inference_mode():
            for sample_index in sorted({row['sample_index'] for row in manifest['records']}):
                sample = samples[sample_index]
                rows = [row for row in manifest['records'] if row['sample_index'] == sample_index]
                if any(row['sample_id'] != sample.sample_id for row in rows):
                    raise ValueError('Source identity mismatch')
                image_sha, swc_sha = digest(sample.volume_path), digest(sample.swc_path)
                volume = read_volume(sample.volume_path)
                width, height, depth, channels = volume.dimensions
                if channels != 1:
                    raise ValueError('Expected one image channel')
                data = volume_data_array(volume)
                nodes = skeleton_to_array(swc_to_skeleton_records(parse_swc(sample.swc_path)))
                for row in rows:
                    with np.load(row['resolved_path'], allow_pickle=False) as cached:
                        raw_points, raw_guide = cached['points'].copy(), cached['image_guide_xyz'].copy()
                        metadata = json.loads(str(cached['metadata']))
                        np.testing.assert_array_equal(raw_points, indices_to_point_array(cached['candidate_indices'], data, width, height))
                    if (metadata['volume_sha256'], metadata['swc_sha256']) != (image_sha, swc_sha):
                        raise ValueError('Sources changed since cache generation')
                    points = torch.tensor(raw_points, device=args.device)[None]
                    guide = torch.tensor(raw_guide, device=args.device)[None]
                    count = pipeline['selected_points']; seed = metadata['seed'] + pipeline['evaluation_seed']
                    learned = sampler(points, guide, count, seed=seed)
                    fixed_selection = fixed(points, guide, count, seed=seed)
                    generator = torch.Generator(device=args.device).manual_seed(seed)
                    chosen = torch.randperm(points.shape[1], generator=generator, device=args.device)[:count]
                    random_points = points[:, chosen]
                    predictions = proposal(learned.points, encoder(learned.points))
                    centers = predictions.center_proposals[0].cpu().numpy()
                    scores = predictions.objectness_logits[0].softmax(-1)[:, 1].cpu().numpy()
                    radii = predictions.radius[0].cpu().numpy()
                    if not all(np.isfinite(a).all() for a in (centers, scores, radii)):
                        raise FloatingPointError('Nonfinite checkpoint output')
                    learned_xyz = learned.points[0].cpu().numpy()
                    fixed_xyz = fixed_selection.points[0].cpu().numpy()
                    random_xyz = random_points[0].cpu().numpy()
                    center = np.asarray(metadata['patch_center']); radius = manifest['config']['patch_radius']
                    lower = np.maximum(0, np.floor(center - radius)).astype(int)
                    upper = np.minimum(np.array([width, height, depth]) - 1, np.ceil(center + radius)).astype(int)
                    curve, segments, crossings = clipped_reference(nodes, lower, upper)
                    x0, y0, z0 = lower; x1, y1, z1 = upper + 1
                    crop = data.reshape(depth, height, width)[z0:z1, y0:y1, x0:x1].copy()
                    result = {key: row[key] for key in ('sample_index', 'patch_index', 'partition', 'sample_id', 'label_nodes')}
                    shared = len(set(learned.indices[0].tolist()) & set(fixed_selection.indices[0].tolist()))
                    result.update(guide_vertices=len(raw_guide), candidate_rows=len(raw_points), selected_rows=count,
                                  reference_curve_points=len(curve), boundary_crossing_edges=crossings,
                                  learned_same_as_fixed=(shared == count),
                                  learned_same_order_as_fixed=bool(torch.equal(learned.indices, fixed_selection.indices)),
                                  learned_fixed_shared_candidates=shared,
                                  score_min=float(scores.min()), score_max=float(scores.max()), score_mean=float(scores.mean()),
                                  above_threshold=int((scores >= args.score_threshold).sum()),
                                  offset_median_voxels=float(np.median(np.linalg.norm(centers - learned_xyz[:, :3], axis=1))),
                                  reference_support={name: reference_support(curve, xyz, spacing) for name, xyz in
                                      [('full_guide', raw_guide), ('candidates', raw_points[:, :3]), ('random', random_xyz[:, :3]),
                                       ('fixed', fixed_xyz[:, :3]), ('learned', learned_xyz[:, :3]),
                                       ('above_threshold_centers', centers[scores >= args.score_threshold])]},
                                  source_volume_sha256=image_sha, source_swc_sha256=swc_sha)
                    name = f"sample_{sample_index:04d}_patch_{row['patch_index']:03d}"
                    asset = args.output_dir / 'artifacts' / f'{name}.npz'
                    np.savez_compressed(asset, raw_image=crop, lower_xyz=lower, upper_xyz=upper, candidate_points=raw_points,
                                        full_guide_xyz=raw_guide, learned_points=learned_xyz, fixed_points=fixed_xyz,
                                        random_points=random_xyz, centers=centers, scores=scores, radii=radii,
                                        reference_curve=curve, reference_segments=np.asarray(segments).reshape(-1, 2, 3))
                    result.update(artifact=asset.relative_to(args.output_dir).as_posix(), artifact_sha256=digest(asset))
                    if row['partition'] not in plotted and row['label_nodes'] > 1:
                        figure = args.output_dir / f'{name}.png'
                        draw(figure, result, crop, lower, upper, curve, segments, raw_guide, learned_xyz, random_xyz,
                             centers, scores, args.score_threshold)
                        result['figure'] = figure.name; plotted.add(row['partition'])
                    report['patches'].append(result); write(report_path, report)
                    print(f"{row['partition']} {sample_index}/{row['patch_index']}: guide={len(raw_guide)} "
                          f"learned=fixed {result['learned_same_as_fixed']} max_score={scores.max():.4f} "
                          f"above_threshold={result['above_threshold']}", flush=True)
                del volume, data
        if digest(args.checkpoint) != report['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during inspection')
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        write(report_path, report)
    print(f'Inspection: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
