"""Short local GPU forward/backward diagnostic; performs no optimizer updates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, masked_chamfer
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from scripts.pointneuron2.train_sampling import proposal_objective


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def measure(points, guide, mode, repeats, full_step=None):
    chunk = 256 if mode == 'full_blocked' else None
    sampler = AdaptiveMedialSampler(distance_chunk_size=chunk).cuda().train()
    if mode == 'capped_dense':
        indices = np.sort(np.random.default_rng(0).choice(guide.shape[1], min(512, guide.shape[1]), replace=False))
        guide = guide[:, indices]
    mask = torch.ones(guide.shape[:2], dtype=torch.bool, device='cuda')
    def step():
        sampler.zero_grad(set_to_none=True)
        if full_step:
            full_step[0].zero_grad(set_to_none=True); full_step[1].zero_grad(set_to_none=True)
        torch.manual_seed(5)
        selected = sampler(points, guide, 512).points
        auxiliary = masked_chamfer(selected[..., :3] / 96, guide / 96, mask, chunk)
        loss = auxiliary
        if full_step:
            encoder, proposal, nodes, node_mask, recipe = full_step
            loss = proposal_objective(proposal(selected, encoder(selected)), selected, nodes, node_mask, recipe) + .05 * auxiliary
        loss.backward()
        if not torch.isfinite(loss) or sampler.log_alpha.grad is None or not torch.isfinite(sampler.log_alpha.grad):
            raise FloatingPointError('Invalid diagnostic loss/gradient')
        return float(loss.detach()), float(sampler.log_alpha.grad.detach())
    step()  # Warm kernels; not part of timing.
    elapsed, allocated, reserved = [], [], []
    loss, gradient = None, None
    for _ in range(repeats):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss, gradient = step()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        allocated.append(torch.cuda.max_memory_allocated() / 2**20)
        reserved.append(torch.cuda.max_memory_reserved() / 2**20)
    return {'mode': mode, 'guide_vertices': guide.shape[1], 'batch_size': len(points),
            'candidate_points': points.shape[1], 'selected_points': 512, 'distance_chunk_size': chunk,
            'median_forward_backward_ms': statistics.median(elapsed), 'max_allocated_mib': max(allocated),
            'max_reserved_mib': max(reserved), 'loss': loss, 'alpha_gradient': gradient,
            'optimizer_updates': 0, 'individual_ms': elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=REPO_ROOT / 'tmp/pointneuron2/foreground_comparison_seed0/comparison.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.repeats < 1:
        parser.error('A local CUDA device and positive repeat count are required')
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output folder')
    comparison = json.loads(args.comparison.read_text(encoding='utf-8'))
    if comparison['status'] != 'completed' or any(p['partition'] != 'train' for p in comparison['patches']):
        raise ValueError('Requires a completed training-only comparison')
    cases = [(p, 'background_otsu') for p in comparison['patches'] if p['reference_curve_points']]
    largest = max(((p, method) for p in comparison['patches'] for method in p['methods']),
                  key=lambda case: case[0]['methods'][case[1]]['full_guide_vertices'])
    if not any(p['sample_index'] == largest[0]['sample_index'] and p['patch_index'] == largest[0]['patch_index'] and method == largest[1]
               for p, method in cases):
        cases.append(largest)
    args.output_dir.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'GPU forward/backward and memory diagnostic; no optimizer, training run or quality evaluation',
              'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'repeats': args.repeats, 'comparison_sha256': digest(args.comparison), 'cases': [], 'failures': [],
              'source_sha256': {path: digest(REPO_ROOT / path) for path in
                  ('scripts/pointneuron2/profile_full_guides.py', 'src/pointneuron/models/guide_distances.py',
                   'src/pointneuron/models/adaptive_sampling.py')},
              'batch_rule': 'Each saved patch repeated twice; timings are diagnostic, not representative dataset throughput.',
              'coordinate_rule': 'Shared patch origin subtracted from candidates and guide; distances retain voxel units.',
              'memory_rule': 'PyTorch allocated/reserved peaks include input tensors and live model state, exclude other processes and driver allocations.'}
    path = args.output_dir / 'profile.json'; write(path, report)
    for case, method in cases:
        metadata = case['methods'][method]
        artifact = args.comparison.parent / metadata['artifact']
        if digest(artifact) != metadata['artifact_sha256']:
            raise ValueError('Saved comparison artifact changed')
        with np.load(artifact, allow_pickle=False) as data:
            origin = data['lower_xyz'].astype(np.float32)
            raw_points = data['points'].astype(np.float32).copy(); raw_points[:, :3] -= origin
            raw_guide = data['full_guide_xyz'].astype(np.float32) - origin
        if len(raw_points) != 1024 or not len(raw_guide):
            raise ValueError('Expected 1024 candidates and a nonempty saved full guide')
        points = torch.tensor(raw_points).cuda()[None].repeat(2, 1, 1)
        guide = torch.tensor(raw_guide).cuda()[None].repeat(2, 1, 1)
        row = {'sample_index': case['sample_index'], 'patch_index': case['patch_index'], 'mask_method': method,
               'artifact_sha256': metadata['artifact_sha256'], 'full_guide_vertices': len(raw_guide), 'measurements': []}
        print(f"Training patch {case['sample_index']}/{case['patch_index']} {method}: full guide {len(raw_guide)}", flush=True)
        for mode in ('capped_dense', 'full_dense', 'full_blocked'):
            try:
                result = measure(points, guide, mode, args.repeats)
                row['measurements'].append(result)
                print(f"  {mode}: {result['median_forward_backward_ms']:.1f} ms; {result['max_allocated_mib']:.1f} MiB", flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect(); torch.cuda.empty_cache()
                report['failures'].append({'sample_index': case['sample_index'], 'mode': mode, 'error': str(exc)})
        report['cases'].append(row); write(path, report)
        if case['sample_index'] == 0:
            cache = REPO_ROOT / 'tmp/pointneuron2/sampling_training_cache_smoke/sample_0000_patch_001.npz'
            with np.load(cache, allow_pickle=False) as data:
                metadata_cache = json.loads(str(data['metadata']))
                centers = np.asarray(metadata_cache['patch_center']); radius = metadata_cache['config']['patch_radius']
                if case['patch_index'] != 1 or not np.array_equal(np.maximum(0, centers - radius), origin):
                    raise ValueError('Reference cache does not match profiled patch')
                raw_nodes = data['skeleton_nodes'].astype(np.float32).copy(); raw_nodes[:, 1:4] -= origin
            nodes = torch.tensor(raw_nodes).cuda()[None].repeat(2, 1, 1)
            node_mask = torch.ones(nodes.shape[:2], dtype=torch.bool, device='cuda')
            recipe = json.loads((REPO_ROOT / 'configs/pointneuron2/proposal_baseline.json').read_text())['training']
            torch.manual_seed(7)
            encoder = DGCNNEncoder(k=recipe['k']).cuda().train()
            proposal = SkeletonProposalHead(encoder.output_dim + 3, coordinate_mode=recipe['proposal_coordinate_mode']).cuda().train()
            proposal.initialize_conservative()
            report['full_proposal_step'] = measure(points, guide, 'full_blocked', args.repeats,
                                                   (encoder, proposal, nodes, node_mask, recipe))
            report['full_proposal_step'].update(scope='fresh DGCNN/proposal plus sampler and supervised/Chamfer backward; repeated chick patch',
                                                 reference_cache_sha256=digest(cache))
            print(f"  DGCNN/proposal/full-guide step: {report['full_proposal_step']['median_forward_backward_ms']:.1f} ms; "
                  f"{report['full_proposal_step']['max_allocated_mib']:.1f} MiB", flush=True)
            del encoder, proposal, nodes, node_mask
        del points, guide; gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'failed' if report['failures'] else 'completed'
    write(path, report)
    print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
