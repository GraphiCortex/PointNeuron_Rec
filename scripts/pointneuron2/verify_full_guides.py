"""Check blocked full-guide values and gradients against direct dense distances."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, masked_chamfer


def direct_dense_distances(*args, **kwargs):
    # Diagnostic reference only: preserve legacy production/checkpoint behavior.
    kwargs['compute_mode'] = 'donot_use_mm_for_euclid_dist'
    return ORIGINAL_CDIST(*args, **kwargs)


ORIGINAL_CDIST = torch.cdist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=REPO_ROOT / 'tmp/pointneuron2/foreground_comparison_seed0/comparison.json')
    parser.add_argument('--profile', type=Path, default=REPO_ROOT / 'tmp/pointneuron2/full_guide_profile_seed0/profile.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh verification output')
    comparison = json.loads(args.comparison.read_text())
    profile = json.loads(args.profile.read_text())
    if profile['status'] != 'completed' or comparison['status'] != 'completed':
        raise ValueError('Requires completed comparison and GPU profile')
    rows = []
    for row in profile['cases']:
        case = next(p for p in comparison['patches'] if (p['sample_index'], p['patch_index']) == (row['sample_index'], row['patch_index']))
        metadata = case['methods'][row['mask_method']]
        artifact = args.comparison.parent / metadata['artifact']
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row['artifact_sha256']:
            raise ValueError('Saved input changed')
        with np.load(artifact, allow_pickle=False) as data:
            raw = data['points'].copy(); raw[:, :3] -= data['lower_xyz']
            guide = data['full_guide_xyz'] - data['lower_xyz']
        points = torch.tensor(raw, dtype=torch.float32, device='cuda')[None].repeat(2, 1, 1)
        guide = torch.tensor(guide, dtype=torch.float32, device='cuda')[None].repeat(2, 1, 1)
        mask = torch.ones(guide.shape[:2], dtype=torch.bool, device='cuda')
        results = []
        for chunk in (None, 256):
            sampler = AdaptiveMedialSampler(distance_chunk_size=chunk).cuda().train()
            torch.manual_seed(5)
            with patch('torch.cdist', side_effect=direct_dense_distances):
                selected = sampler(points, guide, 512)
                loss = masked_chamfer(selected.points[..., :3] / 96, guide / 96, mask, chunk)
                loss.backward()
            results.append((selected.probabilities.detach(), selected.indices.detach(), loss.detach(), sampler.log_alpha.grad.detach()))
        dense, blocked = results
        torch.testing.assert_close(blocked[0], dense[0], atol=1e-7, rtol=1e-5)
        assert torch.equal(blocked[1], dense[1]), 'Hard selections changed'
        torch.testing.assert_close(blocked[2], dense[2], atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(blocked[3], dense[3], atol=1e-7, rtol=1e-4)
        rows.append({'sample_index': row['sample_index'], 'patch_index': row['patch_index'],
                     'guide_vertices': guide.shape[1], 'hard_selections_identical': True,
                     'dense_loss': float(dense[2]), 'blocked_loss': float(blocked[2]),
                     'dense_alpha_gradient': float(dense[3]), 'blocked_alpha_gradient': float(blocked[3])})
        print(f"Verified direct dense vs blocked: {guide.shape[1]} vertices", flush=True)
    report = {'status': 'completed', 'scope': 'Direct dense reference, full-guide forward/backward; zero optimizer updates',
              'distance_rule': 'Direct Euclidean distances for both implementations; legacy default matrix shortcut may perturb voxel-distance ties',
              'cases': rows, 'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'profile_sha256': hashlib.sha256(args.profile.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
