"""Read-only cached validation confidence check; no training or reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint
from scripts.pointneuron2.sampling_ablation import checked_manifest, check_matches, digest, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    plan = read(args.experiment / 'experiment.json')
    if plan['status'] not in ('trained', 'evaluated'):
        raise ValueError('Requires a completed matched training pilot')
    manifest = checked_manifest(plan, args.experiment, allow_historical_reader=True)
    runs = [read(args.experiment / arm['name'] / 'run.json') for arm in plan['arms']]
    check_matches(runs)
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier diagnostics; choose a fresh output directory')
    report = {'scope': 'all cached validation patches; raw proposal rows before NMS; no reconstruction scoring',
              'experiment': str(args.experiment.resolve()), 'manifest_sha256': plan['manifest_sha256'],
              'diagnostic_source_sha256': digest(Path(__file__)), 'device': args.device,
              'matching_checks': 'passed', 'arms': []}
    for arm, run in zip(plan['arms'], runs):
        checkpoint_path = args.experiment / arm['name'] / 'proposal.pt'
        if digest(checkpoint_path) != arm['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        recipe, pipeline = checkpoint['args'], checkpoint['sampling_pipeline']
        encoder = DGCNNEncoder(k=recipe['k']).to(args.device).eval()
        head = SkeletonProposalHead(in_channels=encoder.output_dim + 3,
                                    coordinate_mode=recipe['proposal_coordinate_mode']).to(args.device).eval()
        encoder.load_state_dict(checkpoint['encoder']); head.load_state_dict(checkpoint['proposal'])
        sampler = sampler_from_checkpoint(checkpoint, device=args.device)
        fixed = AdaptiveMedialSampler(**pipeline['sampler_config']).to(args.device).eval() if sampler else None
        rows = []
        with torch.inference_mode():
            for row in manifest['records']:
                if row['partition'] != 'val':
                    continue
                with np.load(row['resolved_path'], allow_pickle=False) as data:
                    metadata = json.loads(str(data['metadata']))
                    points = torch.as_tensor(data['points'], device=args.device)[None]
                    guide = torch.as_tensor(data['image_guide_xyz'], device=args.device)[None]
                seed = metadata['seed'] + pipeline['evaluation_seed']
                if sampler:
                    selection = sampler(points, guide, pipeline['selected_points'], seed=seed)
                    chosen, selected = selection.indices[0], selection.points
                    initial_chosen = fixed(points, guide, pipeline['selected_points'], seed=seed).indices[0]
                    replaced = int(pipeline['selected_points'] - torch.isin(chosen, initial_chosen).sum())
                else:
                    generator = torch.Generator(device=args.device).manual_seed(seed)
                    chosen = torch.randperm(points.shape[1], device=args.device, generator=generator)[:pipeline['selected_points']]
                    selected = points[:, chosen]; replaced = None
                scores = head(selected, encoder(selected)).objectness_logits.softmax(-1)[0, :, 1]
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite proposal confidence')
                rows.append({'sample_index': row['sample_index'], 'patch_index': row['patch_index'],
                             'label_nodes': row['label_nodes'], 'maximum_score': float(scores.max()),
                             'raw_rows_ge_0_5': int((scores >= .5).sum()), 'raw_rows_ge_0_85': int((scores >= .85).sum()),
                             'selected_candidates_replaced_vs_initial_alpha': replaced})
        result = {'name': arm['name'], 'checkpoint_epoch': checkpoint['epoch'], 'val_proposal': checkpoint['val_loss'],
                  'checkpoint_sha256': arm['checkpoint_sha256'], 'training_seconds': run['elapsed_seconds'],
                  'initial_alpha': run['initial_alpha'], 'final_alpha': run['epochs'][-1]['alpha'],
                  'epochs': run['epochs'], 'patches': rows,
                  'patches_ge_0_5': sum(r['raw_rows_ge_0_5'] > 0 for r in rows),
                  'patches_ge_0_85': sum(r['raw_rows_ge_0_85'] > 0 for r in rows),
                  'maximum_score': max(r['maximum_score'] for r in rows)}
        report['arms'].append(result)
        print(f"{arm['name']}: checkpoint_epoch={checkpoint['epoch']} val_proposal={checkpoint['val_loss']:.4f} "
              f"max_score={result['maximum_score']:.4f} patches_ge_0.5={result['patches_ge_0_5']}/{len(rows)} "
              f"patches_ge_0.85={result['patches_ge_0_85']}/{len(rows)}", flush=True)
    args.output_dir.mkdir(parents=True)
    report['status'] = 'completed'; write(args.output_dir / 'summary.json', report)
    print(f'Report: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
