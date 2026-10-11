"""Inspect saved fixed-crop proposals, normalization and dropout; no optimization or checkpoint writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from pointneuron.data.query_cache import file_digest, load_query_manifest, QueryCacheDataset, collate_query_records
from pointneuron.models.query_pipeline import models_from_checkpoint, forward_proposals
from scripts.pointneuron2.train_query_comparison import prepare_batch, inspect_fit, objective


def proposal_metrics(output, nodes, mask, arm, config, recipe, points):
    crops = []
    for i in range(len(nodes)):
        targets = nodes[i, mask[i], 1:4]
        distances = torch.cdist(output.center_proposals[i], targets)
        scores = output.objectness_logits[i].softmax(-1)[:, 1]
        chosen = scores >= .5
        target_distance = distances.min(0).values
        filtered = distances[chosen].min(0).values if bool(chosen.any()) else torch.full_like(target_distance, float('inf'))
        crops.append({'targets': len(targets), 'slots': len(scores), 'accepted_at_0_5': int(chosen.sum()),
                      'mean_target_distance': float(target_distance.mean()),
                      'coverage_6_all_slots': float((target_distance <= 6).float().mean()),
                      'coverage_6_at_0_5': float((filtered <= 6).float().mean()),
                      'proposal_precision_6_at_0_5': float((distances[chosen].min(1).values <= 6).float().mean()) if bool(chosen.any()) else None,
                      'score_max': float(scores.max()), 'target_nearest_distances': target_distance.tolist(),
                      'slot_axis_std': output.center_proposals[i].std(0).tolist(),
                      'target_axis_std': targets.std(0).tolist()})
    loss = objective(arm, output, points, nodes, mask, config, recipe)
    result = {'geometry': inspect_fit(output, nodes, mask), 'crops': crops,
              'loss': {key: float(getattr(loss, key)) for key in ('total', 'center', 'objectness', 'radius')}}
    if arm == 'query':
        result['matched'] = []
        for i, (pred, target) in enumerate(loss.assignments):
            pred = torch.tensor(pred, dtype=torch.long, device=nodes.device)
            target = torch.tensor(target, dtype=torch.long, device=nodes.device)
            scores = output.objectness_logits[i].softmax(-1)[:, 1]
            unmatched = torch.ones(len(scores), dtype=torch.bool, device=nodes.device); unmatched[pred] = False
            error = torch.linalg.vector_norm(output.center_proposals[i, pred] - nodes[i, mask[i]][target, 1:4], dim=-1)
            result['matched'].append({'mean_error': float(error.mean()), 'max_error': float(error.max()),
                                      'scores': scores[pred].tolist(), 'errors': error.tolist(),
                                      'mean_score': float(scores[pred].mean()),
                                      'unmatched_mean_score': float(scores[unmatched].mean()),
                                      'unmatched_max_score': float(scores[unmatched].max())})
    return result


def audit(models_dir, manifest_path, output_dir, partition, device='cpu'):
    if output_dir.exists():
        raise FileExistsError('Use a fresh audit output directory')
    comparison_path = models_dir / 'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    if comparison['status'] != 'completed' or comparison['paired_checks'] != 'passed':
        raise ValueError('Requires a completed paired check')
    manifest = load_query_manifest(manifest_path, partition)
    rows = {r['path']: r for r in manifest['records']}
    report = {'scope': 'two fitted training crops; no validation or optimization', 'device': device,
              'comparison_sha256': file_digest(comparison_path), 'manifest_sha256': file_digest(manifest_path),
              'tolerance': 6., 'score_threshold': .5, 'arms': []}
    arrays = {}
    for record in comparison['arms']:
        assert record['manifest_sha256'] == report['manifest_sha256']
        assert record['scope'] == 'fixed-training-crop execution check'
        chosen = [rows[p] for p in record['crop_paths']]
        assert all(r['partition'] == 'train' and r['target_count'] > 0 for r in chosen)
        dataset = QueryCacheDataset(chosen)
        batch = collate_query_records([dataset[i] for i in range(len(chosen))])
        config, recipe, arm = record['config'], record['pointwise_recipe'], record['arm']
        checkpoint_path = models_dir / arm / 'proposal.pt'
        before = file_digest(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # Reproduce the CUDA randperm used in the original run. Random-index generation
        # only: neural forwards and all diagnostic computations remain on CPU.
        selected = prepare_batch(batch, config, record['device'], 0)[0].cpu()
        assert hashlib.sha256(selected.numpy().tobytes()).hexdigest() == record['inputs_sha256']
        points = selected.to(device)
        nodes, mask, centers = (batch[k].to(device) for k in ('skeleton_nodes', 'skeleton_mask', 'patch_centers'))
        entry = {'arm': arm, 'checkpoint_sha256': before, 'updates': record['updates'], 'modes': {}}
        modes = ('eval_pair', 'eval_single', 'global_batch_stats_pair', 'encoder_batch_stats_pair', 'all_batch_stats_pair')
        if arm == 'query':
            modes += ('head_train_no_dropout',) + tuple(f'{prefix}_{seed}' for prefix in ('dropout_only', 'training') for seed in range(3))
        for mode in modes:
            torch.manual_seed(1000 + int(mode.rsplit('_', 1)[1]) if mode.startswith(('dropout_only_', 'training_')) else 1000)
            encoder, head = models_from_checkpoint(checkpoint, device)
            if mode in ('encoder_batch_stats_pair', 'all_batch_stats_pair'):
                for layer in encoder.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.train()
            if mode == 'global_batch_stats_pair':
                for layer in encoder.global_mlp.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.train()
            if mode == 'all_batch_stats_pair':
                for layer in head.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.train()
            if mode.startswith('training_'):
                encoder.train(); head.train()
            elif mode.startswith('dropout_only_'):
                head.train()
            elif mode == 'head_train_no_dropout':
                head.train()
                for layer in head.modules():
                    if isinstance(layer, nn.Dropout): layer.p = 0.
                    elif isinstance(layer, nn.MultiheadAttention): layer.dropout = 0.
            with torch.no_grad():
                if mode == 'eval_single':
                    outputs = [forward_proposals(head, arm, points[i:i+1], encoder(points[i:i+1]), centers[i:i+1], config['patch_radius']) for i in range(len(points))]
                    from dataclasses import fields
                    output = type(outputs[0])(**{f.name: torch.cat([getattr(o, f.name) for o in outputs]) for f in fields(outputs[0])})
                else:
                    output = forward_proposals(head, arm, points, encoder(points), centers, config['patch_radius'])
                entry['modes'][mode] = proposal_metrics(output, nodes, mask, arm, config, recipe, points)
                if mode == 'eval_pair' and device == record['device']:
                    for key in ('mean_target_distance', 'target_coverage_6'):
                        assert abs(entry['modes'][mode]['geometry'][key] - record['final_fit'][key]) < .001
                arrays[f'{arm}_{mode}_centers'] = output.center_proposals.cpu().numpy()
                arrays[f'{arm}_{mode}_scores'] = output.objectness_logits.softmax(-1)[..., 1].cpu().numpy()
            print(f"{arm} {mode}: {json.dumps(entry['modes'][mode]['geometry'])}", flush=True)
        assert file_digest(checkpoint_path) == before
        entry['checkpoint_unchanged'] = True
        report['arms'].append(entry)
    arrays.update(points=points.cpu().numpy(), nodes=nodes.cpu().numpy(), mask=mask.cpu().numpy(), centers=centers.cpu().numpy())
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / 'proposals.npz', **arrays)
    (output_dir / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Audit: {output_dir.resolve()}; checkpoints unchanged')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--partition', type=Path, default=ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.models_dir, args.manifest, args.output_dir, args.partition, args.device)


if __name__ == '__main__':
    main()
