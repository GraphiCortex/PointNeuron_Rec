"""Bounded representative-batch timing for the approved pilot; no saved training model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))
from pointneuron.data.query_cache import load_query_manifest, QueryCacheDataset, collate_query_records, file_digest
from pointneuron.models.query_pipeline import build_models, forward_proposals
from scripts.pointneuron2.train_query_comparison import (prepare_run_config, prepare_batch, objective, apply_update,
                                                       state_digest, validation_quality, summarize_validation_quality)


def representative_pairs(rows, quantiles):
    known = sorted((r for r in rows if r['target_count'] > 0), key=lambda r: (r['target_count'], r['path']))
    if len(known) < 2:
        raise ValueError('Timing requires at least two annotated crops per role')
    return [known[min(int(q * (len(known)-2)), len(known)-2):min(int(q * (len(known)-2)), len(known)-2)+2]
            for q in quantiles]


def time_arm(arm, config, recipe, pairs, device):
    encoder, head = build_models(config, arm, config['seed'])
    initial = state_digest(encoder)
    encoder, head = encoder.to(device), head.to(device)
    parameters = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    torch.manual_seed(config['seed']+4001)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    timings = {'train': [], 'val': []}
    updates = 0
    for role in ('warmup', 'train', 'val'):
        training = role != 'val'
        encoder.train(training); head.train(training)
        chosen_pairs = pairs['train'][:1] if role == 'warmup' else pairs[role]
        for j, rows in enumerate(chosen_pairs):
            if device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            dataset = QueryCacheDataset(rows)
            batch = collate_query_records([dataset[0], dataset[1]])
            points, nodes, mask, centers = prepare_batch(batch, config, device,
                config['seed']+103000+j if training else 0,
                config['seed']+102000+j if training and config['augment'] else None)
            with torch.set_grad_enabled(training):
                output = forward_proposals(head, arm, points, encoder(points), centers, config['patch_radius'])
                loss = objective(arm, output, points, nodes, mask, config, recipe)
                if not bool(torch.isfinite(loss.total)):
                    raise FloatingPointError('Nonfinite representative-batch loss')
                if training:
                    updates += apply_update(loss, optimizer, parameters)
                else:
                    summarize_validation_quality(validation_quality(output, nodes, mask, batch['metadata']))
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if role != 'warmup':
                timings[role].append({'seconds': elapsed, 'paths': [r['path'] for r in rows],
                                      'target_counts': [r['target_count'] for r in rows]})
            print(f'{arm} timing {role} batch={j+1}/{len(chosen_pairs)} targets={[r["target_count"] for r in rows]} seconds={elapsed:.4f}', flush=True)
    return {'arm': arm, 'initial_encoder_sha256': initial, 'throwaway_optimizer_updates': updates,
            'timings': timings, 'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot-config', type=Path, default=ROOT/'configs/pointneuron2/query_pilot_layernorm.json')
    parser.add_argument('--partition', type=Path, default=ROOT/'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError('Use a fresh timing directory')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    torch.set_num_threads(2)
    manifest = load_query_manifest(args.manifest, args.partition)
    config = prepare_run_config(manifest['config'], pilot_config=json.loads(args.pilot_config.read_text()))
    recipe = json.loads((ROOT/config['pointwise_recipe']).read_text())['training']
    rows = {role: [r for r in manifest['records'] if r['partition']==role] for role in ('train','val')}
    pairs = {'train': representative_pairs(rows['train'], (.1,.5,.9,1.)),
             'val': representative_pairs(rows['val'], (.1,.5,1.))}
    counts = {role: (len(r)+config['batch_size']-1)//config['batch_size'] for role,r in rows.items()}
    result = {'scope': 'representative timing only; fresh throwaway models; no checkpoints or quality experiment',
              'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
              'manifest_sha256': file_digest(args.manifest), 'pilot_config_sha256': file_digest(args.pilot_config),
              'config': config, 'epochs_per_arm': config['epochs'], 'batches_per_epoch': counts,
              'timing_policy': 'one warmup + four train updates and three validation forwards per arm; annotated crops at declared target-count quantiles; data preparation and validation diagnostics included',
              'arms': []}
    for arm in ('pointwise','query'):
        result['arms'].append(time_arm(arm, config, recipe, pairs, args.device))
    if result['arms'][0]['initial_encoder_sha256'] != result['arms'][1]['initial_encoder_sha256']:
        raise ValueError('Unmatched timing encoder initialization')
    estimates = {}
    for name, reduce in (('sample_minimum', np.min), ('sample_mean', np.mean), ('sample_maximum', np.max)):
        estimates[name] = sum(config['epochs'] * sum(counts[role] * float(reduce([t['seconds'] for t in arm['timings'][role]]))
                                                   for role in ('train','val')) for arm in result['arms'])
    result.update(status='completed', estimated_total_seconds=estimates,
                  estimate_note='rough projection, not a bound; assumes all batches supervised and includes measured data preparation/metrics; excludes process startup, checkpoint writes, thermal changes and other workload')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Timing complete: total pilot projection seconds={json.dumps(estimates)}; report={args.output_dir.resolve()/"timing.json"}', flush=True)


if __name__ == '__main__':
    main()
