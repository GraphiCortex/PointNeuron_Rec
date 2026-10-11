"""Prepare/run a paired neuron-21 graph cutoff check with shared proposals.

Preparation performs only provenance checks. Run is a separate user-launched
stage: one full-volume proposal inference, two reconstructions, spatial/topology
evaluation. No model training or production-preset edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.vaa3d_raw import read_header
from scripts.pointneuron2.sampling_ablation import checked_manifest, check_matches, digest, live, read, write


def verify(plan):
    for name, sha in plan['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Source/settings changed: {name}; prepare a new comparison')
    for name in ['checkpoint', 'volume', 'reference', 'partition']:
        if digest(Path(plan[name])) != plan[name + '_sha256']:
            raise ValueError(f'Frozen {name} changed')
    if digest(Path(plan['manifest'])) != plan['manifest_sha256']:
        raise ValueError('Original cache manifest changed')


def aggregation_command(plan, output, device):
    config = plan['reconstruction']
    command = [sys.executable, '-u', str(REPO_ROOT / 'scripts/aggregate_proposals.py'),
               '--root', plan['data_root'], '--sample-index', str(plan['sample_index']),
               '--checkpoint', plan['checkpoint'], '--max-patches', '256', '--patch-selection', 'coverage',
               '--threshold-fraction', str(config['proposal_threshold_fraction']),
               '--score-threshold', str(config['proposal_score_threshold']),
               '--top-proposals-per-patch', str(config['proposal_top_per_patch']),
               '--global-top-proposals', str(config['proposal_global_top']),
               '--point-sample-strategy', config['point_sample_strategy'],
               '--local-nms-mode', 'sphere', '--global-nms-mode', 'sphere',
               '--local-nms-radius', str(config['proposal_local_nms_radius']),
               '--global-nms-radius', str(config['proposal_global_nms_radius']),
               '--device', device, '--output', str(output), '--html-output', str(output.with_suffix('.html'))]
    return command


def reconstruction_command(plan, output, shared, cutoff, device):
    config = dict(plan['reconstruction'], graph_min_proposal_score=cutoff)
    command = [sys.executable, '-u', str(REPO_ROOT / 'scripts/run_end_to_end.py'),
               '--root', plan['data_root'], '--sample-index', str(plan['sample_index']),
               '--checkpoint', plan['checkpoint'], '--output-root', str(output),
               '--existing-proposal-dir', str(shared), '--device', device, '--fail-fast']
    for key, value in config.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    return command


def prepare(experiment, output):
    if output.exists():
        raise FileExistsError('Use a fresh comparison directory')
    original = read(experiment / 'experiment.json')
    if original['status'] not in ('trained', 'evaluated'):
        raise ValueError('Requires completed paired coordinate training')
    for name, sha in original['source_sha256'].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f'Original source/settings changed: {name}')
    manifest = checked_manifest(original, experiment)
    check_matches([read(experiment / arm['name'] / 'run.json') for arm in original['arms']])
    index = 21
    rows = [r for r in manifest['records'] if r['partition'] == 'val' and r['sample_index'] == index]
    if not rows:
        raise ValueError('Neuron 21 is not in the frozen validation cache')
    arm = next(a for a in original['arms'] if a['name'] == 'repaired_local')
    checkpoint = (experiment / arm['name'] / 'proposal.pt').resolve()
    if digest(checkpoint) != arm['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after training')
    sample = scan_gold166(original['data_root'])[index]
    with np.load(rows[0]['resolved_path'], allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
    if (sample.sample_id != rows[0]['sample_id'] or digest(sample.volume_path) != metadata['volume_sha256']
            or digest(sample.swc_path) != metadata['swc_sha256']):
        raise ValueError('Validation image/reference no longer match cached sources')
    import torch
    from pointneuron.models.sampling_pipeline import validate_inference_reader
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_inference_reader(saved)
    if saved['args']['proposal_coordinate_mode'] != 'local':
        raise ValueError('Requires the repaired local-coordinate checkpoint')
    files = sorted((REPO_ROOT / 'src').rglob('*.py')) + sorted((REPO_ROOT / 'scripts').rglob('*.py'))
    files += sorted((REPO_ROOT / 'configs/pointneuron2').glob('*.json'))
    plan = {'status': 'prepared', 'scope': 'one internal validation development case; no training/test data',
            'original_experiment': str(experiment.resolve()), 'sample_index': index, 'sample_id': sample.sample_id,
            'data_root': original['data_root'], 'checkpoint': str(checkpoint),
            'checkpoint_sha256': arm['checkpoint_sha256'], 'checkpoint_epoch': saved['epoch'],
            'volume': str(sample.volume_path.resolve()), 'volume_sha256': metadata['volume_sha256'],
            'reference': str(sample.swc_path.resolve()), 'reference_sha256': metadata['swc_sha256'],
            'partition': original['partition'], 'partition_sha256': digest(original['partition']),
            'manifest': str((experiment / 'cache/manifest.json').resolve()), 'manifest_sha256': original['manifest_sha256'],
            'dimensions_xyzc': list(read_header(sample.volume_path).dimensions),
            'reconstruction': original['config']['reconstruction'], 'cutoffs': [.85, .5],
            'sampling_pipeline': saved['sampling_pipeline'],
            'source_sha256': {f.relative_to(REPO_ROOT).as_posix(): digest(f) for f in files},
            'results': [], 'optimizer_steps': 0}
    output.mkdir(parents=True)
    write(output / 'comparison.json', plan)
    print(f"Prepared neuron={index} {sample.sample_id}; one shared proposal set, graph cutoffs 0.85/0.5")
    print(f'Comparison: {output.resolve()}')


def run(output, device):
    path = output / 'comparison.json'; plan = read(path)
    if plan['status'] != 'prepared':
        raise ValueError('Run requires a fresh prepared comparison; preserve partial/completed runs')
    verify(plan)
    shared = output / 'shared_proposals'
    if shared.exists() or any((output / f'cutoff_{str(c).replace(".", "_")}').exists() for c in plan['cutoffs']):
        raise FileExistsError('Preserve prior run outputs; prepare a fresh comparison')
    plan.update(status='running', device=device); write(path, plan)
    start = time.perf_counter()
    try:
        shared.mkdir()
        proposal = shared / f"sample_{plan['sample_index']:04d}_proposals.npz"
        print('Stage 1/3: full-volume proposal inference (once). No training.', flush=True)
        live(aggregation_command(plan, proposal, device), output / 'aggregation.log')
        if not proposal.exists():
            raise ValueError('Aggregation did not produce the shared proposal file')
        proposal_sha = digest(proposal)
        with np.load(proposal, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if (meta['sample_id'] != plan['sample_id']
                    or Path(meta['checkpoint']).resolve() != Path(plan['checkpoint']).resolve()
                    or any(not np.isfinite(data[k]).all() for k in ['centers', 'radii', 'scores'])):
                raise ValueError('Invalid shared proposal identity/coordinates')
            plan['shared_proposals'] = {'path': str(proposal.resolve()), 'sha256': proposal_sha,
                                        'nodes': len(data['centers']), 'patches': meta['patches']}
        write(path, plan)
        for number, cutoff in enumerate(plan['cutoffs'], 2):
            name = f'cutoff_{str(cutoff).replace(".", "_")}'; arm = output / name
            print(f'Stage {number}/3: reconstruct shared proposals with graph cutoff={cutoff}', flush=True)
            live(reconstruction_command(plan, arm, shared, cutoff, device), output / f'{name}.log')
            summary = read(arm / 'summary.json')
            if summary['failures'] or len(summary['samples']) != 1:
                raise ValueError('Reconstruction failed or changed the comparison denominator')
            row = summary['samples'][0]
            if (row['sample_index'] != plan['sample_index'] or not row['swc_valid']
                    or Path(row['proposal_path']).resolve() != proposal.resolve()
                    or digest(proposal) != proposal_sha):
                raise ValueError('Reconstruction did not use unchanged shared proposals or produced invalid SWC')
            with np.load(row['graph_path'], allow_pickle=False) as data:
                graph = json.loads(str(data['metadata']))
            config = plan['reconstruction']
            if (graph['min_proposal_score'] != cutoff or graph['max_nodes'] != config['graph_max_nodes']
                    or graph['selection_mode'] != config['graph_selection_mode']
                    or graph['nms_distance'] != config['graph_nms_distance']
                    or graph['nms_mode'] != config['graph_nms_mode']):
                raise ValueError('Graph settings disagree with the controlled comparison')
            metrics = {}
            for kind, script in [('spatial', 'evaluate_paper_style_swc.py'), ('topology', 'evaluate_geodesic_baseline.py')]:
                live([sys.executable, '-u', str(REPO_ROOT / 'scripts' / script), '--root', plan['data_root'],
                      '--summary', str(arm / 'summary.json'), '--json-output', str(arm / f'{kind}.json'),
                      '--csv-output', str(arm / f'{kind}.csv')], output / f'{name}_{kind}.log')
                report = read(arm / f'{kind}.json')
                if len(report['samples']) != 1 or report['samples'][0]['sample_index'] != plan['sample_index']:
                    raise ValueError('Metric denominator changed')
                metrics[kind] = report['samples'][0]
            result = {'name': name, 'cutoff': cutoff, 'shared_proposal_sha256': proposal_sha,
                      'reconstruction': row, 'graph_metadata': graph, 'metrics': metrics,
                      'swc_sha256': digest(row['swc_path']), 'graph_sha256': digest(row['graph_path'])}
            plan['results'].append(result); write(path, plan)
            spatial = metrics['spatial']; topology = metrics['topology']
            print(f"cutoff={cutoff}: spatial_F1={spatial['f1']:.4f} precision={spatial['precision']:.4f} "
                  f"recall={spatial['recall']:.4f} topology_F1={topology['edge_f1']:.4f} "
                  f"bridges={row['bridge_edges']} foreground_cap_satisfied={row['foreground_cap_satisfied']}", flush=True)
        verify(plan)
        if digest(proposal) != proposal_sha:
            raise ValueError('Shared proposals changed during evaluation')
        plan.update(status='completed', paired_checks='passed', elapsed_seconds=time.perf_counter() - start)
    except BaseException as exc:
        plan.update(status='failed_or_interrupted', error=str(exc), elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        write(path, plan)
    print(f"Status: {plan['status']}; paired checks: {plan['paired_checks']}; elapsed: {plan['elapsed_seconds']:.1f}s")
    print(f'Results: {path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    parser.add_argument('--experiment', type=Path, help='Original trained coordinate experiment (prepare only)')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh comparison root at preparation; same root at run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == 'prepare':
        if args.experiment is None:
            parser.error('prepare requires --experiment')
        prepare(args.experiment.resolve(), output)
    else:
        run(output, args.device)


if __name__ == '__main__':
    main()
