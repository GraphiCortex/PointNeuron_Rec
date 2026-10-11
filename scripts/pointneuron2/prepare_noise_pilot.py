"""Small training-only pilot reusing exact saved background scores and patch boxes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import label

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.foreground import noise_hysteresis_mask
from pointneuron.data.medial_sampling import MedialSamplingConfig
from scripts.pointneuron2.compare_foreground import digest, process_mask, render, summarize, write_json
from scripts.pointneuron2.audit_sampling import reference_support
from scripts.pointneuron2.inspect_foreground_slices import render_slices, signed_reference_contrast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--case', required=True, action='append', help='Training sample:patch indices, e.g. 0:3')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/pointneuron2/foreground_noise_comparison.json')
    parser.add_argument('--partition', type=Path, default=REPO_ROOT / 'configs/pointneuron2/partition_seed0.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.comparison.read_text(encoding='utf-8-sig'))
    config = json.loads(args.config.read_text())
    partition = json.loads(args.partition.read_text())
    train = {(x['sample_index'], x['sample_id']) for x in partition['partitions']['train']}
    cases = [tuple(map(int, value.split(':'))) for value in args.case]
    if len(set(cases)) != len(cases) or any(len(c) != 2 for c in cases):
        raise ValueError('Cases must be unique sample:patch pairs')
    if source['status'] != 'completed' or source['failures'] or source['partition_sha256'] != digest(args.partition):
        raise ValueError('Requires a completed comparison from the frozen partition')
    rows = [next(x for x in source['patches'] if (x['sample_index'], x['patch_index']) == case) for case in cases]
    if any(x['partition'] != 'train' or (x['sample_index'], x['sample_id']) not in train for x in rows):
        raise ValueError('Pilot is restricted to training patches')
    medial = MedialSamplingConfig(**source['medial_config'])
    if config['guide_points'] is not None or source['config']['guide_points'] is not None:
        raise ValueError('Pilot requires full guides')
    if config['background_sigma'] != source['config']['background_sigma']:
        raise ValueError('Saved scores use a different background scale')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True); (args.output_dir / 'artifacts').mkdir()
    report = deepcopy(source)
    report.update(status='running', scope='four-case preparation pilot; deliberately selected failure/contrast cases, not a broad benchmark',
                  config=config, config_sha256=digest(args.config), patches=[], summary={}, failures=[], promoted_method=None,
                  pilot_cases=cases, source_comparison=str(args.comparison.resolve()), source_comparison_sha256=digest(args.comparison),
                  baseline_source_sha256=source['source_sha256'],
                  source_sha256={name: digest(REPO_ROOT / name) for name in
                                  ['scripts/pointneuron2/prepare_noise_pilot.py', 'scripts/pointneuron2/compare_foreground.py',
                                   'scripts/pointneuron2/inspect_foreground_slices.py', 'src/pointneuron/data/foreground.py',
                                   'src/pointneuron/data/medial_sampling.py']})
    report_path = args.output_dir / 'comparison.json'; write_json(report_path, report)
    for original in rows:
        row = deepcopy(original); row['methods'] = {}; outputs = {}; artifacts = {}
        for method in ('fixed', 'background_otsu'):
            meta = deepcopy(original['methods'][method])
            path = (args.comparison.parent / meta['artifact']).resolve()
            if digest(path) != meta['artifact_sha256']:
                raise ValueError('Saved baseline artifact changed')
            meta['artifact'] = str(path); meta['reused_from_previous_run'] = True
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            artifacts[method] = arrays
            row['methods'][method] = meta
            outputs[method] = (arrays['mask'], arrays, meta)
        raw = artifacts['background_otsu']
        image, lower = raw['raw_image'], raw['lower_xyz']
        start = time.perf_counter()
        # No reference fields enter either mask creation or guide/pool generation.
        mask, info = noise_hysteresis_mask(raw['score'], image, config['noise_high'], config['noise_low'])
        arrays = process_mask(image, mask, lower, medial, config,
                              config['seed'] + row['sample_index'] * 10000 + row['patch_index'])
        sets = {'all_foreground': arrays.pop('foreground_xyz'), 'full_guide': arrays['full_guide_xyz'],
                'candidate_pool': arrays['points'][:, :3], 'bounded_guide': arrays['image_guide_xyz']}
        roles = {k: raw['reference_' + k] for k in ('terminals', 'junctions')}
        meta = {**info, 'method': 'noise_hysteresis', 'foreground_voxels': int(mask.sum()),
                'foreground_fraction': float(mask.mean()), 'connected_components_26': int(label(mask, structure=np.ones((3, 3, 3)))[1]),
                'full_guide_vertices': len(arrays['full_guide_xyz']), 'guide_edges': len(arrays['guide_edges']),
                'candidate_rows': len(arrays['points']), 'bounded_guide_vertices': len(arrays['image_guide_xyz']),
                'reference_support': {k: reference_support(raw['reference_curve'], v, np.ones(3)) for k, v in sets.items()},
                'role_support': {role: {k: reference_support(coords, v, np.ones(3)) for k, v in sets.items()} for role, coords in roles.items()},
                'seconds': time.perf_counter() - start, 'background_score_artifact_sha256': original['methods']['background_otsu']['artifact_sha256']}
        arrays.update({k: raw[k] for k in ('raw_image', 'lower_xyz', 'reference_curve', 'reference_segments', 'reference_terminals', 'reference_junctions')})
        arrays.update(mask=mask, score=raw['score'])
        stem = f"sample_{row['sample_index']:04d}_patch_{row['patch_index']:03d}"
        asset = args.output_dir / 'artifacts' / (stem + '_noise_hysteresis.npz')
        np.savez_compressed(asset, **arrays)
        meta.update(artifact=asset.relative_to(args.output_dir).as_posix(), artifact_sha256=digest(asset))
        row['methods']['noise_hysteresis'] = meta; artifacts['noise_hysteresis'] = arrays
        outputs['noise_hysteresis'] = (mask, arrays, meta)
        upper = np.asarray(row['upper_xyz'])
        render(args.output_dir / (stem + '.png'), image, lower, upper, raw['reference_segments'], outputs, row)
        row['figure'] = stem + '.png'
        row['slice_figure'] = stem + '_slices.png'
        row['slices'] = render_slices(args.output_dir / row['slice_figure'], artifacts, row)
        row['contrast_diagnostic'] = signed_reference_contrast(image, lower, raw['reference_curve'])
        report['patches'].append(row); report['summary'] = summarize(report['patches'], config['methods'])
        write_json(report_path, report)
        support = meta['reference_support']['full_guide']
        print(f"case={row['sample_index']}:{row['patch_index']} noise_sigma={info['noise_sigma']:.3f} "
              f"foreground={mask.sum()} guide={meta['full_guide_vertices']} "
              f"support={None if support is None else round(support['within_3'], 3)}", flush=True)
    report['status'] = 'completed'; write_json(report_path, report)
    print(f'Pilot report: {report_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
