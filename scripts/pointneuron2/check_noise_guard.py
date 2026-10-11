"""Check the bounded noise repair on saved training scores; no extraction/training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from pointneuron.data.foreground import guarded_noise_hysteresis_mask


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', action='append', help='sample:patch; default four known training failure/contrast cases')
    args = parser.parse_args()
    source = json.loads(args.comparison.read_text(encoding='utf-8-sig'))
    partition = REPO_ROOT / 'configs/pointneuron2/partition_seed0.json'
    if source['status'] != 'completed' or source['failures'] or source['partition_sha256'] != digest(partition):
        raise ValueError('Requires a completed comparison from the frozen partition')
    cases = [tuple(map(int, case.split(':'))) for case in (args.case or ['10:0', '10:6', '9:5', '1:7'])]
    report = {'scope': 'targeted mask-only regression; no guide extraction or reconstruction evidence',
              'comparison': str(args.comparison.resolve()), 'comparison_sha256': digest(args.comparison),
              'source_sha256': {'foreground.py': digest(REPO_ROOT / 'src/pointneuron/data/foreground.py'),
                                'check_noise_guard.py': digest(Path(__file__))}, 'cases': []}
    train = {(r['sample_index'], r['sample_id']) for r in json.loads(partition.read_text())['partitions']['train']}
    for case in cases:
        row = next(r for r in source['patches'] if (r['sample_index'], r['patch_index']) == case)
        if row['partition'] != 'train' or (row['sample_index'], row['sample_id']) not in train:
            raise ValueError('Diagnostic is training-only')
        meta = row['methods']['background_otsu']
        path = args.comparison.parent / meta['artifact']
        if digest(path) != meta['artifact_sha256']:
            raise ValueError('Saved artifact changed')
        with np.load(path, allow_pickle=False) as data:
            mask, info = guarded_noise_hysteresis_mask(data['score'], data['raw_image'])
            bound = bool(np.all(~mask | data['mask']))
        if not bound:
            raise AssertionError('Guard escaped background Otsu foreground')
        result = {'sample_index': case[0], 'patch_index': case[1], 'sample_id': row['sample_id'],
                  'background_otsu_voxels': meta['foreground_voxels'],
                  'historical_noise_voxels': row['methods']['noise_hysteresis']['foreground_voxels'],
                  'guarded_noise_voxels': int(mask.sum()), 'subset_check': bound, 'metadata': info}
        report['cases'].append(result)
        print(f"case={case} Otsu={result['background_otsu_voxels']} old_noise={result['historical_noise_voxels']} "
              f"guard={result['guarded_noise_voxels']} MAD_above_floor={info['noise_estimate_above_floor']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report['status'] = 'completed'
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
