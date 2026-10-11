"""Read corrected training-mask artifacts and compare exactly matched guide counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from scripts.pointneuron2.audit_sampling import reference_support
from scripts.pointneuron2.compare_foreground import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding='utf-8'))
    if (report['status'] != 'completed' or report['failures']
            or report.get('volume_reader_version') != VOLUME_READER_VERSION
            or any(r['partition'] != 'train' for r in report['patches'])):
        raise ValueError('Requires a completed corrected-reader training-only comparison')
    if args.budget < 1 or args.repeats < 2:
        raise ValueError('Require a positive budget and at least two draws')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    methods = report['config']['methods']
    rows, hashes = [], {}
    for row in report['patches']:
        arrays = {}
        for method in methods:
            meta = row['methods'][method]
            source = args.comparison.parent / meta['artifact']
            actual = digest(source)
            if actual != meta['artifact_sha256']:
                raise ValueError('Saved mask/guide artifact has changed')
            hashes[meta['artifact']] = actual
            with np.load(source, allow_pickle=False) as data:
                arrays[method] = (np.unique(data['full_guide_xyz'], axis=0), data['reference_curve'].copy())
        curve = arrays[methods[0]][1]
        if any(not np.array_equal(curve, arrays[m][1]) for m in methods):
            raise ValueError('Methods have different diagnostic references')
        count = min(args.budget, *(len(arrays[m][0]) for m in methods))
        result = {k: row[k] for k in ('sample_index', 'patch_index', 'reference_curve_points')}
        result.update(common_guide_points=count, methods={})
        for method in methods:
            guide = arrays[method][0]
            measurements = []
            if len(curve):
                for repetition in range(args.repeats):
                    seed = report['config']['seed'] + row['sample_index'] * 10000 + row['patch_index'] * 100 + repetition
                    rng = np.random.default_rng(seed)
                    selected = guide[np.sort(rng.choice(len(guide), count, replace=False))]
                    measurements.append(reference_support(curve, selected, np.ones(3)))
            result['methods'][method] = {
                'full_guide_points': len(guide),
                'full_guide_support': row['methods'][method]['reference_support']['full_guide'],
                'matched_mean_support': {f'within_{r}': float(np.mean([v[f'within_{r}'] for v in measurements]))
                                         for r in (1, 3, 6)} if measurements else None,
                'matched_std_support': {f'within_{r}': float(np.std([v[f'within_{r}'] for v in measurements]))
                                        for r in (1, 3, 6)} if measurements else None}
        rows.append(result)
    annotated = [r for r in rows if r['reference_curve_points']]
    summary = {m: {field: float(np.mean([r['methods'][m][field]['within_3'] for r in annotated]))
                   for field in ('full_guide_support', 'matched_mean_support')}
               for m in methods} if annotated else {}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'matched_guides.json', {
        'status': 'completed', 'scope': 'training-only saved-guide point-count control; no extraction or training',
        'comparison_sha256': digest(args.comparison), 'script_sha256': digest(__file__),
        'artifact_sha256': hashes, 'maximum_budget': args.budget, 'draws_per_method': args.repeats,
        'rule': 'Per patch, sample the same min(maximum budget, every method unique guide count); no padding',
        'metric_units': 'voxel proximity to available sampled SWC curve; not reconstruction F1 or precision',
        'mean_rule': 'unweighted annotated-patch mean; also retains the six-reference-point fly crop',
        'patches': rows, 'summary_within_3': summary})
    if annotated:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        positions = np.arange(len(annotated))
        width = .8 / len(methods)
        for i, method in enumerate(methods):
            xs = positions - .4 + width * (i + .5)
            for ax, field in zip(axes, ('full_guide_support', 'matched_mean_support')):
                values = [r['methods'][method][field]['within_3'] for r in annotated]
                errors = [r['methods'][method]['matched_std_support']['within_3'] for r in annotated] if field.startswith('matched') else None
                ax.bar(xs, values, width, label=method, yerr=errors, capsize=2)
        for ax, title in zip(axes, ('All extracted guide vertices: counts differ',
                                  'Exactly the same point count per crop: mean of eight random subsets')):
            ax.set_ylim(0, 1.08); ax.set_ylabel('Reference coverage within 3 voxels')
            ax.set_title(title); ax.legend(); ax.grid(axis='y', alpha=.2)
        axes[1].set_xticks(positions, [f"{r['sample_index']}/{r['patch_index']}\nN={r['common_guide_points']}" for r in annotated])
        axes[1].set_xlabel('TRAIN neuron / patch; N = common guide count. Nine annotated crops; three unannotated crops excluded from these bars.')
        fig.savefig(args.output_dir / 'matched_guides.png', dpi=130)
        plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Report: {args.output_dir / "matched_guides.json"}', flush=True)


if __name__ == '__main__':
    main()
