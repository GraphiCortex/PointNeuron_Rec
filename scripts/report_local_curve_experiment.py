"""Read completed local-curve artifacts, render figures, retain portable evidence.

No model forward, optimizer, input filtering or parameter selection. Run only
after the bounded worker exits. Generated plots stay under tmp; readable result
and portable numeric evidence stay in docs.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pn2.inputs.node_cache import digest, write_once
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.evaluation.local_curve import consolidate
from local_curve_experiment import References


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'tmp/local_curve_head_20261104')
    args = parser.parse_args()
    base = args.root.resolve()
    if not base.is_relative_to(ROOT/'tmp'):
        raise ValueError('Report source must stay under PN2/tmp')
    run = base/'train4_dense'
    evaluations = [read(path) for path in sorted(run.glob('eval_step*_training/metrics.json'))]
    interrupted = not (run/'report.json').exists()
    if interrupted:
        stop = read(base/'stop_receipt.json')
        terminal = {'status': 'interrupted_wall_limit', 'updates': evaluations[-1]['step'],
            'config_fingerprint': read(base/'cpucheck/report.json')['config_fingerprint'],
            'last_logged_update': 1700, 'last_complete_checkpoint_update': evaluations[-1]['step'],
            'gate': {'passed': None, 'terminal_gate_evaluated': False}, 'stop_receipt': stop,
            'training': evaluations[-1]['summary'], 'development': None}
    else:
        terminal = read(run/'report.json')
    config = read(ROOT/'configs/local_curve_head.json')
    prepare = read(base/'prepare/report.json')
    cpu = read(base/'cpucheck/report.json')
    timing = read(base/'timing_dense/report.json')
    folder = base/'figures'
    folder.mkdir(exist_ok=False)
    steps = [r['step'] for r in evaluations]
    updates = [json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for key, label in [('loss', 'total'), ('support_loss', 'support'), ('offset_loss', 'offset')]:
        # Full cycles of four boxes, then a 100-update display average.
        values = np.array([r[key] for r in updates])
        block = 100
        n = len(values)//block
        axes[0].plot(np.arange(1, n+1)*block, values[:n*block].reshape(n, block).mean(1), label=label)
    axes[0].set(title='Training loss (100-update display means)', xlabel='Optimizer updates', ylabel='Loss')
    axes[0].legend()
    for key, label in [('precision_2', 'point agreement'), ('coverage_2', 'curve coverage')]:
        axes[1].plot(steps, [r['summary']['macro'][key] for r in evaluations], 'o-', label=label)
    axes[1].axhline(.9, color='C0', ls=':', label='agreement floor')
    required = max(.8, .9*prepare['summaries']['four']['oracle']['macro']['coverage_2'])
    axes[1].axhline(required, color='C1', ls=':', label='coverage gate')
    axes[1].set(title='Four-box TRAINING metrics at two voxels', xlabel='Optimizer updates', ylim=(0, 1.03))
    axes[1].legend(fontsize=8)
    labels = ['Sato ranked\n512 cap', 'Classical nodes\n512 cap', 'Dense head\nlatest complete eval', 'Reference-only\ndiagnostic']
    values = [prepare['summaries']['four']['sato512']['macro'],
              prepare['summaries']['four']['classical512']['macro'],
              evaluations[-1]['summary']['macro'], prepare['summaries']['four']['oracle']['macro']]
    x = np.arange(len(values))
    axes[2].bar(x-.18, [v['precision_2'] for v in values], .36, label='point agreement')
    axes[2].bar(x+.18, [v['coverage_2'] for v in values], .36, label='curve coverage')
    axes[2].set(xticks=x, xticklabels=labels, ylim=(0, 1.03), title='Counts/policies differ: descriptive context')
    axes[2].tick_params(axis='x', labelsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle('PN2 interrupted curve-head fit — latest complete evaluation at update 1,000; training only'
                 if interrupted else 'PN2 bounded curve-head experiment — training is not development or graph quality')
    fig.tight_layout()
    fig.savefig(folder/'four_box_learning.png', dpi=150)
    plt.close(fig)

    selection = read(base/'prepare/selection.json')['four']
    data = NodeBudgetDataset(ROOT, ROOT/config['data_root'])
    refs = References(config, data)
    final_folder = run/f'eval_step{terminal["updates"]:06d}_training'
    per_box = []
    with np.load(final_folder/'predictions.npz', allow_pickle=False) as predictions:
        fig, axes = plt.subplots(4, 3, figsize=(14, 15))
        for i, key in enumerate(selection):
            cid, tile = key
            image = data.image(cid, tile, 2048)
            reference = data.reference(cid, tile)
            p = predictions[f'box{i}_points']
            s = predictions[f'box{i}_scores']
            keep = consolidate(p, s, cutoff=.25, identities=image['packet']['flat_indices'])
            near = refs.get(cid)['distances'](p[keep]) <= 2.
            src = image['packet']['native_xyz']
            for j, (a, b, plane) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
                ax = axes[i, j]
                ax.scatter(src[:, a], src[:, b], s=2, c=image['packet']['attributes'][:, 3],
                           cmap='Greys', vmin=0, vmax=1, alpha=.35, rasterized=True)
                for curve in reference['reference_runs']:
                    ax.plot(curve[:, a], curve[:, b], color='forestgreen', lw=1.1)
                ax.scatter(p[keep[near], a], p[keep[near], b], s=5, color='darkorange', alpha=.7, rasterized=True)
                ax.scatter(p[keep[~near], a], p[keep[~near], b], s=5, color='crimson', alpha=.7, rasterized=True)
                ax.set_title(f'Alias {data.cases[cid]["alias"]}, box {tile}: {plane}; {len(keep)} accepted', fontsize=10)
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_xlabel('XYZ'[a]+' (native voxel)')
                ax.set_ylabel('XYZ'[b]+' (native voxel)')
            per_box.append({'alias': data.cases[cid]['alias'], 'tile_id': tile,
                            **evaluations[-1]['rows'][i]['metrics']})
        fig.suptitle('Gray: sampled image points (not raw-image MIPs); green: reference; accepted orange/red: within/beyond two voxels\n'
                     'Orthogonal TRAINING views; no predicted edges, connectivity or SWC claim', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(folder/'four_box_votes.png', dpi=150)
        plt.close(fig)

    evidence = {'schema': 'pn2-local-curve-head-results-v1', 'scope': 'bounded local proposal experiment',
        'config_sha256': digest(ROOT/'configs/local_curve_head.json'), 'config_fingerprint': terminal['config_fingerprint'],
        'cpu': cpu, 'timing': timing, 'fit': terminal, 'per_box': per_box,
        'prepare_seconds': prepare['seconds'], 'oracle': {g: s['oracle']['macro'] for g, s in prepare['summaries'].items()},
        'controls': {g: {n: s[n]['macro'] for n in ('sato', 'sato512', 'classical', 'classical512')}
                     for g, s in prepare['summaries'].items()},
        'saved_position_domains': read(base/'saved_position_domains.json'),
        'evaluations': [{'step': e['step'], 'summary': e['summary'],
                         'raw_accepted_summary': e['raw_accepted_summary'], 'raw_all_summary': e['raw_all_summary'],
                         'accepted512_summary': e['accepted512_summary'], 'top512_summary': e['top512_summary'],
                         'diagnostic_cutoffs': e['diagnostic_cutoffs'], 'offset_diagnostics': e['offset_diagnostics']}
                        for e in evaluations],
        'artifact_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                           for p in ((base/'stop_receipt.json' if interrupted else run/'report.json'),
                                     final_folder/'metrics.json', final_folder/'predictions.npz',
                                     base/'prepare/manifest.json', base/'prepare/report.json',
                                     folder/'four_box_learning.png', folder/'four_box_votes.png')},
        'test_sources': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                         for p in ROOT.glob('tests/test_local_curve*.py')},
        'full_cohort_training': False, 'gat_training': False, 'phase1_restart': False,
        'reserved_access': False, 'development_accuracy_claim': False,
        'final_reconstruction_quality_claim': False}
    write_once(ROOT/'docs/evidence/local_curve_head_20261104.json', evidence)
    print(json.dumps({'gate': terminal['gate'], 'figures': str(folder),
                      'portable_evidence': 'docs/evidence/local_curve_head_20261104.json'}))


if __name__ == '__main__':
    main()
