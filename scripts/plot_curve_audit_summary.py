"""Scientific summary of completed CPU audit; no models or metric recomputation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(audit, baseline, output):
    a = json.loads(audit.read_text());b = json.loads(baseline.read_text())
    if a['status'] != 'completed' or b['status'] != 'completed':raise ValueError('Completed reports required')
    s = a['summary']['partitions'];classical = b['summary']['partitions']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    positions = np.arange(4);width = .34
    labels = ['Train\n2048', 'Train\n8192', 'Development\n2048', 'Development\n8192']
    pairs = [('training', '2048'), ('training', '8192'), ('development', '2048'), ('development', '8192')]
    ax = axes[0, 0]
    for shift, tolerance, color in [(-width/2, '2.0', '#4477aa'), (width/2, '6.0', '#66bbaa')]:
        values = [100*s[p]['input'][budget][tolerance] for p, budget in pairs]
        bars = ax.bar(positions+shift, values, width, label=f'Within {int(float(tolerance))} voxels', color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 112), ylabel='Reference arclength (%)',
           title='Input proximity: spatial evidence, not a prediction ceiling')
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[0, 1];positions = np.arange(2)
    for shift, key, label, color in [(-width/2, 'reference_target_grid_point_support_2', 'Isolated target-node support', '#aa8844'),
        (width/2, 'reference_target_graph_oracle_f1_2', 'Target graph + REFERENCE edges: curve F1', '#4477aa')]:
        values = [100*s[p][key] for p in ('training', 'development')]
        bars = ax.bar(positions+shift, values, width, label=label, color=color);ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set(xticks=positions, xticklabels=['Training', 'Development'], ylim=(0, 112), ylabel='Percent',
           title='Reference discretization: two different measurements')
    ax.legend(fontsize=7, loc='lower right')
    ax = axes[1, 0];x = np.arange(4)
    pairs = [('training', 'epoch25_original'), ('training', 'epoch25_consistent'),
             ('development', 'epoch25_original'), ('development', 'epoch25_consistent')]
    near = np.array([s[p]['predictions'][m]['accepted']['on_reference_geometry_fraction_2'] for p, m in pairs])
    beyond = np.array([s[p]['predictions'][m]['accepted']['on_curve_beyond_grid_fraction_2'] for p, m in pairs])
    ax.bar(x, 100*(near-beyond), color='#40a9c5', label='On curve and near target grid')
    ax.bar(x, 100*beyond, bottom=100*(near-beyond), color='#e8b343', label='On curve, beyond target grid')
    ax.bar(x, 100*(1-near), bottom=100*near, color='#d76770', label='Beyond 2 voxels from annotated geometry')
    ax.set(xticks=x, xticklabels=['Train\nOriginal', 'Train\nCandidate', 'Dev\nOriginal', 'Dev\nCandidate'],
           ylim=(0, 108), ylabel='Accepted proposals (%)',
           title='Saved nodes: reference-present boxes, 2-voxel tolerance')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.14))
    ax = axes[1, 1];x = np.arange(4);barwidth = .25
    pairs = [('training', 'raw'), ('training', 'pruned'), ('development', 'raw'), ('development', 'pruned')]
    for shift, metric, label, color in [(-barwidth, 'full_grid_precision', 'Precision', '#4477aa'),
        (0., 'recall', 'Recall', '#66bbaa'), (barwidth, 'full_grid_f1', 'F1', '#aa8844')]:
        values = [100*classical[p][variant]['2.0'][metric] for p, variant in pairs]
        bars = ax.bar(x+shift, values, barwidth, label=label, color=color);ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    ax.set(xticks=x, xticklabels=['Train panel\nRaw', 'Train panel\nPruned', 'Development\nRaw', 'Development\nPruned'],
           ylim=(0, 112), ylabel='Arclength agreement (%)',
           title='Automatic classical graph: segment metrics, not learned-node F1')
    ax.legend(fontsize=8)
    fig.suptitle('PN2 formulation audit — no model training\nMacro means over neurons; native voxel units; geometry does not certify connectivity', fontsize=13)
    fig.savefig(output, dpi=170);plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True);p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();main(a.audit, a.baseline, a.output)
