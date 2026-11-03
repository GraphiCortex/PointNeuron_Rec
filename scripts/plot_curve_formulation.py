"""Saved-output orthogonal views; original image backgrounds, no model forward.

Green curves are REFERENCE geometry. No GT-derived predicted edges are drawn.
Fixed four previously used training boxes; this is not favorable-case selection.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.inputs.node_cache import digest, write_once
from pn2.inputs.cases import ImageCase
from pn2.inputs.image import decode_image
from pn2.evaluation.formulation import distances_to_geometry, runs_graph, graph_segments
from scipy.spatial import cKDTree


def main(output, requested_panel=None):
    if not output.resolve().is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Use PN2/tmp')
    output.mkdir(parents=True, exist_ok=False)
    data = NodeBudgetDataset(ROOT, ROOT/'tmp/node_budget_data_20261102')
    panel = [(32, 27), (32, 20), (22, 2), (22, 5)] if requested_panel is None else requested_panel
    raw_cache = {};evidence = []
    run = ROOT/'tmp/consistent_node_fit_train_20261103_155059'
    sources = {}
    control = json.loads((ROOT/'configs/curve_formulation_audit.json').read_text())
    partitions = {case['partition'] for case in data.cases.values() if case['alias'] in {a for a, _ in panel}}
    for partition in partitions:
        for method in ('baseline', 'consistent'):
            folder = run/method/f'eval_stage002300_{partition}'
            name = 'epoch25_original' if method == 'baseline' else 'epoch25_consistent'
            spec = next(s for s in control['predictions'] if s['method'] == name and s['partition'] == partition)
            if digest(folder/'metrics.json') != spec['metrics_sha256'] or digest(folder/'predictions.npz') != spec['predictions_sha256']:
                raise ValueError('Saved prediction packets changed')
            metrics = json.loads((folder/'metrics.json').read_text())
            sources[(partition, method)] = (metrics, np.load(folder/'predictions.npz', allow_pickle=False))
    for alias, tile in panel:
        cid = next(cid for cid, case in data.cases.items() if case['alias'] == alias)
        case = data.cases[cid];box = data.boxes[(cid, tile)]
        cache = (ROOT/box['clouds']['2048']['path']).parent
        metadata = json.loads((cache/'case.json').read_text())
        if digest(cache/'case.json') != case['baseline_case_sha256']:raise ValueError('Cache receipt changed')
        if cid not in raw_cache:
            decoded = decode_image(ImageCase(cid, Path(metadata['image']['image_path']), case['partition']))
            if decoded.provenance['source_sha256'] != case['source_sha256'] or decoded.provenance['decoded_sha256'] != case['decoded_sha256']:
                raise ValueError('Original image changed')
            raw_cache[cid] = decoded.array_zyx
        start = np.array(box['box']['start_xyz']);stop = np.array(box['box']['stop_xyz'])
        low, high = start-.5, stop-.5
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start[::-1], stop[::-1]))
        tile_raw = raw_cache[cid][slices].astype(float)
        lo, hi = case['normalization']['minimum'], case['normalization']['maximum']
        tile_raw = (tile_raw-lo)/(hi-lo) if hi > lo else np.zeros_like(tile_raw)
        ref = data.reference(cid, tile);segments = graph_segments(runs_graph(ref['reference_runs']))
        fig, axes = plt.subplots(4, 3, figsize=(12, 11), constrained_layout=True)
        records = {'alias': alias, 'tile_id': tile, 'case_id': cid, 'models': {}}
        for row, name in enumerate(('2048 input locations', '8192 input locations', 'Original objective +5 epochs', 'Candidate objective +5 epochs')):
            if row < 2:
                points = data.image(cid, tile, (2048, 8192)[row])['packet']['native_xyz']
                scores = None;colors = '#5999dd';rejected = np.empty((0, 3))
                label = f'{name}: {len(points)} real voxels'
            else:
                method = ('baseline', 'consistent')[row-2];m, packet = sources[(case['partition'], method)]
                index = next(i for i, r in enumerate(m['boxes']) if r['case_id'] == cid and r['tile_id'] == tile)
                all_points, all_scores = packet[f'box{index}_centers'], packet[f'box{index}_scores']
                points = all_points[all_scores >= .5];rejected = all_points[all_scores < .5]
                dc = distances_to_geometry(points, segments, ref['target_xyz'])
                dn = cKDTree(ref['target_xyz']).query(points)[0]
                colors = np.where(dc > 2, '#f45a62', np.where(dn > 2, '#f0bb41', '#40c9df'))
                outside = int((~np.all((points >= low) & (points <= high), axis=1)).sum())
                label = f'{name}: {len(points)} accepted; {outside} outside box'
                records['models'][method] = {'accepted': len(points), 'outside_box': outside,
                    'near_reference_geometry_2': int((dc <= 2).sum()), 'near_reference_grid_2': int((dn <= 2).sum())}
            for column, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                ax = axes[row, column];excluded = next(i for i in range(3) if i not in (a, b))
                mip = tile_raw.max(axis=2-excluded)
                ax.imshow(mip, cmap='gray', vmin=0, vmax=1, origin='lower',
                          extent=[low[a], high[a], low[b], high[b]], interpolation='nearest')
                if len(rejected):ax.scatter(rejected[:, a], rejected[:, b], c='#999999', s=2, alpha=.25, linewidths=0)
                if len(points):ax.scatter(points[:, a], points[:, b], c=colors, s=1 if row < 2 else 10,
                                          alpha=.2 if row < 2 else .9, linewidths=0)
                for curve in ref['reference_runs']:
                    ax.plot(curve[:, a], curve[:, b], color='#55d77b', linewidth=.8, alpha=.9)
                ax.set(xlim=(low[a], high[a]), ylim=(low[b], high[b]), xlabel='XYZ'[a]+' (voxel)', ylabel='XYZ'[b]+' (voxel)')
                ax.set_title(('XY', 'XZ', 'YZ')[column], fontsize=9)
                if column == 0:ax.text(0, 1.13, label, transform=ax.transAxes, fontsize=9)
        fig.suptitle(f'{case["partition"].capitalize()} alias {alias}, box {tile}: {len(ref["target_xyz"])} reference targets\n'
                     'Original-image linear-intensity MIPs; fixed crop views; no predicted edges', fontsize=12)
        legend = [Line2D([0], [0], color='#55d77b', label='Reference curve (evaluation only)'),
            Line2D([0], [0], marker='o', linestyle='', color='#40c9df', label='Accepted: near curve and target grid'),
            Line2D([0], [0], marker='o', linestyle='', color='#f0bb41', label='Accepted: near curve, beyond grid'),
            Line2D([0], [0], marker='o', linestyle='', color='#f45a62', label='Accepted: >2 voxels from reference geometry')]
        fig.legend(handles=legend, loc='outside lower center', ncol=2, fontsize=8)
        destination = output/f'alias{alias}_box{tile}.png';fig.savefig(destination, dpi=160);plt.close(fig)
        records['image'] = destination.name;evidence.append(records)
    for _, packet in sources.values():packet.close()
    write_once(output/'report.json', {'status': 'completed', 'views': evidence,
        'training_or_filter_executed': False, 'original_image_redecoded': True,
        'prediction_edges_drawn': False, 'posthoc_diagnostic_selection': requested_panel is not None,
        'display_contrast': 'unchanged whole-image affine normalization, linear gray'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel', nargs='+', help='Explicit diagnostic alias:tile pairs; default is the four fixed training boxes')
    args = p.parse_args()
    panel = [tuple(map(int, value.split(':'))) for value in args.panel] if args.panel else None
    if panel is not None and any(len(pair) != 2 for pair in panel):p.error('Expected alias:tile pairs')
    main(args.output, panel)
