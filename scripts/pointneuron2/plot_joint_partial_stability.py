"""Plot recorded geometry probes only; no neural-model imports or training."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def run():
    source=ROOT/'tmp/pointneuron2/local_runs/joint_partial_stability_20261020/comparison/report.json'
    report=json.loads(source.read_text(encoding='utf-8'))
    if report['status']!='completed':raise ValueError('Expected completed paired control')
    target=ROOT/'docs/pointneuron2/joint_partial_stability_results_seed0/geometry.png'
    if target.exists():raise FileExistsError('Preserve existing figure')
    target.parent.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True,layout='constrained')
    for i,axis in enumerate(axes.flat):
        for arm,color,label in zip(report['arms'],('#c65b35','#2677b5'),('LR 0.0001','LR 0.00001')):
            x=[p['fork_update'] for p in arm['probes']]
            y=[p['crops'][i]['matched_distance_mean'] for p in arm['probes']]
            axis.plot(x,y,marker='o',markersize=3,color=color,label=label)
        crop=report['config']['training_crops'][i]
        axis.set_title(f'Training crop {crop[0]}/{crop[1]}')
        axis.axhline(2,color='#777777',linestyle='--',linewidth=1,label='2-voxel per-node trust limit')
        axis.set_ylim(0,2.4);axis.grid(alpha=.2)
    for axis in axes[1]:axis.set_xlabel('New updates from the same saved checkpoint')
    for axis in axes[:,0]:axis.set_ylabel('Mean matched-center error (voxels)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    figure.suptitle('Smaller steps stabilize geometry on four fixed training crops')
    figure.savefig(target,dpi=160);plt.close(figure)
    print(target)


if __name__=='__main__':run()
