"""Read-only image and supervision audit for neuron-79 off-reference groups.

Decode one frozen image, replay image-defined patch geometry, inspect cached
labels and test the existing objective on synthetic CPU tensors. No model
inference, gradient updates, new negative labels or reconstruction changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import maximum_filter
import torch

REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT/'src'))
sys.path.insert(0,str(REPO_ROOT))
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_volume
from pointneuron.data.training_cache import volume_data_array
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.models.proposal import SkeletonProposalOutput
from scripts.aggregate_proposals import build_patch_specs
from scripts.pointneuron2.train_sampling import proposal_objective
from scripts.pointneuron2.compare_component_allocation import verify_inputs
from scripts.pointneuron2.audit_full_volume_selection import reference_targets,distances
from scripts.pointneuron2.audit_hybrid_graph_paths import voxel_values
from scripts.pointneuron2.sampling_ablation import read,write,digest


def gradient_probe(recipe):
    xyz=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[20.,0.,0.],[25.,0.,0.]],
        [[100.,0.,0.],[101.,0.,0.],[120.,0.,0.],[125.,0.,0.]]])
    points=torch.cat([xyz,torch.ones(2,4,1)],dim=-1)
    logits=torch.zeros(2,4,2,requires_grad=True)
    output=SkeletonProposalOutput(offsets=torch.zeros(2,4,3,requires_grad=True),objectness_logits=logits,
        radius=torch.ones(2,4,1,requires_grad=True),center_proposals=xyz.clone().requires_grad_(),
        raw=torch.zeros(2,4,6,requires_grad=True))
    nodes=torch.tensor([[[1.,0.,0.,0.,1.]],[[0.,0.,0.,0.,0.]]])
    mask=torch.tensor([[True],[False]])
    loss=proposal_objective(output,points,nodes,mask,recipe)
    gradient=torch.autograd.grad(loss,logits)[0]
    labeled=float(gradient[0].abs().sum())
    unlabeled=float(gradient[1].abs().sum())
    if labeled<=0 or unlabeled!=0:
        raise ValueError('Unexpected existing supervision gradients')
    return {'method':'synthetic mixed batch, existing objective, CPU autograd; no parameter update',
        'annotated_crop_objectness_gradient_l1':labeled,'unlabeled_crop_objectness_gradient_l1':unlabeled,
        'unlabeled_crop_treated_as_background':False}


def inspect(path):
    plan=read(path)
    original,trace,comparison=verify_inputs(plan)
    baseline=comparison['baseline']
    manifest=load_sampling_manifest(baseline['manifest'],baseline['partition'])
    if digest(baseline['manifest'])!=baseline['manifest_sha256']:
        raise ValueError('Training manifest differs from frozen experiment')
    run_path=Path(baseline['checkpoint']).parent/'run.json'
    run=read(run_path)
    if run['manifest_sha256']!=baseline['manifest_sha256'] or run['status']!='completed':
        raise ValueError('Training run provenance differs')
    cache=[]
    for row in manifest['records']:
        with np.load(row['resolved_path'],allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            count=len(z['skeleton_nodes'])
        if count!=row['label_nodes']:
            raise ValueError('Cached label count differs from manifest')
        cache.append({'partition':row['partition'],'sample_index':row['sample_index'],
            'patch_index':row['patch_index'],'center':meta['patch_center'],'label_nodes':count})
    counts={role:{'records':sum(c['partition']==role for c in cache),
        'with_local_reference_nodes':sum(c['partition']==role and c['label_nodes']>0 for c in cache),
        'without_local_reference_nodes':sum(c['partition']==role and c['label_nodes']==0 for c in cache)}
        for role in ['train','val']}
    for role in counts:
        if counts[role]['with_local_reference_nodes']!=run['epochs'][-1][role]['annotated_records']:
            raise ValueError('Training log annotated count differs from cache')
    print('Training cache hashes/identities verified. Decoding the unchanged neuron-79 image once on CPU.',flush=True)
    volume=read_volume(baseline['volume']);w,h,z,c=volume.dimensions
    if c!=1:
        raise ValueError('Expected frozen single-channel image')
    data=volume_data_array(volume)
    image=data.reshape(z,h,w)
    with np.load(original['proposal']['path'],allow_pickle=False) as proposals:
        proposal_meta=json.loads(str(proposals['metadata']))
        patch_indices=proposals['patch_indices'].copy()
    cfg=proposal_meta['sampling_pipeline']
    specs=build_patch_specs(np.flatnonzero(data>proposal_meta['threshold']),data,w,h,z,
        cfg['patch_radius'],cfg['stride'],proposal_meta['threshold'],cfg['min_foreground_points'],256,'coverage')
    if len(specs)!=proposal_meta['patches']:
        raise ValueError('Inference patch geometry differs from saved inference')
    reference=parse_swc(baseline['reference'])
    node_xyz=np.asarray([[n.x,n.y,n.z] for n in reference.nodes],dtype=np.float32)
    targets=reference_targets(baseline['reference'])[0]
    rows=[]
    neuron_cache=[row for row in cache if row['sample_index']==79]
    for patch_index,spec in specs:
        labels=int(np.count_nonzero(np.all(np.abs(node_xyz-spec['center'])<=cfg['patch_radius'],axis=1)))
        matched=[r for r in neuron_cache if np.array_equal(r['center'],spec['center'])]
        if len(matched)>1 or (matched and matched[0]['label_nodes']!=labels):
            raise ValueError('Cached and inference patch centers/labels disagree')
        rows.append({'inference_patch_index':patch_index,'center':spec['center'].tolist(),'reference_nodes':labels,
            'in_validation_cache':bool(matched),'validation_cache_patch_index':matched[0]['patch_index'] if matched else None})
    with np.load(trace['matrix_artifact']['path'],allow_pickle=False) as pool:
        pool_xyz=pool['centers'].copy();pool_scores=pool['scores'].copy();pool_source=pool['proposal_indices'].copy()
    groups=[]
    nearby=maximum_filter(image,size=5,mode='constant',cval=0)
    for component in trace['component_trace']['components'][:3]:
        idx=np.asarray(component['candidate_indices'],dtype=np.int64)
        xyz=pool_xyz[idx]
        exact=voxel_values(image,xyz);local=voxel_values(nearby,xyz)
        d=distances(targets['curve'],xyz)
        source=patch_indices[pool_source[idx]]
        patches=sorted(set(map(int,source)))
        source_rows=[rows[i] for i in patches]
        if any(rows[i]['inference_patch_index']!=i for i in patches):
            raise ValueError('Inference patch identities are not sequential')
        groups.append({'component_rank':component['component_rank'],'candidates':len(idx),
            'mean_score':float(pool_scores[idx].mean()),'reference_within_6_candidates':int(np.count_nonzero(d<=6)),
            'median_reference_distance':float(np.median(d)),
            'median_rounded_voxel_intensity':float(np.nanmedian(exact)),
            'rounded_voxel_above15_fraction':float(np.mean(exact>15)),
            'nearby_cube_radius2_above15_fraction':float(np.mean(local>15)),
            'nearby_cube_radius2_above51_fraction':float(np.mean(local>51)),
            'source_patch_count':len(patches),'source_patches':source_rows,
            'candidates_from_patches_without_local_reference_nodes':int(sum(rows[int(i)]['reference_nodes']==0 for i in source)),
            'source_patches_in_validation_cache':sum(r['in_validation_cache'] for r in source_rows),
            'source_patches_without_local_reference_nodes':sum(r['reference_nodes']==0 for r in source_rows)})
    report={'status':'completed','verification':'passed','comparison_sha256':digest(path),
        'inspection_source_sha256':digest(Path(__file__)),'image_sha256':baseline['volume_sha256'],
        'reference_sha256':baseline['reference_sha256'],'manifest_sha256':baseline['manifest_sha256'],
        'training_run_sha256':digest(run_path),'image_dimensions_xyzc':list(volume.dimensions),
        'cache_supervision_counts':counts,'neuron79_validation_cache':neuron_cache,
        'inference_patches':len(rows),'inference_patches_without_reference_nodes':sum(r['reference_nodes']==0 for r in rows),
        'components':groups,'existing_objective_gradient_probe':gradient_probe(run['recipe']),
        'recipe':run['recipe'],'scope':'one source image plus saved train/validation cache labels; no reserved test use',
        'limitations':['Empty local reference labels do not establish verified biological background.',
            'Bright foreground may be another structure or an unannotated part of the target neuron.',
            'This audit cannot establish whether the reference annotation is complete.',
            'Zero supervision on unlabeled crops is confirmed; its causal contribution to this model error is not isolated.',
            'Inference patch indices are local to their selected list; cache matching uses actual centers, not index equality.',
            'Validation labels are used for diagnosis only, not training or inference decisions.',
            'XY image projections hide depth; local intensity support does not prove neurite identity.']}
    return report,image,targets,pool_xyz,trace


def figures(output,report,image,targets,xyz,trace):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,7),constrained_layout=True)
    q=targets['curve']
    for ax,component,color in zip(axes,trace['component_trace']['components'][:3],['#38d5e5','#e469d2','#f4c833']):
        points=xyz[component['candidate_indices']]
        lo=np.maximum(np.floor(points.min(0)-25).astype(int),0)
        hi=np.minimum(np.ceil(points.max(0)+25).astype(int),np.asarray(image.shape[::-1])-1)
        slab=image[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1].max(0)
        ax.imshow(slab,cmap='gray',origin='lower',extent=[lo[0]-.5,hi[0]+.5,lo[1]-.5,hi[1]+.5],vmin=0,vmax=255)
        nearby=np.all((q>=lo)&(q<=hi),axis=1)
        ax.scatter(q[nearby,0],q[nearby,1],s=1,c='#fa5b4d',alpha=.7,label='Reference samples')
        ax.scatter(points[:,0],points[:,1],s=12,facecolors='none',edgecolors=color,linewidths=.6,label='Pool candidates')
        ax.set_aspect('equal');ax.set_xlabel('X (voxels)');ax.set_ylabel('Y (voxels)')
        ax.set_title(f"Component {component['component_rank']} | mean score {component['mean_score']:.3f}\nXY maximum projection over z={lo[2]}..{hi[2]}")
        ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Neuron 79: source image behind the main and two off-reference components\nCrop bounds follow each pool component; independent axes; projections do not prove identity')
    fig.savefig(output/'source_components.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier reports; use a fresh output directory')
    report,image,targets,xyz,trace=inspect(args.comparison)
    args.output_dir.mkdir(parents=True)
    figures(args.output_dir,report,image,targets,xyz,trace)
    write(args.output_dir/'summary.json',report)
    print('Cache supervision:',json.dumps(report['cache_supervision_counts']),flush=True)
    for row in report['components']:
        print(json.dumps({k:v for k,v in row.items() if k!='source_patches'},indent=2),flush=True)
    print('CPU gradient check:',report['existing_objective_gradient_probe'],flush=True)
    print(f'Report: {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
