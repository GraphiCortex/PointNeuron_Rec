"""Read-only neuron-79 graph/path and source-image audit, with fixed outputs.

References define evaluation regions only. No geodesic recomputation, learned
inference, training, parameter changes or output reconstruction is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import maximum_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT/'src'))
sys.path.insert(0,str(REPO_ROOT))
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_volume
from pointneuron.data.training_cache import volume_data_array
from scripts.evaluate_paper_style_swc import resample_swc_points, deterministic_downsample
from scripts.initialize_geodesic_graph import filter_proposals
from scripts.pointneuron2.audit_full_volume_selection import reference_targets, distances
from scripts.pointneuron2.compare_hybrid_global import source_indices
from scripts.pointneuron2.sampling_ablation import digest, read, write
from scripts.pointneuron2.summarize_hybrid_swc import inspect


def fragment_labels(tree):
    """Root ID for every reference sample, in the evaluator's exact point order."""
    nodes = {n.node_id:n for n in tree.nodes}
    roots = {}
    for node in tree.nodes:
        chain=[]; current=node
        while current.node_id not in roots and current.parent_id in nodes:
            if current.node_id in chain:
                raise ValueError('Reference parent cycle')
            chain.append(current.node_id); current=nodes[current.parent_id]
        root=roots.get(current.node_id,current.node_id)
        roots[current.node_id]=root
        for nid in chain:
            roots[nid]=root
    labels=[]
    for n in tree.nodes:
        if n.parent_id not in nodes:
            count=1
        else:
            p=nodes[n.parent_id]
            delta=np.asarray([n.x,n.y,n.z],dtype=np.float32)-np.asarray([p.x,p.y,p.z],dtype=np.float32)
            count=max(1,int(np.ceil(float(np.linalg.norm(delta)))))+1
        labels.extend([roots[n.node_id]]*count)
    return np.asarray(labels,dtype=np.int64)


def voxel_values(volume, xyz):
    ijk=np.rint(xyz).astype(np.int64)
    shape=np.array(volume.shape[::-1])
    valid=np.all((ijk>=0)&(ijk<shape),axis=1)
    values=np.full(len(xyz),np.nan)
    q=ijk[valid]
    values[valid]=volume[q[:,2],q[:,1],q[:,0]]
    return values


def signal_record(mask, exact, nearby):
    n=int(mask.sum())
    if not n:
        return {'samples':0}
    a,b=exact[mask],nearby[mask]
    return {'samples':n,'median_rounded_voxel_intensity':float(np.nanmedian(a)),
        'rounded_voxel_above_5_fraction':float(np.mean(a>5)),
        'rounded_voxel_above_15_fraction':float(np.mean(a>15)),
        'within_cube_radius2_above_5_fraction':float(np.mean(b>5)),
        'within_cube_radius2_above_15_fraction':float(np.mean(b>15)),
        'within_cube_radius2_above_51_fraction':float(np.mean(b>51))}


def figures(output, image, targets, labels, fragment_rows, arm_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scripts.pointneuron2.summarize_graph_cutoffs import segments
    # Post-evaluation diagnostic examples: largest counts in each category.
    examples=[]
    for field,label in [('gained','Most newly supported samples'),('lost','Most newly missed samples'),
                        ('hybrid_represented_but_missed','Most represented but untraced samples')]:
        winner=max(fragment_rows,key=lambda r:(r[field],-r['root_id']))
        examples.append((winner,label))
    fig,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    allref=targets['curve']
    for rr,(winner,label) in enumerate(examples):
        fragment=allref[labels==winner['root_id']]
        lo=np.floor(fragment.min(0)-20).astype(int); hi=np.ceil(fragment.max(0)+20).astype(int)
        lo=np.maximum(lo,0); hi=np.minimum(hi,np.array(image.shape[::-1])-1)
        slab=image[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1].max(0)
        refmask=np.all((allref>=lo)&(allref<=hi),axis=1)
        for col,arm in enumerate(arm_data):
            ax=axes[rr,col]
            ax.imshow(slab,cmap='gray',origin='lower',extent=[lo[0]-.5,hi[0]+.5,lo[1]-.5,hi[1]+.5],vmin=0,vmax=255)
            ax.scatter(allref[refmask,0],allref[refmask,1],s=2,c='#ef6554',alpha=.6,label='Reference')
            seg=segments(arm['tree'])
            mask=np.all((seg.mean(1)>=lo)&(seg.mean(1)<=hi),axis=1)
            ax.add_collection(LineCollection(seg[mask,...,:2],colors='#36cced',linewidths=1.,alpha=.8))
            pool=arm['pool_xyz']; pm=np.all((pool>=lo)&(pool<=hi),axis=1)
            anchors=arm['anchor_xyz']; am=np.all((anchors>=lo)&(anchors<=hi),axis=1)
            ax.scatter(pool[pm,0],pool[pm,1],s=6,c='#f6d24b',alpha=.45,label='Pool nodes')
            ax.scatter(anchors[am,0],anchors[am,1],s=30,facecolors='none',edgecolors='#78ed5d',label='Graph anchors')
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
            ax.set_xlabel('X (voxels)');ax.set_ylabel('Y (voxels)')
            ax.set_title(f"{label} | fragment root {winner['root_id']}\n{arm['name']} | gained {winner['gained']}, lost {winner['lost']}",fontsize=9)
    axes[0,0].legend(fontsize=7,loc='best')
    fig.suptitle('Neuron 79: saved graph anchors and traces on source-image XY projections\nCyan: traced SWC; red: reference; yellow: input pool; green rings: graph anchors\nExamples chosen after evaluation; projected depth overlap does not prove a correct path')
    fig.savefig(output/'image_path_examples.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh directory; preserve prior audits')
    plan=read(args.comparison)
    verified=inspect(plan)
    targets,roles=reference_targets(plan['baseline']['reference']);q=targets['curve']
    reference=parse_swc(plan['baseline']['reference']);labels=fragment_labels(reference)
    if len(labels)!=len(q) or len(q)>250000:
        raise ValueError('Fragment labels do not match the fixed reference target order')
    print('Saved pair verified; inspecting source identities, paths and reference fragments.',flush=True)
    with np.load(plan['results'][0]['reconstruction']['proposal_path']) as z:
        allxyz,ar,sc=(z[k].copy() for k in ['all_centers','all_radii','all_scores'])
        baseline_indices=source_indices(allxyz,ar,sc,{k:z[k] for k in ['centers','radii','scores']})
    arms=[];raw=[]
    for arm in plan['results']:
        row=arm['reconstruction']
        with np.load(row['proposal_path']) as z,np.load(row['graph_path']) as g:
            pool=filter_proposals(z['centers'],z['radii'],z['scores'],z['features'],min_score=.5,
                nms_distance=18,nms_mode='distance',iou_threshold=.1,max_nodes=128,
                selection_mode='adaptive_connected_coverage_nms')[-1]
            source=source_indices(allxyz,ar,sc,{k:z[k] for k in ['centers','radii','scores']})
            selected_source=source[g['original_indices']]
            tree=parse_swc(row['swc_path']);swc=deterministic_downsample(resample_swc_points(tree,1.),250000)
            ds={name:distances(x,q) for name,x in [('global',z['centers']),('pool',z['centers'][pool]),
                ('anchors',g['centers']),('swc',swc)]}
            missing=ds['swc']>6;represented=missing&(ds['global']<=6)
            # Descriptive strata only: sparse-anchor distance does not prove causality.
            strata={'represented_but_missed':int(represented.sum()),
                'represented_missed_pool_farther_than18':int((represented&(ds['pool']>18)).sum()),
                'represented_missed_pool_within18_anchor_farther_than18':int((represented&(ds['pool']<=18)&(ds['anchors']>18)).sum()),
                'represented_missed_anchor_within18':int((represented&(ds['pool']<=18)&(ds['anchors']<=18)).sum())}
            if sum(v for k,v in strata.items() if k!='represented_but_missed')!=strata['represented_but_missed']:
                raise ValueError('Missing-support strata do not partition the target set')
            paths=g['edge_path_points'].copy()
            path_distance=distances(q,paths)
            offsets=g['edge_path_offsets'];edges=g['edges']
            if offsets.shape!=(len(edges)+1,) or offsets[0]!=0 or offsets[-1]!=len(paths) or np.any(np.diff(offsets)<0):
                raise ValueError('Saved edge-path offsets invalid')
            edge_records=[]
            for i,(a,b) in enumerate(edges):
                s,e=int(offsets[i]),int(offsets[i+1])
                d=path_distance[s:e]
                edge_records.append({'edge_index':i,'graph_nodes':[int(a),int(b)],'path_voxels':e-s,
                    'path_voxel_fraction_farther_than6_from_reference':float(np.mean(d>6)) if len(d) else None,
                    'reachable':bool(g['edge_geodesic_reachable'][i]),
                    'euclidean_length':float(g['edge_euclidean_distance'][i]),
                    'saved_geodesic_cost':float(g['edge_tree_distance'][i]) if np.isfinite(g['edge_tree_distance'][i]) else None})
            anchor_roots=labels[np.asarray(__import__('scipy').spatial.cKDTree(q).query(g['centers'])[1])]
            arms.append({'name':arm['name'],'missing_reference_samples':int(missing.sum()),'missing_strata':strata,
                'selected_anchors':128,'anchor_source_indices':selected_source.tolist(),
                'anchors_from_original_score_control':int(np.isin(selected_source,baseline_indices).sum()),
                'anchors_from_protected2048':int(np.isin(selected_source,baseline_indices[:2048]).sum()),
                'anchors_below_score085':int((g['scores']<.85).sum()),
                'nearest_reference_fragment_count':len(np.unique(anchor_roots)),
                'edge_paths':edge_records})
            raw.append({'name':arm['name'],'tree':tree,'ds':ds,'pool_xyz':z['centers'][pool].copy(),
                'anchor_xyz':g['centers'].copy(),'anchor_roots':anchor_roots,'paths':paths})
    gained=(raw[0]['ds']['swc']>6)&(raw[1]['ds']['swc']<=6)
    lost=(raw[0]['ds']['swc']<=6)&(raw[1]['ds']['swc']>6)
    represented=(raw[1]['ds']['global']<=6)&(raw[1]['ds']['swc']>6)
    fragments=[]
    for root in np.unique(labels):
        mask=labels==root
        fragments.append({'root_id':int(root),'samples':int(mask.sum()),'gained':int((mask&gained).sum()),
            'lost':int((mask&lost).sum()),'hybrid_represented_but_missed':int((mask&represented).sum()),
            'current_support6':float(np.mean(raw[0]['ds']['swc'][mask]<=6)),
            'hybrid_support6':float(np.mean(raw[1]['ds']['swc'][mask]<=6)),
            'current_nearest_assigned_anchors':int((raw[0]['anchor_roots']==root).sum()),
            'hybrid_nearest_assigned_anchors':int((raw[1]['anchor_roots']==root).sum())})
    print('Decoding the unchanged image once on CPU for local signal evidence; no geodesic rerun.',flush=True)
    volume=read_volume(plan['baseline']['volume']);w,h,z,c=volume.dimensions
    if c!=1:
        raise ValueError('This audit requires the frozen single-channel neuron-79 image')
    image=volume_data_array(volume).reshape(z,h,w)
    exact=voxel_values(image,q)
    localmax=maximum_filter(image,size=5,mode='constant',cval=0)
    nearby=voxel_values(localmax,q)
    if np.isnan(exact).any() or np.isnan(nearby).any():
        raise ValueError('Reference samples outside image')
    image_stats={name:signal_record(mask,exact,nearby) for name,mask in [
        ('all_reference',np.ones(len(q),dtype=bool)),('newly_supported',gained),('newly_missed',lost),
        ('hybrid_globally_represented_but_untraced',represented),('hybrid_all_missing',raw[1]['ds']['swc']>6)]}
    for arm,data in zip(arms,raw):
        values=voxel_values(image,data['paths'])
        arm['path_image_signal']={'sampled_path_voxels':len(values),'outside_image':int(np.isnan(values).sum()),
            'rounded_voxel_above15_fraction':float(np.mean(values>15)),'median_intensity':float(np.nanmedian(values))}
    report={'status':'completed','scope':'fixed neuron-79 saved-output and image audit; no selection/tracing changes',
        'comparison_path':str(args.comparison.resolve()),'comparison_sha256':digest(args.comparison),
        'audit_source_sha256':digest(Path(__file__)),'verified_summary':verified,
        'reference_roots':roles['reference_roots'],'arms':arms,'fragment_records':fragments,
        'shared_graph_anchor_source_rows':len(set(arms[0]['anchor_source_indices'])&set(arms[1]['anchor_source_indices'])),
        'image_dimensions_xyzc':list(volume.dimensions),'image_maximum':int(image.max()),
        'image_sha256':plan['baseline']['volume_sha256'],'image_signal':image_stats,
        'limitations':['Local intensity and cube-radius2 maxima do not establish neurite identity or connectivity.',
            'Missing-support distance strata are descriptive; sparse anchors can trace distant reference geometry.',
            'Saved final graphs do not retain the full pool geodesic matrix; component-budget allocation is not replayed.',
            'Reference fragments and nearest-fragment anchor assignments are evaluation diagnostics, not biological branches.']}
    args.output_dir.mkdir(parents=True)
    figures(args.output_dir,image,targets,labels,fragments,raw)
    write(args.output_dir/'summary.json',report)
    for arm in arms:
        print(arm['name'],json.dumps({k:v for k,v in arm.items() if k not in ['anchor_source_indices','edge_paths']}),flush=True)
    print('Shared graph anchors:',report['shared_graph_anchor_source_rows'],flush=True)
    print('Image evidence:',json.dumps(image_stats),flush=True)
    print(f'Report: {args.output_dir.resolve()}',flush=True)


if __name__=='__main__':
    main()
