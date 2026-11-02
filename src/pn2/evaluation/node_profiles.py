"""Evaluation-only apparent width/contrast profiles, never model inputs or losses."""
import json

import numpy as np

from .branch_evidence import arc_windows,profile_window
from .cloud_support import polyline_samples
from pn2.supervision.curve_nodes import _chain_geometry

STRATA=('unresolved','apparent_thin_dim','apparent_thin_bright','apparent_thick_dim','apparent_thick_bright')


def weighted_cutoff(values,weights,valid,fraction=.25):
    values=np.asarray(values);weights=np.asarray(weights);valid=np.asarray(valid,dtype=bool)
    if values.shape!=weights.shape or values.shape!=valid.shape or not 0<fraction<=1:
        raise ValueError('Invalid training percentile inputs')
    v,w=values[valid],weights[valid]
    if not len(v):return None
    if not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid profile values/weights')
    order=np.argsort(v,kind='stable');v,w=v[order],w[order]
    return float(v[min(np.searchsorted(np.cumsum(w),w.sum()*fraction,side='left'),len(v)-1)])


def classify(width,width_valid,contrast,contrast_valid,cutoffs):
    labels=np.zeros(len(width),dtype=np.int8)
    if cutoffs['width_q25'] is None or cutoffs['contrast_q25'] is None:return labels
    valid=np.asarray(width_valid)&np.asarray(contrast_valid)
    thin=np.asarray(width)<=cutoffs['width_q25'];dim=np.asarray(contrast)<=cutoffs['contrast_q25']
    labels[valid&thin&dim]=1;labels[valid&thin&~dim]=2
    labels[valid&~thin&dim]=3;labels[valid&~thin&~dim]=4
    return labels


def tangent_lookup(morphology):
    _,_,chains=_chain_geometry(morphology);pairs={};incident={}
    for ids,points,_,arc in chains:
        key=(int(morphology.ids[ids[0]]),int(morphology.ids[ids[-1]]));pairs[key]=(points,arc)
        incident.setdefault(key[0],[]).append((key,0.));incident.setdefault(key[1],[]).append((key,float(arc[-1])))
    def tangent(identity):
        if identity[0]=='chain_arc':key=tuple(identity[1:3]);distance=float(identity[3])
        elif identity[0]=='node' and len(incident.get(identity[1],[]))==1:
            key,distance=incident[identity[1]][0]
        else:return None
        points,arc=pairs[key];positive=np.flatnonzero(np.diff(arc)>0)
        if not len(positive):return None
        # Same outgoing-segment convention as arc_windows at an exact bend.
        index=positive[min(np.searchsorted(arc[positive+1],distance,side='right'),len(positive)-1)]
        delta=points[index+1]-points[index]
        return delta/np.linalg.norm(delta)
    return tangent


def build_profiles(image,reference,box,normalization,tangent):
    xyz=[];weights=[];meta=[];fine=[];fine_weights=[];fine_index=[]
    start,stop=box['start_xyz'],box['stop_xyz'];span=normalization['maximum']-normalization['minimum']
    for run in reference['reference_runs']:
        windows=arc_windows(run,4.);offset=len(xyz)
        for window in windows:
            xyz.append(window['xyz']);weights.append(window['weight'])
            meta.append(profile_window(image,window['xyz'],window['tangent'],start,stop)['metadata'])
        points,w=polyline_samples(run,.5)
        if len(windows) and len(w):
            arcs=np.cumsum(w)-w/2;indices=np.minimum((arcs/windows[0]['weight']).astype(int),len(windows)-1)+offset
            fine.append(points);fine_weights.append(w);fine_index.append(indices)
    node_meta=[]
    for point,identity in zip(reference['target_xyz'],reference['target_packet']['identities_json']):
        direction=tangent(json.loads(str(identity)))
        node_meta.append(profile_window(image,point,direction,start,stop)['metadata'] if direction is not None
                         else {'apparent_width_voxel':None,'inner_peak_contrast':None,'profile_status':'ambiguous_or_zero_tangent'})
    def arrays(items,prefix):
        return {prefix+'width':np.asarray([m['apparent_width_voxel'] or 0. for m in items],dtype=float),
            prefix+'width_valid':np.asarray([m['apparent_width_voxel'] is not None for m in items],dtype=bool),
            prefix+'contrast':np.asarray([(m['inner_peak_contrast'] or 0.)/span if span else 0. for m in items],dtype=float),
            prefix+'contrast_valid':np.asarray([m['inner_peak_contrast'] is not None and span>0 for m in items],dtype=bool),
            prefix+'status':np.asarray([m['profile_status'] for m in items],dtype=str)}
    return {'window_xyz':np.asarray(xyz,dtype=float).reshape(-1,3),'window_weights':np.asarray(weights,dtype=float),
        'curve_xyz':np.concatenate(fine) if fine else np.empty((0,3)),
        'curve_weights':np.concatenate(fine_weights) if fine_weights else np.empty(0),
        'curve_window_index':np.concatenate(fine_index).astype(np.int64) if fine_index else np.empty(0,dtype=np.int64),
        **arrays(meta,'window_'),**arrays(node_meta,'node_')}
