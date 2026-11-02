"""Complete-grid proposal reference agreement; no topology or biological labels."""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .node_proposals import one_to_one
from .cloud_support import polyline_samples, longest_run
from .geometry import _CurveIndex

TOLERANCES = (2., 6.)
STRUCTURAL_ROLES = ('root','leaf','junction','crop_boundary')


def ratios(matched,predicted,reference):
    return {'matched':int(matched),'predicted':int(predicted),'reference':int(reference),
        'precision':float(matched/predicted) if predicted else 0.,
        'recall':float(matched/reference) if reference else None,
        'f1':float(2*matched/(predicted+reference)) if reference else None}


def score_box(case_id,tile_id,centers,scores,reference,*,cutoff=.5):
    """Score native-voxel centers at one fixed cutoff, with an explicit target-grid baseline."""
    centers=np.asarray(centers,dtype=float);scores=np.asarray(scores,dtype=float)
    target=np.asarray(reference['target_xyz'],dtype=float);roles=reference['target_roles']
    if (centers.ndim!=2 or centers.shape[1]!=3 or scores.shape!=(len(centers),)
            or not np.isfinite(centers).all() or not np.isfinite(scores).all()
            or np.any((scores<0)|(scores>1)) or not 0<=cutoff<=1 or len(roles)!=len(target)):
        raise ValueError('Invalid proposal geometry/confidence/reference roles')
    accepted=centers[scores>=cutoff];runs=reference['reference_runs']
    samples=[polyline_samples(run,.5) for run in runs]
    distances=[cKDTree(accepted).query(p)[0] if len(accepted) else np.full(len(p),np.inf) for p,_ in samples]
    baseline=[cKDTree(target).query(p)[0] if len(target) else np.full(len(p),np.inf) for p,_ in samples]
    length=sum(float(w.sum()) for _,w in samples)
    segments=[np.stack((r[:-1],r[1:]),axis=1) for r in runs if len(r)>1]
    segments=np.concatenate(segments) if segments else np.empty((0,2,3))
    segments=segments[np.linalg.norm(segments[:,1]-segments[:,0],axis=1)>0]
    point_distance=np.full(len(accepted),np.inf)
    if len(accepted) and len(target):point_distance=np.minimum(point_distance,cKDTree(target).query(accepted)[0])
    if len(accepted) and len(segments):point_distance=np.minimum(point_distance,_CurveIndex(segments,4.).distances(accepted))
    result={'case_id':case_id,'tile_id':tile_id,'cutoff':cutoff,'query_count':len(centers),
        'accepted_count':len(accepted),'target_count':len(target),'reference_present':bool(len(target)),
        'unknown_box_accepted':len(accepted) if not len(target) else 0,
        'duplicate_accepted_pairs_within_two_voxels':len(cKDTree(accepted).query_pairs(2.)) if len(accepted)>1 else 0,
        'annotation_status':'reference_agreement_not_certified_foreground',
        'units':'native_voxels','tolerances':{}}
    for tolerance in TOLERANCES:
        nodes,mask=one_to_one(target,accepted,tolerance);all_nodes,_=one_to_one(target,centers,tolerance)
        covered=sum(float(w[d<=tolerance].sum()) for d,(_,w) in zip(distances,samples))
        base_covered=sum(float(w[d<=tolerance].sum()) for d,(_,w) in zip(baseline,samples))
        result['tolerances'][str(tolerance)]={'nodes':nodes,'all_query_matched':all_nodes['matched'],
            'structural':{role:{'reference':sum(role in r for r in roles),
                'matched':sum(bool(m) and role in r for m,r in zip(mask,roles))} for role in STRUCTURAL_ROLES},
            'curve_length':length,'covered_curve_length':covered,'target_grid_covered_curve_length':base_covered,
            'curve_to_accepted_node_support':covered/length if length else None,
            'curve_to_target_grid_support':base_covered/length if length else None,
            'longest_unsupported_cropped_run':max((longest_run(d>tolerance,w) for d,(_,w) in zip(distances,samples)),default=None),
            'accepted_near_reference_geometry':int((point_distance<=tolerance).sum()) if len(target) else None}
    return result


def summarize(dataset,rows,case_ids=None):
    """Require every intended image box, then pool per neuron before macro averages."""
    selected=set(dataset.cases if case_ids is None else case_ids)
    if not selected or not selected<=set(dataset.cases):raise ValueError('Unknown/missing evaluation cases')
    expected={key for key in dataset.boxes if key[0] in selected};observed={}
    for row in rows:
        key=(row['case_id'],row['tile_id'])
        if key in observed or key not in expected:raise ValueError('Duplicate/unexpected prediction box')
        if row['target_count']!=dataset.boxes[key]['target']['count']:
            raise ValueError('Prediction/evaluation target denominator differs from shared cache')
        observed[key]=row
    if set(observed)!=expected:raise ValueError('Evaluation must include every image box, including unknown boxes')
    by_case=defaultdict(list)
    for (cid,_),row in observed.items():by_case[cid].append(row)
    neurons=[]
    for cid in sorted(selected):
        case=dataset.cases[cid];boxes=by_case[cid]
        neuron={'case_id':cid,'partition':case['partition'],'alias':case['alias'],'source_group':case['source_group'],
            'boxes':len(boxes),'reference_present_boxes':sum(r['reference_present'] for r in boxes),
            'unknown_boxes':sum(not r['reference_present'] for r in boxes),
            'unknown_box_accepted':sum(r['unknown_box_accepted'] for r in boxes),
            'duplicate_accepted_pairs_within_two_voxels':sum(r['duplicate_accepted_pairs_within_two_voxels'] for r in boxes),
            'tolerances':{}}
        for tolerance in TOLERANCES:
            key=str(tolerance); values=[b['tolerances'][key] for b in boxes]
            counts=[v['nodes'] for v in values]
            pooled=ratios(sum(v['matched'] for v in counts),sum(v['predicted'] for v in counts),sum(v['reference'] for v in counts))
            present=[v['tolerances'][key]['nodes'] for v in boxes if v['reference_present']]
            length=sum(v['curve_length'] for v in values)
            geometry_count=sum(v['accepted_near_reference_geometry'] or 0 for v in values)
            geometry_denominator=sum(b['accepted_count'] for b in boxes if b['reference_present'])
            neuron['tolerances'][key]={'full_grid':pooled,
                'reference_present_only':ratios(sum(v['matched'] for v in present),sum(v['predicted'] for v in present),sum(v['reference'] for v in present)),
                'all_query_recall':sum(v['all_query_matched'] for v in values)/pooled['reference'] if pooled['reference'] else None,
                'curve_length':length,'curve_to_accepted_node_support':sum(v['covered_curve_length'] for v in values)/length if length else None,
                'curve_to_target_grid_support':sum(v['target_grid_covered_curve_length'] for v in values)/length if length else None,
                'accepted_point_reference_geometry_agreement_in_reference_present_boxes':geometry_count/geometry_denominator if geometry_denominator else None,
                'longest_unsupported_cropped_run':max((v['longest_unsupported_cropped_run'] for v in values if v['longest_unsupported_cropped_run'] is not None),default=None),
                'structural':{role:{'matched':sum(v['structural'][role]['matched'] for v in values),
                    'reference':sum(v['structural'][role]['reference'] for v in values)} for role in STRUCTURAL_ROLES}}
        neurons.append(neuron)
    partitions={}
    for partition in sorted({n['partition'] for n in neurons}):
        group=[n for n in neurons if n['partition']==partition];tolerances={}
        for tolerance in TOLERANCES:
            key=str(tolerance); full=[n['tolerances'][key]['full_grid'] for n in group]
            if any(v['reference']==0 for v in full):raise ValueError('A reference-free neuron cannot silently enter a recall macro')
            tolerances[key]={'macro_precision':float(np.mean([v['precision'] for v in full])),
                'macro_recall':float(np.mean([v['recall'] for v in full])),
                'macro_f1':float(np.mean([v['f1'] for v in full])),
                'minimum_neuron_f1':min(v['f1'] for v in full),
                'pooled_reference_agreement':ratios(sum(v['matched'] for v in full),sum(v['predicted'] for v in full),sum(v['reference'] for v in full))}
        partitions[partition]={'neurons':len(group),'tolerances':tolerances,
            'unknown_box_accepted':sum(n['unknown_box_accepted'] for n in group)}
    return {'partitions':partitions,'neurons':neurons,'scored_boxes':len(rows),
        'annotation_status':'reference_agreement_not_certified_foreground',
        'limitations':['Counts pool a tiled node task, not unique whole-neuron graph nodes',
            'Structural counts use one geometric matching and may be ambiguous near coincident nodes',
            'Unknown-box outputs are disclosed; unmatched outputs are not certified biological false positives',
            'Native-voxel agreement is not calibrated anatomy, connectivity or SWC reconstruction quality']}
