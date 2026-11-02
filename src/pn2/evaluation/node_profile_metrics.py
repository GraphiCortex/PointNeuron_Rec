"""Supplementary apparent-stratum recall/curve support; no precision labels for extras."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pn2.inputs.node_cache import digest
from pn2.inputs.node_dataset import checked_packet
from .node_profiles import classify,STRATA
from .node_proposals import one_to_one


class ProfileCache:
    def __init__(self,root,folder,data):
        self.root=Path(root);self.folder=Path(folder)
        marker=json.loads((self.folder/'complete.json').read_text())
        if digest(self.folder/'manifest.json')!=marker['manifest_sha256']:raise ValueError('Profile manifest changed')
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if (self.manifest['status']!='completed' or self.manifest['cutoff_partition']!='training_only'
                or self.manifest['data_manifest_sha256']!=digest(data.folder/'manifest.json')):
            raise ValueError('Profiles do not belong to this data or training-only definition')
        for name,sha in self.manifest['source_sha256'].items():
            if digest(self.root/name)!=sha:raise ValueError('Profile construction source changed')
        self.records={(r['case_id'],r['tile_id']):r for r in self.manifest['records']}
        if len(self.records)!=len(self.manifest['records']) or set(self.records)!=set(data.boxes):
            raise ValueError('Profiles must cover the complete shared image grid')

    def load(self,cid,tile):
        row=self.records[(cid,tile)]
        return checked_packet(self.root,{'path':(self.folder/row['packet']).relative_to(self.root).as_posix(),'sha256':row['sha256']})


def score_profiles(centers,scores,reference,profiles,cutoffs):
    accepted=np.asarray(centers)[np.asarray(scores)>=.5];target=reference['target_xyz']
    node_labels=classify(profiles['node_width'],profiles['node_width_valid'],profiles['node_contrast'],profiles['node_contrast_valid'],cutoffs)
    window_labels=classify(profiles['window_width'],profiles['window_width_valid'],profiles['window_contrast'],profiles['window_contrast_valid'],cutoffs)
    points=profiles['curve_xyz'];weights=profiles['curve_weights'];indices=profiles['curve_window_index']
    if (len(node_labels)!=len(target) or len(points)!=len(weights) or len(indices)!=len(points)
            or (len(indices) and (indices.min()<0 or indices.max()>=len(window_labels)))):
        raise ValueError('Profile nodes/curve samples disagree with shared targets')
    labels=window_labels[indices]
    distances=cKDTree(accepted).query(points)[0] if len(accepted) else np.full(len(points),np.inf)
    baseline=cKDTree(target).query(points)[0] if len(target) else np.full(len(points),np.inf)
    result={}
    for tolerance in (2.,6.):
        _,matched=one_to_one(target,accepted,tolerance);rows={}
        for index,name in enumerate(STRATA):
            nodes=node_labels==index;curve=labels==index
            rows[name]={'reference_nodes':int(nodes.sum()),'matched_nodes':int((nodes&matched).sum()),
                'curve_length':float(weights[curve].sum()),
                'covered_curve_length':float(weights[curve&(distances<=tolerance)].sum()),
                'target_grid_covered_curve_length':float(weights[curve&(baseline<=tolerance)].sum())}
        result[str(tolerance)]=rows
    return result


def summarize_profiles(data,rows):
    neurons=[]
    for cid in sorted({r['case_id'] for r in rows}):
        group=[r for r in rows if r['case_id']==cid];result={}
        for tolerance in ('2.0','6.0'):
            result[tolerance]={}
            for name in STRATA:
                values=[r['profiles'][tolerance][name] for r in group]
                sums={k:sum(v[k] for v in values) for k in values[0]}
                sums['node_recall']=sums['matched_nodes']/sums['reference_nodes'] if sums['reference_nodes'] else None
                sums['curve_support']=sums['covered_curve_length']/sums['curve_length'] if sums['curve_length'] else None
                sums['target_grid_curve_support']=sums['target_grid_covered_curve_length']/sums['curve_length'] if sums['curve_length'] else None
                result[tolerance][name]=sums
        neurons.append({'case_id':cid,'partition':data.cases[cid]['partition'],'tolerances':result})
    return {'neurons':neurons,'interpretation':'Apparent native-voxel profile strata; unresolved included; not calibrated biological caliber',
        'precision':'Global reference-agreement precision is reported by the core scorer; unmatched proposals have no known reference branch class',
        'matching_limitation':'Stratum recall uses one geometric matching and may be ambiguous at coincident/close identities'}
