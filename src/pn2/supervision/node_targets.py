"""Shared reference-only node targets for both cloud budgets; no image access."""
import json
from pathlib import Path

import numpy as np

from .curve_nodes import rectangular_curve_nodes
from pn2.evaluation.swc import parse_swc
from pn2.inputs.node_cache import digest, write_once

ROLES = ('root', 'leaf', 'junction', 'crop_boundary', 'arc_sample')


def target_arrays(target):
    runs = target.reference_runs
    lengths = [len(r) for r in runs]
    return {'xyz': target.xyz, 'radii': target.radii, 'edges': target.edges,
        'roles': np.asarray([[role in r for role in ROLES] for r in target.roles], dtype=bool).reshape(-1,len(ROLES)),
        'identities_json': np.asarray([json.dumps(key, separators=(',', ':')) for key in target.identities], dtype=str),
        'curve_points': np.concatenate(runs) if runs else np.empty((0,3),dtype=np.float64),
        'curve_offsets': np.r_[0,np.cumsum(lengths)].astype(np.int64)}


def build_target_case(reference, reference_path, boxes, expected_boxes, folder, *, step=8., capacity=512):
    """Preserve all identities/edges/runs; reference-empty means unknown."""
    role = reference['partition']
    if (role not in ('training','development')
            or reference['gradient_training_allowed'] != (role == 'training')
            or reference['coordinates'] != {'mode':'native','units':'voxel'}):
        raise ValueError('Reference role, gradient eligibility or coordinates are invalid')
    if digest(reference_path) != reference['reference_sha256']:
        raise ValueError('Reference content changed')
    morphology = parse_swc(reference_path, type_policy='opaque_integer')
    folder = Path(folder); folder.mkdir(exist_ok=False); rows = []
    expected = {b['tile_id']: b for b in expected_boxes}
    if {b['tile_id'] for b in boxes} != set(expected):
        raise ValueError('Reference audit and complete image grid differ')
    for b in boxes:
        target = rectangular_curve_nodes(morphology, b['box']['start_xyz'], b['box']['stop_xyz'], step=step, capacity=capacity)
        check = expected[b['tile_id']]
        if (len(target.xyz) != check['targets'] or b['box']['start_xyz'] != check['start_xyz']
                or b['box']['stop_xyz'] != check['stop_xyz']):
            raise ValueError('Targets/grid differ from the frozen cohort preflight')
        arrays = target_arrays(target); path = folder/f'tile{b["tile_id"]:05d}.npz'
        np.savez_compressed(path, **arrays)
        with np.load(path,allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k],v) or saved[k].dtype!=v.dtype for k,v in arrays.items()):
                raise ValueError('Target serialization round trip failed')
        rows.append({'tile_id':b['tile_id'],'box':b['box'],'packet':path.name,'packet_sha256':digest(path),
            'count':len(target.xyz),'supervision':'reference_present' if len(target.xyz) else 'unknown',
            'gradient_training_allowed':role=='training' and bool(len(target.xyz)),
            'coincident_identities':len(target.xyz)-len(np.unique(target.xyz,axis=0)),
            'role_counts':{r:sum(r in roles for roles in target.roles) for r in ROLES}})
    if digest(reference_path) != reference['reference_sha256']:
        raise ValueError('Reference changed during target construction')
    receipt = {'schema':'pn2-shared-node-targets-v1','case_id':reference['case_id'],'partition':role,
        'reference_sha256':reference['reference_sha256'],'target_step_voxel':step,
        'query_capacity':capacity,'roles':list(ROLES),'records':rows,'images_opened':0,
        'radius_status':'metadata_only_unsupervised','identity_merging':False}
    write_once(folder/'case.json',receipt)
    write_once(folder/'complete.json',{'case_json_sha256':digest(folder/'case.json')})
    return receipt
