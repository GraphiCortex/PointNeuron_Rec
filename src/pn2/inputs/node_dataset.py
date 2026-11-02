"""NumPy-only paired-budget loader with explicit image/target separation."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .node_cache import digest
from pn2.supervision.node_targets import ROLES


def checked_packet(root, spec):
    path = (Path(root)/spec['path']).resolve()
    if not path.is_relative_to((Path(root)/'tmp').resolve()):
        raise ValueError('Model data packets must stay under the workspace tmp directory')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec['sha256']:
        raise ValueError(f'Packet content changed: {path}')
    with np.load(io.BytesIO(data),allow_pickle=False) as packet:
        return {k:packet[k].copy() for k in packet.files}


class NodeBudgetDataset:
    def __init__(self, root, folder):
        self.root = Path(root).resolve(); self.folder = Path(folder).resolve()
        marker = json.loads((self.folder/'complete.json').read_text())
        if digest(self.folder/'manifest.json') != marker['manifest_sha256']:
            raise ValueError('Data manifest is incomplete or changed')
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['schema'] != 'pn2-paired-node-data-v1':
            raise ValueError('Unsupported dataset schema')
        self.cases = {}; self.boxes = {}
        for case in self.manifest['cases']:
            cid = case['case_id']
            if cid in self.cases or case['partition'] not in ('training','development'):
                raise ValueError('Duplicate, reserved or unknown case')
            self.cases[cid] = case
            for b in case['boxes']:
                key = (cid,b['tile_id']); target = b['target']
                if key in self.boxes: raise ValueError('Duplicate image box')
                expected = case['partition']=='training' and target['count']>0
                if (target['gradient_training_allowed'] != expected
                        or target['supervision'] != ('reference_present' if target['count'] else 'unknown')
                        or target['count']>512):
                    raise ValueError('Invalid target/supervision contract')
                self.boxes[key] = b
        self.training_cases = sorted(cid for cid,c in self.cases.items() if c['partition']=='training')

    def image(self, case_id, tile_id, budget):
        """Load model inputs only. Does not open the target packet."""
        case = self.cases[case_id]; b = self.boxes[(case_id,tile_id)]
        spec = b['clouds'][str(budget)]; arrays = checked_packet(self.root,spec)
        count = min(budget,b['source_voxels'])
        ids = arrays['flat_indices']; xyz = arrays['native_xyz']; local = arrays['local_xyz']; attrs = arrays['attributes']
        shape = tuple(case['dimensions_xyz'][::-1])
        start = np.asarray(b['box']['start_xyz']); stop = np.asarray(b['box']['stop_xyz']); center = (start+stop-1.)/2.
        if (count!=spec['count'] or ids.shape!=(count,) or len(np.unique(ids))!=count
                or not np.all((ids>=0)&(ids<np.prod(shape)))):
            raise ValueError('Source point counts/IDs are invalid')
        expected = np.asarray(np.unravel_index(ids,shape))[::-1].T
        if (not np.array_equal(xyz,expected) or not np.all((xyz>=start)&(xyz<stop))
                or local.dtype!=np.float32 or attrs.dtype!=np.float32 or attrs.shape!=(count,4)
                or not np.isfinite(attrs).all() or not np.array_equal(attrs[:,:3],local)
                or not np.array_equal(local,((xyz-center)/64).astype(np.float32))):
            raise ValueError('Image coordinate/feature contract failed')
        lo,hi=case['normalization']['minimum'],case['normalization']['maximum']
        intensity=np.zeros(count,dtype=np.float32) if hi==lo else ((arrays['raw_intensity'].astype(np.float64)-lo)/(hi-lo)).astype(np.float32)
        if not np.array_equal(attrs[:,3],intensity): raise ValueError('Intensity normalization changed')
        return {'metadata':{'case_id':case_id,'partition':case['partition'],'alias':case['alias'],
                    'tile_id':tile_id,'center_xyz':center.tolist(),'distinct_points':count},'packet':arrays}

    def reference(self, case_id, tile_id):
        b=self.boxes[(case_id,tile_id)]; arrays=checked_packet(self.root,b['target'])
        xyz=arrays['xyz']; roles=arrays['roles']; offsets=arrays['curve_offsets']; curves=arrays['curve_points']
        if (xyz.shape!=(b['target']['count'],3) or not np.isfinite(xyz).all()
                or roles.shape!=(len(xyz),len(ROLES)) or roles.dtype!=bool
                or offsets.ndim!=1 or not len(offsets) or offsets[0]!=0 or offsets[-1]!=len(curves)
                or np.any(np.diff(offsets)<0) or curves.shape!=(len(curves),3) or not np.isfinite(curves).all()):
            raise ValueError('Target or reference-run contract failed')
        return {'target_xyz':xyz,'target_roles':[frozenset(r for r,on in zip(ROLES,row) if on) for row in roles],
                'reference_runs':[curves[a:z] for a,z in zip(offsets[:-1],offsets[1:])],
                'target_packet':arrays,'supervision':b['target']['supervision']}

    def training(self, case_id, tile_id, budget):
        case=self.cases[case_id]; b=self.boxes[(case_id,tile_id)]
        if case['partition']!='training' or not b['target']['gradient_training_allowed']:
            raise ValueError('Development or unknown boxes must not enter gradient training')
        return {**self.image(case_id,tile_id,budget),**self.reference(case_id,tile_id)}

    def schedule(self):
        packet=checked_packet(self.root,self.manifest['schedule'])
        order=packet['epoch_case_tile']; case_ids=packet['case_ids'].tolist()
        epochs=self.manifest['epochs']
        eligible={(case_ids.index(cid),tile) for cid,tile in self.boxes
                  if cid in case_ids and self.boxes[(cid,tile)]['target']['gradient_training_allowed']}
        if (case_ids!=self.training_cases or order.shape!=(len(eligible)*epochs,3)
                or not np.issubdtype(order.dtype,np.integer)):
            raise ValueError('Exposure schedule identities/counts differ')
        for epoch in range(epochs):
            rows=order[order[:,0]==epoch,1:]
            if len(rows)!=len(eligible) or set(map(tuple,rows))!=eligible:
                raise ValueError('Unknown/development/duplicate/missing box in exposure schedule')
        return case_ids,order
