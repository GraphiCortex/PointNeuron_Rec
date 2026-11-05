"""Bound source packets, image feature caches and training-only C1 labels."""
import json
import os
import time
from pathlib import Path
import uuid
import numpy as np
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.selection.features import static_edge_features,edge_matrix
from pn2.supervision.candidate_labels import label_candidates


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+'_'+uuid.uuid4().hex+'.npz')
    np.savez_compressed(temp,**arrays);os.replace(temp,path)


class Store:
    def __init__(self,root,output,config,data):
        self.root,self.output,self.config,self.data=Path(root),Path(output),config,data
        stage1=self.root/config['stage1_root']
        self.pools={tuple(p['key']):p for p in read(stage1/'candidate_manifest.json')['packets']}
        self.labels={tuple(p['key']):p for p in read(stage1/'label_manifest.json')['packets']}
        self.features={tuple(p['key']):p for p in read(self.output/'prepare/manifest.json')['packets']}
        self.pn={tuple(p['key']):p for p in read(self.root/config['pointnext_root']/'complete.json')['packets']}
        self.cache={};self.verified={};self.guides={};self.points={};self.cold_feature_seconds=0.
    def checked(self,spec):
        path=Path(spec['path'])
        if str(path) not in self.verified:
            if digest(path)!=spec['sha256']:raise ValueError('Changed packet: '+str(path))
            self.verified[str(path)]=spec['sha256']
        elif self.verified[str(path)]!=spec['sha256']:raise ValueError('Conflicting packet receipt')
        with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
    def get(self,key):
        key=tuple(key)
        if key not in self.cache:
            p=self.checked(self.pools[key]);s=self.features[key];a=self.checked(s)
            if s['candidate_sha256']!=self.pools[key]['sha256'] or len(p['xyz'])!=len(a['node_features']):raise ValueError('Misaligned features')
            self.cache[key]=(p,a)
        return self.cache[key]
    def training_labels(self,key):
        key=tuple(key)
        if self.data.cases[key[0]]['partition']!='training' or key not in self.labels:
            raise ValueError('Development labels prohibited')
        s=self.labels[key]
        if s['candidate_sha256']!=self.pools[key]['sha256']:raise ValueError('Changed label alignment')
        return self.checked(s)
    def pointnext(self,key):
        key=tuple(key)
        if key not in self.points:
            p,a=self.get(key);spec=self.pn[key]
            if spec['candidate_sha256']!=self.pools[key]['sha256']:raise ValueError('PointNeXt geometry differs')
            value=self.checked(spec)
            if not np.array_equal(value['candidate_xyz'],p['xyz']):raise ValueError('PointNeXt coordinate order differs')
            self.points[key]=value['candidate_features']
        return self.points[key]
    def guide(self,key):
        spec=self.features[tuple(key)]
        if key[0] not in self.guides:
            path=Path(spec['guide_path'])
            if digest(path)!=spec['guide_sha256']:raise ValueError('Changed cached Sato guide')
            self.guides[key[0]]=np.load(path,mmap_mode='r')
        return self.guides[key[0]],spec['guide_maximum']
    def edge_features(self,key,kept,pointnext=False):
        key=tuple(key);pool,a=self.get(key);edge_ids=np.flatnonzero(np.all(kept[pool['edges']],axis=1))
        spec=self.features[key];path=self.output/'edge_cache'/f"box{spec['index']:04d}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                if str(f['binding'])!=fingerprint({'config':fingerprint(self.config),'pool':self.pools[key]['sha256']}):raise ValueError('Changed edge cache binding')
                done=f['done'].copy();values=f['values'].copy()
        else:
            from pn2.selection.features import STATIC_EDGE_NAMES
            done=np.zeros(len(pool['edges']),bool);values=np.zeros((len(done),len(STATIC_EDGE_NAMES)),np.float32)
        missing=edge_ids[~done[edge_ids]]
        if len(missing):
            began=time.perf_counter()
            rawspec={'path':spec['raw_crop_path'],'sha256':spec['raw_crop_sha256']};crop=self.checked(rawspec)
            guide,maximum=self.guide(key)
            for start in range(0,len(missing),256):
                ids=missing[start:start+256]
                values[ids]=static_edge_features(pool,a['node_features'],a['tangent'],ids,crop['raw'],crop['origin_xyz'],
                    self.data.cases[key[0]]['normalization'],guide,maximum)
                done[ids]=True
            atomic_npz(path,done=done,values=values,binding=np.array(fingerprint({'config':fingerprint(self.config),'pool':self.pools[key]['sha256']})))
            self.cold_feature_seconds+=time.perf_counter()-began
            print(f'IMAGE EDGE FEATURES box{spec["index"]:04d}: +{len(missing)} cached; cold feature seconds={self.cold_feature_seconds:.1f}',flush=True)
        matrix=edge_matrix(pool,a['node_features'],values[edge_ids],kept,edge_ids,self.pointnext(key) if pointnext else None)
        return edge_ids,matrix
    def targets(self,key,kept,edge_ids,contract,references):
        key=tuple(key)
        if self.data.cases[key[0]]['partition']!='training':raise ValueError('C1 generation is training-only')
        original=self.training_labels(key)
        if contract=='C0':return original['edge_label'][edge_ids]
        if contract!='C1':raise ValueError('Only approved C0/C1 contracts')
        pool,_=self.get(key);ids=np.flatnonzero(kept);mapping=np.full(len(kept),-1,int);mapping[ids]=np.arange(len(ids))
        binding=fingerprint({'config':fingerprint(self.config),'key':list(key),'kept':ids.tolist(),
                             'edges':edge_ids.tolist(),'original_labels':self.labels[key]['sha256']})
        path=self.output/'C1_training_labels'/f'{binding}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as a:return a['edge_label'].copy()
        arrays,stats=label_candidates(pool['xyz'][ids],mapping[pool['edges'][edge_ids]],
                                      references.get(key[0])['morphology'],partition='training')
        if not np.array_equal(arrays['node_label'],original['node_label'][ids]):raise ValueError('C1 changed node semantics')
        target=arrays['edge_label'];target[original['edge_label'][edge_ids]<0]=-1
        atomic_npz(path,edge_label=target,original_edge_indices=edge_ids,retained_indices=ids,
                   binding=np.array(binding),junctions=np.array(stats['junctions_with_anchor']))
        return target
