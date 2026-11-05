"""Reference-assisted editing diagnostic. These functions are NEVER inference."""
import numpy as np
from scipy.spatial import cKDTree
from .post_stage2 import graph,subset,reference_index
from .selection_oracle import edge_agreement
from .vote_scaffold import minimum_spanning_forest


def make_forests(classical,pool,reference,low,high):
    """Prune-only uses exact classical positions/edge inventory.

    Augmentation maps pool classical representatives to exact original vertices
    within1,adds nonclassical Sato/learned vertices and existing pool edges touching
    them. Never add a classical-classical shortcut. Every added component must
    attach to one classical component;no additions join two rooted components.
    """
    index=reference_index(reference)
    agreement,length=edge_agreement(classical.xyz,classical.edges,low,high,index)
    eligible=(agreement>=.5)&(length>0)
    edges=classical.edges[eligible];cost=length[eligible]*(1+4*(1-agreement[eligible]))
    forest,_=minimum_spanning_forest(classical.xyz,edges,cost)
    a=graph(classical.xyz,forest)
    mapping=np.full(len(pool['xyz']),-1,int);classical_pool=np.flatnonzero(pool['source_flags']&1)
    if len(classical.xyz) and len(classical_pool):
        d,ids=cKDTree(classical.xyz).query(pool['xyz'][classical_pool]);ok=d<=1.+1e-9
        mapping[classical_pool[ok]]=ids[ok]
    novel=np.flatnonzero((pool['source_flags']&6)>0)
    novel=novel[(pool['source_flags'][novel]&1)==0]
    mapping[novel]=len(classical.xyz)+np.arange(len(novel));xyz=np.concatenate((classical.xyz,pool['xyz'][novel]))
    edges=mapping[pool['edges']];valid=np.all(edges>=0,axis=1)&np.any(edges>=len(classical.xyz),axis=1)&(edges[:,0]!=edges[:,1])
    original_pool_edges=np.flatnonzero(valid);edges=edges[valid]
    overlap,length=edge_agreement(xyz,edges,low,high,index);keep=(overlap>=.5)&(length>0)
    edges=edges[keep];original_pool_edges=original_pool_edges[keep];cost=length[keep]*(1+4*(1-overlap[keep]))
    parent=np.arange(len(xyz));rooted=np.arange(len(xyz))<len(classical.xyz)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def join(u,v):
        u,v=find(u),find(v)
        if u==v:return False
        parent[v]=u;rooted[u]|=rooted[v];return True
    for u,v in forest:join(int(u),int(v))
    extra=[];kept_source=[]
    for i in np.lexsort((original_pool_edges,cost)):
        u,v=map(int,edges[i]);ru,rv=find(u),find(v)
        if ru==rv or (rooted[ru] and rooted[rv]):continue
        join(u,v);extra.append((u,v));kept_source.append(int(original_pool_edges[i]))
    extra=np.asarray(extra,int).reshape(-1,2);all_edges=np.concatenate((forest,extra))
    attached=np.array([rooted[find(int(u))] for u,v in all_edges],bool)
    b=graph(xyz,all_edges[attached])
    return subset(a,np.ones(len(a.edges),bool)),subset(b,np.ones(len(b.edges),bool)),{
        'original_classical_vertices':len(classical.xyz),'original_classical_edges':len(classical.edges),
        'prune_only_forest_edges':len(forest),'additional_pool_edge_indices_before_unanchored_drop':kept_source,
        'added_pool_vertex_indices':novel.tolist(),'pool_classical_to_original_mapping':mapping[classical_pool].tolist(),
        'classical_geometry_moved':False,'new_classical_classical_edges':False,
        'edges_reference_overlap_2_min':.5,'rooted_component_merges_allowed':False,
        'source':'existing classical and Stage1 candidate geometry only;reference scores are diagnostic-only'}
