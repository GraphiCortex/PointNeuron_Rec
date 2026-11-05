"""Image-derived candidate geometry only. No references or labels in this API."""
import numpy as np
from scipy.spatial import cKDTree

NODE_SOURCES = {'classical': 1, 'sato': 2, 'learned': 4}
EDGE_SOURCES = {'classical': 1, 'sato': 2, 'learned': 4, 'knn6': 8, 'knn10': 16}


def clipped_edges(xyz, edges, low, high):
    """Return original edge IDs and positive-length clipped geometry."""
    xyz=np.asarray(xyz,float).reshape(-1,3);edges=np.asarray(edges,int).reshape(-1,2)
    a=xyz[edges[:,0]];delta=xyz[edges[:,1]]-a
    lo=np.zeros(len(edges));hi=np.ones(len(edges));valid=np.ones(len(edges),bool)
    for axis in range(3):
        moving=delta[:,axis]!=0
        valid &= moving | ((a[:,axis]>=low[axis]) & (a[:,axis]<=high[axis]))
        u=np.full(len(edges),-np.inf);v=np.full(len(edges),np.inf)
        first=(low[axis]-a[moving,axis])/delta[moving,axis]
        last=(high[axis]-a[moving,axis])/delta[moving,axis]
        u[moving]=np.minimum(first,last);v[moving]=np.maximum(first,last)
        lo=np.maximum(lo,u);hi=np.minimum(hi,v)
    valid &= hi>lo
    ids=np.flatnonzero(valid)
    segments=np.stack((a[ids]+lo[ids,None]*delta[ids],a[ids]+hi[ids,None]*delta[ids]),axis=1)
    positive=np.linalg.norm(segments[:,1]-segments[:,0],axis=1)>0
    return ids[positive],segments[positive]


def crop_classical(xyz,edges,low,high):
    """Keep in-box vertices plus original endpoints of intersecting edges (halo)."""
    xyz=np.asarray(xyz,float);edges=np.asarray(edges,int).reshape(-1,2)
    edge_ids,_=clipped_edges(xyz,edges,np.asarray(low),np.asarray(high))
    inside=np.all((xyz>=low)&(xyz<=high),axis=1)
    keep=inside.copy();keep[edges[edge_ids].ravel()]=True
    ids=np.flatnonzero(keep);mapping=np.full(len(xyz),-1,int);mapping[ids]=np.arange(len(ids))
    return xyz[ids],mapping[edges[edge_ids]],ids


def merge(points,flags,radius=1.):
    """Greedy fixed source priority/XYZ order. Never transitive; keep real centers."""
    p=np.asarray(points,float).reshape(-1,3);flags=np.asarray(flags,np.uint8)
    if flags.shape!=(len(p),) or not np.isfinite(p).all() or radius!=1.:
        raise ValueError('Invalid candidate merge input/constants')
    tree=cKDTree(p);order=np.lexsort((np.arange(len(p)),p[:,2],p[:,1],p[:,0],flags))
    assignment=np.full(len(p),-1,int);representatives=[];combined=[];counts=[]
    for i in order:
        if assignment[i]>=0:continue
        members=np.asarray(tree.query_ball_point(p[i],1.),int)
        members=members[assignment[members]<0]
        assignment[members]=len(representatives);representatives.append(int(i))
        combined.append(int(np.bitwise_or.reduce(flags[members])))
        counts.append([int(np.count_nonzero(flags[members]&v)) for v in (1,2,4)])
    ids=np.asarray(representatives,int)
    return {'xyz':p[ids],'source_flags':np.asarray(combined,np.uint8),'source_counts':np.asarray(counts,int).reshape(-1,3),
            'representative_input_indices':ids,'input_to_merged':assignment}


def knn(points,k):
    """Exact KD-tree acceleration with deterministic distance/index tie handling."""
    p=np.asarray(points,float).reshape(-1,3);n=len(p);k=min(k,n-1)
    if k<=0:return np.empty((0,2),np.int64)
    tree=cKDTree(p);distance,_=tree.query(p,k=k+1)
    radius=distance[:,-1]
    margin=128*np.finfo(float).eps*(np.linalg.norm(p,axis=1)+radius+1)
    candidates=tree.query_ball_point(p,np.nextafter(radius+margin,np.inf))
    pairs=set()
    for i,raw in enumerate(candidates):
        ids=np.asarray(raw,int);ids=ids[ids!=i]
        d2=((p[ids]-p[i])**2).sum(1)
        for j in ids[np.lexsort((ids,d2))[:k]]:pairs.add((min(i,int(j)),max(i,int(j))))
    return np.asarray(sorted(pairs),np.int64).reshape(-1,2)


def pool(sources):
    """sources ordered by fixed source identity, each with XYZ and inherited edges."""
    parts=[];flags=[];inherited=[];offset=0
    for name in ('classical','sato','learned'):
        if name not in sources:continue
        xyz,edges=sources[name]
        xyz=np.asarray(xyz,float).reshape(-1,3);edges=np.asarray(edges,int).reshape(-1,2)
        parts.append(xyz);flags.extend([NODE_SOURCES[name]]*len(xyz))
        inherited.append((edges+offset,EDGE_SOURCES[name]));offset+=len(xyz)
    raw=np.concatenate(parts) if parts else np.empty((0,3))
    result=merge(raw,flags);mapping=result['input_to_merged'];pairs={};dropped=0
    for edges,flag in inherited:
        for u,v in mapping[edges]:
            if u==v:dropped+=1;continue
            pair=(min(int(u),int(v)),max(int(u),int(v)))
            pairs[pair]=pairs.get(pair,0)|flag
    for k in (6,10):
        for pair in map(tuple,knn(result['xyz'],k)):
            pairs[pair]=pairs.get(pair,0)|EDGE_SOURCES[f'knn{k}']
    ordered=sorted(pairs)
    result.update(edges=np.asarray(ordered,np.int64).reshape(-1,2),
                  edge_flags=np.asarray([pairs[e] for e in ordered],np.uint8),
                  input_xyz=raw,input_source_flags=np.asarray(flags,np.uint8),
                  inherited_self_edges_dropped=dropped)
    return result
