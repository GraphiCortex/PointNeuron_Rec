"""Training-label geometry diagnostics; no changes to the frozen label generator."""
import numpy as np
from scipy.spatial import cKDTree
from .candidate_pool import clipped_edges
from .geometry import _sample_arclength,_CurveIndex
from .bounded_agreement import bounded_distances


def memberships(xyz,node,anchors,index):
    result={}
    for i in np.flatnonzero(node==1):
        status,q,_=index.project(xyz[i])
        if status!=1:raise ValueError('Saved positive projection differs')
        if q[1]=='isolated':result[i]={};continue
        ei,t=q[2],q[3];ci,a,b=index.edge_chain[ei]
        vertex=int(index.edges[ei,int(t)]) if t in (0.,1.) else None
        result[i]=dict(index.vertex_chains[vertex]) if vertex in index.vertex_chains else {ci:a+t*(b-a)}
        if anchors[i]>=0:result[i]=dict(index.vertex_chains.get(int(anchors[i]),[]))
    return result


def skip_types(xyz,edges,node,member):
    """0 same-chain skip within1;1 same-chain other;2 cross-chain.

    A skip witness must project strictly between endpoints on a shared chain
    AND lie within1 of the segment interior. Geometric witnesses from any
    positive branch are counted separately, avoiding a forced false dichotomy.
    """
    xyz=np.asarray(xyz,float);edges=np.asarray(edges,int).reshape(-1,2)
    positive=np.flatnonzero(node==1);tree=cKDTree(xyz[positive]) if len(positive) else None
    kind=np.full(len(edges),1,np.int8);geometric=np.zeros(len(edges),bool);witness=np.full(len(edges),-1,int)
    for k,(u,v) in enumerate(edges):
        shared=set(member[u])&set(member[v])
        if not shared:kind[k]=2
        a,b=xyz[u],xyz[v];delta=b-a;l2=float(delta@delta)
        if l2==0 or tree is None:continue
        ids=positive[tree.query_ball_point((a+b)/2,np.sqrt(l2)/2+1.+1e-10)]
        ids=ids[(ids!=u)&(ids!=v)]
        if not len(ids):continue
        t=(xyz[ids]-a)@delta/l2;d=np.linalg.norm(xyz[ids]-(a+t[:,None]*delta),axis=1)
        ids=ids[(t>1e-9)&(t<1-1e-9)&(d<=1.)]
        geometric[k]=bool(len(ids))
        for j in sorted(ids):
            if any(ci in member[j] and min(member[u][ci],member[v][ci])+1e-9<member[j][ci]<max(member[u][ci],member[v][ci])-1e-9 for ci in shared):
                kind[k]=0;witness[k]=j;break
    return kind,geometric,witness


def overlap_summary(overlap,length):
    p=np.asarray(overlap,float);length=np.asarray(length,float);valid=length>0;p=p[valid];l=length[valid]
    bounds=[0.,.25,.5,.75,.9,.99,1.0000000001]
    return {'edges':len(length),'in_box_edges':int(valid.sum()),'no_in_box_length':int((~valid).sum()),
            'in_box_length':float(l.sum()),'supported_length':float((p*l).sum()),
            'length_weighted_overlap':float((p*l).sum()/l.sum()) if l.sum() else None,
            'edge_mean_overlap':float(p.mean()) if len(p) else None,
            'quantiles':{str(q):float(np.quantile(p,q)) for q in [0,.1,.25,.5,.75,.9,1]} if len(p) else {},
            'bins':{f'[{a},{min(b,1.)}{"]" if b>1 else ")"}':int(((p>=a)&(p<b)).sum()) for a,b in zip(bounds[:-1],bounds[1:])},
            'overlap_ge_half':int((p>=.5).sum()),'overlap_lt_half':int((p<.5).sum()),
            'overlap_ge_90pct':int((p>=.9).sum()),'overlap_ge_99pct':int((p>=.99).sum())}


def metric_score(reference,predicted,tolerances=(.5,1.,2.),step=.1):
    """Same midpoint/projection convention in explicitly transformed coordinates."""
    reference=np.asarray(reference,float).reshape(-1,2,3);predicted=np.asarray(predicted,float).reshape(-1,2,3)
    reference=reference[np.linalg.norm(reference[:,1]-reference[:,0],axis=1)>0]
    predicted=predicted[np.linalg.norm(predicted[:,1]-predicted[:,0],axis=1)>0]
    rp,rw=_sample_arclength(reference,step);pp,pw=_sample_arclength(predicted,step)
    ri=_CurveIndex(reference,4.) if len(reference) else None;pi=_CurveIndex(predicted,4.) if len(predicted) else None
    rd=bounded_distances(rp,pi);pd=bounded_distances(pp,ri);rl=float(rw.sum());pl=float(pw.sum());scores={}
    for t in tolerances:
        rc=float(rw[rd<=t].sum());pc=float(pw[pd<=t].sum());p=pc/pl if pl else 0.;r=rc/rl if rl else 0.
        scores[str(t)]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,
                        'supported_reference_length':rc,'supported_prediction_length':pc}
    return {'reference_length':rl,'predicted_length':pl,'length_ratio':pl/rl if rl else None,'tolerances':scores}


def physical_edge_overlap(xyz,edges,low,high,reference,spacing):
    ids,segments=clipped_edges(xyz,edges,low,high);segments=segments*np.asarray(spacing)
    index=_CurveIndex(np.asarray(reference)*spacing,4.) if len(reference) else None
    scores={str(t):np.zeros(len(edges)) for t in (.5,1.,2.)};length=np.zeros(len(edges))
    if not len(ids):return scores,length
    points,weights=_sample_arclength(segments,.1)
    counts=np.array([max(1,int(np.ceil(float(np.linalg.norm(b-a))/.1))) for a,b in segments]);offset=np.r_[0,np.cumsum(counts)[:-1]]
    distance=bounded_distances(points,index);sums=np.add.reduceat(weights,offset);length[ids]=sums
    for t in (.5,1.,2.):scores[str(t)][ids]=np.add.reduceat(weights*(distance<=t),offset)/sums
    return scores,length
