"""Fixed CPU diagnostic geometry. No learner, inference policy or label writer."""
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d,map_coordinates
from scipy.spatial import cKDTree
from .resampled_skeleton import uniform_samples,resampled_scores
from .bounded_agreement import BoundedAgreement,bounded_distances
from .geometry import _CurveIndex,_sample_arclength
from .formulation import graph_segments
from .node_proposals import one_to_one


def graph(xyz,edges):
    return SimpleNamespace(xyz=np.asarray(xyz,float).reshape(-1,3),edges=np.asarray(edges,int).reshape(-1,2))


def reference_index(g):
    s=graph_segments(g);s=s[np.linalg.norm(s[:,1]-s[:,0],axis=1)>0]
    return _CurveIndex(s,4.) if len(s) else None


def subset(g,keep_edges,keep_isolates=False):
    edges=g.edges[np.asarray(keep_edges,bool)];mask=np.zeros(len(g.xyz),bool)
    if len(edges):mask[edges.ravel()]=True
    if keep_isolates:
        degree=np.bincount(g.edges.ravel(),minlength=len(g.xyz));mask|=degree==0
    ids=np.flatnonzero(mask);mapping=np.full(len(mask),-1,int);mapping[ids]=np.arange(len(ids))
    return graph(g.xyz[ids],mapping[edges])


def chains(g):
    """Maximal degree-two paths with edge provenance, including pure cycles."""
    adj=[[] for _ in g.xyz]
    for ei,(u,v) in enumerate(g.edges):adj[u].append((int(v),ei));adj[v].append((int(u),ei))
    degree=np.array([len(a) for a in adj]);visited=np.zeros(len(g.edges),bool);result=[]
    def walk(start,v,ei):
        ids=[int(start),v];es=[ei];visited[ei]=True;current=v;previous=ei
        while degree[current]==2 and current!=start:
            nxt,ni=next((n,e) for n,e in adj[current] if e!=previous)
            if visited[ni]:break
            visited[ni]=True;ids.append(nxt);es.append(ni);current,previous=nxt,ni
        p=g.xyz[ids];length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        result.append({'nodes':np.asarray(ids,int),'edges':np.asarray(es,int),'length':length,
                       'terminal':bool(degree[ids[0]]==1 or degree[ids[-1]]==1),
                       'spur':bool(length<6. and ((degree[ids[0]]==1 and degree[ids[-1]]>=3) or (degree[ids[-1]]==1 and degree[ids[0]]>=3)))})
    for start in np.flatnonzero(degree!=2):
        for v,ei in sorted(adj[start]):
            if not visited[ei]:walk(start,v,ei)
    for start in range(len(g.xyz)):
        for v,ei in sorted(adj[start]):
            if not visited[ei]:walk(start,v,ei)
    return result,degree


def duplicate_chains(g,paths):
    """Shorter-chain redundancy flag:80% .5-voxel samples near a longer path.

    Need distance<=2 AND absolute tangent cosine>=.9. Shared endpoint samples
    within2 Euclidean voxels are excluded. True close parallel branches remain possible.
    """
    samples=[];directions=[];owners=[]
    for i,p in enumerate(paths):
        es=g.edges[p['edges']];delta=g.xyz[es[:,1]]-g.xyz[es[:,0]]
        for (u,v),d in zip(es,delta):
            length=float(np.linalg.norm(d))
            if length==0:continue
            n=max(1,int(np.ceil(length/.5)));q=g.xyz[u]+((np.arange(n)+.5)/n)[:,None]*d
            samples.extend(q);directions.extend(np.repeat((d/length)[None,:],n,axis=0));owners.extend([i]*n)
    flagged=np.zeros(len(paths),bool)
    if not samples:return flagged
    points=np.asarray(samples);direction=np.asarray(directions);owner=np.asarray(owners);tree=cKDTree(points)
    rank=sorted(range(len(paths)),key=lambda i:(-paths[i]['length'],i));earlier=set()
    for i in rank:
        ids=np.flatnonzero(owner==i);endpoints=g.xyz[paths[i]['nodes'][[0,-1]]]
        if len(ids):ids=ids[np.min(np.linalg.norm(points[ids,None]-endpoints[None,:],axis=2),axis=1)>2.]
        near=0
        for j in ids:
            ns=np.asarray(tree.query_ball_point(points[j],2.),int)
            mask=np.array([int(owner[k]) in earlier for k in ns],bool)
            near+=bool(np.any(mask&(np.abs(direction[ns]@direction[j])>=.9)))
        if len(ids) and near/len(ids)>=.8:flagged[i]=True
        else:earlier.add(i)
    return flagged


def ablations(g):
    paths,degree=chains(g);spur=np.zeros(len(g.edges),bool);duplicate=np.zeros(len(g.edges),bool)
    flags=duplicate_chains(g,paths)
    for i,p in enumerate(paths):
        if p['spur']:spur[p['edges']]=True
        if flags[i]:duplicate[p['edges']]=True
    variants={'original':g,'minus_isolates':subset(g,np.ones(len(g.edges),bool)),
              'minus_spurs':subset(g,~spur,True),'minus_duplicate_paths':subset(g,~duplicate,True),
              'minus_isolates_and_spurs':subset(g,~spur),
              'minus_all_three':subset(g,~(spur|duplicate))}
    return variants,{'isolated_nodes':int((degree==0).sum()),'spur_chains':sum(p['spur'] for p in paths),
                     'spur_length':float(sum(p['length'] for p in paths if p['spur'])),
                     'duplicate_path_flags':int(flags.sum()),'duplicate_flagged_length':float(sum(p['length'] for i,p in enumerate(paths) if flags[i])),
                     'overlapping_categories_edges':int((spur&duplicate).sum()),
                     'interpretation':'independent ablations;F1 changes are not additive;duplicate flags may be true parallel branches'}


def scores(reference_graph,predicted):
    reference=graph_segments(reference_graph);segments=BoundedAgreement(reference).score(graph_segments(predicted))
    return {'segments':segments,'resampled':resampled_scores(uniform_samples(reference_graph),predicted),
            'length_ratio':segments['predicted_length']/segments['reference_length'] if segments['reference_length'] else None}


def intensity_ridges(raw,normalization,origin=(0,0,0),core=None):
    """Operational ridge proxy,sigma1,Hessian transverse NMS at +/-1 voxel.

    Input includes halo>=6 except true image boundaries. Integer ridge voxels,
    no reference direction or projection is used to create a ridge.
    """
    lo,hi=normalization['minimum'],normalization['maximum']
    if hi<=lo:return np.empty((0,3))
    x=np.clip((np.asarray(raw,np.float32)-lo)/(hi-lo),0,1)
    smooth=gaussian_filter(x,1.,truncate=4.,mode='reflect')
    zz,yy,xx=np.indices(x.shape,dtype=np.float32)
    h=np.empty(x.shape+(3,3),np.float32)
    for i in range(3):
        for j in range(i,3):
            order=[0,0,0];order[i]+=1;order[j]+=1
            value=gaussian_filter(x,1.,order=order,truncate=4.,mode='reflect')
            if i==j:
                # The truncated second-derivative kernel has a nonzero DC sum.
                # Remove that response so a flat tube axis is not false curvature.
                dc=float(gaussian_filter1d(np.ones(9),1.,order=2,truncate=4.,mode='reflect')[4])
                value=value-dc*smooth
            h[...,i,j]=h[...,j,i]=value
    values,vectors=np.linalg.eigh(h);valid=(values[...,1]<-1e-6)&(smooth>=.05)
    # Eigenvectors follow array ZYX axes, as does the Hessian.
    grid=np.stack((zz,yy,xx))
    for axis in (0,1):
        direction=np.moveaxis(vectors[...,axis],-1,0)
        for sign in (-1.,1.):
            neighbor=map_coordinates(smooth,grid+sign*direction,order=1,mode='reflect',prefilter=False)
            valid &= smooth>=neighbor
    if core is not None:
        mask=np.zeros_like(valid);mask[core]=True;valid&=mask
    return np.argwhere(valid)[:,::-1]+np.asarray(origin)


def weighted_offsets(distances,weights,limit=12.,thresholds=(1.,2.,3.)):
    d=np.asarray(distances,float);w=np.asarray(weights,float);total=float(w.sum());finite=np.isfinite(d)
    order=np.argsort(d);cumulative=np.cumsum(w[order])
    def quantile(q):
        value=float(d[order[min(len(order)-1,int(np.searchsorted(cumulative,q*total)))]]) if len(order) else np.inf
        return value if np.isfinite(value) else None
    return {'reference_length':total,'share_over':{str(t):float(w[d>t].sum()/total) if total and t<limit else None for t in thresholds},
            'mean_distance_lower_bound':float(np.dot(np.minimum(d,limit),w)/total) if total else None,
            'q50':quantile(.5),'q90':quantile(.9),'q95':quantile(.95),
            'no_ridge_within_limit_share':float(w[~finite].sum()/total) if total else None,'search_limit':limit,
            'censoring':'missing distances exceed search_limit;null quantiles are censored,not exact infinite distances'}


def compact_prune(g,reference_graph,reference_length,budget=1.25,anchors=None):
    """Bounded reference-assisted local search; feasible headroom,not global optimum.

    At most128 edits. Evaluate16 terminal paths per edit,ranked by unsupported
    length then length. No vertex moves or edge insertions. If over budget after
    local search,drop complete components in deterministic worst-support order.
    """
    reference_samples=uniform_samples(reference_graph);index=reference_index(reference_graph)
    current=subset(g,np.ones(len(g.edges),bool));cap=budget*reference_length;edits=0
    def objective(q):
        return one_to_one(reference_samples,uniform_samples(q),2.)[0]['f1'] or 0.
    value=objective(current)
    for iteration in range(128):
        paths,_=chains(current);candidates=[]
        for i,p in enumerate(paths):
            if not p['terminal']:continue
            seg=current.xyz[current.edges[p['edges']]];q,w=_sample_arclength(seg,.5)
            unsupported=float(w[bounded_distances(q,index)>2.].sum())
            candidates.append((-unsupported,-p['length'],i))
        length=float(np.linalg.norm(np.diff(current.xyz[current.edges],axis=1),axis=2).sum())
        best=None
        for _,__,i in sorted(candidates)[:16]:
            mask=np.ones(len(current.edges),bool);mask[paths[i]['edges']]=False;q=subset(current,mask)
            v=objective(q);key=(-v,-paths[i]['length'],i)
            if best is None or key<best[0]:best=(key,q,v)
        if best is None or (length<=cap+1e-9 and best[2]<=value+1e-12):break
        current,value=best[1],best[2];edits+=1
    # Complete-component dropping guarantees a feasible length bound,even empty.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n=len(current.xyz);a=current.edges
    adjacency=csr_matrix((np.ones(2*len(a)),(np.r_[a[:,0],a[:,1]],np.r_[a[:,1],a[:,0]])),shape=(n,n))
    count,labels=connected_components(adjacency,directed=False);parts=[]
    for i in range(count):
        mask=labels[a[:,0]]==i;q=subset(current,mask);length=float(np.linalg.norm(q.xyz[q.edges[:,1]]-q.xyz[q.edges[:,0]],axis=1).sum())
        samples,w=_sample_arclength(graph_segments(q),.5);support=float(w[bounded_distances(samples,index)<=2.].sum())
        parts.append((support/max(length,1e-12),-length,i,mask))
    keep=np.ones(len(current.edges),bool);length=sum(-p[1] for p in parts);dropped=0
    for _,negative_length,i,mask in sorted(parts,key=lambda p:p[:3]):
        if length<=cap+1e-9:break
        keep[mask]=False;length+=negative_length;dropped+=1
    current=subset(current,keep)
    if anchors is not None and len(current.xyz):
        # Preserve only components containing an original classical anchor.
        a=current.edges;n=len(current.xyz);adjacency=csr_matrix((np.ones(2*len(a)),(np.r_[a[:,0],a[:,1]],np.r_[a[:,1],a[:,0]])),shape=(n,n))
        _,labels=connected_components(adjacency,directed=False)
        near=cKDTree(np.asarray(anchors)).query(current.xyz)[0]<=1e-8
        good=np.unique(labels[near]);current=subset(current,np.isin(labels[a[:,0]],good))
    return current,{'accepted_prunes':edits,'local_search_limit':128,'component_drops_for_budget':dropped,
                    'globally_optimal':False,'length_budget':cap,'reference_assisted':True}
