"""Fixed native-voxel skeleton resampling and diagnostic node suppression."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from .candidate_pool import clipped_edges
from .node_proposals import one_to_one


def clip_graph(xyz,edges,low,high):
    """Clip geometry without welding coincident components or distinct edges.

    Original in-box vertex identities are retained. New boundary identities
    are local to their original edge; no coordinate-based connectivity repair.
    """
    xyz=np.asarray(xyz,float).reshape(-1,3);edges=np.asarray(edges,int).reshape(-1,2)
    low,high=np.asarray(low),np.asarray(high)
    inside=np.all((xyz>=low)&(xyz<=high),axis=1);ids=np.flatnonzero(inside)
    points=[p.copy() for p in xyz[ids]];mapping={int(j):i for i,j in enumerate(ids)};out=[]
    selected,segments=clipped_edges(xyz,edges,low,high)
    for ei,segment in zip(selected,segments):
        pair=[]
        for q in segment:
            original=next((int(v) for v in edges[ei] if inside[v] and np.linalg.norm(q-xyz[v])<=1e-9),None)
            if original is not None:pair.append(mapping[original])
            else:pair.append(len(points));points.append(q.copy())
        out.append(pair)
    for u,v in edges:
        if inside[u] and inside[v] and np.array_equal(xyz[u],xyz[v]):out.append([mapping[int(u)],mapping[int(v)]])
    return SimpleNamespace(xyz=np.asarray(points,float).reshape(-1,3),edges=np.asarray(out,int).reshape(-1,2))


def uniform_samples(graph,spacing=2.):
    """Every2 voxels along maximal degree-two chains, plus structural endpoints.

    Junctions/endpoints/isolates count once by TOPOLOGICAL identity. Only
    exactly zero-length connected edges collapse; disconnected coincidences
    remain separate samples. Pure cycles use deterministic XYZ/index phase.
    """
    if spacing!=2.:raise ValueError('Only frozen2-voxel spacing is supported')
    p=np.asarray(graph.xyz,float).reshape(-1,3);edges=np.asarray(graph.edges,int).reshape(-1,2);n=len(p)
    if not np.isfinite(p).all() or (len(edges) and (edges.min()<0 or edges.max()>=n)):raise ValueError('Invalid skeleton')
    parent=np.arange(n)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for u,v in edges:
        if np.array_equal(p[u],p[v]):
            a,b=find(u),find(v);parent[max(a,b)]=min(a,b)
    roots=np.array([find(i) for i in range(n)],int);unique=sorted(set(roots));remap={v:i for i,v in enumerate(unique)}
    p=p[unique];mapped=np.array([[remap[roots[u]],remap[roots[v]]] for u,v in edges],int).reshape(-1,2)
    mapped=mapped[mapped[:,0]!=mapped[:,1]];adj=[[] for _ in p]
    for i,(u,v) in enumerate(mapped):adj[u].append((int(v),i));adj[v].append((int(u),i))
    key=lambda i:(*map(float,p[i]),int(i))
    critical=sorted((i for i,a in enumerate(adj) if len(a)!=2),key=key)
    samples=[p[i].copy() for i in critical];visited=np.zeros(len(mapped),bool)
    def emit(ids,closed=False):
        points=p[ids]
        if not closed and key(ids[-1])<key(ids[0]):points=points[::-1]
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];length=arc[-1]
        if closed:samples.append(points[0].copy())
        # Structural endpoints are already present, including a short remainder.
        for position in np.arange(2.,max(2.,length-1e-9),2.):
            j=min(len(arc)-2,int(np.searchsorted(arc,position,side='right')-1))
            t=(position-arc[j])/(arc[j+1]-arc[j]);samples.append(points[j]+t*(points[j+1]-points[j]))
    def walk(start,first,eid):
        ids=[start,first];visited[eid]=True;previous=eid;current=first
        while len(adj[current])==2 and current!=start:
            nxt,ni=next((v,i) for v,i in adj[current] if i!=previous)
            if visited[ni]:break
            visited[ni]=True;ids.append(nxt);current,previous=nxt,ni
        return ids
    for start in critical:
        for neighbor,eid in sorted(adj[start],key=lambda a:(key(a[0]),a[1])):
            if not visited[eid]:emit(walk(start,neighbor,eid))
    # Components consisting entirely of degree-two vertices are cycles.
    for start in sorted(range(len(p)),key=key):
        for neighbor,eid in sorted(adj[start],key=lambda a:(key(a[0]),a[1])):
            if not visited[eid]:emit(walk(start,neighbor,eid),closed=True)
    return np.asarray(samples,float).reshape(-1,3)


def resampled_scores(reference_samples,predicted_graph):
    predicted=uniform_samples(predicted_graph)
    return {'spacing':2.,'units':'native_voxel','reference_samples':len(reference_samples),'predicted_samples':len(predicted),
            'tolerances':{str(t):one_to_one(reference_samples,predicted,t)[0] for t in (2.,6.)}}


def greedy_nms(xyz,probability,source_flags):
    """Frozen radius2,probability/source/XYZ/index order; no moved nodes/edges."""
    xyz=np.asarray(xyz,float).reshape(-1,3);probability=np.asarray(probability,float);flags=np.asarray(source_flags)
    if probability.shape!=(len(xyz),) or flags.shape!=probability.shape or not np.isfinite(probability).all():raise ValueError('Invalid NMS inputs')
    priority=np.where(flags&1,0,np.where(flags&2,1,np.where(flags&4,2,3)))
    eligible=np.flatnonzero(probability>=.5)
    order=eligible[np.lexsort((eligible,xyz[eligible,2],xyz[eligible,1],xyz[eligible,0],priority[eligible],-probability[eligible]))]
    removed=np.zeros(len(xyz),bool);keep=np.zeros(len(xyz),bool);tree=cKDTree(xyz)
    for i in order:
        if removed[i]:continue
        keep[i]=True;near=np.asarray(tree.query_ball_point(xyz[i],2.+1e-10),int)
        near=near[np.linalg.norm(xyz[near]-xyz[i],axis=1)<=2.];removed[near]=True
    return keep


def summarize(rows):
    neurons=[]
    for cid in sorted({r['case_id'] for r in rows}):
        group=[r for r in rows if r['case_id']==cid];rl=sum(r['segments']['reference_length'] for r in group);pl=sum(r['segments']['predicted_length'] for r in group)
        node={}
        for t in ('2.0','6.0'):
            values=[r['resampled']['tolerances'][t] for r in group]
            m=sum(v['matched'] for v in values);p=sum(v['predicted'] for v in values);q=sum(v['reference'] for v in values)
            node[t]={'matched':m,'predicted':p,'reference':q,'precision':m/p if p else 0.,'recall':m/q if q else None,'f1':2*m/(p+q) if q else None}
        neurons.append({'case_id':cid,'reference_length':rl,'predicted_length':pl,'length_ratio':pl/rl if rl else None,'resampled':node})
    return {'neurons':neurons,'pooled_length_ratio':sum(n['predicted_length'] for n in neurons)/sum(n['reference_length'] for n in neurons),
            'macro_length_ratio':float(np.mean([n['length_ratio'] for n in neurons if n['length_ratio'] is not None])),
            'macro_resampled':{t:{k:float(np.mean([n['resampled'][t][k] for n in neurons if n['resampled'][t][k] is not None])) for k in ('precision','recall','f1')} for t in ('2.0','6.0')}}
