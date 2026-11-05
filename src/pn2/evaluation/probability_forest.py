"""Probability-only decoder. No reference, target or label inputs."""
import numpy as np
from .vote_scaffold import minimum_spanning_forest


def prune_spurs(points,edges):
    """One pass removing leaf-to-junction paths <2 voxels; preserve components."""
    p=np.asarray(points);edges=np.asarray(edges,int).reshape(-1,2)
    adjacency=[[] for _ in p]
    for u,v in edges:adjacency[u].append(int(v));adjacency[v].append(int(u))
    degree=np.asarray([len(a) for a in adjacency]);remove=np.zeros(len(p),bool)
    for leaf in np.flatnonzero(degree==1):
        path=[int(leaf)];previous=-1;current=int(leaf);length=0.
        while True:
            following=next((v for v in adjacency[current] if v!=previous),None)
            if following is None:break
            length+=float(np.linalg.norm(p[following]-p[current]))
            if length>=2.:break
            previous,current=current,following
            if degree[current]!=2:
                if degree[current]>=3:remove[path]=True
                break
            path.append(current)
    kept=np.flatnonzero(~remove);mapping=np.full(len(p),-1,int);mapping[kept]=np.arange(len(kept))
    retained_edges=edges[np.all(~remove[edges],axis=1)]
    pruned=edges[~np.all(~remove[edges],axis=1)]
    return kept,mapping[retained_edges],float(np.linalg.norm(p[pruned[:,1]]-p[pruned[:,0]],axis=1).sum())


def decode(points,edges,node_probability,edge_probability):
    """Threshold .5, then spanning forest of eligible components, then fixed pruning.

    Spanning the eligible components is explicit: minimizing nonnegative costs
    over arbitrary subsets would otherwise choose the empty graph.
    """
    p=np.asarray(points,float);edges=np.asarray(edges,int).reshape(-1,2)
    node_probability=np.asarray(node_probability,float);edge_probability=np.asarray(edge_probability,float)
    if node_probability.shape!=(len(p),) or edge_probability.shape!=(len(edges),):raise ValueError('Unaligned probabilities')
    if (not np.isfinite(node_probability).all() or not np.isfinite(edge_probability).all()
            or ((node_probability<0)|(node_probability>1)).any() or ((edge_probability<0)|(edge_probability>1)).any()):
        raise ValueError('Invalid probabilities')
    selected=np.flatnonzero(node_probability>=.5)
    mapping=np.full(len(p),-1,int);mapping[selected]=np.arange(len(selected))
    eligible=np.flatnonzero((edge_probability>=.5)&np.all(mapping[edges]>=0,axis=1))
    mapped=mapping[edges[eligible]]
    cost=-np.log(np.clip(edge_probability[eligible],1e-6,1.))
    cost+=.5*(-np.log(np.clip(node_probability[edges[eligible,0]],1e-6,1.))
              -np.log(np.clip(node_probability[edges[eligible,1]],1e-6,1.)))
    forest,components=minimum_spanning_forest(p[selected],mapped,cost)
    kept,pruned,length=prune_spurs(p[selected],forest)
    return {'selected_indices':selected[kept],'edges':pruned,'eligible_edges':eligible,
            'pre_prune_nodes':len(selected),'pre_prune_edges':len(forest),'pre_prune_components':components,
            'pruned_nodes':len(selected)-len(kept),'pruned_length':length}
