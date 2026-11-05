"""Image-only deterministic interpolation of saved per-input PointNeXt features."""
import numpy as np
from scipy.spatial import cKDTree


def interpolate(source,features,queries):
    """Three nearest inputs, inverse-distance weights; exact copy at zero.

    Ties use source array index. No inference coordinate is moved. Distances
    are returned so extrapolation from sparse/crop-halo inputs is visible.
    """
    source=np.asarray(source,float).reshape(-1,3);queries=np.asarray(queries,float).reshape(-1,3)
    features=np.asarray(features)
    if not len(source) or features.ndim!=2 or len(features)!=len(source) or not all(
            np.isfinite(a).all() for a in (source,features,queries)):
        raise ValueError('Invalid source features/coordinates')
    tree=cKDTree(source);n=min(3,len(source));out=[];distances=[];neighbors=[];weights=[]
    for point in queries:
        d,_=tree.query(point,k=n);radius=float(np.max(d))
        ids=np.asarray(tree.query_ball_point(point,radius+1e-9),int)
        dist=np.linalg.norm(source[ids]-point,axis=1);order=np.lexsort((ids,dist))[:n]
        ids,dist=ids[order],dist[order];w=np.zeros(n)
        if dist[0]<=1e-12:w[0]=1.
        else:w=1/dist;w/=w.sum()
        out.append(w@features[ids]);distances.append(dist);neighbors.append(ids);weights.append(w)
    return {'candidate_features':np.asarray(out,np.float32).reshape(-1,features.shape[1]),
            'neighbor_indices':np.asarray(neighbors,np.int64).reshape(-1,n),
            'neighbor_distances':np.asarray(distances,float).reshape(-1,n),
            'neighbor_weights':np.asarray(weights,float).reshape(-1,n)}
