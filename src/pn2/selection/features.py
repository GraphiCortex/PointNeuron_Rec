"""Image/candidate-only features. No reference or label input is accepted."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from pn2.evaluation.vote_scaffold import guide_at,image_edge_costs

NODE_NAMES=['sato','intensity','source_classical','source_sato','source_learned',
            'log_classical_members','log_sato_members','log_learned_members',
            'nearest_input_distance','nearest_input_intensity','nearest_candidate',
            'median_six_spacing','sixth_spacing','log_neighbors_2','log_neighbors_4',
            'linearity','planarity','scattering','log_raw_vote_support_2',
            'distance_crop_boundary','outside_image']
STATIC_EDGE_NAMES=['length','abs_dx','abs_dy','abs_dz','straight_sato_cost','mean_sato',
                   'minimum_sato','std_sato','mean_intensity','minimum_intensity','std_intensity',
                   'bounded_geodesic_cost','geodesic_over_length','geodesic_capped',
                   'intervening_full','edge_over_full_spacing','tangent_u_alignment',
                   'tangent_v_alignment','endpoint_tangent_alignment']


def local_geometry(xyz):
    p=np.asarray(xyz,float).reshape(-1,3);n=len(p);out=np.zeros((n,8),float);tangent=np.zeros((n,3))
    if n<2:return out,tangent
    tree=cKDTree(p);k=min(7,n);dist,ids=tree.query(p,k=k)
    out[:,:3]=np.column_stack((dist[:,1],np.median(dist[:,1:],axis=1),dist[:,-1]))
    for col,r in [(3,2.),(4,4.)]:out[:,col]=np.log1p(np.maximum(0,tree.query_ball_point(p,r,return_length=True)-1))
    delta=p[ids[:,1:]]-p[:,None,:];cov=np.einsum('nki,nkj->nij',delta,delta)/max(1,k-1)
    values,vectors=np.linalg.eigh(cov);den=np.maximum(values[:,-1],1e-12)
    out[:,5:]=np.column_stack(((values[:,2]-values[:,1])/den,(values[:,1]-values[:,0])/den,values[:,0]/den))
    tangent=vectors[:,:,-1]
    return out,tangent


def intervening_counts(xyz,edges):
    p=np.asarray(xyz,float).reshape(-1,3);edges=np.asarray(edges,int).reshape(-1,2);tree=cKDTree(p);counts=np.zeros(len(edges),float)
    for i,(u,v) in enumerate(edges):
        delta=p[v]-p[u];length2=float(delta@delta)
        if length2==0:continue
        ids=np.asarray(tree.query_ball_point((p[u]+p[v])/2,np.sqrt(length2)/2+1.+1e-10),int)
        ids=ids[(ids!=u)&(ids!=v)]
        t=(p[ids]-p[u])@delta/length2
        d=np.linalg.norm(p[ids]-(p[u]+t[:,None]*delta),axis=1)
        counts[i]=np.count_nonzero((t>1e-9)&(t<1-1e-9)&(d<=1.))
    return counts


def intensity_at(raw,origin,normalization,xyz):
    p=np.asarray(xyz,float)-np.asarray(origin);lo,hi=normalization['minimum'],normalization['maximum']
    values=map_coordinates(np.asarray(raw,np.float32),p[:,::-1].T,order=1,mode='constant',cval=0.,prefilter=False)
    return np.clip((values.astype(float)-lo)/(hi-lo),0,1) if hi>lo else np.zeros(len(p))


def node_features(pool,source,raw,origin,normalization,guide,maximum,raw_votes,box,image_shape_xyz):
    p=pool['xyz'];n=len(p);g,tangent=local_geometry(p);nearest,ids=cKDTree(source['native_xyz']).query(p)
    native_intensity=source['attributes'][:,3];flags=pool['source_flags'];counts=pool['source_counts']
    support=cKDTree(raw_votes).query_ball_point(p,2.,return_length=True) if len(raw_votes) else np.zeros(n)
    low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
    boundary=np.minimum(p-low,high-p).min(1);outside=np.any((p<0)|(p>np.asarray(image_shape_xyz)-1),axis=1)
    features=np.column_stack((guide_at(guide,p,maximum),intensity_at(raw,origin,normalization,p),
               (flags&1)>0,(flags&2)>0,(flags&4)>0,np.log1p(counts),nearest,native_intensity[ids],
               g,np.log1p(support),boundary,outside)).astype(np.float32)
    if features.shape!=(n,len(NODE_NAMES)) or not np.isfinite(features).all():raise ValueError('Invalid handcrafted node features')
    return features,tangent.astype(np.float32)


def bounded_geodesic(a,b,guide,maximum,max_cells=32768,padding=2):
    """26-neighbor voxel geodesic inside endpoint bbox+2, native spacing.

    Grid endpoints are nearest voxels. Large boxes/outside endpoints return
    NaN with a cap flag; never silently substitute a straight cost.
    """
    from skimage.graph import MCP_Geometric
    p=np.rint(np.stack((a,b))).astype(int);shape=np.asarray(guide.shape[::-1])
    if np.any(p<0) or np.any(p>=shape):return np.nan,1.
    low=np.maximum(0,p.min(0)-padding);high=np.minimum(shape,p.max(0)+padding+1)
    if int(np.prod(high-low))>max_cells:return np.nan,1.
    q=np.asarray(guide[low[2]:high[2],low[1]:high[1],low[0]:high[0]],float)
    q=np.clip(q/maximum,0,1) if maximum>0 else np.zeros_like(q)
    cost=1.+4.*(1.-q);start=tuple((p[0]-low)[::-1]);end=tuple((p[1]-low)[::-1])
    distance,_=MCP_Geometric(cost,fully_connected=True).find_costs([start],[end])
    value=float(distance[end]);return (value,0.) if np.isfinite(value) else (np.nan,1.)


def static_edge_features(pool,node,tangent,edge_ids,raw,origin,normalization,guide,maximum):
    raw=np.asarray(raw,np.float32)
    xyz=pool['xyz'];edges=pool['edges'][edge_ids];delta=xyz[edges[:,1]]-xyz[edges[:,0]];length=np.linalg.norm(delta,axis=1)
    unit=delta/np.maximum(length[:,None],1e-12);cost,mean=image_edge_costs(xyz,edges,guide,maximum)
    count=intervening_counts(xyz,edges);spacing=node[:,11]
    out=np.zeros((len(edges),len(STATIC_EDGE_NAMES)),np.float32)
    for i,(u,v) in enumerate(edges):
        steps=max(1,int(np.ceil(length[i]/.5)));points=xyz[u]+((np.arange(steps)+.5)/steps)[:,None]*delta[i]
        sato=guide_at(guide,points,maximum);intensity=intensity_at(raw,origin,normalization,points)
        geod,capped=bounded_geodesic(xyz[u],xyz[v],guide,maximum)
        scale=max(float((spacing[u]+spacing[v])/2),1e-6)
        out[i]=[length[i],*np.abs(unit[i]),cost[i],mean[i],sato.min(),sato.std(),
                intensity.mean(),intensity.min(),intensity.std(),geod,geod/max(length[i],1e-6),capped,
                count[i],length[i]/scale,abs(tangent[u]@unit[i]),abs(tangent[v]@unit[i]),abs(tangent[u]@tangent[v])]
    return out


def edge_matrix(pool,node,static,kept,edge_ids,pointnext=None):
    """Same representative geometry in both feature arms. No labels accepted."""
    edges=pool['edges'][edge_ids];ids=np.flatnonzero(kept);mapping=np.full(len(kept),-1,int);mapping[ids]=np.arange(len(ids))
    if len(edges) and not np.all(kept[edges]):raise ValueError('Edge touches suppressed node')
    geometry,_=local_geometry(pool['xyz'][ids]);mapped=mapping[edges]
    count=intervening_counts(pool['xyz'][ids],mapped);spacing=geometry[:,1]
    local=np.maximum((spacing[mapped[:,0]]+spacing[mapped[:,1]])/2,1e-6)
    length=static[:,0]
    u,v=edges[:,0],edges[:,1]
    pieces=[static,(node[u]+node[v])/2,np.abs(node[u]-node[v]),
            count[:,None],(length/local)[:,None]]
    if pointnext is not None:pieces +=[(pointnext[u]+pointnext[v])/2,np.abs(pointnext[u]-pointnext[v])]
    return np.concatenate(pieces,axis=1).astype(np.float32)
