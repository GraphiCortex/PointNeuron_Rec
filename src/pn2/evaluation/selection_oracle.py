"""Explicit reference-assisted diagnostic. NEVER an inference selector."""
import numpy as np
from .geometry import _sample_arclength
from .bounded_agreement import bounded_distances
from .candidate_pool import clipped_edges


def edge_agreement(xyz,edges,low,high,reference_index):
    """Same clipped-segment .5-voxel quadrature; individual-edge P at two voxels."""
    ids,segments=clipped_edges(xyz,edges,np.asarray(low),np.asarray(high))
    score=np.zeros(len(edges));inside_length=np.zeros(len(edges))
    if not len(ids):return score,inside_length
    points,weights=_sample_arclength(segments,.5)
    # Match the scalar norm/ceil arithmetic of _sample_arclength exactly.
    counts=np.asarray([max(1,int(np.ceil(float(np.linalg.norm(b-a))/.5))) for a,b in segments])
    offset=np.r_[0,np.cumsum(counts)[:-1]]
    near=bounded_distances(points,reference_index)<=2.
    covered=np.add.reduceat(weights*near,offset);length=np.add.reduceat(weights,offset)
    score[ids]=covered/length;inside_length[ids]=length
    return score,inside_length


from .probability_forest import decode, prune_spurs


def headroom_gate(f1):
    if not np.isfinite(f1) or not 0<=f1<=1:raise ValueError('Invalid headroom F1')
    status='proceed' if f1>=.8 else 'conditional_candidate_recall_work' if f1>=.65 else 'stop'
    return {'status':status,'union_oracle_f1_2':float(f1),'thresholds':{'proceed':.8,'conditional':.65},
            'selects_model_or_threshold':False,'requires_candidate_recall_work':status=='conditional_candidate_recall_work'}
