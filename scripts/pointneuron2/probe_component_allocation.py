"""Reproduce candidate-component anchor starvation with synthetic geometry.

Calls the unchanged production selector. This is not an observed allocation
on neuron 79: its full input-pool geodesic matrix was not saved.
"""
from pathlib import Path
import argparse
import sys

import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT/'src'))
sys.path.insert(0,str(REPO_ROOT))
from scripts.initialize_geodesic_graph import connected_coverage_selection
from scripts.pointneuron2.sampling_ablation import digest,write


def probe(sizes,budget=128):
    n=sum(sizes)
    xyz=np.zeros((n,3),dtype=np.float32)
    labels=np.repeat(np.arange(len(sizes)),sizes)
    geodesic=np.full((n,n),np.inf,dtype=np.float32)
    start=0
    for component,size in enumerate(sizes):
        idx=np.arange(start,start+size)
        xyz[idx,0]=component*1000+np.arange(size)
        geodesic[np.ix_(idx,idx)]=np.abs(np.arange(size)[:,None]-np.arange(size)[None,:])
        start+=size
    scores=np.linspace(1,.5,n,dtype=np.float32)
    euclidean=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1)
    keep=connected_coverage_selection(xyz,scores,geodesic,euclidean,budget,12.)
    if len(keep)!=budget or len(np.unique(keep))!=budget:
        raise ValueError('Unexpected selected count')
    return {'eligible_component_sizes':sizes,'budget':budget,
        'observed_allocations':np.bincount(labels[keep],minlength=len(sizes)).tolist(),
        'fixed_total_proportional_shares':(budget*np.asarray(sizes)/n).tolist(),
        'zero_anchor_components':np.flatnonzero(np.bincount(labels[keep],minlength=len(sizes))==0).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior probe output')
    cases=[probe([300,250,75]),probe([250,250,125])]
    report={'status':'completed','scope':'synthetic counterexamples; not neuron-79 component measurements',
        'source_sha256':digest(REPO_ROOT/'scripts/initialize_geodesic_graph.py'),
        'probe_source_sha256':digest(Path(__file__)),'cases':cases,
        'explanation':'The numerator stays at max_nodes while remaining_source shrinks. A later large component can consume all remaining slots before smaller eligible components are visited.',
        'production_changed':False,'proposed_fix_implemented':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,report)
    for case in cases:
        print(f"sizes={case['eligible_component_sizes']} allocations={case['observed_allocations']} proportional_shares={case['fixed_total_proportional_shares']}")


if __name__=='__main__':
    main()
