"""Reference-assisted evaluation only. Never used by feature/inference code."""
import numpy as np
from pn2.supervision.candidate_labels import TrainingPaths
from pn2.evaluation.formulation import graph_segments
from pn2.evaluation.geometry import _sample_arclength,_CurveIndex
from pn2.evaluation.bounded_agreement import bounded_distances


def reconstruction_risks(predicted,reference,reference_runs,index=None):
    index=index or TrainingPaths(reference);projection=[index.project(p) for p in predicted.xyz]
    counts={k:0 for k in ('edges','off_trace_endpoint','ambiguous_endpoint','cross_component_unknown',
            'scorable_same_component','same_chain','junction_adjacent_cross_chain','nonincident_cross_chain','reference_inconsistent_shortcut')}
    for u,v in predicted.edges:
        counts['edges']+=1;a,b=projection[u],projection[v]
        if a[0]==0 or b[0]==0:counts['off_trace_endpoint']+=1;continue
        if a[0]<0 or b[0]<0:counts['ambiguous_endpoint']+=1;continue
        qa,qb=a[1],b[1]
        if qa[1]!='edge' or qb[1]!='edge':counts['ambiguous_endpoint']+=1;continue
        ea,eb=qa[2],qb[2]
        if index.component[index.edges[ea,0]]!=index.component[index.edges[eb,0]]:counts['cross_component_unknown']+=1;continue
        counts['scorable_same_component']+=1;ca,cb=index.edge_chain[ea][0],index.edge_chain[eb][0]
        if ca==cb:counts['same_chain']+=1
        else:
            ia=index.chain_info[ca][0];ib=index.chain_info[cb][0]
            ends_a={ia[0],ia[-1]};ends_b={ib[0],ib[-1]}
            counts['junction_adjacent_cross_chain' if ends_a&ends_b else 'nonincident_cross_chain']+=1
        path,_=index.path((ea,qa[3],qa[4]),(eb,qb[3],qb[4]));length=float(np.linalg.norm(predicted.xyz[u]-predicted.xyz[v]))
        if path>2*length+4.:counts['reference_inconsistent_shortcut']+=1
    den=counts['scorable_same_component']
    counts['shortcut_fraction_of_scorable']=counts['reference_inconsistent_shortcut']/den if den else None
    counts['nonincident_cross_chain_fraction_of_scorable']=counts['nonincident_cross_chain']/den if den else None
    counts['cross_chain_fraction_of_scorable']=(counts['junction_adjacent_cross_chain']+counts['nonincident_cross_chain'])/den if den else None
    counts['different_component_fraction_of_all_edges']=counts['cross_component_unknown']/counts['edges'] if counts['edges'] else None
    segments=graph_segments(predicted);target=_CurveIndex(segments,4.) if len(segments) else None;breaks={}
    for tolerance in (2.,6.):
        total=0.;runs=0;longest=0.
        for points in reference_runs:
            p=np.asarray(points,float)
            if len(p)<2:continue
            samples,weights=_sample_arclength(np.stack((p[:-1],p[1:]),axis=1),.5)
            missing=bounded_distances(samples,target)>tolerance;current=0.
            for bad,weight in zip(missing,weights):
                if bad:
                    if current==0:runs+=1
                    current+=float(weight);total+=float(weight);longest=max(longest,current)
                else:current=0.
        breaks[str(tolerance)]={'uncovered_reference_length':total,'uncovered_intervals':runs,'longest_uncovered_interval':longest}
    return {'edge_relation_counts':counts,'breaks':breaks,
            'biological_wrong_join_rate':None,
            'interpretation':'shortcut flag/path>2*chord+4 and nonincident-chain counts are conditional reference diagnostics;fragment relations unknown;cross-chain transitions need not be wrong'}
