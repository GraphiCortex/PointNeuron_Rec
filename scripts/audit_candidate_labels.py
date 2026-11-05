"""CPU diagnostic only: saved-label oracle, fragmentation audit, fixed-feature AP.

No labels, candidates, features, thresholds or model are selected or changed.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import numpy as np
from scipy.spatial import cKDTree

import local_curve_experiment as e
from score_saved_curve_transfer import cached_record
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.label_audit import precision_recall_area,label_oracle_gate
from pn2.evaluation.bounded_agreement import BoundedAgreement,bounded_distances
from pn2.evaluation.formulation import graph_segments,runs_graph,clip_segments
from pn2.evaluation.selection_oracle import edge_agreement
from pn2.evaluation.probability_forest import decode
from pn2.evaluation.vote_scaffold import image_edge_costs,export_forest
from pn2.evaluation.fixed_curve_graph import summarize
from pn2.supervision.candidate_labels import TrainingPaths
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/candidate_label_audit_20261105.json'


def control():
    c=e.read(CONFIG)
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed label-audit binding: '+name)
    if c['schema']!='pn2-candidate-label-audit-v1' or c['wall_seconds']!=3600 or c['maximum_drop']!=.03:
        raise ValueError('Changed diagnostic scope')
    scientific=e.control();data,_,_=e.prepared(scientific);stage1=ROOT/c['stage1_root']
    report=e.read(stage1/'report.json');manifest=e.read(stage1/'candidate_manifest.json');labels=e.read(stage1/'label_manifest.json')
    if (report['status']!='completed' or digest(stage1/'candidate_manifest.json')!=report['candidate_manifest_sha256']
            or digest(stage1/'label_manifest.json')!=report['label_manifest_sha256']):raise ValueError('Changed completed Stage1')
    packets=[p for p in manifest['packets'] if p['partition']=='training'];labels={tuple(p['key']):p for p in labels['packets']}
    if len(packets)!=460 or len(labels)!=460 or len({p['key'][0] for p in packets})!=33:raise ValueError('Changed training cohort')
    for p in packets:
        key=tuple(p['key']);label=labels[key]
        if (data.cases[key[0]]['partition']!='training' or label['partition']!='training'
                or label['candidate_sha256']!=p['sha256'] or p['references_used_for_construction']):
            raise ValueError('Invalid training/candidate binding')
        for a in (p,label):
            if digest(Path(a['path']))!=a['sha256']:raise ValueError('Changed saved packet')
    return c,scientific,data,packets,labels,report


def decoded_score(xyz,edges,node_p,edge_p,guide,agreement,low,high,folder):
    result=decode(xyz,edges,node_p,edge_p);selected=result['selected_indices']
    folder.mkdir(parents=True)
    exported=export_forest(folder/'diagnostic_only.swc',xyz[selected],result['edges'],guide[selected])
    segments=graph_segments(SimpleNamespace(xyz=xyz[selected],edges=result['edges']))
    metrics=agreement.score(clip_segments(segments,low,high))
    return {'segments':metrics,'decoded':exported,'swc':str(folder/'diagnostic_only.swc'),
            'swc_sha256':digest(folder/'diagnostic_only.swc'),
            'pruning':{k:result[k] for k in ('pre_prune_nodes','pre_prune_edges','pre_prune_components','pruned_nodes','pruned_length')}}


def single_feature_scores(xyz,edges,source_xyz,guide,maximum):
    """Fixed lower-is-better image features; no labels accepted."""
    nearest=cKDTree(source_xyz).query(xyz)[0]
    source_distance=np.max(nearest[edges],axis=1) if len(edges) else np.empty(0)
    costs=[]
    for start in range(0,len(edges),2048):
        cost,_=image_edge_costs(xyz,edges[start:start+2048],guide,maximum)
        costs.append(cost)
    return {'sato_straight_path_cost':np.concatenate(costs) if costs else np.empty(0),
            'max_endpoint_nearest_input_distance':source_distance}


def run_packets(c,scientific,data,packets,labels,output):
    refs=e.References(scientific,data);guides={};maxima={};paths={};rows=[]
    for i,packet in enumerate(packets):
        key=tuple(packet['key']);cid,tile=key;label=labels[key]
        binding=fingerprint({'config':fingerprint(c),'key':list(key),'candidate':packet['sha256'],'labels':label['sha256']})
        def compute():
            with np.load(packet['path'],allow_pickle=False) as a:
                xyz,edges,guide_at_nodes=a['xyz'].copy(),a['edges'].copy(),a['image_guide'].copy()
            with np.load(label['path'],allow_pickle=False) as a:
                node,edge=a['node_label'].copy(),a['edge_label'].copy()
            if node.shape!=(len(xyz),) or edge.shape!=(len(edges),):raise ValueError('Unaligned labels')
            image=data.image(cid,tile,2048)['packet'];e.sato_scores(data,key,image,guides)
            if cid not in maxima:maxima[cid]=float(guides[cid].max())
            features=single_feature_scores(xyz,edges,image['native_xyz'],guides[cid],maxima[cid])
            morphology=refs.get(cid)['morphology']
            if cid not in paths:paths.clear();paths[cid]=TrainingPaths(morphology)
            index=paths[cid];component=np.full(len(xyz),-1,int)
            for ni in np.flatnonzero(node==1):
                status,q,_=index.project(xyz[ni])
                if status!=1:raise ValueError('Saved positive projection cannot be replayed')
                vertex=q[2] if q[1]=='isolated' else index.edges[q[2],0]
                component[ni]=index.component[vertex]
            both_positive=np.all(node[edges]==1,axis=1)
            cross=both_positive&(component[edges[:,0]]!=component[edges[:,1]])
            fragment={'reference_components':morphology.component_count,'both_positive_edges':int(both_positive.sum()),
                      'both_positive_unknown':int((both_positive&(edge<0)).sum()),
                      'cross_component_positive_endpoint_edges':int(cross.sum()),
                      'cross_component_masked':int((cross&(edge<0)).sum()),
                      'cross_component_negative':int((cross&(edge==0)).sum()),
                      'cross_component_positive':int((cross&(edge==1)).sum())}
            box=data.boxes[key]['box'];low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
            reference=graph_segments(runs_graph(data.reference(cid,tile)['reference_runs']))
            agreement=BoundedAgreement(reference)
            folder=output/'boxes'/f'box{i:04d}_{uuid.uuid4().hex[:8]}';folder.mkdir(parents=True)
            # Unknowns are ineligible, not relabeled negatives in saved supervision.
            oracle=decoded_score(xyz,edges,(node==1).astype(float),(edge==1).astype(float),
                                  guide_at_nodes,agreement,low,high,folder/'label_oracle')
            positive_union=agreement.score(clip_segments(graph_segments(SimpleNamespace(xyz=xyz,edges=edges[edge==1])),low,high))
            # New same-TRAINING-cohort comparator separates target-rule and cohort differences.
            geometric_node=(refs.get(cid)['distances'](xyz)<=2.).astype(float)
            geometric_edge,in_length=edge_agreement(xyz,edges,low,high,agreement.index)
            geometric=decoded_score(xyz,edges,geometric_node,np.where(in_length>0,geometric_edge,0.),
                                     guide_at_nodes,agreement,low,high,folder/'geometric_oracle')
            feature_path=folder/'fixed_feature_scores.npz'
            np.savez_compressed(feature_path,edge_label=edge,**features)
            return {'row':{'case_id':cid,'tile_id':tile,'alias':data.cases[cid]['alias'],'label_oracle':oracle,
                           'geometric_oracle':geometric,'positive_edge_union':positive_union,'fragment_audit':fragment,
                           'feature_scores':str(feature_path),'feature_scores_sha256':digest(feature_path),
                           'candidate_sha256':packet['sha256'],'label_sha256':label['sha256']}}
        record,reused=cached_record(output/'records'/f'box{i:04d}.json',binding,compute)
        row=record['row']
        for field,sha in [('feature_scores','feature_scores_sha256')]:
            if digest(Path(row[field]))!=row[sha]:raise ValueError('Changed cached diagnostic features')
        rows.append(row)
        if (i+1)%10==0 or i==459:print(f'LABEL AUDIT {i+1}/460 reused={reused}',flush=True)
    return rows


def assemble(c,rows,stage1):
    summaries={name:summarize([{'case_id':r['case_id'],'segments':r[name]['segments']} for r in rows])
               for name in ('label_oracle','geometric_oracle')}
    summaries['positive_edge_union']=summarize(rows,'positive_edge_union')
    neurons=[];features=('sato_straight_path_cost','max_endpoint_nearest_input_distance')
    for n in stage1['neurons']:
        group=[r for r in rows if r['case_id']==n['case_id']];y=[];scores={f:[] for f in features}
        for r in group:
            with np.load(r['feature_scores'],allow_pickle=False) as a:
                known=a['edge_label']>=0;y.append(a['edge_label'][known].copy())
                for f in features:scores[f].append(-a[f][known].copy())
        y=np.concatenate(y);ranking={f:precision_recall_area(y,np.concatenate(scores[f])) for f in features}
        fragments={k:sum(r['fragment_audit'][k] for r in group) for k in group[0]['fragment_audit'] if k!='reference_components'}
        fragments['reference_components']=group[0]['fragment_audit']['reference_components']
        fragments['masked_fraction_both_positive']=fragments['both_positive_unknown']/fragments['both_positive_edges'] if fragments['both_positive_edges'] else None
        neurons.append({**n,'fixed_feature_heldout_edge_AP':ranking,'fragment_audit':fragments,
                        'known_fragmented_alias':n['alias'] in (23,66,75)})
    stage0=e.read(ROOT/c['stage0_report'])['summaries']['union']['oracle']['macro']['2.0']['f1']
    gate=label_oracle_gate(summaries['label_oracle']['macro']['2.0']['f1'],stage0)
    violation=sum(n['fragment_audit']['cross_component_negative']+n['fragment_audit']['cross_component_positive'] for n in neurons)
    gate['cross_component_known_label_violations']=violation
    gate['passed']=bool(gate['passed'] and violation==0)
    return {'summaries':summaries,'neurons':neurons,'gate':gate,
            'fixed_feature_macro_AP':{f:float(np.mean([n['fixed_feature_heldout_edge_AP'][f]['average_precision'] for n in neurons])) for f in features}}


def worker(output):
    c,scientific,data,packets,labels,stage1=control();output.mkdir(parents=True,exist_ok=False);(output/'records').mkdir()
    write_once(output/'binding.json',{'config_fingerprint':fingerprint(c),'stage':'diagnostic only'})
    rows=run_packets(c,scientific,data,packets,labels,output);result=assemble(c,rows,stage1)
    report={'status':'completed','config_fingerprint':fingerprint(c),'rows':rows,**result,
            'GPU_executed':False,'model_executed':False,'training_executed':False,'labels_changed':False,
            'development_labels_created':False,'development_scored_again':False,
            'AUPRC_convention':'primary stepwise average precision, ties grouped; trapezoidal PR area also reported',
            'baseline_scope':'fixed lower-is-better image-only rankings, scored separately on each held-out training neuron; no fitted parameters/thresholds or selector training',
            'Sato_cost':'preserved straight-segment scaffold cost; NOT geodesic path cost',
            'CV_limit':'conditional on frozen upstream trained on all33 training neurons; not full-pipeline CV',
            'zero_positive_boxes':{kind:[{'alias':data.cases[r['case_id']]['alias'],'key':r['key']} for r in stage1['rows'] if not r['counts'][kind]['positive']] for kind in ('node','edge')}}
    write_once(output/'report.json',report)
    lines=['# Saved-label oracle and fixed-feature diagnostic','','No labels/geometry/constants changed. Training only; no model training.',
           '| Oracle | Macro F1 at2 | Macro F1 at6 |','|---|---:|---:|']
    for name in ('label_oracle','geometric_oracle'):
        a=result['summaries'][name]['macro'];lines.append(f"| {name} | {100*a['2.0']['f1']:.2f}% | {100*a['6.0']['f1']:.2f}% |")
    lines+=['',json.dumps(result['gate'],indent=2),'','Sato feature is straight-segment cost, not geodesic. AP uses fixed score signs, no fitting.',
            '| Alias | Components | Edge positive rate known | Node masked | Edge masked | Positive-endpoint masked | Cross-component mislabeled | Sato AP | Nearest-input AP |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in result['neurons']:
        a,b=n['counts']['node'],n['counts']['edge'];f=n['fragment_audit'];ap=n['fixed_feature_heldout_edge_AP']
        fmt=lambda v:f'{100*v:.2f}%' if v is not None else 'n/a'
        lines.append(f"| {n['alias']} | {f['reference_components']} | {fmt(b['positive']/(b['positive']+b['negative']))} | {fmt(a['masked_fraction'])} | {fmt(b['masked_fraction'])} | {fmt(f['masked_fraction_both_positive'])} | {f['cross_component_negative']+f['cross_component_positive']} | {fmt(ap['sato_straight_path_cost']['average_precision'])} | {fmt(ap['max_endpoint_nearest_input_distance']['average_precision'])} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if 'torch' in sys.modules:raise RuntimeError('CPU diagnostic imported Torch')
    print('LABEL AUDIT COMPLETE '+json.dumps({'gate':result['gate'],'report':str(output/'report.json'),'automatic_training':False}),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();c,*_=control();output=e.output_path(a.output or ROOT/'tmp/candidate_label_audit_20261105')
    if a.worker:worker(output);return
    if output.exists():raise FileExistsError('Preserve previous diagnostics; no automatic retry')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    print(f'LABEL AUDIT CPU ONLY; OUTPUT {output}; cap3600s; no changes/training',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(output)],
        output.with_name(output.name+'_guard_'+suffix),c['wall_seconds'],{'stage':'label audit','config':fingerprint(c),'output':str(output)})


if __name__=='__main__':main()
