"""CPU-only saved-training-label overlap audit and unadopted draft oracles."""
import argparse
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import numpy as np

import audit_candidate_labels as prior
import local_curve_experiment as e
from score_saved_curve_transfer import cached_record
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.edge_label_geometry import memberships,skip_types,overlap_summary,metric_score,physical_edge_overlap
from pn2.evaluation.selection_oracle import edge_agreement
from pn2.evaluation.bounded_agreement import BoundedAgreement
from pn2.evaluation.formulation import graph_segments,runs_graph,clip_segments
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.fixed_curve_graph import summarize
from pn2.evaluation.probability_forest import decode
from pn2.evaluation.vote_scaffold import export_forest
from pn2.supervision.candidate_labels import TrainingPaths
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/edge_label_geometry_audit_20261105.json'


def control():
    c=e.read(CONFIG)
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed geometry-audit binding: '+name)
    if c['schema']!='pn2-edge-label-geometry-audit-v1' or c['trigger_share']!=.3 or c['on_trace_overlap']!=.5:
        raise ValueError('Changed predeclared audit')
    _,scientific,data,packets,labels,_=prior.control()
    for spec in c['spacing_metadata'].values():
        if spec and digest(Path(spec['path']))!=spec['sha256']:raise ValueError('Changed spacing sidecar')
    return c,scientific,data,packets,labels


def audit_packets(c,scientific,data,packets,labels,old,output):
    refs=e.References(scientific,data);indices={};rows=[]
    for i,packet in enumerate(packets):
        key=tuple(packet['key']);cid,tile=key;label=labels[key]
        binding=fingerprint({'config':fingerprint(c),'key':list(key),'candidate':packet['sha256'],'label':label['sha256']})
        def compute():
            with np.load(packet['path'],allow_pickle=False) as a:xyz,edges,guide=a['xyz'].copy(),a['edges'].copy(),a['image_guide'].copy()
            with np.load(label['path'],allow_pickle=False) as a:node,edge,reason,anchors=(a[k].copy() for k in ('node_label','edge_label','edge_reason','junction_anchor_vertex'))
            neg=np.flatnonzero(reason==7);pos=np.flatnonzero(edge==1);extra=np.flatnonzero(reason==6)
            selected=np.r_[neg,pos,extra];selected_edges=edges[selected]
            reference=graph_segments(runs_graph(data.reference(cid,tile)['reference_runs']));agreement=BoundedAgreement(reference)
            box=data.boxes[key]['box'];low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
            overlap,length=edge_agreement(xyz,selected_edges,low,high,agreement.index)
            if cid not in indices:indices.clear();indices[cid]=TrainingPaths(refs.get(cid)['morphology'])
            member=memberships(xyz,node,anchors,indices[cid]);kind,geo,witness=skip_types(xyz,edges[neg],node,member)
            folder=output/'boxes'/f'box{i:04d}_{uuid.uuid4().hex[:8]}';folder.mkdir(parents=True)
            arrays={'edge_indices':selected,'overlap_2':overlap,'in_box_length':length,
                    'negative_count':np.array(len(neg)),'positive_count':np.array(len(pos)),
                    'negative_kind':kind,'geometric_intervening_positive':geo,'skip_witness_index':witness}
            physical={};spacing=c['spacing_metadata'].get(cid)
            if spacing:
                scores,plength=physical_edge_overlap(xyz,selected_edges,low,high,reference,spacing['spacing_xyz_um'])
                arrays['physical_length_um']=plength
                for t,p in scores.items():arrays['overlap_um_'+t]=p
                for name in ('label_oracle','geometric_oracle'):
                    spec=old[key][name];swc=Path(spec['swc'])
                    if digest(swc)!=spec['swc_sha256']:raise ValueError('Changed original oracle SWC')
                    prediction=clip_segments(graph_segments(parse_swc(swc)),low,high)
                    physical[name]=metric_score(reference*np.asarray(spacing['spacing_xyz_um']),prediction*np.asarray(spacing['spacing_xyz_um']))
            path=folder/'audit_arrays.npz';np.savez_compressed(path,**arrays)
            return {'row':{'case_id':cid,'alias':data.cases[cid]['alias'],'tile_id':tile,'arrays':str(path),'sha256':digest(path),
                            'negative_count':len(neg),'positive_count':len(pos),'physical_oracles':physical,
                            'candidate_sha256':packet['sha256'],'label_sha256':label['sha256']}}
        record,reused=cached_record(output/'records'/f'box{i:04d}.json',binding,compute);row=record['row']
        if digest(Path(row['arrays']))!=row['sha256']:raise ValueError('Changed audit arrays')
        rows.append(row)
        if (i+1)%10==0 or i==459:print(f'EDGE LABEL AUDIT {i+1}/460 reused={reused}',flush=True)
    return rows


def summarize_edges(rows):
    def group_summary(group):
        collect={k:[] for k in ('negative_p','negative_l','positive_p','positive_l','kind','geometric')};physical={}
        for row in group:
            with np.load(row['arrays'],allow_pickle=False) as a:
                n=int(a['negative_count']);p=int(a['positive_count']);s=slice(n,n+p)
                for name,value in [('negative_p',a['overlap_2'][:n]),('negative_l',a['in_box_length'][:n]),
                                   ('positive_p',a['overlap_2'][s]),('positive_l',a['in_box_length'][s]),
                                   ('kind',a['negative_kind']),('geometric',a['geometric_intervening_positive'])]:collect[name].append(value.copy())
                if 'physical_length_um' in a:
                    for t in ('.5','1.0','2.0'):
                        field='overlap_um_'+str(float(t));physical.setdefault(str(float(t)),{'negative_p':[],'negative_l':[],'positive_p':[],'positive_l':[]})
                        for name,value in [('negative_p',a[field][:n]),('negative_l',a['physical_length_um'][:n]),('positive_p',a[field][s]),('positive_l',a['physical_length_um'][s])]:physical[str(float(t))][name].append(value.copy())
        a={k:np.concatenate(v) for k,v in collect.items()};result={}
        for k in ('negative','positive'):result[k]=overlap_summary(a[k+'_p'],a[k+'_l'])
        n=len(a['kind']);result['negative_categories']={}
        for name,mask in [('same_chain_skip_within_1',a['kind']==0),('same_chain_other',a['kind']==1),('cross_branch_chain_pairs',a['kind']==2)]:
            result['negative_categories'][name]=overlap_summary(a['negative_p'][mask],a['negative_l'][mask])
        result['negative_geometric_witness_any_branch']=int(a['geometric'].sum())
        result['cross_branch_with_geometric_witness']=int((a['geometric']&(a['kind']==2)).sum())
        result['negative_on_trace_share_all_edges']=result['negative']['overlap_ge_half']/n if n else None
        result['physical']={t:{k:overlap_summary(np.concatenate(v[k+'_p']),np.concatenate(v[k+'_l'])) for k in ('negative','positive')} for t,v in physical.items()}
        return result
    neurons=[{'case_id':cid,'alias':next(r['alias'] for r in rows if r['case_id']==cid),**group_summary([r for r in rows if r['case_id']==cid])} for cid in sorted({r['case_id'] for r in rows})]
    return {'pooled':group_summary(rows),'neurons':neurons}


def draft_oracles(c,scientific,data,packets,labels,rows,old,output):
    refs=None;result=[]
    for i,(packet,row) in enumerate(zip(packets,rows)):
        key=tuple(packet['key']);cid,tile=key
        with np.load(packet['path'],allow_pickle=False) as a:xyz,edges,guide=a['xyz'],a['edges'],a['image_guide']
        with np.load(labels[key]['path'],allow_pickle=False) as a:node,edge=a['node_label'],a['edge_label']
        with np.load(row['arrays'],allow_pickle=False) as a:
            selected=a['edge_indices'];overlap=a['overlap_2'];length=a['in_box_length'];n=int(a['negative_count'])
            conflict=selected[:n][(overlap[:n]>=.5)&(length[:n]>0)]
        folder=output/'draft_only'/f'box{i:04d}';folder.mkdir(parents=True)
        # Draft A masks conflicting nonconsecutive negatives. Ineligible in BOTH
        # old and draft oracle; rerun decode and verify exact SWC score replay.
        masked=edge.copy();masked[conflict]=-1
        a=decode(xyz,edges,(node==1).astype(float),(masked==1).astype(float));ids=a['selected_indices']
        apath=folder/'mask_conflicts_DIAGNOSTIC_ONLY.swc';export_forest(apath,xyz[ids],a['edges'],guide[ids])
        if digest(apath)!=old[key]['label_oracle']['swc_sha256']:raise ValueError('Mask-only draft changed binary oracle unexpectedly')
        # Draft B geometric binary labels on known positive-endpoint edges only.
        # Every original unknown mask is retained, including fragment relations.
        ep=np.zeros(len(edges));ep[selected]=(overlap>=.5)&(length>0)
        if np.any((ep>0)&(edge<0)):raise ValueError('Draft geometric arm changed unknown mask')
        box=data.boxes[key]['box'];low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
        reference=graph_segments(runs_graph(data.reference(cid,tile)['reference_runs']));agreement=BoundedAgreement(reference)
        b=prior.decoded_score(xyz,edges,(node==1).astype(float),ep,guide,agreement,low,high,folder/'geometric_known_edges')
        write_once(folder/'diagnostic_receipt.json',{'key':list(key),'original_label_sha256':labels[key]['sha256'],
                 'mask_conflicts':len(conflict),'no_training_labels_written':True,'adopted':False,'mask_oracle_exact_SWC_replay':True})
        result.append({'case_id':cid,'tile_id':tile,'mask_conflicts':len(conflict),
                       'mask_only':old[key]['label_oracle']['segments'],'geometric_known':b['segments'],
                       'mask_swc':str(apath),'mask_swc_sha256':digest(apath),'geometric_swc':b['swc'],'geometric_swc_sha256':b['swc_sha256']})
        if (i+1)%20==0 or i==459:print(f'DRAFT ORACLES {i+1}/460; NOT ADOPTED',flush=True)
    return {'adopted':False,'training_labels_written':False,'rows':result,
            'summaries':{name:summarize(result,name) for name in ('mask_only','geometric_known')}}


def length_ledger(c,output,drafts=None,physical_rows=None):
    rows=[]
    def walk(value,source,pointer,units='native_voxel',context=None):
        context=dict(context or {})
        if isinstance(value,dict):
            for k in ('case_id','tile_id','arm','policy','partition'):
                if k in value and isinstance(value[k],(str,int)):context[k]=value[k]
            if all(isinstance(value.get(k),(int,float)) for k in ('reference_length','predicted_length')):
                rl,pl=value['reference_length'],value['predicted_length']
                rows.append({'source_report':source,'metric_pointer':pointer,'reference_length':rl,'predicted_length':pl,
                             'predicted_div_reference':pl/rl if rl else None,'units':units,'zero_reference_length':rl==0,
                             **{k:context.get(k) for k in ('case_id','tile_id','arm','policy','partition')}})
            for k,v in value.items():walk(v,source,pointer+'/'+str(k),units,context)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,source,pointer+'/'+str(i),units,context)
    for source in c['length_sources']:
        walk(e.read(ROOT/source),source,'')
    if drafts:walk(drafts,'CURRENT_DRAFT_DIAGNOSTICS','/drafts')
    for i,row in enumerate(physical_rows or []):
        walk(row['physical_oracles'],'CURRENT_REPORTED_SPACING_VIEW',f'/rows/{i}/physical_oracles','um_from_unverified_sidecar')
    path=output/'all_prior_length_ratios.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'path':str(path),'sha256':digest(path),'metric_entries':len(rows),'sources':c['length_sources'],
            'includes_nested_per_box_and_aggregate_entries':True,'independent_experiments':False,
            'isolated_point_only_results':'no predicted segment length; ratio not applicable',
            'prior_ratio_summaries':{s:{'entries':sum(r['source_report']==s for r in rows)} for s in c['length_sources']}}


def worker(output):
    c,scientific,data,packets,labels=control();output.mkdir(parents=True,exist_ok=False);(output/'records').mkdir()
    write_once(output/'binding.json',{'config_fingerprint':fingerprint(c),'labels_changed':False})
    old_report=e.read(ROOT/'tmp/candidate_label_audit_20261105/report.json');old={(r['case_id'],r['tile_id']):r for r in old_report['rows']}
    rows=audit_packets(c,scientific,data,packets,labels,old,output);summary=summarize_edges(rows)
    if summary['pooled']['negative']['edges']!=180269 or summary['pooled']['positive']['edges']!=163536:raise ValueError('Changed requested label counts')
    trigger=summary['pooled']['negative_on_trace_share_all_edges']>=.3
    # Training-only draft diagnostics, never a replacement label manifest.
    drafts=draft_oracles(c,scientific,data,packets,labels,rows,old,output) if trigger else None
    ledger=length_ledger(c,output,drafts,rows)
    report={'status':'completed','config_fingerprint':fingerprint(c),'rows':rows,**summary,'triggered':trigger,
            'trigger':'at least30% of ALL nonconsecutive negatives have positive in-box length and overlap>=.5',
            'drafts':drafts,'length_ledger':ledger,'spacing_metadata':c['spacing_metadata'],
            'native_scoring_spacing':[1.,1.,1.],'native_units':'voxel','micron_metadata_verified':False,
            'micron_tolerances':[.5,1.,2.],'micron_quadrature_step':.1,'development_labels_created':False,
            'Stage2_work_executed':False,'model_executed':False,'GPU_executed':False,'training_executed':False,
            'original_labels_changed':False,'draft_adopted':False}
    # Original packet integrity rechecked after all diagnostics.
    for p in labels.values():
        if digest(Path(p['path']))!=p['sha256']:raise ValueError('Original labels changed')
    write_once(output/'report.json',report)
    lines=['# Edge-label geometry audit — training only','','No labels changed or draft adopted; no Stage2 work.','',
           f"Revision trigger: {trigger}; on-trace share={100*summary['pooled']['negative_on_trace_share_all_edges']:.2f}%.",'',
           '| Alias | Negative mean overlap | Negative length-weighted overlap | Negative >=.5 | Same-chain skip <=1 | Cross-branch | Positive mean overlap | Positive <.5 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in summary['neurons']:
        a,b=n['negative'],n['positive'];cat=n['negative_categories']
        fmt=lambda x:f'{100*x:.2f}%' if x is not None else 'n/a'
        lines.append(f"| {n['alias']} | {fmt(a['edge_mean_overlap'])} | {fmt(a['length_weighted_overlap'])} | {a['overlap_ge_half']}/{a['edges']} | {cat['same_chain_skip_within_1']['edges']} | {cat['cross_branch_chain_pairs']['edges']} | {fmt(b['edge_mean_overlap'])} | {b['overlap_lt_half']} |")
    lines+=['','Complete distributions, draft oracles and conditional micron view are in report.json.','Reported sidecar spacing only, not verified calibration; missing cases are not assigned microns.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if 'torch' in sys.modules:raise RuntimeError('CPU label audit unexpectedly imported Torch')
    print('EDGE GEOMETRY AUDIT COMPLETE '+json.dumps({'report':str(output/'report.json'),'triggered':trigger,'draft_adopted':False}),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    c,*_=control();output=e.output_path(a.output or ROOT/'tmp/edge_label_geometry_audit_20261105')
    if a.worker:worker(output);return
    if output.exists():raise FileExistsError('Preserve completed/partial diagnostics; no automatic retry')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    print(f'CPU EDGE-LABEL AUDIT; OUTPUT {output}; cap3600s; no Stage2/model/training',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(output)],output.with_name(output.name+'_guard_'+suffix),3600,
        {'stage':'edge label geometry audit','output':str(output),'config':fingerprint(c)})


if __name__=='__main__':main()
