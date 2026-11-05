"""CPU metric backfill plus fixed radius2 NMS oracle diagnostics. No training."""
import argparse
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

import local_curve_experiment as e
from score_saved_curve_transfer import cached_record
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.resampled_skeleton import clip_graph,uniform_samples,resampled_scores,greedy_nms,summarize as summarize_new
from pn2.evaluation.formulation import graph_segments,runs_graph
from pn2.evaluation.bounded_agreement import BoundedAgreement
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.probability_forest import decode
from pn2.evaluation.vote_scaffold import export_forest
from pn2.evaluation.fixed_curve_graph import summarize as summarize_segments
from pn2.evaluation.selection_oracle import headroom_gate
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/resampled_compact_audit_20261105.json'
HEAD='tmp/candidate_headroom_20261105T084249_809302Z'
LABEL='tmp/candidate_label_audit_20261105'
EDGE='tmp/edge_label_geometry_audit_20261105'
SCAFFOLD='tmp/saved_vote_scaffold_20261104T182353_539123Z'
FIT='tmp/curve_transfer_fit_20261104T193753_641182Z'
DEV='tmp/curve_transfer_development_20261104T195033_977395Z'
DEV_SCORE='tmp/curve_transfer_scoring_20261104T211240_017254Z'
BROAD='tmp/broad_curve_head_20261105'


def control():
    c=e.read(CONFIG)
    if c['schema']!='pn2-resampled-compact-audit-v1' or c['spacing']!=2. or c['nms_radius']!=2. or c['wall_seconds']!=3600:
        raise ValueError('Changed fixed audit constants')
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed audit binding: '+name)
    scientific=e.control();data,_,_=e.prepared(scientific)
    return c,scientific,data


def graph_spec(path,sha,kind='swc',xyz=None,edges=None):
    return {'kind':kind,'path':str(path),'sha256':sha,'xyz_field':xyz,'edge_field':edges}


def inventory(data):
    jobs=[]
    def add(method,key,spec,segments=None,note=None):
        key=tuple(key);part=data.cases[key[0]]['partition']
        jobs.append({'method':method+'/'+part,'key':list(key),'graph':spec,'segments':segments,'note':note})
    head=e.read(ROOT/HEAD/'report.json')
    for r in head['rows']:add('stage0_'+r['arm'],(r['case_id'],r['tile_id']),graph_spec(ROOT/HEAD/r['swc'],r['swc_sha256']),r['segments'],'reference-assisted oracle')
    label=e.read(ROOT/LABEL/'report.json')
    for r in label['rows']:
        for name in ('label_oracle','geometric_oracle'):
            s=r[name];add(name,(r['case_id'],r['tile_id']),graph_spec(Path(s['swc']),s['swc_sha256']),s['segments'])
    # Historical rejected/held drafts are metrics-only; NEVER rerun/adopt labels.
    drafts=e.read(ROOT/EDGE/'report.json')['drafts']
    for r in drafts['rows']:
        key=(r['case_id'],r['tile_id'])
        add('historical_HELD_A',key,graph_spec(Path(r['mask_swc']),r['mask_swc_sha256']),r['mask_only'],'held; historical artifact only')
        add('historical_REJECTED_B',key,graph_spec(Path(r['geometric_swc']),r['geometric_swc_sha256']),r['geometric_known'],'rejected; historical artifact only')
    def scaffold(manifest,report,scope,field=None):
        packets=e.read(ROOT/manifest)['packets'];r=e.read(ROOT/report)
        rows=r[field]['rows'] if field else r['rows'];by={(row['policy'],row['case_id'],row['tile_id']):row for row in rows}
        for p in packets:
            key=tuple(p['key']);row=by[(p['policy'],*key)];inf=p['inference']
            add(scope+'_'+p['policy'],key,graph_spec(Path(p['folder'])/'scaffold.swc',inf['swc_sha256']),row['segments'])
    scaffold(SCAFFOLD+'/C_graphs/inference_complete.json',SCAFFOLD+'/report.json','fit4_scaffold','C')
    scaffold(FIT+'/all_inference_complete.json',FIT+'/report.json','fit4_controls','measurement')
    scaffold(DEV+'/all_inference_complete.json',DEV_SCORE+'/report.json','fit4_transfer','measurement')
    scaffold(BROAD+'/evaluation/inference_complete.json',BROAD+'/evaluation/report.json','broad')
    # Existing raw/pruned classical geometry: no extraction or filtering.
    classical={p['case_id']:p for p in e.read(ROOT/'tmp/candidate_stage1_20261105/classical_manifest.json')['packets']}
    expected=e.read(e.BASE/'prepare/report.json')['classical_packet_hashes']
    for report in sorted((ROOT/'tmp/curve_classical_baseline_20261103').glob('case*/report.json')):
        r=e.read(report);cid=r['case_id']
        if data.cases[cid]['partition']=='development':
            p=report.parent/'graphs.npz'
            if digest(p)!=expected[cid]:raise ValueError('Changed classical development source')
            classical[cid]={'path':str(p),'sha256':expected[cid]}
    if len(classical)!=40:raise ValueError('Incomplete cached classical cohort')
    for cid,p in classical.items():
        receipt=Path(p['path']).parent/'report.json';by={}
        if receipt.exists():by={b['tile_id']:b for b in e.read(receipt)['boxes']}
        for b in data.cases[cid]['boxes']:
            if not b['target']['count']:continue  # annotation-unknown regions disclosed, not new labels
            key=(cid,b['tile_id'])
            for variant in ('raw','pruned'):
                old=by.get(b['tile_id'],{}).get('variants',{}).get(variant,{}).get('segments')
                add('classical_'+variant,key,graph_spec(Path(p['path']),p['sha256'],'npz',variant+'_xyz',variant+'_edges'),old)
    # Existing eight-voxel target graph with REFERENCE edges, metric oracle only.
    for key,b in data.boxes.items():
        if b['target']['count']:
            p=b['target'];add('reference_edge_target_oracle',key,graph_spec(ROOT/p['path'],p['sha256'],'npz','xyz','edges'),None,'existing target cache; reference edges; not inference')
    return jobs


class Metrics:
    def __init__(self,c,scientific,data,output):
        self.c,self.data,self.output=c,data,output;self.verified={};self.npz_cache={};self.ref_cache={};self.morph_cache={}
        manifest=ROOT/scientific['reference_manifest'];self.reference_root=manifest.parent
        self.references={s['case_id']:s for s in e.read(manifest)['cases']}
    def verify(self,spec):
        path=Path(spec['path'])
        if str(path) not in self.verified:
            sha=digest(path)
            if sha!=spec['sha256']:raise ValueError('Changed graph artifact: '+str(path))
            self.verified[str(path)]=sha
        elif self.verified[str(path)]!=spec['sha256']:raise ValueError('Conflicting artifact bindings')
    def graph(self,spec):
        self.verify(spec);path=Path(spec['path'])
        if spec['kind']=='swc':return parse_swc(path,type_policy='opaque_integer')
        if str(path) not in self.npz_cache:
            with np.load(path,allow_pickle=False) as a:
                self.npz_cache={str(path):{k:a[k].copy() for k in a.files if k in ('xyz','edges','raw_xyz','raw_edges','pruned_xyz','pruned_edges')}}
        a=self.npz_cache[str(path)]
        return SimpleNamespace(xyz=a[spec['xyz_field']],edges=a[spec['edge_field']])
    def reference(self,key):
        key=tuple(key)
        if key not in self.ref_cache:
            cid,tile=key;spec=self.references[cid]
            if cid not in self.morph_cache:
                path=self.reference_root/spec['reference_swc']
                if digest(path)!=spec['reference_sha256']:raise ValueError('Changed scoring reference')
                self.morph_cache={cid:parse_swc(path,type_policy='opaque_integer')}
            b=self.data.boxes[key]['box'];low=np.asarray(b['start_xyz'])-.5;high=np.asarray(b['stop_xyz'])-.5
            m=self.morph_cache[cid];graph=clip_graph(m.xyz,m.edges,low,high)
            reference=graph_segments(runs_graph(self.data.reference(cid,tile)['reference_runs']))
            self.ref_cache[key]={'samples':uniform_samples(graph),'segments':reference,'agreement':None,'low':low,'high':high}
        return self.ref_cache[key]
    def measure(self,job):
        key=tuple(job['key']);spec=job['graph'];self.verify(spec)
        identity={k:spec[k] for k in ('kind','sha256','xyz_field','edge_field')}
        token=fingerprint({'config':fingerprint(self.c),'key':list(key),'graph':identity})
        def compute():
            original=self.graph(spec);ref=self.reference(key);pred=clip_graph(original.xyz,original.edges,ref['low'],ref['high'])
            predicted_segments=graph_segments(pred);pl=float(np.linalg.norm(predicted_segments[:,1]-predicted_segments[:,0],axis=1).sum())
            rl=float(np.linalg.norm(ref['segments'][:,1]-ref['segments'][:,0],axis=1).sum())
            segments=job['segments']
            if segments is not None:
                if not np.isclose(segments['predicted_length'],pl,rtol=1e-10,atol=1e-7) or not np.isclose(segments['reference_length'],rl,rtol=1e-10,atol=1e-7):
                    raise ValueError('Existing metric does not match graph/crop length: '+job['method']+str(key))
            else:
                if ref['agreement'] is None:ref['agreement']=BoundedAgreement(ref['segments'])
                segments=ref['agreement'].score(predicted_segments)
            return {'metric':{'case_id':key[0],'tile_id':key[1],'segments':segments,
                              'length_ratio':pl/rl if rl else None,'resampled':resampled_scores(ref['samples'],pred)}}
        record,reused=cached_record(self.output/'metric_cache'/f'{token}.json',token,compute)
        return {**record['metric'],'method':job['method'],'graph':spec,'note':job.get('note'),'metric_cache_reused':reused}


def compact(c,data,metrics,output):
    candidates=e.read(ROOT/HEAD/'candidate_manifest.json')['packets'];head=e.read(ROOT/HEAD/'report.json')
    by_head={(r['case_id'],r['tile_id']):r for r in head['rows'] if r['arm']=='union'}
    pools=e.read(ROOT/'tmp/candidate_stage1_20261105/candidate_manifest.json')['packets']
    labels={tuple(p['key']):p for p in e.read(ROOT/'tmp/candidate_stage1_20261105/label_manifest.json')['packets']}
    label_old={(r['case_id'],r['tile_id']):r for r in e.read(ROOT/LABEL/'report.json')['rows']}
    work=[('compact_union',p,by_head[tuple(p['key'])]) for p in candidates if p['arm']=='union']
    work += [('compact_label',p,label_old[tuple(p['key'])]) for p in pools if p['partition']=='training']
    rows=[]
    for i,(name,packet,old) in enumerate(work):
        key=tuple(packet['key']);path=ROOT/HEAD/packet['path'] if name=='compact_union' else Path(packet['path'])
        if digest(path)!=packet['sha256']:raise ValueError('Changed fixed candidate graph')
        binding=fingerprint({'config':fingerprint(c),'name':name,'candidate':packet['sha256'],'key':list(key)})
        def compute():
            with np.load(path,allow_pickle=False) as a:xyz,edges,flags,guide=(a[k].copy() for k in ('xyz','edges','source_flags','image_guide'))
            if name=='compact_union':
                old_swc=ROOT/HEAD/old['swc'];old_sha=old['swc_sha256'];probs=old_swc.parent/'oracle_scores_DO_NOT_USE_IN_INFERENCE.npz'
                with np.load(probs,allow_pickle=False) as a:node_p,edge_p,inside=a['node_probability'].copy(),a['edge_probability'].copy(),a['edge_scored_length'].copy()
                edge_p=np.where(inside>0,edge_p,0.)
            else:
                spec=labels[key]
                if digest(Path(spec['path']))!=spec['sha256']:raise ValueError('Changed training label')
                with np.load(spec['path'],allow_pickle=False) as a:node_p=(a['node_label']==1).astype(float);edge_p=(a['edge_label']==1).astype(float)
                old_swc=Path(old['label_oracle']['swc']);old_sha=old['label_oracle']['swc_sha256']
            folder=output/'compact_graphs'/f'{name}_{i:04d}';folder.mkdir(parents=True)
            replay=decode(xyz,edges,node_p,edge_p);ids=replay['selected_indices']
            export_forest(folder/'original_oracle_replay.swc',xyz[ids],replay['edges'],guide[ids])
            if digest(folder/'original_oracle_replay.swc')!=old_sha or digest(old_swc)!=old_sha:raise ValueError('Original oracle probability/geometry replay differs')
            keep=greedy_nms(xyz,node_p,flags);node=np.where(keep,node_p,0.)
            result=decode(xyz,edges,node,edge_p);ids=result['selected_indices'];swc=folder/'compact_diagnostic_only.swc'
            export_forest(swc,xyz[ids],result['edges'],guide[ids])
            return {'job':{'method':name+'/'+data.cases[key[0]]['partition'],'key':list(key),'graph':graph_spec(swc,digest(swc)),
                           'segments':None,'note':'diagnostic NMS2; original edges only; no remapping or relabeling'},
                    'nms':{'eligible_before':int((node_p>=.5).sum()),'kept_before_forest':int(keep.sum()),
                           'original_edges':len(edges),'edges_between_NMS_nodes':int(np.all(keep[edges],axis=1).sum()),
                           'decoded_nodes':len(ids),'decoded_edges':len(result['edges']),
                           'original_oracle_replay_exact':True,'coordinates_moved':False,'edges_rewired':False}}
        record,reused=cached_record(output/'compact_records'/f'graph{i:04d}.json',binding,compute)
        row=metrics.measure(record['job']);row['nms']=record['nms'];rows.append(row)
        if (i+1)%20==0 or i==610:print(f'COMPACT ORACLES {i+1}/611',flush=True)
    union=[r for r in rows if r['method']=='compact_union/development']
    gate=headroom_gate(summarize_segments(union)['macro']['2.0']['f1'])
    checked_manifest(output/'compact_gate.json',{'gate':gate,'original_stage0_gate_passed_at_length_ratio':2.397850994747163,
                                               'original_stage0_gate_changed':False,'NMS_adopted':False})
    print('COMPACT GATE '+json.dumps(gate),flush=True)
    return rows,gate


def checked_manifest(path,value):
    if path.exists():
        if e.read(path)!=value:raise ValueError('Existing manifest/gate differs: '+str(path))
    else:write_once(path,value)


def worker(output,resume):
    c,scientific,data=control();output.mkdir(parents=True,exist_ok=resume)
    binding={'config_fingerprint':fingerprint(c),'draftB':'rejected','draftA':'held','new_development_label_packets':False}
    if (output/'binding.json').exists():
        if e.read(output/'binding.json')!=binding:raise ValueError('Different audit output binding')
    else:write_once(output/'binding.json',binding)
    for name in ('metric_cache','compact_records'):(output/name).mkdir(exist_ok=True)
    metrics=Metrics(c,scientific,data,output)
    if (output/'report.json').exists():raise ValueError('Audit already complete; do not repeat it')
    compact_rows,gate=compact(c,data,metrics,output)
    jobs=inventory(data)
    if fingerprint(jobs)!=c['inventory_fingerprint']:raise ValueError('Changed historical graph inventory')
    checked_manifest(output/'inventory.json',{'jobs':jobs,'scope':'all saved curve-stage decodes/oracles plus raw/pruned classical graphs and cached target-edge oracle;reference-present boxes only',
        'candidate_edge_unions':'opportunity diagnostics,not skeleton outputs; original length ledger retained',
        'unknown_regions':'not relabeled; excluded from matched reference-present metrics'})
    rows=compact_rows
    for i,job in enumerate(jobs):
        rows.append(metrics.measure(job))
        if (i+1)%50==0 or i+1==len(jobs):print(f'METRIC BACKFILL {i+1}/{len(jobs)}',flush=True)
    summaries={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        summaries[method]={'boxes':len(group),'segments':summarize_segments(group),'new_metrics':summarize_new(group)}
    report={'status':'completed',**binding,'rows':rows,'summaries':summaries,'compact_gate':gate,
            'NMS_adopted':False,'label_contract_implemented':False,'Stage2_features_implemented':False,
            'GPU_executed':False,'training_executed':False,'model_executed':False,
            'resampling':'maximal degree-two chains at2 native voxels; structural endpoints once;cycles deterministic;isolates once',
            'historical_scope':'per-box reference-present graphs; historical rejected B receives metrics only,not adoption',
            'original_stage0_gate_passed_at_length_ratio':2.397850994747163}
    write_once(output/'report.json',report)
    lines=['# Resampled node matching and compact-oracle audit','','Original Stage0 gate passed at2.39785x reference length. DraftB rejected; DraftA held.','',
           '| Method/cohort | Boxes | Segment F1 at2 | Segment F1 at6 | Resampled F1 at2 | Resampled F1 at6 | Pooled length ratio |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summaries.items():
        a=s['segments']['macro'];b=s['new_metrics']
        lines.append(f"| {method} | {s['boxes']} | {100*a['2.0']['f1']:.2f}% | {100*a['6.0']['f1']:.2f}% | {100*b['macro_resampled']['2.0']['f1']:.2f}% | {100*b['macro_resampled']['6.0']['f1']:.2f}% | {b['pooled_length_ratio']:.4f} |")
    lines+=['',json.dumps(gate,indent=2),'','No NMS/label/decoder change adopted; no model/CV fitting.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (output/'per_box_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['method','case_id','tile_id','length_ratio','segment_F1_2','segment_F1_6','resampled_F1_2','resampled_F1_6']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({'method':r['method'],'case_id':r['case_id'],'tile_id':r['tile_id'],'length_ratio':r['length_ratio'],
           'segment_F1_2':r['segments']['tolerances']['2.0']['f1'],'segment_F1_6':r['segments']['tolerances']['6.0']['f1'],
           'resampled_F1_2':r['resampled']['tolerances']['2.0']['f1'],'resampled_F1_6':r['resampled']['tolerances']['6.0']['f1']})
    if 'torch' in sys.modules:raise RuntimeError('CPU audit unexpectedly imported Torch')
    print('RESAMPLED AUDIT COMPLETE '+json.dumps({'report':str(output/'report.json'),'compact_gate':gate,'automatic_training':False}),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path);a=p.parse_args()
    c,_,_=control()
    if a.output and a.resume:p.error('Use output OR resume')
    output=e.output_path(a.resume or a.output or ROOT/'tmp/resampled_compact_audit_20261105')
    if a.worker:worker(output,bool(a.resume));return
    if output.exists() and not a.resume:raise FileExistsError('Preserve old outputs; explicit resume is required')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    print(f'CPU RESAMPLED METRICS/NMS AUDIT; OUTPUT {output}; cap3600s; no model/training',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--resume' if a.resume else '--output',str(output)],
        output.with_name(output.name+'_guard_'+suffix),3600,{'stage':'resampled metric and compact oracle','output':str(output),'config':fingerprint(c)})


if __name__=='__main__':main()
