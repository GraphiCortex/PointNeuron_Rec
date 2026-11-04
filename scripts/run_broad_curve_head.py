"""Explicit authorized broad same-head actions; original experiments unchanged.

prepare: CPU labels only; timing/train: GPU; score: CPU terminal metrics only.
No action is implicit. No accuracy checkpoint selection or automatic retry.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

import local_curve_experiment as e
import diagnose_broad_curve_controls as diagnostics
from pn2.inputs.node_cache import digest, fingerprint, write_once
from pn2.evaluation.local_curve import consolidate
from pn2.evaluation.formulation import graph_segments, runs_graph
from pn2.evaluation.fixed_curve_graph import build, score, summarize, gate
from pn2.training.guarded_run import run

ROOT=e.ROOT
CONFIG=ROOT/'configs/broad_curve_head_20261105.json'


def control():
    c=e.read(CONFIG)
    if (c['schema']!='pn2-broad-curve-head-v1' or c['epochs']!=20 or c['updates']!=9200
            or c['maximum_wall_seconds']!=14400 or c['training_accuracy_gated']
            or c['cutoff']!=.25 or c['consolidation_radius']!=1. or c['decoding_k']!=6):
        raise ValueError('Changed authorized broad scope')
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Changed broad source/evidence: {name}')
    diagnostics.control()
    scientific=e.control();data,prepared,_=e.prepared(scientific)
    train=sorted(k for k,b in data.boxes.items() if b['target']['gradient_training_allowed'])
    dev=[(cid,b['tile_id']) for cid,case in data.cases.items() if case['partition']=='development'
         for b in case['boxes'] if b['target']['count']]
    if len(train)!=460 or len({k[0] for k in train})!=33 or len(dev)!=151:
        raise ValueError('Training/development identities differ')
    comparison=e.read(ROOT/c['diagnostics_root']/'report.json')
    if comparison['status']!='completed' or comparison['parameters_selected']:
        raise ValueError('Required CPU diagnostics incomplete/selected parameters')
    return c,scientific,data,prepared,train,dev,comparison['comparison']['controls']


def prepare(c,scientific,data,original,train,dev,output):
    output.mkdir(parents=True,exist_ok=False);(output/'labels').mkdir()
    available={(r['case_id'],r['tile_id']):r for r in original['labels'] if r['partition']=='training'}
    refs=e.References(scientific,data);labels=[];reused=0;began=time.perf_counter();last=None
    for i,key in enumerate(train):
        if key in available:
            spec=available[key]
            if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Old training label changed')
            labels.append(spec);reused+=1
        else:
            if last!=key[0]:refs.cache.clear();last=key[0]
            source=data.image(*key,2048)['packet']['native_xyz']
            values=e.make_labels(scientific,refs,key[0],source)
            from pn2.supervision.local_curve import distance_bands
            path=output/'labels'/f'box{i:04d}.npz'
            np.savez_compressed(path,**values,bands=distance_bands(values['source_distance']))
            labels.append({'case_id':key[0],'tile_id':key[1],'partition':'training',
                           'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)})
        if (i+1)%10==0:print(f'PREPARE LABELS {i+1}/460 reused={reused} elapsed={time.perf_counter()-began:.1f}s',flush=True)
    case_ids,order=data.schedule()
    schedule=[(case_ids[int(cid)],int(tile)) for epoch,cid,tile in order]
    if len(schedule)!=9200 or any(k not in set(train) for k in schedule):raise ValueError('Unexpected schedule')
    for epoch in range(20):
        keys=schedule[epoch*460:(epoch+1)*460]
        if len(set(keys))!=460 or set(keys)!=set(train):raise ValueError('Unequal exposures/unknown schedule row')
    manifest={'schema':'pn2-broad-curve-preparation-v1','config_fingerprint':fingerprint(c),
        'scientific_config_fingerprint':fingerprint(scientific),'labels':labels,'train':train,'development':dev,
        'schedule':schedule,'schedule_sha256':data.manifest['schedule']['sha256'],
        'data_sha256':digest(data.folder/'manifest.json'),'reused_labels':reused,'generated_labels':460-reused,
        'training_cases':33,'development_never_gradient':True,'seconds':time.perf_counter()-began}
    write_once(output/'manifest.json',manifest)
    write_once(output/'complete.json',{'manifest_sha256':digest(output/'manifest.json')})
    print('BROAD PREPARATION COMPLETE '+json.dumps({'labels':len(labels),'reused':reused,'generated':460-reused}),flush=True)


def load_prepared(c,folder,data):
    p=folder/'prepare/manifest.json';m=e.read(p)
    if digest(p)!=e.read(folder/'prepare/complete.json')['manifest_sha256']:
        raise ValueError('Preparation marker/hash mismatch')
    case_ids,order=data.schedule()
    expected=[(case_ids[int(cid)],int(tile)) for epoch,cid,tile in order]
    if (m['config_fingerprint']!=fingerprint(c) or len(m['labels'])!=460
            or [tuple(k) for k in m['schedule']]!=expected or len(expected)!=9200
            or len(set(tuple(k) for k in m['train']))!=460
            or {(s['case_id'],s['tile_id']) for s in m['labels']}!=set(expected)):
        raise ValueError('Changed preparation/cohort/schedule')
    return m


def move(batch,device):
    import torch
    return {k:move(v,device) if isinstance(v,dict) else v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def cpu_records(scientific,data,manifest,keys):
    # Same original dense loss batches, retained on CPU and streamed one at a time.
    return {key:batch for key,record,batch,labels in e.train_records(scientific,data,manifest,keys,'cpu')}


def timing(c,scientific,data,manifest,output):
    import torch
    import sampler_node_common as common
    e.runtime(scientific,'cuda:0')
    if torch.cuda.get_device_name(0)!=c['device']:raise ValueError('Unexpected GPU')
    output.mkdir(exist_ok=False)
    keys=[tuple(k) for k in manifest['schedule'][:16]]
    records=cpu_records(scientific,data,manifest,list(dict.fromkeys(keys)))
    model=e.model_factory(scientific,'dense','cuda:0');optimizer=common.make_optimizer(model,scientific)
    initial=common.state_digest(model.state_dict());times=[];loss=[]
    for i,key in enumerate(keys):
        torch.cuda.synchronize();began=time.perf_counter()
        value=e.update(scientific,model,optimizer,move(records[key],'cuda:0'),'dense')
        torch.cuda.synchronize();times.append(time.perf_counter()-began);loss.append(value)
        print(f'GPU TIMING {i+1}/16 seconds={times[-1]:.3f} loss={value["loss"]:.5f}',flush=True)
    conservative=float(np.quantile(times[2:],.9))
    training_estimate=conservative*9200
    # Prior 453-graph three-policy diagnostic took ~16min including exports.
    # Reserve at least 40min for 611 single-policy terminal exports/scoring/I/O.
    eval_reserve=2400
    wall_seconds=int(math.ceil(1.5*(training_estimate+eval_reserve)/60)*60)
    report={'status':'passed' if wall_seconds<=14400 else 'resource_projection_failed',
        'config_fingerprint':fingerprint(c),'device':torch.cuda.get_device_name(0),'throwaway_updates':16,
        'initial_model_sha256':initial,'update_seconds':times,'losses':loss,
        'conservative_update_seconds':conservative,'training_estimate_seconds':training_estimate,
        'terminal_evaluation_reserve_seconds':eval_reserve,'wall_seconds_with_50_percent_margin':wall_seconds,
        'maximum_wall_seconds':14400,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'timing_model_adopted':False,'real_training_updates':0}
    write_once(output/'report.json',report);print('BROAD TIMING COMPLETE '+json.dumps(report),flush=True)
    if report['status']!='passed':raise RuntimeError('Projected broad run exceeds cap; no training')


def train(c,scientific,data,manifest,root):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import image_batch
    timing_report=e.read(root/'timing/report.json')
    if timing_report['status']!='passed' or timing_report['config_fingerprint']!=fingerprint(c):raise ValueError('No valid timing')
    e.runtime(scientific,'cuda:0')
    if torch.cuda.get_device_name(0)!=c['device']:raise ValueError('Unexpected GPU')
    output=root/'train';output.mkdir(exist_ok=False)
    keys=[tuple(k) for k in manifest['train']]
    records=cpu_records(scientific,data,manifest,keys)
    model=e.model_factory(scientific,'dense','cuda:0');optimizer=common.make_optimizer(model,scientific)
    if common.state_digest(model.state_dict())!=timing_report['initial_model_sha256'] or optimizer.state:
        raise ValueError('Fresh initializer/Adam differs from throwaway timing start')
    bindings={'maximum_updates':9200,'config_fingerprint':fingerprint(c),'scientific_config_fingerprint':fingerprint(scientific),
        'preparation_sha256':digest(root/'prepare/manifest.json'),'schedule_sha256':manifest['schedule_sha256'],
        'arm':'dense','rung':'full460'}
    state.save(output/'checkpoint_step000000.pt',model,optimizer,bindings,0)
    began=time.perf_counter();losses=[]
    with (output/'updates.jsonl').open('x',encoding='utf-8') as stream:
        for step,key in enumerate(manifest['schedule'],1):
            metrics=e.update(scientific,model,optimizer,move(records[tuple(key)],'cuda:0'),'dense')
            stream.write(json.dumps({'step':step,'epoch':(step-1)//460+1,'key':key,**metrics})+'\n')
            losses.append(metrics['loss'])
            if step%50==0:
                stream.flush();print(f'BROAD TRAIN {step}/9200 epoch={(step-1)//460+1} loss={np.mean(losses[-50:]):.5f} '
                                    f'elapsed={time.perf_counter()-began:.1f}s',flush=True)
            if step%920==0:state.save(output/f'checkpoint_step{step:06d}.pt',model,optimizer,bindings,step)
    clocks=[int(float(s['step'])) for s in optimizer.state.values()]
    if not clocks or min(clocks)!=9200 or max(clocks)!=9200:raise ValueError('Adam cursor mismatch')
    write_once(output/'training_complete.json',{'updates':9200,'epochs':20,'config_fingerprint':fingerprint(c),
        'seconds':time.perf_counter()-began,'optimizer_states':len(clocks),'Adam_clock_min':min(clocks),'Adam_clock_max':max(clocks),
        'last50_loss':float(np.mean(losses[-50:])),'checkpoint_sha256':digest(output/'checkpoint_step009200.pt'),
        'training_accuracy_gated':False,'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    # Terminal ONLY image-only inference; no references/targets passed into forward.
    model.eval();before=common.state_digest(model.state_dict());rng=common.nested_digest(common.rng_state())
    pred=output/'terminal_predictions';pred.mkdir();packets=[]
    with torch.inference_mode():
        for i,key in enumerate(manifest['train']+manifest['development']):
            image=data.image(*key,2048)
            votes,support=e.unpack(model(image_batch(image,'cuda:0')),'dense')
            if not np.isfinite(votes).all() or not np.isfinite(support).all():raise ValueError('Nonfinite terminal output')
            path=pred/f'box{i:04d}.npz';np.savez_compressed(path,points=votes,scores=support)
            packets.append({'key':key,'partition':image['metadata']['partition'],'path':path.name,'sha256':digest(path)})
            if (i+1)%25==0:print(f'TERMINAL INFERENCE {i+1}/611',flush=True)
    if before!=common.state_digest(model.state_dict()) or rng!=common.nested_digest(common.rng_state()):
        raise ValueError('Terminal inference mutated model/RNG')
    write_once(pred/'complete.json',{'packets':packets,'config_fingerprint':fingerprint(c),'model_step':9200,
                                   'state_unchanged':True,'checkpoint_sha256':digest(output/'checkpoint_step009200.pt')})
    print('BROAD GPU WORK COMPLETE; terminal checkpoint only',flush=True)


def evaluate(c,scientific,data,manifest,controls,root):
    output=root/'evaluation';output.mkdir(exist_ok=False);(output/'records').mkdir()
    pred=root/'train/terminal_predictions';complete=e.read(pred/'complete.json')
    if complete['config_fingerprint']!=fingerprint(c) or complete['model_step']!=9200 or len(complete['packets'])!=611:
        raise ValueError('Incomplete terminal predictions')
    guides,packets,maxima={},[],{}
    for i,spec in enumerate(complete['packets']):
        key=tuple(spec['key']);path=pred/spec['path']
        if digest(path)!=spec['sha256']:raise ValueError('Terminal packet changed')
        image=data.image(*key,2048)['packet'];e.sato_scores(data,key,image,guides)
        if key[0] not in maxima:maxima[key[0]]=float(guides[key[0]].max())
        with np.load(path,allow_pickle=False) as a:votes,support=a['points'],a['scores']
        indices=consolidate(votes,support,radius=1.,identities=image['flat_indices']);indices=indices[support[indices]>=.25]
        packet=build(votes[indices],guides[key[0]],maxima[key[0]],output/'graphs'/f'box{i:04d}',key,'dense',indices)
        packet['partition']=spec['partition'];packets.append(packet)
        if (i+1)%10==0:print(f'TERMINAL GRAPH {i+1}/611',flush=True)
    write_once(output/'inference_complete.json',{'packets':packets,'references_used_for_graph_inference':False})
    from pn2.evaluation.reference_shortcuts import ReferencePaths
    from pn2.evaluation.node_profile_metrics import ProfileCache
    refs=e.References(scientific,data);profiles=ProfileCache(ROOT,ROOT/scientific['profile_root'],data)
    paths={};probes={};rows=[];point_rows=[];previous=None
    for i,packet in enumerate(packets):
        key=tuple(packet['key']);reference=data.reference(*key)
        if key[0]!=previous:
            refs.cache.clear();paths.clear();probes.clear();previous=key[0]
        row=score(packet,graph_segments(runs_graph(reference['reference_runs'])),data.boxes[key]['box'])
        row['partition']=packet['partition']
        context=refs.get(key[0]);box=data.boxes[key]['box']
        with np.load(Path(packet['folder'])/'graph.npz',allow_pickle=False) as a:
            points=a['xyz']
            point=e.score(data,refs,key,points);point['partition']=packet['partition']
            row['thin_dim_point_support']=e.thin_metrics(points,profiles.load(*key),profiles.manifest['cutoffs'])
            if key[0] not in probes:probes[key[0]]=e.incident_probes(context['morphology'])
            row['incident_point_support']=e.incident_metrics(points,probes[key[0]],box)
            if key[0] not in paths:paths[key[0]]=ReferencePaths(context['morphology'])
            shortcut=paths[key[0]].score(points,a['forest_edges'],np.asarray(box['start_xyz'])-.5,np.asarray(box['stop_xyz'])-.5)
        write_once(output/'records'/f'graph{i:04d}.json',{'row':row,'point_metrics':point,'shortcuts':shortcut})
        row['shortcut_summary']={k:v for k,v in shortcut.items() if k!='edges'}
        rows.append(row);point_rows.append(point)
        if (i+1)%10==0:print(f'TERMINAL SCORE {i+1}/611',flush=True)
    summary={part:summarize([r for r in rows if r['partition']==part]) for part in ['training','development']}
    point_summary={part:e.summarize([r for r in point_rows if r['partition']==part]) for part in ['training','development']}
    decision=gate(summary['development'],controls)
    report={'status':'completed','config_fingerprint':fingerprint(c),'updates':9200,'epochs':20,'training_cases':33,
        'training_boxes':460,'development_boxes':151,'development_neurons':7,'rows':rows,'summary':summary,
        'point_summary':point_summary,'controls':controls,'gate':decision,'training_accuracy_gated':False,
        'terminal_checkpoint_only':True,'head_recipe_changed':False,'whole_image_merge':False,'topology_certified':False,
        'next_decision':'whole-image assembly and intended graph stage' if decision['passed'] else 'stop tuning this head',
        'automatic_new_run':False}
    write_once(output/'report.json',report)
    lines=['# Broad curve head terminal result','','| Partition | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |','|---|---:|---:|---:|---:|---:|---:|']
    for part,s in summary.items():lines.append('| '+part+' | '+' | '.join(f'{100*s["macro"][t][k]:.2f}%' for t in ['2.0','6.0'] for k in ['precision','recall','f1'])+' |')
    lines+=['',json.dumps(decision,indent=2),'','Training is reported, never gated. Terminal checkpoint only. Geometry does not certify topology.',report['next_decision']]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('BROAD TERMINAL VERDICT '+json.dumps(decision),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['prepare','timing','train','score'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.action in ['prepare','score']:os.environ['CUDA_VISIBLE_DEVICES']=''
    c,scientific,data,original,train_keys,dev,controls=control();root=e.output_path(a.root)
    if a.check:print('Authorized broad sources/cohort/controls verified; no action executed');return
    if a.worker:
        if a.action=='prepare':prepare(c,scientific,data,original,train_keys,dev,root/'prepare')
        else:
            manifest=load_prepared(c,root,data)
            if a.action=='timing':timing(c,scientific,data,manifest,root/'timing')
            elif a.action=='train':train(c,scientific,data,manifest,root)
            else:evaluate(c,scientific,data,manifest,controls,root)
        return
    root.mkdir(parents=True,exist_ok=True)
    if a.action in ['train','score']:
        timing_report=e.read(root/'timing/report.json');timing_guard=e.read(root/'timing_guard/report.json')
        if timing_report['status']!='passed' or timing_guard['status']!='completed':raise ValueError('No valid measured timing')
        deadline=timing_guard['started_epoch']+timing_report['wall_seconds_with_50_percent_margin']
        remaining=int(deadline-time.time())
        if remaining<60:raise RuntimeError('Broad overall deadline exhausted; no stage launch')
        seconds=min(3600 if a.action=='score' else 14400,remaining)
    else:seconds=1800 if a.action=='prepare' else 600
    print(f'BROAD ACTION {a.action}; ROOT {root}; cap={seconds}s',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--action',a.action,'--root',str(root),'--worker'],
        root/(a.action+'_guard'),seconds,{'action':a.action,'config_fingerprint':fingerprint(c),'root':str(root)})


if __name__=='__main__':main()
