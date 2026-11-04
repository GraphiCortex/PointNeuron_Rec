"""Execution-only broad-run loader repair; original scientific recipe unchanged.

Keep original failed streaming timing and label preparation untouched. Validate
GPU-resident batches against its exact 16-update trace before real training.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import run_broad_curve_head as base
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.training.guarded_run import run

ROOT=base.ROOT
CONFIG=ROOT/'configs/broad_curve_resident_execution_20261105.json'


def control():
    execution=base.e.read(CONFIG)
    for name,sha in {**execution['source_sha256'],**execution['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Changed execution source/evidence: {name}')
    if execution['maximum_wall_seconds']!=21600 or execution['updates']!=9200:
        raise ValueError('Changed execution-only scope')
    return execution,base.control()


def exact_prefix(old,new):
    if old!=new:raise ValueError('Resident loader changed saved streamed loss/gradient prefix')


def cache(scientific,data,manifest):
    cpu=base.cpu_records(scientific,data,manifest,[tuple(k) for k in manifest['train']])
    result={}
    for i,(key,batch) in enumerate(cpu.items(),1):
        result[key]=base.move(batch,'cuda:0')
        if i%25==0:print(f'GPU RESIDENT CACHE {i}/460',flush=True)
    return result


def model_start(scientific,c):
    import torch
    import sampler_node_common as common
    base.e.runtime(scientific,'cuda:0')
    if torch.cuda.get_device_name(0)!=c['device']:raise ValueError('Unexpected GPU')
    model=base.e.model_factory(scientific,'dense','cuda:0')
    optimizer=common.make_optimizer(model,scientific)
    return model,optimizer


def timing(execution,c,scientific,data,manifest,root):
    import torch
    import sampler_node_common as common
    old=base.e.read(root/'timing/report.json')
    output=root/'resident_timing';output.mkdir(exist_ok=False)
    model,optimizer=model_start(scientific,c)
    initial=common.state_digest(model.state_dict())
    if initial!=old['initial_model_sha256']:raise ValueError('Original initializer differs')
    records=cache(scientific,data,manifest)
    before=common.nested_digest(records);times=[];losses=[]
    for i,key in enumerate(manifest['schedule'][:16]):
        torch.cuda.synchronize();start=time.perf_counter()
        metrics=base.e.update(scientific,model,optimizer,records[tuple(key)],'dense')
        torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(metrics)
        exact_prefix(old['losses'][:i+1],losses)
        print(f'RESIDENT TIMING {i+1}/16 seconds={times[-1]:.3f} original_prefix_exact=True',flush=True)
    if before!=common.nested_digest(records):raise ValueError('Model mutated resident input batches')
    seconds=float(np.quantile(times[2:],.9));estimate=seconds*9200
    wall=int(math.ceil(1.5*(estimate+2400)/60)*60)
    report={'status':'passed' if wall<=execution['maximum_wall_seconds'] else 'resource_projection_failed',
        'execution_config_fingerprint':fingerprint(execution),'config_fingerprint':fingerprint(c),
        'initial_model_sha256':initial,'losses':losses,'update_seconds':times,'original_prefix_exact':True,
        'resident_inputs_unchanged':True,'resident_input_sha256':before,'conservative_update_seconds':seconds,
        'training_estimate_seconds':estimate,'terminal_evaluation_reserve_seconds':2400,
        'wall_seconds_with_50_percent_margin':wall,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'throwaway_updates':16,'real_training_updates':0,'timing_model_adopted':False}
    write_once(output/'report.json',report);print('RESIDENT TIMING COMPLETE '+json.dumps(report),flush=True)
    if report['status']!='passed':raise RuntimeError('Measured resource plan exceeds maximum; no training')


def train(execution,c,scientific,data,manifest,root):
    import torch
    import sampler_node_common as common
    from pn2.training import node_fit_state as state
    from pn2.training.node_runtime import image_batch
    timing=base.e.read(root/'resident_timing/report.json')
    if timing['status']!='passed' or timing['execution_config_fingerprint']!=fingerprint(execution):raise ValueError('No valid resident timing')
    output=root/'train';output.mkdir(exist_ok=False)
    model,optimizer=model_start(scientific,c)
    if common.state_digest(model.state_dict())!=timing['initial_model_sha256'] or optimizer.state:
        raise ValueError('Fresh initialization/Adam mismatch')
    records=cache(scientific,data,manifest);input_sha=common.nested_digest(records)
    if input_sha!=timing['resident_input_sha256']:raise ValueError('Resident input cache changed')
    bindings={'maximum_updates':9200,'config_fingerprint':fingerprint(c),'execution_config_fingerprint':fingerprint(execution),
        'scientific_config_fingerprint':fingerprint(scientific),'preparation_sha256':digest(root/'prepare/manifest.json'),
        'schedule_sha256':manifest['schedule_sha256'],'arm':'dense','rung':'full460'}
    state.save(output/'checkpoint_step000000.pt',model,optimizer,bindings,0)
    began=time.perf_counter();losses=[];prefix=[]
    with (output/'updates.jsonl').open('x',encoding='utf-8') as stream:
        for step,key in enumerate(manifest['schedule'],1):
            metrics=base.e.update(scientific,model,optimizer,records[tuple(key)],'dense')
            if step<=16:
                prefix.append(metrics);exact_prefix(timing['losses'][:step],prefix)
            stream.write(json.dumps({'step':step,'epoch':(step-1)//460+1,'key':key,**metrics})+'\n');losses.append(metrics['loss'])
            if step%50==0:
                stream.flush();print(f'BROAD TRAIN {step}/9200 epoch={(step-1)//460+1} loss={np.mean(losses[-50:]):.5f} elapsed={time.perf_counter()-began:.1f}s',flush=True)
            if step%920==0:state.save(output/f'checkpoint_step{step:06d}.pt',model,optimizer,bindings,step)
    if input_sha!=common.nested_digest(records):raise ValueError('Training mutated resident input batches')
    clocks=[int(float(s['step'])) for s in optimizer.state.values()]
    if not clocks or min(clocks)!=9200 or max(clocks)!=9200:raise ValueError('Adam cursor mismatch')
    write_once(output/'training_complete.json',{'updates':9200,'epochs':20,'config_fingerprint':fingerprint(c),
        'execution_config_fingerprint':fingerprint(execution),'seconds':time.perf_counter()-began,
        'optimizer_states':len(clocks),'Adam_clock_min':min(clocks),'Adam_clock_max':max(clocks),
        'last50_loss':float(np.mean(losses[-50:])),'checkpoint_sha256':digest(output/'checkpoint_step009200.pt'),
        'training_accuracy_gated':False,'resident_inputs_unchanged':True,'original_streamed_prefix_exact':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    model.eval();before=common.state_digest(model.state_dict());rng=common.nested_digest(common.rng_state())
    pred=output/'terminal_predictions';pred.mkdir();packets=[]
    with torch.inference_mode():
        for i,key in enumerate(manifest['train']+manifest['development']):
            image=data.image(*key,2048);votes,support=base.e.unpack(model(image_batch(image,'cuda:0')),'dense')
            if not np.isfinite(votes).all() or not np.isfinite(support).all():raise ValueError('Nonfinite terminal output')
            path=pred/f'box{i:04d}.npz';np.savez_compressed(path,points=votes,scores=support)
            packets.append({'key':key,'partition':image['metadata']['partition'],'path':path.name,'sha256':digest(path)})
            if (i+1)%25==0:print(f'TERMINAL INFERENCE {i+1}/611',flush=True)
    if before!=common.state_digest(model.state_dict()) or rng!=common.nested_digest(common.rng_state()):raise ValueError('Inference mutated model/RNG')
    write_once(pred/'complete.json',{'packets':packets,'config_fingerprint':fingerprint(c),'model_step':9200,'state_unchanged':True,
        'execution_config_fingerprint':fingerprint(execution),'checkpoint_sha256':digest(output/'checkpoint_step009200.pt')})
    print('BROAD GPU WORK COMPLETE; terminal checkpoint only',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['timing','train','score'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.action=='score':os.environ['CUDA_VISIBLE_DEVICES']=''
    execution,(c,scientific,data,_,_,_,controls)=control();root=base.e.output_path(a.root)
    manifest=base.load_prepared(c,root,data)
    if a.check:print('Resident execution sources/base science/preparation verified; no GPU action');return
    if a.worker:
        if a.action=='timing':timing(execution,c,scientific,data,manifest,root)
        elif a.action=='train':train(execution,c,scientific,data,manifest,root)
        else:base.evaluate(c,scientific,data,manifest,controls,root)
        return
    if a.action=='timing':seconds=600
    else:
        timing_report=base.e.read(root/'resident_timing/report.json');guard=base.e.read(root/'resident_timing_guard/report.json')
        if timing_report['status']!='passed' or guard['status']!='completed':raise ValueError('No completed resource timing')
        remaining=int(guard['started_epoch']+timing_report['wall_seconds_with_50_percent_margin']-time.time())
        if remaining<60:raise RuntimeError('Overall measured deadline exhausted')
        seconds=min(3600,remaining) if a.action=='score' else remaining
    print(f'BROAD RESIDENT ACTION {a.action}; ROOT {root}; cap={seconds}s',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--action',a.action,'--root',str(root),'--worker'],
        root/('resident_'+a.action+'_guard'),seconds,{'action':a.action,'execution_config_fingerprint':fingerprint(execution)})


if __name__=='__main__':main()
