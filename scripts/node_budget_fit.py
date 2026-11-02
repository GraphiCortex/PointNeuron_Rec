"""Two fresh 20-epoch node fits or explicit stateful recovery; manual GPU launch."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import node_budget_integration_timing as integration
import sampler_node_common as common
import geometric_query_experiment as geometric
import groupnorm_node_experiment as gn
from pn2.training.process_control import launch_worker
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.node_profile_metrics import ProfileCache

ROOT=common.ROOT
CONFIG=ROOT/'configs/node_budget_fit.json'


def load_control(path):
    c=json.loads(Path(path).read_text())
    if (c['schema_version']!=1 or c['budgets']!=[2048,8192] or c['epochs']!=20 or c['updates_per_arm']!=9200
            or c['checkpoint_every_updates']!=460 or c['development_steps']!=[0,2300,4600,9200]
            or c['driver_sha256']!=digest(__file__)):
        raise ValueError('Expected the agreed fixed paired fit and checkpoint policy')
    for name,sha in c['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Fit source changed: {name}')
    ic,recipe,data=integration.load_control(ROOT/c['integration_config'])
    if (c['optimizer']!=recipe['optimizer'] or c['loss']!=recipe['loss']
            or c['gradient_clip']!=recipe['gradient_clip'] or c['confidence_cutoff']!=.5
            or c['query_count']!=512):
        raise ValueError('Loss/optimizer/proposal settings differ from the frozen comparison')
    prior=common.checked_json(c['integrated_timing']['path'],c['integrated_timing']['sha256'])
    if prior['status']!='completed' or any(prior[k]!=v for k,v in integration.identity(ic,recipe).items()):
        raise ValueError('Passed current integrated timing is required')
    profiles=ProfileCache(ROOT,ROOT/c['profile_root'],data)
    if digest(profiles.folder/'manifest.json')!=c['profile_manifest_sha256']:
        raise ValueError('Training-defined profiles changed')
    if profiles.manifest['reference_manifest_sha256']!=c['reference_manifest_sha256']:
        raise ValueError('Profile references differ from the shared target definition')
    ids,schedule=data.schedule()
    if len(schedule)!=9200 or len(ids)!=33:raise ValueError('Shared exposure schedule changed')
    return c,recipe,data,profiles,prior


def identity(c,recipe,data):
    return {'fit_control_fingerprint':fingerprint(c),'model_identity':geometric.identity(recipe),
        'data_manifest_sha256':digest(data.folder/'manifest.json'),'profile_manifest_sha256':c['profile_manifest_sha256'],
        'schedule_sha256':data.manifest['schedule']['sha256'],'driver_sha256':digest(__file__)}


def bindings(c,recipe,data,budget):
    return {'maximum_updates':9200,'budget':budget,**identity(c,recipe,data)}


def screening(training,development):
    # Original prospective milestones, not selected from learned results.
    train=training['summary']['neurons'];dev=development['summary']['partitions']['development']['tolerances']['2.0']
    precision=float(np.mean([n['tolerances']['2.0']['reference_present_only']['precision'] for n in train]))
    recall=float(np.mean([n['tolerances']['2.0']['reference_present_only']['recall'] for n in train]))
    return {'training_reference_present_macro_precision_2':precision,
        'training_reference_present_macro_recall_2':recall,'development_full_grid':dev,
        'broader_training_gate':precision>=.9 and recall>=.9,
        'development_pilot_gate':dev['macro_precision']>=.8 and dev['macro_recall']>=.8 and dev['minimum_neuron_f1']>=.6,
        'meaning':'Prospective engineering screening only; not biological certification, independent test or final SWC quality'}


def resume_payload(path,expected):
    import torch
    path=Path(path).resolve()
    marker=json.loads(path.with_suffix('.complete.json').read_text())
    if marker['checkpoint']!=path.name or marker['sha256']!=digest(path):raise ValueError('Incomplete/changed resume checkpoint')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('bindings')!=expected or payload['step']!=marker['step']
            or not 0<=payload['step']<=9200 or payload['step']%460):
        raise ValueError('Resume requires the same budget/data/profile/code and a completed epoch checkpoint')
    optimizer=payload['optimizer'];parameter_ids=[p for g in optimizer['param_groups'] for p in g['params']]
    clocks=optimizer['state']
    if payload['step']==0:
        if clocks:raise ValueError('Initial checkpoint unexpectedly has Adam history')
    elif (set(clocks)!=set(parameter_ids) or any(float(v['step'])!=payload['step']
            or not all(bool(torch.isfinite(v[k]).all()) for k in ('exp_avg','exp_avg_sq')) for v in clocks.values())):
        raise ValueError('Resume requires complete finite Adam moments and matching update clocks')
    return payload,path


def worker(c,recipe,data,profiles,prior,args):
    from pn2.training.node_runtime import training_batch
    from pn2.training import node_fit_state as state
    from pn2.training.node_fit_eval import evaluate
    for package,version in c['runtime_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError(f'Fit runtime changed: {package}')
    torch=common.runtime(recipe,'cuda:0');folder=Path(args.output);began=time.perf_counter()
    model=geometric.make_model(recipe,'cuda:0');optimizer=common.make_optimizer(model,recipe)
    expected=bindings(c,recipe,data,args.budget);start=0;resume=None
    initial=common.state_digest(model.state_dict());old=next(r for r in prior['arms'] if r['budget']==args.budget)
    if initial!=old['initial_state_sha256']:raise ValueError('Fresh experiment initialization changed')
    if args.checkpoint:
        payload,path=resume_payload(args.checkpoint,expected)
        start=state.restore(payload,model,optimizer,expected)
        resume={'path':str(path),'sha256':digest(path),'step':start};del payload
    ids,schedule=data.schedule();development=[cid for cid,v in data.cases.items() if v['partition']=='development']
    evaluations=[];states=[];prefix_verified=start>=460;torch.cuda.reset_peak_memory_stats()
    write_once(folder/'launch.json',{'budget':args.budget,'dataset':'33 training / seven development; all-grid evaluation',
        'maximum_total_updates':9200,'starting_step':start,'resume':resume,'bindings':expected,
        'device':torch.cuda.get_device_name(0),'runtime_versions':c['runtime_versions']})
    def assess(step,selected,name):
        before=gn.prefix_identity(model,optimizer)
        result=evaluate(model,data,profiles,args.budget,'cuda:0',selected,folder/f'eval_step{step:06d}_{name}',step)
        if gn.prefix_identity(model,optimizer)!=before:raise ValueError('Scoring changed model/Adam/RNG')
        evaluations.append(result);return result
    if start==0:
        checkpoint=folder/'checkpoint_step000000.pt';state.save(checkpoint,model,optimizer,expected,0);states.append(str(checkpoint))
        assess(0,development,'development')
    with (folder/'updates.jsonl').open('x',encoding='utf-8') as log:
        for index in range(start,9200):
            epoch,case_index,tile=map(int,schedule[index]);cid=ids[case_index];step=index+1
            record=data.training(cid,tile,args.budget);batch=training_batch(record,'cuda:0')
            metrics=common.update(model,optimizer,batch,recipe)
            log.write(json.dumps({'step':step,'epoch':epoch,'case_id':cid,'tile_id':tile,**metrics})+'\n');log.flush()
            if step==8:
                if gn.prefix_identity(model,optimizer)!=old['prefix_identity']:
                    raise ValueError('First eight fit updates differ from passed integrated timing')
                prefix_verified=True;print('PREFIX step=8 model/Adam/RNG matches passed integrated timing exactly',flush=True)
            if step%50==0 or step==start+1:
                print(f'UPDATE budget={args.budget} step={step}/9200 epoch={epoch+1}/20 loss={metrics["loss"]:.5f} elapsed={time.perf_counter()-began:.1f}s',flush=True)
            if step%460==0:
                if not prefix_verified:raise ValueError('Unverified training prefix cannot produce epoch checkpoint')
                checkpoint=folder/f'checkpoint_step{step:06d}.pt';state.save(checkpoint,model,optimizer,expected,step);states.append(str(checkpoint))
                print(f'SAVED complete epoch checkpoint step={step}: {checkpoint}',flush=True)
            if step in (2300,4600,9200):assess(step,development,'development')
    if start==9200:terminal_dev=assess(9200,development,'development')
    else:terminal_dev=evaluations[-1]
    terminal_train=assess(9200,ids,'training')
    result={'status':'completed',**identity(c,recipe,data),'budget':args.budget,'total_updates':9200,
        'updates_this_run':9200-start,'starting_step':start,'resume':resume,'experiment_initial_state_sha256':initial,
        'prefix_verified':prefix_verified,'checkpoints':states,'evaluations':evaluations,
        'screening':screening(terminal_train,terminal_dev),'fixed_terminal_checkpoint':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.perf_counter()-began,'development_gradient_updates':0,
        'automatic_extension':False,'final_graph_or_swc_evaluated':False}
    write_once(folder/'worker_report.json',result)
    print(f'COMPLETED fit budget={args.budget} total_updates=9200 screening={result["screening"]}',flush=True)


def paired_delta(arms):
    if {r['budget'] for r in arms}!={2048,8192}:return None
    def dev(arm):return next(e for e in arm['evaluations'] if e['step']==9200 and 'development' in e['summary']['partitions'])
    a=next(r for r in arms if r['budget']==2048);b=next(r for r in arms if r['budget']==8192)
    x=dev(a)['summary']['partitions']['development']['tolerances']['2.0'];y=dev(b)['summary']['partitions']['development']['tolerances']['2.0']
    return {'8192_minus_2048':{k:y[k]-x[k] for k in ('macro_precision','macro_recall','macro_f1','minimum_neuron_f1')},
        'interpretation':'One-seed development comparison of node proposals; no universal winner or final reconstruction claim'}


def controller(c,recipe,data,profiles,prior,args):
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Fit output must stay under PN2/tmp')
    budgets=[args.budget] if args.budget else [2048,8192]
    if args.checkpoint:
        if not args.budget:raise ValueError('Resume needs one explicit budget')
        resume_payload(args.checkpoint,bindings(c,recipe,data,args.budget))
    folder.mkdir(exist_ok=False);write_once(folder/'control.json',c);arms=[];status='completed'
    for budget in budgets:
        dest=folder/f'points{budget}';dest.mkdir()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker','--config',str(Path(args.config).resolve()),
            '--output',str(dest),'--budget',str(budget)]
        if args.checkpoint:command+=['--checkpoint',str(Path(args.checkpoint).resolve())]
        code,worker_status=launch_worker(command,dest,c['worker_timeout_seconds'][str(budget)])
        path=dest/'worker_report.json';report=json.loads(path.read_text()) if path.exists() else {}
        if (worker_status!='completed' or report.get('status')!='completed' or report.get('budget')!=budget
                or report.get('total_updates')!=9200 or not report.get('prefix_verified')
                or any(report.get(k)!=v for k,v in identity(c,recipe,data).items())):
            status='failed';arms.append({'budget':budget,'worker_status':worker_status,'exit_code':code});break
        arms.append(report)
    result={'status':status,**identity(c,recipe,data),'arms':arms,'automatic_retry':False,
        'paired_delta':paired_delta(arms) if status=='completed' else None}
    write_once(folder/'report.json',result)
    print(f'FINISHED paired fit status={status} report={folder/"report.json"}',flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('train','resume','_worker'),required=True)
    parser.add_argument('--config',default=str(CONFIG));parser.add_argument('--output',required=True)
    parser.add_argument('--budget',type=int,choices=(2048,8192));parser.add_argument('--checkpoint');args=parser.parse_args()
    try:
        c,recipe,data,profiles,prior=load_control(args.config)
        if args.action=='resume' and not args.checkpoint:raise ValueError('Resume requires a complete checkpoint')
        if args.action=='train' and args.checkpoint:raise ValueError('Use explicit resume action for saved state')
        if args.action=='_worker':
            if args.budget is None:raise ValueError('Explicit worker budget required')
            worker(c,recipe,data,profiles,prior,args)
        else:raise SystemExit(controller(c,recipe,data,profiles,prior,args))
    except Exception:
        traceback.print_exc();raise SystemExit(1)
