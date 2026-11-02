"""Manual integrated GPU timing through the real loader and all development boxes.

Eight throwaway training updates per budget; no fit/resume action or retained
trained checkpoint. Core reference metrics are execution diagnostics only.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import geometric_query_experiment as geometric
import sampler_node_common as common
import groupnorm_node_experiment as gn
from sampler_node_experiment import launch_worker
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.evaluation.node_cohort import score_box,summarize

ROOT=common.ROOT
CONFIG=ROOT/'configs/node_budget_integration_timing.json'


def load_control(path):
    c=json.loads(Path(path).read_text())
    if (c['schema_version']!=1 or c['budgets']!=[2048,8192] or c['updates']!=8
            or c['driver_sha256']!=common.digest(__file__)):
        raise ValueError('Expected the declared integrated resource diagnostic')
    for name,sha in c['source_sha256'].items():
        if common.digest(ROOT/name)!=sha:raise ValueError(f'Integration source changed: {name}')
    recipe=geometric.load_recipe(ROOT/c['model_config'])
    geometric.require_cpu(recipe)
    if common.fingerprint(recipe)!=c['model_recipe_fingerprint']:raise ValueError('Model recipe changed')
    prior=common.checked_json(c['resource_timing']['path'],c['resource_timing']['sha256'])
    if (prior['status']!='completed' or [r['budget'] for r in prior['arms']]!=[2048,8192]
            or prior['model_identity']!=geometric.identity(recipe)):
        raise ValueError('The passed two-budget resource timing is required')
    proof=common.checked_json(c['cpu_integration']['path'],c['cpu_integration']['sha256'])
    data=NodeBudgetDataset(ROOT,ROOT/c['data_root']);data.schedule()
    if (common.digest(data.folder/'manifest.json')!=c['data_manifest_sha256'] or proof['status']!='passed'
            or proof['data_manifest_sha256']!=c['data_manifest_sha256']
            or proof['real_data_optimizer_updates']!=0 or not proof['next_synthetic_update_model_adam_rng_exact']):
        raise ValueError('Real paired data/model CPU proof is missing or stale')
    for name,sha in proof['source_sha256'].items():
        if common.digest(ROOT/name)!=sha:raise ValueError('CPU integration source changed')
    verify=common.checked_json(c['data_verification']['path'],c['data_verification']['sha256'])
    if verify['status']!='passed' or verify['manifest_sha256']!=c['data_manifest_sha256']:
        raise ValueError('Complete paired-data verification is required')
    return c,recipe,data


def identity(control,recipe):
    return {'control_fingerprint':common.fingerprint(control),'model_identity':geometric.identity(recipe),
            'data_manifest_sha256':control['data_manifest_sha256'],'driver_sha256':common.digest(__file__)}


def evaluate(model,data,budget,device,folder):
    import torch
    from pn2.training.node_runtime import image_batch
    model.eval();rows=[];arrays={};inference_seconds=0.;began=time.perf_counter();n=0
    selected=sorted(cid for cid,c in data.cases.items() if c['partition']=='development')
    sync=lambda:torch.cuda.synchronize() if str(device).startswith('cuda') else None
    with torch.inference_mode():
        for ci,cid in enumerate(selected):
            for b in data.cases[cid]['boxes']:
                image=data.image(cid,b['tile_id'],budget);batch=image_batch(image,device)
                sync();one=time.perf_counter();output=model(batch);sync();inference_seconds+=time.perf_counter()-one
                centers=output.center_proposals[0].detach().cpu().numpy()
                scores=output.objectness_logits[0].softmax(-1)[:,1].detach().cpu().numpy()
                if centers.shape!=(512,3):raise ValueError('Expected all 512 proposals before fixed-cutoff scoring')
                # Reference files are opened only after the model forward.
                reference=data.reference(cid,b['tile_id'])
                rows.append(score_box(cid,b['tile_id'],centers,scores,reference))
                arrays[f'box{n}_centers']=centers;arrays[f'box{n}_scores']=scores
                arrays[f'box{n}_identity']=np.asarray([ci,b['tile_id']],dtype=np.int32);n+=1
                if n%16==0:print(f'DEVELOPMENT timing points={budget} boxes={n}/415 elapsed={time.perf_counter()-began:.1f}s',flush=True)
    summary=summarize(data,rows,selected)
    summary['case_order']=selected
    common.write(folder/'core_metrics.json',{'scope':'eight-update execution diagnostic, not development-trained quality evidence',
        'summary':summary,'boxes':rows,'apparent_width_contrast_strata':'separate evaluation cache not yet implemented'})
    np.savez_compressed(folder/'predictions.npz',**arrays)
    return {'boxes':n,'neurons':len(selected),'inference_seconds':inference_seconds,
        'total_evaluation_seconds':time.perf_counter()-began,'summary_sha256':common.digest(folder/'core_metrics.json'),
        'predictions_sha256':common.digest(folder/'predictions.npz')}


def worker(control,recipe,data,args):
    import torch
    from pn2.training.node_runtime import training_batch
    from pn2.training import node_fit_state as state
    torch=common.runtime(recipe,'cuda:0');folder=Path(args.output)
    model=geometric.make_model(recipe,'cuda:0');optimizer=common.make_optimizer(model,recipe)
    initial=common.state_digest(model.state_dict());case_ids,schedule=data.schedule();durations=[]
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    with (folder/'updates.jsonl').open('x',encoding='utf-8') as log:
        for step,row in enumerate(schedule[:8]):
            epoch,case_index,tile=map(int,row);cid=case_ids[case_index]
            torch.cuda.synchronize();one=time.perf_counter()
            record=data.training(cid,tile,args.budget);batch=training_batch(record,'cuda:0')
            metrics=common.update(model,optimizer,batch,recipe)
            torch.cuda.synchronize();seconds=time.perf_counter()-one;durations.append(seconds)
            log.write(json.dumps({'step':step+1,'case_id':cid,'tile_id':tile,'seconds':seconds,**metrics})+'\n');log.flush()
            print(f'THROWAWAY integrated points={args.budget} update={step+1}/8 targets={len(record["target_xyz"])} seconds={seconds:.3f}',flush=True)
    prefix=gn.prefix_identity(model,optimizer)
    evaluation=evaluate(model,data,args.budget,'cuda:0',folder)
    if prefix!=gn.prefix_identity(model,optimizer):raise ValueError('Development evaluation changed model/Adam/RNG')
    binding={'maximum_updates':9200,'budget':args.budget,'data_manifest_sha256':control['data_manifest_sha256'],
        'schedule_sha256':data.manifest['schedule']['sha256'],'model_identity':geometric.identity(recipe)}
    stream=io.BytesIO();torch.save(state.capture(model,optimizer,binding,8),stream)
    restored=geometric.make_model(recipe,'cuda:0');restored_opt=common.make_optimizer(restored,recipe)
    stream.seek(0);payload=torch.load(stream,map_location='cpu',weights_only=False)
    restored_step=state.restore(payload,restored,restored_opt,binding)
    if restored_step!=8 or prefix!=gn.prefix_identity(restored,restored_opt):
        raise ValueError('Integrated model/Adam/RNG/cursor reload failed')
    next_row=list(map(int,schedule[restored_step]))
    result={'status':'completed',**identity(control,recipe),'budget':args.budget,'updates':8,
        'initial_state_sha256':initial,'update_seconds_including_loader_and_transfer':durations,
        'mean_last_four_update_seconds':float(np.mean(durations[4:])),
        'prefix_identity':prefix,'next_schedule_row':next_row,'evaluation':evaluation,
        'evaluation_preserved_model_adam_rng':True,'reload_model_adam_rng_cursor_exact':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.perf_counter()-began,'trained_checkpoint_retained':False,
        'quality_gate':None,'profile_metrics_included':False,
        'projection_update_hours':float(np.mean(durations[4:]))*9200/3600,
        'projection_four_development_evaluations_hours':evaluation['total_evaluation_seconds']*4/3600,
        'limitation':'Eight training boxes and one all-grid development pass; terminal training scoring, profile metrics, checkpoint IO and sustained performance remain unmeasured'}
    common.write(folder/'worker_report.json',result)
    print(f'COMPLETED integrated timing points={args.budget} development={evaluation["boxes"]} boxes/{evaluation["neurons"]} neurons reload=True',flush=True)


def controller(control,recipe,data,args):
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Output must stay under PN2/tmp')
    folder.mkdir(exist_ok=False);arms=[];status='completed'
    common.write(folder/'control.json',control)
    for budget in control['budgets']:
        dest=folder/f'points{budget}';dest.mkdir()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker',
            '--config',str(Path(args.config).resolve()),'--output',str(dest),'--budget',str(budget)]
        code,worker_status=launch_worker(command,dest,control['worker_timeout_seconds'][str(budget)])
        path=dest/'worker_report.json';r=json.loads(path.read_text()) if path.exists() else {}
        if (worker_status!='completed' or r.get('status')!='completed' or r.get('budget')!=budget or r.get('updates')!=8
                or r.get('evaluation',{}).get('boxes')!=415 or r.get('evaluation',{}).get('neurons')!=7
                or not r.get('reload_model_adam_rng_cursor_exact')
                or not r.get('evaluation_preserved_model_adam_rng')
                or any(r.get(k)!=v for k,v in identity(control,recipe).items())):
            status='failed';arms.append({'budget':budget,'worker_status':worker_status,'exit_code':code});break
        arms.append(r)
    if status=='completed' and arms[0]['initial_state_sha256']!=arms[1]['initial_state_sha256']:
        status='failed_initialization_mismatch'
    common.write(folder/'report.json',{'status':status,**identity(control,recipe),'arms':arms,
        'automatic_retry':False,'training_action_exists':False,'learned_quality_comparison':False})
    print(f'FINISHED integrated timing status={status} report={folder/"report.json"}',flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('timing','_worker'),default='timing')
    parser.add_argument('--config',default=str(CONFIG));parser.add_argument('--output',required=True)
    parser.add_argument('--budget',type=int,choices=(2048,8192));args=parser.parse_args()
    try:
        c,recipe,data=load_control(args.config)
        if args.action=='_worker':
            if args.budget is None:raise ValueError('Explicit worker budget required')
            worker(c,recipe,data,args)
        else:raise SystemExit(controller(c,recipe,data,args))
    except Exception:
        traceback.print_exc();raise SystemExit(1)
