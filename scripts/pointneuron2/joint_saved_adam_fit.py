"""Bounded retained joint fit: saved proposal Adam state, fresh GAT Adam state."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_timing as timing
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng,restore_rng
from pointneuron.graph.proposal_graph import CandidateGraph
from pointneuron.graph.connectivity_decoder import decode_forest

runner=timing.runner
CONFIG=ROOT/'configs/pointneuron2/joint_saved_adam_fit.json'
SCHEMA='pointneuron2-joint-saved-adam-fit-v1'
LEGACY_CONTROLLER_SHA256='2e9030270ec558846c7ec328d1912c5762b28c7e0d51264fd943212f78082c66'
CHECK_DIRECTORY='check_io_repair_v1'


def write(path,value):
    """Keep atomic JSON replacement; retry only bounded Windows sharing/access errors."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    delays=(.05,.1,.2,.4,.8)
    for attempt in range(len(delays)+1):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==len(delays):raise
            time.sleep(delays[attempt])


def source_transition(saved,current):
    """Permit only the documented legacy controller -> I/O repair transition."""
    if saved==current:return None
    controller=str(Path(__file__).resolve())
    old=saved.get('sources',{});new=current.get('sources',{})
    if (old.get(controller)!=LEGACY_CONTROLLER_SHA256 or new.get(controller)!=runner.base.file_digest(Path(controller))
        or set(old)!=set(new) or {k:v for k,v in old.items() if k!=controller}!={k:v for k,v in new.items() if k!=controller}
        or {k:v for k,v in saved.items() if k!='sources'}!={k:v for k,v in current.items() if k!='sources'}):
        raise ValueError('Checkpoint experiment/model/backend/source differs beyond approved I/O repair')
    return {'policy':'legacy_controller_to_windows_json_retry_v1','path':controller,
        'before_sha256':old[controller],'after_sha256':new[controller]}


def validate_config(config):
    if (config['schema']!=SCHEMA or config['updates']!=4000 or config['snapshots']!=[0,500,1000,2000,4000]
        or config['prefix_replay_update']!=8 or config['proposal_initial_cursor']!=4000 or config['gat_initial_cursor']!=0
        or config['training_crops']!=[[0,0],[0,1],[92,4],[86,3]] or config['seed']!=0 or config['torch_threads']!=2
        or config['automatic_extension'] is not False):raise ValueError('Different agreed capped fit contract')


def validate_optimizer(model,optimizer,step,recipe):
    if type(step) is not int or not 0<=step<=4000:raise ValueError('Invalid new-stage cursor')
    parameters=list(model.parameters());proposal=list(model.proposal_model.parameters())
    if [id(p) for p in parameters[:len(proposal)]]!=[id(p) for p in proposal]:raise ValueError('Different parameter order')
    expected=set(parameters if step else proposal)
    if set(optimizer.state)!=expected or timing.stability.cursors(model,optimizer)!=timing.expected_cursors(step):
        raise ValueError('Incomplete/mismatched mixed optimizer state')
    wanted=runner.optimizer_for(model,recipe).state_dict()['param_groups']
    if optimizer.state_dict()['param_groups']!=wanted:raise ValueError('Optimizer options/order differ')
    for p,state in optimizer.state.items():
        if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Different Adam state layout')
        for name in ('exp_avg','exp_avg_sq'):
            v=state[name]
            if v.shape!=p.shape or v.dtype!=p.dtype or v.device!=p.device or not bool(torch.isfinite(v).all()):
                raise ValueError('Invalid/nonfinite optimizer moments')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative Adam second moment')


def checkpoint(model,optimizer,step,config,recipe,provenance,*,purpose='fit',prefix_verified=False):
    validate_config(config);validate_optimizer(model,optimizer,step,recipe)
    if purpose not in ('fit','check') or (step>8 and not prefix_verified) or (step<8 and prefix_verified):raise ValueError('Invalid checkpoint purpose/prefix proof')
    return {'schema':SCHEMA,'purpose':purpose,'resumable':purpose=='fit' and step<4000,'stage_step':step,'next_crop':step%4,
        'prefix_verified':bool(prefix_verified),'config':config,'training_recipe':recipe,'provenance':provenance,
        'parameter_names':[n for n,_ in model.named_parameters()],
        'proposal_spec':model.proposal_model.model_spec(),'graph_spec':model.graph_model.model_spec(),
        'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}


def restore(cp,model,config,recipe,provenance,*,resume):
    validate_config(config);step=cp.get('stage_step')
    source_transition(cp.get('provenance',{}),provenance)
    if (cp.get('schema')!=SCHEMA or cp.get('config')!=config or cp.get('training_recipe')!=recipe
        or type(step) is not int or not 0<=step<=4000 or cp.get('next_crop')!=step%4
        or cp.get('purpose') not in ('fit','check') or cp.get('resumable')!=(cp['purpose']=='fit' and step<4000)
        or type(cp.get('prefix_verified')) is not bool or (step>8 and not cp['prefix_verified']) or (step<8 and cp['prefix_verified'])
        or cp.get('parameter_names')!=[n for n,_ in model.named_parameters()]
        or cp.get('proposal_spec')!=model.proposal_model.model_spec() or cp.get('graph_spec')!=model.graph_model.model_spec()):
        raise ValueError('Checkpoint experiment/model/backend/cursor differs')
    if resume and (cp['purpose']!='fit' or not cp['resumable']):raise ValueError('Terminal/check-only checkpoint cannot resume')
    model.load_state_dict(cp['model'])
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise ValueError('Nonfinite saved weights')
    optimizer=runner.optimizer_for(model,recipe);optimizer.load_state_dict(cp['optimizer'])
    validate_optimizer(model,optimizer,step,recipe);restore_rng(cp['rng'])
    return optimizer,step,cp['prefix_verified']


def save_state(model,optimizer,step,config,recipe,provenance,report,directory):
    path=directory/f'step_{step:04d}.pt';temporary=path.with_suffix('.pt.writing')
    if path.exists() or temporary.exists():raise FileExistsError('Preserve checkpoint evidence')
    torch.save(checkpoint(model,optimizer,step,config,recipe,provenance,prefix_verified=report['prefix_verified']),temporary)
    temporary.replace(path)
    report.update(last_checkpoint=str(path),last_checkpoint_step=step,last_checkpoint_sha256=runner.base.file_digest(path))
    write(directory/'report.json',report)
    return path


def snapshot(model,optimizer,step,config,recipe,provenance,report,directory,records,oracle,original,device):
    save_state(model,optimizer,step,config,recipe,provenance,report,directory)
    # Fallible graph/geometry/SWC evaluation happens only after complete state is safe.
    report['evaluations'].append(runner.evaluate(model,records,oracle,original,recipe,device,step,directory))
    write(directory/'report.json',report)


def compare_prefix(directory,timed_directory,records,original):
    maxima={};audit=runner.audit
    for i,record in enumerate(records):
        with np.load(directory/f'step_0008_crop_{i}.npz') as a,np.load(timed_directory/f'step_0008_crop_{i}.npz') as b:
            actual={k:a[k].copy() for k in a.files};expected={k:b[k].copy() for k in b.files}
        for key in ('centers','radii','logits','decoder_features','edge_logits','retained_edge_logits'):
            torch.testing.assert_close(torch.from_numpy(actual[key]),torch.from_numpy(expected[key]),rtol=1e-5,atol=1e-5)
            maxima[key]=max(maxima.get(key,0.),float(np.abs(actual[key]-expected[key]).max(initial=0.)))
        for key in ('candidate_edges','edge_labels','pred_indices','target_indices','retained_node_ids','retained_edges','evaluation_labels'):
            if not np.array_equal(actual[key],expected[key]):raise ValueError(f'Timing-prefix discrete mapping differs: {key}')
        if not np.array_equal(actual['edge_logits']>=0,expected['edge_logits']>=0):raise ValueError('Timing-prefix edge cutoff differs')
        graphs=[]
        for saved in (actual,expected):
            p=audit.make_packet(saved,saved['retained_node_ids'],record['meta']['patch_center'],original['patch_radius'])
            c=CandidateGraph(torch.from_numpy(saved['retained_edges']),16,8.)
            graphs.append(decode_forest(p,c,torch.from_numpy(saved['retained_edge_logits'])))
        if graphs[0]!=graphs[1]:raise ValueError('Timing-prefix forest decisions differ')
    return {'passed':True,'maximum_float_deltas':maxima,'discrete_and_forest_decisions_identical':True,'rtol':1e-5,'atol':1e-5}


def prefix_check(model,optimizer,config,recipe,provenance,report,directory,records,oracle,original,device,timed_directory):
    save_state(model,optimizer,8,config,recipe,provenance,report,directory)
    target=directory/'prefix_replay';target.mkdir()
    report['prefix_evaluation']=runner.evaluate(model,records,oracle,original,recipe,device,8,target)
    report['prefix_replay']=compare_prefix(target,timed_directory,records,original)
    report['prefix_verified']=True;write(directory/'report.json',report)
    print(f"PREFIX stage=8 matched passed GPU timing; deltas={report['prefix_replay']['maximum_float_deltas']}",flush=True)


def zero_update_check(model,optimizer,config,recipe,provenance,report,directory,records,oracle,original):
    before=runner.state_hashes(model);moments=timing.stability.control.optimizer_digests(optimizer)
    report['evaluations'].append(runner.evaluate(model,records,oracle,original,recipe,'cpu',0,directory))
    predictions=runner.prediction_signature(model,records,original,'cpu')
    path=directory/'check_only_reload.pt'
    try:
        torch.save(checkpoint(model,optimizer,0,config,recipe,provenance,purpose='check'),path)
        reloaded,step,verified=restore(torch.load(path,map_location='cpu',weights_only=True),model,config,recipe,provenance,resume=False)
        if step!=0 or verified or before!=runner.state_hashes(model) or moments!=timing.stability.control.optimizer_digests(reloaded):
            raise ValueError('Check reload changed weights/optimizer/cursors')
        runner.require_identical_predictions(predictions,runner.prediction_signature(model,records,original,'cpu'))
    finally:path.unlink(missing_ok=True)
    report.update(cpu_check_passed=True,optimizer_updates=0,weights_and_optimizer_unchanged=True,reload_prediction_max_delta=0.)


def load_inputs(config):
    recipe=runner.recipe_at(ROOT/config['source_recipe']);original,records,oracle,protected=runner.load_inputs(recipe)
    summary_path=ROOT/config['timing_review_summary'];summary=runner.base.read(summary_path)
    timed_path=ROOT/summary['gpu_report'];review_path=ROOT/summary['review_report']
    if (runner.base.file_digest(timed_path)!=summary['gpu_report_sha256'] or runner.base.file_digest(review_path)!=summary['review_report_sha256']
        or not summary['timing_readiness_gate'] or summary['independent_saved_output_verification']['pairs']!=8):
        raise ValueError('Different reviewed GPU readiness result')
    timed=runner.base.read(timed_path)
    if timed['status']!='completed' or not timed['timing_readiness_gate'] or timed['completed_updates']!=8:raise ValueError('Timing not ready')
    protected.update(timed['protected_files_sha256']);protected.update(timed['provenance']['sources'])
    for path in (summary_path,timed_path,review_path):protected[str(path)]=runner.base.file_digest(path)
    for evaluation in timed['evaluations']:
        for i,row in enumerate(evaluation['crops']):
            protected[str(timed_path.parent/f"step_{evaluation['stage_step']:04d}_crop_{i}.npz")]=row['prediction_sha256']
    for name in ('learning_rate','weight_decay','gradient_clip'):
        if config[name]!=recipe[name] or config[name]!=timed['config'][name]:raise ValueError('Changed update settings')
    runner.base.verify_hashes(protected)
    return recipe,original,records,oracle,protected,timed,timed_path


def run(args):
    config=runner.base.read(CONFIG);validate_config(config)
    if args.action!='resume' and args.checkpoint is not None:raise ValueError('Checkpoint only applies to explicit resume')
    torch.set_num_threads(2);runner.configure();device=torch.device('cpu' if args.action=='check' else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
    recipe,original,records,oracle,protected,timed,timed_path=load_inputs(config)
    root=(args.directory or ROOT/config['output']).resolve()
    if not root.is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Outputs belong in ignored tmp')
    sources=runner.base.source_hashes(SimpleNamespace(recipe=CONFIG))
    launcher=ROOT/'scripts/pointneuron2/run_joint_saved_adam_fit.ps1';sources[str(launcher)]=runner.base.file_digest(launcher)
    provenance={'sources':sources,'timing_report_sha256':runner.base.file_digest(timed_path),'backend':runner.backend(device),'experiment_root':str(root)}
    if device.type=='cuda' and provenance['backend']!=timed['provenance']['backend']:raise ValueError('Backend differs from passed GPU timing')
    if args.action!='check':
        checked=runner.base.read(root/CHECK_DIRECTORY/'report.json')
        if (checked['status']!='completed' or not checked['cpu_check_passed'] or checked['config']!=config
            or checked['provenance']['sources']!=sources or checked['provenance']['timing_report_sha256']!=provenance['timing_report_sha256']):
            raise ValueError('Matching completed CPU fit-state check required')
    directory=root/(CHECK_DIRECTORY if args.action=='check' else 'model')
    if args.action=='resume':
        if args.checkpoint is None:raise ValueError('Resume needs an explicit intermediate checkpoint')
        args.checkpoint=args.checkpoint.resolve()
        if (not args.checkpoint.is_relative_to(root) or args.checkpoint.suffix!='.pt'
            or not (args.checkpoint.parent.name=='model' or args.checkpoint.parent.name.startswith('resume_'))):raise ValueError('Checkpoint outside this fit')
        directory=root/f'resume_{args.checkpoint.stem}';number=2
        while directory.exists():directory=root/f'resume_{args.checkpoint.stem}_{number}';number+=1
    if directory.exists():raise FileExistsError('Preserve existing action output')
    directory.mkdir(parents=True);started=time.perf_counter()
    report={'schema':SCHEMA,'status':'running','action':args.action,'config':config,'provenance':provenance,'completed_updates':0,
        'prefix_verified':False,'evaluations':[],'scope':'four fixed TRAINING crops; automatic joint reconstruction; frozen Phase 1',
        'io_policy':{'windows_replace_retries':5,'total_backoff_seconds':1.55,'checkpoint_migration':'only pinned legacy controller; all other sources/backend/config identical'}}
    write(directory/'report.json',report)
    try:
        model=runner.warm_start(recipe,device)
        cp=torch.load(ROOT/recipe['proposal_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
        optimizer=timing.stability.mixed_optimizer(model,cp['optimizer'],recipe);start=0
        validate_optimizer(model,optimizer,0,recipe)
        if args.action=='resume':
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            report['source_transition']=source_transition(saved.get('provenance',{}),provenance)
            optimizer,start,report['prefix_verified']=restore(saved,model,config,recipe,provenance,resume=True)
            report.update(completed_updates=start,resumed_from=str(args.checkpoint),source_checkpoint_sha256=runner.base.file_digest(args.checkpoint))
        print(f"ACTION={args.action} device={provenance['backend']['name']} stage={start}/4000 | saved proposal Adam + fresh-stage GAT Adam | automatic gates fixed",flush=True)
        if args.action=='check':zero_update_check(model,optimizer,config,recipe,provenance,report,directory,records,oracle,original)
        else:
            if start in config['snapshots']:
                snapshot(model,optimizer,start,config,recipe,provenance,report,directory,records,oracle,original,device)
                if start==0:report['historical_warm_replay']=runner.historical_warm_replay(directory,recipe,device)
            if start==8 and not report['prefix_verified']:
                prefix_check(model,optimizer,config,recipe,provenance,report,directory,records,oracle,original,device,timed_path.parent)
            report.setdefault('clipped_updates_this_invocation',0);report.setdefault('maximum_preclip_norm_this_invocation',0.)
            for step in range(start+1,4001):
                attempt={'stage_update':step,'crop':config['training_crops'][(step-1)%4],'optimizer_step_returned':False}
                report['attempted_update']=step
                persist=lambda:write(directory/'active_update.json',attempt)
                timing.instrumented_update(model,optimizer,records[(step-1)%4],original,recipe,device,attempt,persist,directory)
                report['completed_updates']=step;attempt['cursors_after']=timing.stability.cursors(model,optimizer)
                if attempt['cursors_after']!=timing.expected_cursors(step):raise ValueError('Mixed update cursor differs')
                persist()
                compact={k:attempt[k] for k in ('stage_update','crop','proposal_gradient_l2','gat_gradient_l2','preclip_norm','cursors_after')}
                compact['incoming']={k:v for k,v in attempt['incoming'].items() if k not in ('matched_centers','matched_distances','matched_target_indices','matched_query_ids')}
                with (directory/'updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(compact,allow_nan=False)+'\n')
                report['clipped_updates_this_invocation']+=attempt['preclip_norm']>config['gradient_clip']
                report['maximum_preclip_norm_this_invocation']=max(report['maximum_preclip_norm_this_invocation'],attempt['preclip_norm'])
                if step%config['log_every']==0:
                    x=attempt['incoming'];print(f"UPDATE stage={step}/4000 crop={attempt['crop']} loss={x['loss']:.5f} proposal={x['proposal_loss']:.5f} edge={x['edge_loss']:.5f} trusted={x['edge_supervision']['trusted_matches']}/{x['edge_supervision']['targets']} pos/neg={x['edge_supervision']['positive']}/{x['edge_supervision']['negative']} candidates={x['candidate_edges']} elapsed={time.perf_counter()-started:.1f}s",flush=True)
                    write(directory/'report.json',report)
                if step==8:
                    prefix_check(model,optimizer,config,recipe,provenance,report,directory,records,oracle,original,device,timed_path.parent)
                if step in config['snapshots']:
                    snapshot(model,optimizer,step,config,recipe,provenance,report,directory,records,oracle,original,device)
            predictions=runner.prediction_signature(model,records,original,device);before=runner.state_hashes(model)
            moments=timing.stability.control.optimizer_digests(optimizer)
            loaded,stage,_=restore(torch.load(directory/'step_4000.pt',map_location='cpu',weights_only=True),model,config,recipe,provenance,resume=False)
            if stage!=4000 or before!=runner.state_hashes(model) or moments!=timing.stability.control.optimizer_digests(loaded):raise ValueError('Terminal reload state differs')
            runner.require_identical_predictions(predictions,runner.prediction_signature(model,records,original,device))
            report.update(automatic_gate=report['evaluations'][-1]['automatic_gate'],terminal_reload_prediction_delta=0.,
                terminal_cursors=timing.stability.cursors(model,loaded),next_step='Review terminal automatic gates. Stop at cap; no extension, threshold change or production promotion.')
        runner.base.verify_hashes(protected);runner.base.verify_hashes(sources)
        report.update(status='completed',original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);print(f"COMPLETED {args.action}; stage={report['completed_updates']} automatic_gate={report.get('automatic_gate')} report={directory/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);(directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','train','resume'),required=True)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    run(parser.parse_args())
