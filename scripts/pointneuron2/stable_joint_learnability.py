"""One capped stable-LR joint fit, importing historical stage 1000 and ending at 4000."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_partial_stability as control
from pointneuron.models.partial_joint_loss import POLICY,partial_joint_loss
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng,restore_rng

runner=control.runner
timing=control.timing
base=control.base
write=control.write
SCHEMA='pointneuron2-stable-joint-learnability-v1'
CONFIG=ROOT/'configs/pointneuron2/stable_joint_learnability.json'


def validate_config(c):
    fixed={'schema':SCHEMA,'policy':POLICY,'parent_stage':1000,'updates':3000,'terminal_historical_stage':4000,
        'learning_rate':1e-5,'snapshots':[0,64,500,1000,2000,3000],'prefix_updates':64,'early_probe_every':4,
        'later_probe_every':20,'geometry_stop_mean_distance':4.,'geometry_stop_consecutive_probes':3,
        'stability_fraction':.9,'stability_window':8,'torch_threads':2,'log_every':50,'automatic_extension':False,
        'training_crops':[[0,0],[0,1],[92,4],[86,3]]}
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('Different agreed stable-LR fit contract')


def load_inputs(config):
    partial=base.read(ROOT/config['partial_control_recipe'])
    recipe,original,records,oracle,saved,parent,protected,_=control.load_inputs(partial)
    path=ROOT/config['reference_report']
    if base.file_digest(path)!=config['reference_report_sha256']:raise ValueError('Changed reference control')
    reference=base.read(path)
    if reference['status']!='completed' or reference['config']!=partial or not reference['sources_and_parent_unchanged']:
        raise ValueError('Different completed short-control contract')
    arm=reference['arms'][1]
    if arm['name']!='reduced_lr' or arm['proposal_updates']!=64 or not arm['qualification']['stable']:
        raise ValueError('Expected completed reduced-LR stability evidence, not a full qualification pass')
    folder=path.parent/'reduced_lr'
    protected.update(reference['provenance']['sources']);protected[str(path)]=config['reference_report_sha256']
    protected[str(folder/'step_0064_terminal.pt')]=config['reference_terminal_sha256']
    protected[str(folder/'updates.jsonl')]=config['reference_ledger_sha256']
    for evaluation in arm['evaluations']:
        for i,row in enumerate(evaluation['crops']):
            protected[str(folder/f"step_{evaluation['fork_update']:04d}_crop_{i}.npz")]=row['prediction_sha256']
            protected[str(folder/f"step_{evaluation['fork_update']:04d}_crop_{i}.swc")]=row['swc_sha256']
    base.verify_hashes(protected)
    return recipe,original,records,oracle,saved,parent,protected,folder


def validate_state(model,optimizer,q_steps,g_steps,config,recipe):
    validate_config(config)
    if type(q_steps) is not int or type(g_steps) is not int or not 0<=g_steps<=q_steps<=3000:
        raise ValueError('Invalid proposal/GAT counters or cap')
    if timing.stability.cursors(model,optimizer)!={'proposal':[float(5000+q_steps)],'gat':[float(1000+g_steps)]}:
        raise ValueError('Optimizer clocks differ from active update counts')
    if set(optimizer.state)!=set(model.parameters()):raise ValueError('Missing optimizer parameter states')
    expected=runner.optimizer_for(model,dict(recipe,learning_rate=1e-5)).state_dict()['param_groups']
    if optimizer.state_dict()['param_groups']!=expected:raise ValueError('Optimizer options/parameter order changed')
    for p,state in optimizer.state.items():
        if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected Adam state layout')
        for key in ('exp_avg','exp_avg_sq'):
            v=state[key]
            if v.shape!=p.shape or v.dtype!=p.dtype or v.device!=p.device or not bool(torch.isfinite(v).all()):raise ValueError('Invalid Adam moments')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative Adam second moment')
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise ValueError('Nonfinite model state')


def checkpoint(model,optimizer,q_steps,g_steps,prefix,config,recipe,provenance,purpose='fit'):
    validate_state(model,optimizer,q_steps,g_steps,config,recipe)
    if purpose not in ('fit','check','geometry_stop') or type(prefix) is not bool or (q_steps<64 and prefix) or (q_steps>64 and not prefix):
        raise ValueError('Invalid checkpoint purpose/replay proof')
    return {'schema':SCHEMA,'purpose':purpose,'resumable':purpose=='fit' and q_steps<3000,
        'config':config,'training_recipe':recipe,'provenance':provenance,'proposal_updates':q_steps,'graph_updates':g_steps,
        'historical_stage':1000+q_steps,'next_crop':q_steps%4,'prefix_verified':prefix,
        'parameter_names':[n for n,_ in model.named_parameters()],
        'proposal_spec':model.proposal_model.model_spec(),'graph_spec':model.graph_model.model_spec(),
        'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}


def restore(saved,model,config,recipe,provenance,*,resume):
    q=saved.get('proposal_updates');g=saved.get('graph_updates');prefix=saved.get('prefix_verified');purpose=saved.get('purpose')
    if (saved.get('schema')!=SCHEMA or saved.get('config')!=config or saved.get('training_recipe')!=recipe
        or saved.get('provenance')!=provenance or type(q) is not int or type(g) is not int or not 0<=g<=q<=3000
        or saved.get('historical_stage')!=1000+q or saved.get('next_crop')!=q%4
        or type(prefix) is not bool or (q<64 and prefix) or (q>64 and not prefix)
        or purpose not in ('fit','check','geometry_stop') or saved.get('resumable')!=(purpose=='fit' and q<3000)
        or saved.get('parameter_names')!=[n for n,_ in model.named_parameters()]
        or saved.get('proposal_spec')!=model.proposal_model.model_spec() or saved.get('graph_spec')!=model.graph_model.model_spec()):
        raise ValueError('Checkpoint source/backend/policy/model/cursor differs')
    if resume and not saved['resumable']:raise ValueError('Terminal, check-only or geometry-stop state cannot resume')
    model.load_state_dict(saved['model']);optimizer=runner.optimizer_for(model,dict(recipe,learning_rate=1e-5))
    optimizer.load_state_dict(cpu_snapshot(saved['optimizer']));validate_state(model,optimizer,q,g,config,recipe)
    restore_rng(saved['rng']);return optimizer,q,g,prefix


def save_state(model,optimizer,report,config,recipe,provenance,directory,purpose='fit'):
    path=directory/f"stage_{1000+report['proposal_updates']:04d}.pt";temporary=path.with_suffix('.pt.writing')
    if path.exists() or temporary.exists():raise FileExistsError('Preserve full-state checkpoint')
    torch.save(checkpoint(model,optimizer,report['proposal_updates'],report['graph_updates'],report['prefix_verified'],
        config,recipe,provenance,purpose),temporary);temporary.replace(path)
    report.update(last_checkpoint=str(path),last_checkpoint_stage=1000+report['proposal_updates'],last_checkpoint_sha256=base.file_digest(path))
    write(directory/'report.json',report);return path


def snapshot(model,optimizer,report,config,recipe,provenance,directory,records,oracle,original,device,purpose='fit'):
    save_state(model,optimizer,report,config,recipe,provenance,directory,purpose)
    q=report['proposal_updates'];print(f'EVALUATE historical_stage={1000+q} new_updates={q}/3000',flush=True)
    evaluation=control.evaluate(model,records,oracle,original,recipe,device,q,directory)
    evaluation['historical_stage']=1000+q;report['evaluations'].append(evaluation)
    write(directory/'report.json',report)


def verify_prefix(model,optimizer,report,directory,reference,config):
    for i in range(4):
        control.compare_arrays(control.read_arrays(directory/f'step_0064_crop_{i}.npz'),control.read_arrays(reference/f'step_0064_crop_{i}.npz'))
    source=torch.load(reference/'step_0064_terminal.pt',map_location='cpu',weights_only=True)
    if (not control.equal(model.state_dict(),source['model']) or not control.equal(optimizer.state_dict(),source['optimizer'])
        or not control.equal(capture_rng(),source['rng'])):raise ValueError('64-update replay model/Adam/RNG differs from reduced-LR control')
    report['prefix_verified']=True;report['prefix_model_optimizer_rng_exact']=True
    write(directory/'report.json',report);print('PREFIX 64/64: reduced-LR predictions, decisions, model, Adam and RNG reproduced',flush=True)


def apply_update(model,optimizer,total,edge_active,attempt,persist,recipe):
    """Same arithmetic as the short control; persist immediately after Adam returns."""
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite loss')
    total.backward()
    for parameters,active in ((list(model.proposal_model.parameters()),True),(list(model.graph_model.parameters()),edge_active)):
        if active and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Missing/nonfinite active gradients')
        if not active and any(p.grad is not None for p in parameters):raise ValueError('Inactive GAT gradients must remain None')
    attempt.update(proposal_gradient_l2=runner.grad_norm(model.proposal_model.parameters()),gat_gradient_l2=runner.grad_norm(model.graph_model.parameters()))
    attempt['preclip_norm']=float(torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'],error_if_nonfinite=True))
    attempt['phase']='before_optimizer_step';persist();optimizer.step()
    attempt.update(optimizer_step_returned=True,phase='optimizer_step_returned');persist()


def is_probe_step(q):return q%4==0 if q<=64 else q%20==0


def geometry_stop(probes):
    return len(probes)>=3 and any(all(p['crops'][i]['matched_distance_mean']>4. for p in probes[-3:]) for i in range(4))


def stable_tail(report,config):
    window=config['stability_window'];threshold=config['stability_fraction'];probes=report['probes'][-window:]
    stable=len(probes)==window and all(r['trusted_fraction']>=threshold for p in probes for r in p['crops'])
    for crop in config['training_crops']:
        incoming=[a for a in report['updates'] if a['crop']==crop][-window:]
        stable=stable and len(incoming)==window and all(a['incoming']['edge_supervision']['trusted_matches']/a['incoming']['edge_supervision']['targets']>=threshold for a in incoming)
    return bool(stable)


def cpu_check(model,optimizer,report,config,recipe,provenance,directory,records,oracle,original):
    before=runner.state_hashes(model);moments=cpu_snapshot(optimizer.state_dict());rng=capture_rng()
    evaluation=control.evaluate(model,records,oracle,original,recipe,'cpu',0,directory)
    probe=control.probe(model,records,original,recipe,'cpu')
    predictions=runner.prediction_signature(model,records,original,'cpu');path=directory/'temporary_check.pt'
    try:
        torch.save(checkpoint(model,optimizer,0,0,False,config,recipe,provenance,'check'),path)
        loaded,q,g,prefix=restore(torch.load(path,map_location='cpu',weights_only=True),model,config,recipe,provenance,resume=False)
        if q or g or prefix or before!=runner.state_hashes(model) or not control.equal(moments,loaded.state_dict()) or not control.equal(rng,capture_rng()):
            raise ValueError('CPU import/observation/reload state differs')
        runner.require_identical_predictions(predictions,runner.prediction_signature(model,records,original,'cpu'))
    finally:path.unlink(missing_ok=True)
    if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
    report.update(cpu_check_passed=True,optimizer_updates=0,exact_reload_prediction_delta=0.,state_and_rng_unchanged=True,
        baseline=evaluation,baseline_probe=probe,cuda_initialized=False)


def run(args):
    config=base.read(CONFIG);validate_config(config);torch.set_num_threads(2);runner.configure()
    if args.action!='resume' and args.checkpoint is not None:raise ValueError('Checkpoint only applies to explicit resume')
    device=torch.device('cpu' if args.action=='check' else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fitting fallback')
    recipe,original,records,oracle,saved,parent,protected,reference=load_inputs(config)
    root=(args.directory or ROOT/config['output']).resolve()
    if not root.is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Generated files belong in ignored tmp')
    sources=base.source_hashes(SimpleNamespace(recipe=CONFIG));launcher=ROOT/'scripts/pointneuron2/run_stable_joint_learnability.ps1'
    sources[str(launcher)]=base.file_digest(launcher)
    provenance={'sources':sources,'reference_report_sha256':config['reference_report_sha256'],'backend':runner.backend(device),'root':str(root)}
    if device.type=='cuda':
        if provenance['backend']!=parent['provenance']['backend']:raise ValueError('GPU backend differs from the matched controls')
        checked=base.read(root/'check/report.json')
        if checked['status']!='completed' or not checked['cpu_check_passed'] or checked['config']!=config or checked['provenance']['sources']!=sources:
            raise ValueError('Matching completed zero-update CPU check required')
    directory=root/('check' if args.action=='check' else 'model')
    if args.action=='resume':
        if args.checkpoint is None:raise ValueError('An explicit intermediate checkpoint is required')
        args.checkpoint=args.checkpoint.resolve()
        if not args.checkpoint.is_relative_to(root) or args.checkpoint.suffix!='.pt' or not (args.checkpoint.parent.name=='model' or args.checkpoint.parent.name.startswith('resume_')):
            raise ValueError('Checkpoint outside this fit')
        directory=root/f'resume_{args.checkpoint.stem}';number=2
        while directory.exists():directory=root/f'resume_{args.checkpoint.stem}_{number}';number+=1
    if directory.exists():raise FileExistsError('Preserve prior action output')
    directory.mkdir(parents=True);started=time.perf_counter()
    report={'schema':SCHEMA,'status':'running','action':args.action,'config':config,'provenance':provenance,'protected_files_sha256':protected,
        'proposal_updates':0,'graph_updates':0,'prefix_verified':False,'probes':[],'updates':[],'evaluations':[],
        'scope':'four fixed TRAINING crops; bounded stable-LR learnability, no generalization claim'}
    write(directory/'report.json',report)
    try:
        model=runner.warm_start(recipe,device);optimizer=control.imported_state(model,saved,recipe,1e-5,device)
        if args.action=='resume':
            optimizer,q,g,prefix=restore(torch.load(args.checkpoint,map_location='cpu',weights_only=True),model,config,recipe,provenance,resume=True)
            report.update(proposal_updates=q,graph_updates=g,prefix_verified=prefix,resumed_from=str(args.checkpoint),source_checkpoint_sha256=base.file_digest(args.checkpoint))
            # Recover recent diagnostics for an equivalent terminal stability decision.
            previous=base.read(args.checkpoint.parent/'report.json')
            if previous['provenance']!=provenance:raise ValueError('Resume diagnostic provenance differs')
            report['updates']=[a for a in previous['updates'] if a['fork_update']<=q]
            report['probes']=[p for p in previous['probes'] if p['new_updates']<=q]
        print(f"ACTION={args.action} device={provenance['backend']['name']} historical_stage={1000+report['proposal_updates']}/4000 LR=.00001",flush=True)
        if args.action=='check':cpu_check(model,optimizer,report,config,recipe,provenance,directory,records,oracle,original)
        else:
            start=report['proposal_updates'];reference_updates={a['fork_update']:a for a in map(json.loads,(reference/'updates.jsonl').read_text(encoding='utf-8').splitlines())}
            if start in config['snapshots']:
                snapshot(model,optimizer,report,config,recipe,provenance,directory,records,oracle,original,device)
                if start==0:
                    for i in range(4):control.compare_arrays(control.read_arrays(directory/f'step_0000_crop_{i}.npz'),control.read_arrays(reference/f'step_0000_crop_{i}.npz'))
                if start==64 and not report['prefix_verified']:verify_prefix(model,optimizer,report,directory,reference,config)
            for q in range(start+1,3001):
                index=(q-1)%4;model.train();optimizer.zero_grad(set_to_none=True)
                attempt={'fork_update':q,'stage_update':1000+q,'crop':config['training_crops'][index],'optimizer_step_returned':False,'phase':'forward'}
                persist=lambda:write(directory/'active_update.json',attempt)
                persist();output=runner.forward(model,records[index],original,device)
                incoming=timing.describe(output,records[index],original,recipe)
                attempt.update(incoming=incoming,phase='incoming_recorded');persist()
                if q<=64:control.compare_values(incoming,reference_updates[q]['incoming'])
                total,_,edge,_=partial_joint_loss(output,records[index],original['query_loss'],edge_weight=recipe['edge_weight'],match_tolerance=2.)
                active=bool(edge.supervised_crops);attempt['edge_active']=active
                before=control.graph_state(model,optimizer) if not active else None
                apply_update(model,optimizer,total,active,attempt,persist,recipe)
                report['proposal_updates']=q;report['graph_updates']+=int(active)
                validate_state(model,optimizer,q,report['graph_updates'],config,recipe)
                if not active and not control.equal(before,control.graph_state(model,optimizer)):raise ValueError('Inactive GAT state changed')
                attempt['cursors_after']=timing.stability.cursors(model,optimizer);persist()
                if q<=64:control.compare_values(attempt,{k:v for k,v in reference_updates[q].items() if k!='incoming'})
                compact={k:v for k,v in attempt.items() if k!='incoming'}
                compact['incoming']={k:v for k,v in incoming.items() if not k.startswith('matched_') or k in ('matched_distance_mean','matched_distance_max')}
                report['updates'].append(compact)
                with (directory/'updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(compact,allow_nan=False)+'\n')
                stopped=False
                if is_probe_step(q):
                    rows=control.probe(model,records,original,recipe,device);report['probes'].append({'new_updates':q,'historical_stage':1000+q,'crops':rows})
                    stopped=geometry_stop(report['probes'])
                    print(f"PROBE stage={1000+q} trusted={[r['edge_supervision']['trusted_matches'] for r in rows]} mean_distance={[round(r['matched_distance_mean'],3) for r in rows]}",flush=True)
                if q%50==0:print(f"UPDATE stage={1000+q}/4000 new={q}/3000 loss={incoming['loss']:.5f} q/G={attempt['cursors_after']} elapsed={time.perf_counter()-started:.1f}s",flush=True)
                if q in config['snapshots'] or stopped:
                    snapshot(model,optimizer,report,config,recipe,provenance,directory,records,oracle,original,device,'geometry_stop' if stopped else 'fit')
                    if q==64:verify_prefix(model,optimizer,report,directory,reference,config)
                if q%50==0 or is_probe_step(q):write(directory/'report.json',report)
                if stopped:report['status']='stopped_geometry';break
            report['terminal_stable_tail']=stable_tail(report,config)
            report['automatic_gate']=report['evaluations'][-1]['automatic_gate']
            report['learnability_gate']=bool(report['proposal_updates']==3000 and report['automatic_gate'] and report['terminal_stable_tail'])
            predictions=runner.prediction_signature(model,records,original,device);before=runner.state_hashes(model);moments=cpu_snapshot(optimizer.state_dict())
            loaded,q,g,_=restore(torch.load(report['last_checkpoint'],map_location='cpu',weights_only=True),model,config,recipe,provenance,resume=False)
            if q!=report['proposal_updates'] or g!=report['graph_updates'] or before!=runner.state_hashes(model) or not control.equal(moments,loaded.state_dict()):raise ValueError('Final state reload differs')
            runner.require_identical_predictions(predictions,runner.prediction_signature(model,records,original,device))
            report['final_reload_prediction_delta']=0.;report['next_step']='Review original automatic gates; no extension or objective change automatically follows.'
        base.verify_hashes(protected);base.verify_hashes(sources)
        if report['status']=='running':report['status']='completed'
        report.update(sources_and_parent_unchanged=True,elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);print(f"FINISHED {args.action} status={report['status']} stage={1000+report['proposal_updates']} learnability_gate={report.get('learnability_gate')} report={directory/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);(directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if 'output' in locals():timing.dump_prediction(output,directory/'failure_incoming_predictions.npz')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','train','resume'),required=True)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    run(parser.parse_args())
