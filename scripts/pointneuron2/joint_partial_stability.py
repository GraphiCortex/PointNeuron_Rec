"""CPU preparation or two nonresumable 64-update CUDA partial-supervision controls."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_fit as legacy
from pointneuron.models.partial_joint_loss import POLICY,partial_joint_loss,step_active_loss
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng,restore_rng
from pointneuron.graph.proposal_graph import CandidateGraph
from pointneuron.graph.connectivity_decoder import export_decoded_swc

runner=legacy.runner
timing=legacy.timing
base=runner.base
CONFIG=ROOT/'configs/pointneuron2/joint_partial_stability.json'
SCHEMA='pointneuron2-partial-stability-control-v1'
write=legacy.write


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def validate_config(c):
    fixed={'schema':SCHEMA,'policy':POLICY,'updates_per_arm':64,'parent_stage':1000,'proposal_initial_cursor':5000,
        'gat_initial_cursor':1000,'probe_every':4,'geometry_stop_mean_distance':4.,'geometry_stop_consecutive_cycles':3,
        'qualification_fraction':.9,'qualification_window':8,'maximum_arclength_decline':.005,'torch_threads':2,
        'replay_rtol':1e-5,'replay_atol':1e-5,'automatic_extension':False,
        'training_crops':[[0,0],[0,1],[92,4],[86,3]],
        'arms':[{'name':'current_lr','learning_rate':.0001},{'name':'reduced_lr','learning_rate':.00001}]}
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('Different agreed partial-supervision comparison')
    if c.get('parent_checkpoint_sha256')!='c096a0906cc6d69f1c0b525ff105a87b207008383faa09e03f06500710b9e414':
        raise ValueError('Different pinned parent checkpoint')


def observe(model,operation):
    """Fail if an eval probe alters any parameter/buffer or any recorded RNG stream."""
    state=runner.state_hashes(model);rng=capture_rng();modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():result=operation()
    finally:
        for module,training in modes:module.training=training
    if state!=runner.state_hashes(model) or not equal(rng,capture_rng()):
        raise ValueError('Observation changed model/buffers or RNG; stop rather than hide probe interference')
    return result


def load_inputs(config):
    validate_config(config)
    protected={str(ROOT/config[k]):config[k+'_sha256'] for k in ('parent_checkpoint','parent_report','review_summary')}
    parent_dir=(ROOT/config['parent_report']).parent
    protected.update({str(parent_dir/name):digest for name,digest in config['replay_files_sha256'].items()})
    base.verify_hashes(protected)
    report=base.read(ROOT/config['parent_report']);review=base.read(ROOT/config['review_summary'])
    if report['status']!='failed' or report['completed_updates']!=1013 or report['last_checkpoint_step']!=1000:
        raise ValueError('Different stopped parent run')
    if review['source_report_sha256']!=config['parent_report_sha256'] or review['validated_output_pairs']!=12:
        raise ValueError('Different reviewed parent evidence')
    source_config=base.read(legacy.CONFIG)
    recipe,original,records,oracle,upstream,_,_=legacy.load_inputs(source_config);protected.update(upstream)
    saved=torch.load(ROOT/config['parent_checkpoint'],map_location='cpu',weights_only=True)
    if (saved['schema']!=legacy.SCHEMA or saved['purpose']!='fit' or not saved['resumable']
        or saved['stage_step']!=1000 or saved['next_crop']!=0 or not saved['prefix_verified']
        or saved['config']!=source_config or saved['training_recipe']!=recipe or saved['provenance']!=report['provenance']):
        raise ValueError('Invalid parent import contract')
    protected.update(saved['provenance']['sources'])
    for row_index,row in enumerate(report['evaluations'][-1]['crops']):
        for suffix,key in (('npz','prediction_sha256'),('swc','swc_sha256')):
            protected[str(parent_dir/f'step_1000_crop_{row_index}.{suffix}')]=row[key]
    base.verify_hashes(protected)
    return recipe,original,records,oracle,saved,report,protected,parent_dir


def imported_state(model,saved,recipe,lr,device):
    if (saved['parameter_names']!=[n for n,_ in model.named_parameters()]
        or saved['proposal_spec']!=model.proposal_model.model_spec() or saved['graph_spec']!=model.graph_model.model_spec()):
        raise ValueError('Imported model/order differs')
    model.load_state_dict(saved['model'])
    optimizer=runner.optimizer_for(model,recipe)
    optimizer.load_state_dict(cpu_snapshot(saved['optimizer']))
    legacy.validate_optimizer(model,optimizer,1000,recipe)
    if not equal(model.state_dict(),saved['model']) or not equal(optimizer.state_dict(),saved['optimizer']):
        raise ValueError('Parent model/Adam transfer is not exact')
    for group in optimizer.param_groups:group['lr']=lr
    rng=cpu_snapshot(saved['rng'])
    if torch.device(device).type=='cpu':rng['torch_cuda']=[]  # CPU checks never restore/initialize CUDA RNG.
    restore_rng(rng)
    validate_state(model,optimizer,0,0,lr,recipe)
    return optimizer


def validate_state(model,optimizer,proposal_updates,graph_updates,lr,recipe):
    if (type(proposal_updates) is not int or type(graph_updates) is not int
        or not 0<=graph_updates<=proposal_updates<=64 or lr not in (1e-4,1e-5)):
        raise ValueError('Invalid diagnostic cap/cursors/LR')
    wanted={'proposal':[float(5000+proposal_updates)],'gat':[float(1000+graph_updates)]}
    if timing.stability.cursors(model,optimizer)!=wanted or set(optimizer.state)!=set(model.parameters()):
        raise ValueError('Partial-supervision optimizer clocks differ')
    expected=runner.optimizer_for(model,dict(recipe,learning_rate=lr)).state_dict()['param_groups']
    if optimizer.state_dict()['param_groups']!=expected:raise ValueError('Optimizer options or parameter order differ')
    for p,state in optimizer.state.items():
        if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected Adam state')
        for key in ('exp_avg','exp_avg_sq'):
            v=state[key]
            if v.shape!=p.shape or v.dtype!=p.dtype or v.device!=p.device or not bool(torch.isfinite(v).all()):
                raise ValueError('Invalid optimizer moments')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative Adam second moment')
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise ValueError('Nonfinite model state')


def snapshot_state(model,optimizer,q_steps,g_steps,lr,config,recipe,provenance):
    validate_config(config);validate_state(model,optimizer,q_steps,g_steps,lr,recipe)
    return {'schema':SCHEMA,'purpose':'bounded_diagnostic_only','resumable':False,'config':config,'provenance':provenance,
        'training_recipe':recipe,
        'proposal_updates':q_steps,'graph_updates':g_steps,'next_crop':q_steps%4,'learning_rate':lr,
        'parameter_names':[n for n,_ in model.named_parameters()],
        'proposal_spec':model.proposal_model.model_spec(),'graph_spec':model.graph_model.model_spec(),
        'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}


def restore_snapshot(saved,model,config,recipe,provenance):
    if (saved.get('schema')!=SCHEMA or saved.get('purpose')!='bounded_diagnostic_only' or saved.get('resumable') is not False
        or saved.get('config')!=config or saved.get('provenance')!=provenance or saved.get('training_recipe')!=recipe
        or saved.get('parameter_names')!=[n for n,_ in model.named_parameters()]
        or saved.get('proposal_spec')!=model.proposal_model.model_spec() or saved.get('graph_spec')!=model.graph_model.model_spec()
        or saved.get('next_crop')!=saved['proposal_updates']%4):raise ValueError('Different diagnostic checkpoint contract')
    model.load_state_dict(saved['model']);optimizer=runner.optimizer_for(model,dict(recipe,learning_rate=saved['learning_rate']))
    optimizer.load_state_dict(cpu_snapshot(saved['optimizer']))
    validate_config(config);validate_state(model,optimizer,saved['proposal_updates'],saved['graph_updates'],saved['learning_rate'],recipe)
    restore_rng(saved['rng'])
    return optimizer


def save_checkpoint(path,model,optimizer,report,config,recipe,provenance):
    temporary=path.with_suffix('.pt.writing')
    if path.exists() or temporary.exists():raise FileExistsError('Preserve diagnostic checkpoint')
    torch.save(snapshot_state(model,optimizer,report['proposal_updates'],report['graph_updates'],report['learning_rate'],config,recipe,provenance),temporary)
    temporary.replace(path);return base.file_digest(path)


def probe(model,records,original,recipe,device):
    def operation():
        rows=[]
        for record in records:
            row=timing.describe(runner.forward(model,record,original,device),record,original,recipe)
            row['matched_distance_p95']=float(np.quantile(row['matched_distances'],.95))
            row['trusted_fraction']=row['edge_supervision']['trusted_matches']/row['edge_supervision']['targets']
            rows.append(row)
        return rows
    return observe(model,operation)


def evaluate(model,records,oracle,original,recipe,device,step,directory):
    def operation():
        rows=[];audit=runner.audit
        for i,(record,reference) in enumerate(zip(records,oracle)):
            output=runner.forward(model,record,original,device)
            total,proposal,edge,labels=partial_joint_loss(output,record,original['query_loss'],
                edge_weight=recipe['edge_weight'],match_tolerance=recipe['trusted_match_tolerance'])
            selected,c,scores=runner.retained_graph(output);packet=runner.cpu_packet(selected)
            c=CandidateGraph(c.edges.cpu(),c.neighbors,c.radius);scores=scores.cpu()
            pi,ti=map(np.asarray,proposal.assignments[0]);saved={'pred_indices':pi,'target_indices':ti}
            row,decoded,evaluation_labels=audit.assess(packet,c,scores,saved,record,audit.stage.reference_graph(reference))
            q2t=dict(zip(pi.tolist(),ti.tolist()))
            row['retained_trusted_junctions']=int(sum(bool(record['junction_mask'][q2t[int(q)]])
                and np.linalg.norm(packet.positions[j].numpy().astype(float)-record['nodes'][q2t[int(q)],1:4])<=2
                for j,q in enumerate(packet.node_ids) if int(q) in q2t))
            if decoded.status=='decoded':
                path=directory/f'step_{step:04d}_crop_{i}.swc'
                row['swc_roundtrip']=export_decoded_swc(packet,decoded,path,description='PARTIAL-SUPERVISION TRAINING CONTROL; automatic nodes/forest; no reference-selected model inputs.')
                row['swc_sha256']=base.file_digest(path)
            row.update(sample_index=record['identity']['sample_index'],patch_index=record['identity']['patch_index'],
                loss=float(total),proposal_loss=float(proposal.total),edge_loss=float(edge.total),training_edge_supervision=labels.statistics)
            row['automatic_gate']=runner.automatic_gate(row,len(record['nodes']))
            path=directory/f'step_{step:04d}_crop_{i}.npz'
            np.savez_compressed(path,centers=output.proposals.center_proposals.cpu().numpy(),radii=output.proposals.radius.cpu().numpy(),
                logits=output.proposals.objectness_logits.cpu().numpy(),decoder_features=output.proposals.decoder_features.cpu().numpy(),
                pred_indices=pi,target_indices=ti,candidate_edges=output.candidates.edges.cpu().numpy(),edge_logits=output.edge_logits.cpu().numpy(),
                edge_labels=labels.labels.cpu().numpy(),retained_node_ids=packet.node_ids.numpy(),retained_edges=c.edges.numpy(),
                retained_edge_logits=scores.numpy(),evaluation_labels=evaluation_labels)
            row['prediction_sha256']=base.file_digest(path);rows.append(row)
            g=row['geometry']['tolerances']['2.0']
            print(f"EVAL fork_update={step} crop={row['sample_index']}/{row['patch_index']} nodes={row['assigned_nodes']}/{len(record['nodes'])}+{row['unmatched_nodes']} edges={row['decoded_required_edges']}/{row['required_reference_edges']} P/R@2={g['precision']}/{g['recall']} full_gate={row['automatic_gate']}",flush=True)
        return {'fork_update':step,'crops':rows,'automatic_gate':all(r['automatic_gate'] for r in rows)}
    return observe(model,operation)


def read_arrays(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}


def compare_arrays(actual,expected,*,rtol=1e-5,atol=1e-5):
    if actual.keys()!=expected.keys():raise ValueError('Replay prediction fields differ')
    float_keys=('centers','radii','logits','decoder_features','edge_logits','retained_edge_logits')
    for key in actual.keys() & expected.keys():
        if key in float_keys:np.testing.assert_allclose(actual[key],expected[key],rtol=rtol,atol=atol)
        elif not np.array_equal(actual[key],expected[key]):raise ValueError(f'Replay discrete field differs: {key}')
    for key in ('logits','edge_logits'):
        if key in actual and key in expected:
            a,b=actual[key],expected[key]
            da=a[...,1]>=a[...,0] if key=='logits' else a>=0
            db=b[...,1]>=b[...,0] if key=='logits' else b>=0
            if not np.array_equal(da,db):raise ValueError(f'Replay cutoff decision differs: {key}')


def compare_values(actual,expected):
    if isinstance(expected,dict):
        for key,value in expected.items():compare_values(actual[key],value)
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('Replay list length differs')
        for a,b in zip(actual,expected):compare_values(a,b)
    elif isinstance(expected,float):np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
    elif actual!=expected:raise ValueError(f'Replay value differs: {actual!r} vs {expected!r}')


def graph_state(model,optimizer):
    return (runner.state_hashes(model.graph_model),
        {name:cpu_snapshot(optimizer.state[p]) for name,p in model.graph_model.named_parameters()})


def geometry_stop(probes,config):
    n=config['geometry_stop_consecutive_cycles']
    return len(probes)>=n and any(all(p['crops'][i]['matched_distance_mean']>config['geometry_stop_mean_distance']
        for p in probes[-n:]) for i in range(4))


def output_preservation(start,end,config):
    violations=[]
    for i,(a,b) in enumerate(zip(start['crops'],end['crops'])):
        for key in ('unmatched_nodes','decoded_nonreference_edges','retained_junction_incidence_errors',
                    'output_components_spanning_multiple_reference_crop_components'):
            if b[key]>a[key]:violations.append([i,key,'increased'])
        for key in ('assigned_nodes','decoded_required_edges','retained_reference_junctions'):
            if b[key]<a[key]:violations.append([i,key,'decreased'])
        if abs(b['decoding']['components']-b['reference_components'])>abs(a['decoding']['components']-a['reference_components']):
            violations.append([i,'component_count_error','increased'])
        if not b.get('swc_roundtrip',{}).get('passed',False):violations.append([i,'SWC','invalid'])
        for key in ('precision','recall'):
            value=b['geometry']['tolerances']['2.0'][key]
            if value is None or not np.isfinite(value) or value < a['geometry']['tolerances']['2.0'][key]-config['maximum_arclength_decline']:
                violations.append([i,key,'declined_more_than_0.005'])
    return {'passed':not violations,'violations':violations}


def qualification(report,config):
    window=config['qualification_window'];threshold=config['qualification_fraction']
    cycles=report['probes'][-window:]
    stable=len(cycles)==window and all(r['trusted_fraction']>=threshold for p in cycles for r in p['crops'])
    for crop in config['training_crops']:
        selected=[a for a in report['updates'] if a['crop']==crop][-window:]
        stable=stable and len(selected)==window and all(a['incoming']['edge_supervision']['trusted_matches']/a['incoming']['edge_supervision']['targets']>=threshold for a in selected)
    preserved=output_preservation(report['evaluations'][0],report['evaluations'][-1],config)
    return {'completed_cap':report['proposal_updates']==64,'stable':bool(stable),'output_preservation':preserved,
        'qualified':bool(report['proposal_updates']==64 and stable and preserved['passed']),
        'full_automatic_gate':report['evaluations'][-1]['automatic_gate']}


def strictly_dominates(a,b):
    """Does endpoint a improve a reconstruction measure without worsening any?"""
    better=False
    for x,y in zip(a['crops'],b['crops']):
        higher=('assigned_nodes','decoded_required_edges','retained_reference_junctions')
        lower=('unmatched_nodes','decoded_nonreference_edges','retained_junction_incidence_errors','output_components_spanning_multiple_reference_crop_components')
        pairs=[(x[k],y[k]) for k in higher]+[(-x[k],-y[k]) for k in lower]
        pairs+=[(-abs(x['decoding']['components']-x['reference_components']),-abs(y['decoding']['components']-y['reference_components']))]
        pairs+=[(x['geometry']['tolerances']['2.0'][k],y['geometry']['tolerances']['2.0'][k]) for k in ('precision','recall')]
        if any(u<v-1e-6 for u,v in pairs):return False
        better=better or any(u>v+1e-6 for u,v in pairs)
    return better


def decision(arms):
    a,b=arms;qa,qb=a['qualification']['qualified'],b['qualification']['qualified']
    if qb and (not qa or strictly_dominates(b['evaluations'][-1],a['evaluations'][-1])):return 'reduced_lr_supported_locally_review_before_any_longer_fit'
    if qa:return 'retain_current_lr_for_discussion_no_automatic_extension'
    return 'neither_qualified_stop_policy_tuning_revisit_geometry_and_objective_coupling'


def run_arm(arm,config,recipe,original,records,oracle,saved,parent_dir,provenance,directory,device):
    directory.mkdir();report={'name':arm['name'],'learning_rate':arm['learning_rate'],'status':'running','proposal_updates':0,
        'graph_updates':0,'probes':[],'updates':[],'evaluations':[],'replay_verified_steps':[]}
    model=runner.warm_start(recipe,device);optimizer=imported_state(model,saved,recipe,arm['learning_rate'],device)
    old_updates={a['stage_update']:a for a in map(json.loads,(parent_dir/'updates.jsonl').read_text(encoding='utf-8').splitlines())}
    failure=base.read(parent_dir/'active_update.json')
    try:
        report['initial_checkpoint_sha256']=save_checkpoint(directory/'step_0000.pt',model,optimizer,report,config,recipe,provenance)
        report['evaluations'].append(evaluate(model,records,oracle,original,recipe,device,0,directory))
        for i in range(4):compare_arrays(read_arrays(directory/f'step_0000_crop_{i}.npz'),read_arrays(parent_dir/f'step_1000_crop_{i}.npz'))
        write(directory/'report.json',report)
        for step in range(1,65):
            index=(step-1)%4;model.train();optimizer.zero_grad(set_to_none=True)
            attempt={'fork_update':step,'stage_update':1000+step,'crop':config['training_crops'][index],'optimizer_step_returned':False,'phase':'forward'}
            write(directory/'active_update.json',attempt)
            output=runner.forward(model,records[index],original,device)
            incoming=timing.describe(output,records[index],original,recipe)
            attempt.update(incoming=incoming,phase='incoming_recorded');write(directory/'active_update.json',attempt)
            if arm['name']=='current_lr' and step<=13:compare_values(incoming,old_updates[1000+step]['incoming'])
            if arm['name']=='current_lr' and step==14:
                compare_values(incoming,failure['incoming'])
                old=read_arrays(parent_dir/'failure_incoming_predictions.npz')
                actual={k:getattr(output.proposals,k).detach().cpu().numpy() for k in ('center_proposals','radius','objectness_logits','decoder_features')}
                actual.update(candidate_edges=output.candidates.edges.cpu().numpy(),edge_logits=output.edge_logits.detach().cpu().numpy())
                for key in actual:
                    if key=='candidate_edges':
                        if not np.array_equal(actual[key],old[key]):raise ValueError('Guard-transition candidates differ')
                    else:np.testing.assert_allclose(actual[key],old[key],rtol=1e-5,atol=1e-5)
                for key in ('matched_query_ids','matched_target_indices'):
                    if incoming[key]!=failure['incoming'][key]:raise ValueError('Guard-transition assignment differs')
                report['guard_transition_verified']=True
                print('REPLAY current_lr: saved incoming update 1014 reproduced; known-positive-only policy now applies',flush=True)
            total,_,edge,_=partial_joint_loss(output,records[index],original['query_loss'],edge_weight=recipe['edge_weight'],match_tolerance=2.)
            active=bool(edge.supervised_crops);attempt['edge_active']=active
            inactive_before=graph_state(model,optimizer) if not active else None
            attempt['phase']='before_backward';write(directory/'active_update.json',attempt)
            attempt.update(step_active_loss(model,optimizer,total,edge_active=active,gradient_clip=recipe['gradient_clip']))
            attempt.update(optimizer_step_returned=True,phase='optimizer_step_returned')
            report['proposal_updates']=step;report['graph_updates']+=int(active)
            validate_state(model,optimizer,step,report['graph_updates'],arm['learning_rate'],recipe)
            if not active and not equal(inactive_before,graph_state(model,optimizer)):raise ValueError('Inactive GAT state changed')
            attempt['cursors_after']=timing.stability.cursors(model,optimizer)
            write(directory/'active_update.json',attempt)
            if arm['name']=='current_lr' and step<=13:
                compare_values(attempt, {k:v for k,v in old_updates[1000+step].items() if k!='incoming'})
                report['replay_verified_steps'].append(1000+step)
            timing.dump_prediction(output,directory/'last_incoming_predictions.npz')
            compact={k:v for k,v in attempt.items() if k!='incoming'}
            compact['incoming']={k:v for k,v in incoming.items() if not k.startswith('matched_') or k in ('matched_distance_mean','matched_distance_max')}
            report['updates'].append(compact)
            with (directory/'updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(compact,allow_nan=False)+'\n')
            if step==13 and arm['name']=='current_lr':
                save_checkpoint(directory/'step_0013.pt',model,optimizer,report,config,recipe,provenance)
            if step%4==0:
                rows=probe(model,records,original,recipe,device);report['probes'].append({'fork_update':step,'crops':rows})
                print(f"CYCLE arm={arm['name']} updates={step}/64 clocks={attempt['cursors_after']} trusted={[r['edge_supervision']['trusted_matches'] for r in rows]} mean_distance={[round(r['matched_distance_mean'],3) for r in rows]}",flush=True)
                write(directory/'report.json',report)
                if geometry_stop(report['probes'],config):report['status']='stopped_geometry';break
        terminal=report['proposal_updates']
        report['terminal_checkpoint_sha256']=save_checkpoint(directory/f'step_{terminal:04d}_terminal.pt',model,optimizer,report,config,recipe,provenance)
        report['evaluations'].append(evaluate(model,records,oracle,original,recipe,device,terminal,directory))
        report['qualification']=qualification(report,config)
        if report['status']=='running':report['status']='completed'
        if arm['name']=='current_lr' and not report.get('guard_transition_verified'):raise ValueError('Control ended before historical guard transition was verified')
        write(directory/'report.json',report);return report
    except BaseException as exc:
        report.update(status='failed',error=str(exc));write(directory/'report.json',report)
        (directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if 'output' in locals():timing.dump_prediction(output,directory/'failure_incoming_predictions.npz')
        raise


def cpu_check(config,recipe,original,records,oracle,saved,provenance,directory):
    signatures=[];moments=[];rows=[]
    for arm in config['arms']:
        target=directory/arm['name'];target.mkdir()
        model=runner.warm_start(recipe,'cpu');optimizer=imported_state(model,saved,recipe,arm['learning_rate'],'cpu')
        report={'proposal_updates':0,'graph_updates':0,'learning_rate':arm['learning_rate']}
        signatures.append(runner.state_hashes(model));moments.append(cpu_snapshot(optimizer.state_dict()['state']))
        before_rng=capture_rng();before=cpu_snapshot(optimizer.state_dict())
        baseline=evaluate(model,records,oracle,original,recipe,'cpu',0,target)
        probes=probe(model,records,original,recipe,'cpu')
        signature=runner.prediction_signature(model,records,original,'cpu')
        path=target/'check_reload.pt'
        save_checkpoint(path,model,optimizer,report,config,recipe,provenance)
        loaded=restore_snapshot(torch.load(path,map_location='cpu',weights_only=True),model,config,recipe,provenance)
        if not equal(before,loaded.state_dict()) or not equal(before_rng,capture_rng()):raise ValueError('CPU reload/observations changed state/RNG')
        runner.require_identical_predictions(signature,runner.prediction_signature(model,records,original,'cpu'))
        path.unlink();rows.append({'arm':arm,'baseline':baseline,'probes':probes,'optimizer_updates':0,'reload_prediction_delta':0.})
    if signatures[0]!=signatures[1] or not equal(moments[0],moments[1]):raise ValueError('Forks have different initial weights/Adam moments')
    for i in range(4):compare_arrays(read_arrays(directory/'current_lr'/f'step_0000_crop_{i}.npz'),read_arrays(directory/'reduced_lr'/f'step_0000_crop_{i}.npz'),rtol=0,atol=0)
    if torch.cuda.is_initialized():raise ValueError('CPU check unexpectedly initialized CUDA')
    return {'cpu_check_passed':True,'optimizer_updates':0,'fork_weights_and_moments_identical':True,
        'probes_preserve_weights_buffers_and_rng':True,'exact_cpu_reload_predictions':True,'rows':rows,'cuda_initialized':False}


def run(args):
    config=base.read(CONFIG);validate_config(config);torch.set_num_threads(2);runner.configure()
    device=torch.device('cpu' if args.action=='check' else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fitting fallback')
    recipe,original,records,oracle,saved,parent,protected,parent_dir=load_inputs(config)
    root=(args.directory or ROOT/config['output']).resolve()
    if not root.is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Generated outputs belong in ignored tmp')
    sources=base.source_hashes(SimpleNamespace(recipe=CONFIG))
    launcher=ROOT/'scripts/pointneuron2/run_joint_partial_stability.ps1';sources[str(launcher)]=base.file_digest(launcher)
    provenance={'sources':sources,'parent_checkpoint_sha256':config['parent_checkpoint_sha256'],'backend':runner.backend(device),'root':str(root)}
    if device.type=='cuda':
        if provenance['backend']!=parent['provenance']['backend']:raise ValueError('GPU backend differs from parent')
        checked=base.read(root/'check/report.json')
        if checked['status']!='completed' or not checked['cpu_check']['cpu_check_passed'] or checked['config']!=config or checked['provenance']['sources']!=sources:
            raise ValueError('Matching completed CPU check required')
    directory=root/('check' if args.action=='check' else 'comparison')
    if directory.exists():raise FileExistsError('Preserve prior action output; no automatic retry')
    directory.mkdir(parents=True);started=time.perf_counter()
    report={'schema':SCHEMA,'status':'running','action':args.action,'config':config,'provenance':provenance,
        'protected_files_sha256':protected,'scope':'four TRAINING crops; short local policy/LR control, no generalization claim'}
    write(directory/'report.json',report)
    try:
        if args.action=='check':report['cpu_check']=cpu_check(config,recipe,original,records,oracle,saved,provenance,directory)
        else:
            report['arms']=[]
            for arm in config['arms']:
                report['arms'].append(run_arm(arm,config,recipe,original,records,oracle,saved,parent_dir,provenance,directory/arm['name'],device))
                write(directory/'report.json',report)
            report['decision']=decision(report['arms'])
            report['next_step']='Review with user. No automatic longer fit, LR search or production promotion.'
        base.verify_hashes(protected);base.verify_hashes(sources)
        report.update(status='completed',sources_and_parent_unchanged=True,elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);print(f"COMPLETED {args.action}; decision={report.get('decision')} report={directory/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write(directory/'report.json',report);(directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','compare'),required=True)
    parser.add_argument('--directory',type=Path)
    run(parser.parse_args())
