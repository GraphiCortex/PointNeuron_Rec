"""Eight throwaway CPU joint updates: saved proposal Adam, fresh GAT Adam."""
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import compare_proposal_optimizer_state as control
from pointneuron.models.query_fit_state import cpu_snapshot
from pointneuron.models.balanced_query_loss import balanced_query_loss
from pointneuron.data.connectivity_targets import assign_edge_labels,masked_edge_loss

runner=control.runner
CONFIG=ROOT/'configs/pointneuron2/joint_saved_adam_stability.json'


def mixed_optimizer(model,saved,recipe,*,expected_step=4000):
    """One original joint parameter group; only proposal slots inherit history."""
    proposal=list(model.proposal_model.parameters());graph=list(model.graph_model.parameters())
    parameters=list(model.parameters())
    if [id(p) for p in parameters]!=[id(p) for p in proposal+graph]:raise ValueError('Different parameter order/sharing')
    validated=control.make_optimizer(model.proposal_model,saved,restored=True,expected_step=expected_step)
    optimizer=runner.optimizer_for(model,recipe);payload=optimizer.state_dict()
    options=lambda group:{k:v for k,v in group.items() if k!='params'}
    if options(payload['param_groups'][0])!=options(saved['param_groups'][0]):raise ValueError('Different Adam options')
    payload['state']=cpu_snapshot(validated.state_dict()['state'])
    optimizer.load_state_dict(payload)
    if any(p in optimizer.state for p in graph):raise ValueError('GAT must start with fresh Adam state')
    return optimizer


def cursors(model,optimizer):
    def values(module):
        return sorted({float(optimizer.state[p]['step']) if p in optimizer.state else 0. for p in module.parameters()})
    return {'proposal':values(model.proposal_model),'gat':values(model.graph_model)}


def describe(output,record,original,recipe):
    """Record this exact incoming forward before the original loss can stop."""
    with torch.no_grad():
        nodes=torch.from_numpy(record['nodes'])[None]
        proposal=balanced_query_loss(output.proposals,nodes,torch.ones(nodes.shape[:2],dtype=torch.bool),**original['query_loss'])
        labels=assign_edge_labels(output.packet,output.candidates,record['nodes'][:,1:4],record['target_edges'],
            record['target_components'],proposal.assignments[0],maximum_match_error=recipe['trusted_match_tolerance'])
        edge=masked_edge_loss(output.edge_logits,labels.labels)
        pi,ti=map(np.asarray,proposal.assignments[0])
        xyz=output.packet.positions.detach().numpy()[pi]
        distance=np.linalg.norm(xyz.astype(float)-record['nodes'][ti,1:4],axis=1)
        return {'edge_supervision':labels.statistics,'loss':float(proposal.total+recipe['edge_weight']*edge.total),
            'proposal_loss':float(proposal.total),'class_loss':float(proposal.objectness),
            'center_loss':float(proposal.center),'radius_loss':float(proposal.radius),'edge_loss':float(edge.total),
            'candidate_edges':len(output.candidates.edges),'matched_query_ids':pi.tolist(),
            'matched_target_indices':ti.tolist(),'matched_centers':xyz.tolist(),'matched_distances':distance.tolist()}


def instrumented_update(model,optimizer,record,original,recipe,attempt,persist):
    model.train();optimizer.zero_grad(set_to_none=True)
    output=runner.forward(model,record,original,'cpu')
    attempt['before']=describe(output,record,original,recipe);persist()
    try:
        # The loss, class-collapse guard, weights and masks are the original implementation.
        total,_,_,_=runner.loss(output,record,original,recipe)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite joint loss')
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Missing/nonfinite joint gradients')
        attempt['proposal_gradient_l2']=runner.grad_norm(model.proposal_model.parameters())
        attempt['gat_gradient_l2']=runner.grad_norm(model.graph_model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'],error_if_nonfinite=True)
        attempt.update(preclip_norm=float(norm),clip_factor=min(1.,recipe['gradient_clip']/(float(norm)+1e-6)))
        persist();optimizer.step();attempt['optimizer_step_executed']=True;persist()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite joint parameters')
    except BaseException:
        # Predictions are diagnostic evidence, not a resumable trained checkpoint.
        directory=Path(attempt['failure_output_directory'])
        np.savez_compressed(directory/'failure_incoming_predictions.npz',
            centers=output.proposals.center_proposals.detach().numpy(),radii=output.proposals.radius.detach().numpy(),
            logits=output.proposals.objectness_logits.detach().numpy(),features=output.proposals.decoder_features.detach().numpy(),
            candidate_edges=output.candidates.edges.numpy(),edge_logits=output.edge_logits.detach().numpy())
        raise


def stability_gate(report):
    snapshots={s['after_updates']:s['crops'] for s in report['snapshots']}
    completed=report['completed_updates']==8
    crop_zero=all(s in snapshots and snapshots[s][0]['edge_supervision']['trusted_matches']==6 for s in (4,8))
    terminal=8 in snapshots and all(p['edge_supervision']['positive']>0 and p['edge_supervision']['negative']>0 for p in snapshots[8])
    return {'completed_eight':completed,'crop_zero_six_at_four_and_eight':crop_zero,
        'terminal_edge_classes_all_crops':terminal,'passed':bool(completed and crop_zero and terminal)}


def run():
    config=runner.base.read(CONFIG)
    if config['device']!='cpu' or config['updates']!=8 or config['proposal_adam_initial_cursor']!=4000 or config['gat_adam_initial_cursor']!=0:
        raise ValueError('Different agreed CPU diagnostic')
    torch.set_num_threads(config['torch_threads']);runner.configure();started=time.perf_counter()
    recipe=runner.recipe_at(ROOT/config['source_recipe']);original,records,_,protected=runner.load_inputs(recipe)
    if config['training_crops']!=recipe['training_crops']:raise ValueError('Different crop order')
    for key in ('learning_rate','weight_decay','gradient_clip'):
        if config[key]!=recipe[key]:raise ValueError('Changed update hyperparameters')
    failed_path=ROOT/config['failed_fresh_timing'];failed=runner.base.read(failed_path)
    if failed['status']!='failed' or failed['recipe']!=recipe:raise ValueError('Expected original failed joint timing')
    runner.base.verify_hashes(failed['provenance']['sources'])
    summary_path=ROOT/config['optimizer_control_summary'];summary=runner.base.read(summary_path)
    if runner.base.file_digest(ROOT/summary['report'])!=summary['report_sha256'] or summary['arms'][1]['primary_pass'] is not True:
        raise ValueError('Different proposal-state control')
    for p in (failed_path,summary_path,ROOT/summary['report']):protected[str(p)]=runner.base.file_digest(p)
    sources=runner.base.source_hashes(SimpleNamespace(recipe=CONFIG))
    directory=ROOT/config['output']
    if not directory.resolve().is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Output outside ignored tmp')
    if directory.exists():raise FileExistsError('Preserve diagnostic evidence')
    directory.mkdir(parents=True)
    report={'schema':config['schema'],'status':'running','config':config,'sources':sources,
        'protected_files_sha256':protected,'completed_updates':0,'attempts':[],'snapshots':[],
        'gpu_work':False,'retained_trained_checkpoint':False}
    persist=lambda:runner.base.write(directory/'report.json',report)
    persist()
    try:
        model=runner.warm_start(recipe,'cpu');initial_hashes=runner.state_hashes(model)
        if initial_hashes!=failed['initial_state_sha256']:raise ValueError('Different initial weights')
        cp=torch.load(ROOT/recipe['proposal_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
        names=[n for n,_ in model.proposal_model.named_parameters()]
        if cp['step']!=4000 or [n for n in cp['inference_checkpoint']['model'] if n in set(names)]!=names:raise ValueError('Different source optimizer mapping')
        optimizer=mixed_optimizer(model,cp['optimizer'],recipe)
        if cursors(model,optimizer)!={'proposal':[4000.],'gat':[0.]}:raise ValueError('Different initial optimizer counters')
        report['initial_optimizer_state']=control.optimizer_digests(optimizer)
        report['optimizer_options']=optimizer.state_dict()['param_groups']
        initial=control.probe_all(model,records,original,recipe)
        report['snapshots'].append({'after_updates':0,'crops':initial});persist()
        for step in range(1,9):
            attempt={'stage_update':step,'crop':config['training_crops'][(step-1)%4],
                'optimizer_step_executed':False,'failure_output_directory':str(directory)}
            report['attempts'].append(attempt);persist()
            instrumented_update(model,optimizer,records[(step-1)%4],original,recipe,attempt,persist)
            report['completed_updates']=step;attempt['cursors_after']=cursors(model,optimizer)
            if attempt['cursors_after']!={'proposal':[float(4000+step)],'gat':[float(step)]}:raise ValueError('Mixed optimizer cursor differs')
            probes=control.probe_all(model,records,original,recipe,initial)
            report['snapshots'].append({'after_updates':step,'crops':probes});persist()
            print(f"CPU JOINT update={step}/8 trusted={[p['edge_supervision']['trusted_matches'] for p in probes]} pos_neg={[(p['edge_supervision']['positive'],p['edge_supervision']['negative']) for p in probes]} norm={attempt['preclip_norm']:.3f}",flush=True)
        final=runner.state_hashes(model)
        report['parameter_changes']={prefix:any(final[k]!=initial_hashes[k] for k in final if k.startswith(prefix+'.'))
            for prefix in ('proposal_model','graph_model')}
        if not all(report['parameter_changes'].values()):raise ValueError('Both models must update')
        report.update(status='completed',final_optimizer_state=control.optimizer_digests(optimizer),gate=stability_gate(report))
    except BaseException as exc:
        last=report['attempts'][-1] if report['attempts'] else {}
        counts=last.get('before',{}).get('edge_supervision',{})
        policy_stop=counts and (not counts['positive'] or not counts['negative']) and not last.get('optimizer_step_executed')
        report.update(status='stopped_supervision' if policy_stop else 'failed',error=str(exc),gate=stability_gate(report))
        (directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    runner.base.verify_hashes(protected);runner.base.verify_hashes(failed['provenance']['sources']);runner.base.verify_hashes(sources)
    report.update(original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-started);persist()
    print(f"STOPPED AT BUDGET status={report['status']} completed={report['completed_updates']} gate={report['gate']}; report={directory/'report.json'}",flush=True)
    if report['status']=='failed':raise RuntimeError('Diagnostic failed; preserve traceback')


if __name__=='__main__':run()
