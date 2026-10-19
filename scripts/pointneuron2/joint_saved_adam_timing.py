"""Zero-update CPU preparation or eight throwaway CUDA joint timing updates only."""
from pathlib import Path
import argparse
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import check_joint_saved_adam_stability as stability
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng,restore_rng
from pointneuron.models.balanced_query_loss import balanced_query_loss
from pointneuron.data.connectivity_targets import assign_edge_labels,masked_edge_loss

runner=stability.runner
CONFIG=ROOT/'configs/pointneuron2/joint_saved_adam_timing.json'
SCHEMA='pointneuron2-joint-saved-adam-timing-v1'


def describe(output,record,original,recipe):
    """Capture the exact incoming prediction and supervision on either backend."""
    with torch.no_grad():
        nodes=torch.from_numpy(record['nodes'])[None].to(output.packet.positions.device)
        proposal=balanced_query_loss(output.proposals,nodes,torch.ones(nodes.shape[:2],dtype=torch.bool,device=nodes.device),**original['query_loss'])
        labels=assign_edge_labels(output.packet,output.candidates,record['nodes'][:,1:4],record['target_edges'],
            record['target_components'],proposal.assignments[0],maximum_match_error=recipe['trusted_match_tolerance'])
        edge=masked_edge_loss(output.edge_logits,labels.labels)
        pi,ti=map(np.asarray,proposal.assignments[0])
        xyz=output.packet.positions.detach().cpu().numpy()[pi]
        distances=np.linalg.norm(xyz.astype(float)-record['nodes'][ti,1:4],axis=1)
        logits=output.packet.objectness_logits.detach().cpu().numpy();accepted=logits[:,1]>=logits[:,0]
        return {'crop':[record['identity']['sample_index'],record['identity']['patch_index']],
            'edge_supervision':labels.statistics,'proposal_loss':float(proposal.total),'edge_loss':float(edge.total),
            'loss':float(proposal.total+recipe['edge_weight']*edge.total),'class_loss':float(proposal.objectness),
            'center_loss':float(proposal.center),'radius_loss':float(proposal.radius),
            'candidate_edges':len(output.candidates.edges),'matched_query_ids':pi.tolist(),'matched_target_indices':ti.tolist(),
            'matched_centers':xyz.tolist(),'matched_distances':distances.tolist(),
            'matched_distance_mean':float(distances.mean()),'matched_distance_max':float(distances.max()),
            'accepted_assigned':int(accepted[pi].sum()),'accepted_unmatched':int(accepted.sum()-accepted[pi].sum())}


def probe_all(model,records,original,recipe,device):
    model.train()
    with torch.no_grad():return [describe(runner.forward(model,r,original,device),r,original,recipe) for r in records]


def dump_prediction(output,path):
    arrays={name:getattr(output.proposals,name).detach().cpu().numpy() for name in
        ('center_proposals','objectness_logits','radius','decoder_features')}
    np.savez_compressed(path,**arrays,candidate_edges=output.candidates.edges.detach().cpu().numpy(),
        edge_logits=output.edge_logits.detach().cpu().numpy())


def instrumented_update(model,optimizer,record,original,recipe,device,attempt,persist,directory):
    model.train();optimizer.zero_grad(set_to_none=True);output=None
    try:
        attempt['phase']='forward';persist();output=runner.forward(model,record,original,device)
        attempt.update(incoming=describe(output,record,original,recipe),phase='incoming_recorded');persist()
        total,_,_,_=runner.loss(output,record,original,recipe)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite joint loss')
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Missing/nonfinite gradients')
        attempt.update(proposal_gradient_l2=runner.grad_norm(model.proposal_model.parameters()),
            gat_gradient_l2=runner.grad_norm(model.graph_model.parameters()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'],error_if_nonfinite=True)
        attempt.update(preclip_norm=float(norm),phase='before_optimizer_step');persist()
        optimizer.step();attempt.update(optimizer_step_returned=True,phase='optimizer_step_returned');persist()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
    except BaseException:
        if output is not None:dump_prediction(output,directory/'failure_incoming_predictions.npz')
        raise


def expected_cursors(step):return {'proposal':[float(4000+step)],'gat':[float(step)]}


def timing_checkpoint(model,optimizer,step,config,provenance):
    if type(step) is not int or not 0<=step<=8 or stability.cursors(model,optimizer)!=expected_cursors(step):
        raise ValueError('Different diagnostic stage/optimizer cursors')
    return {'schema':SCHEMA,'purpose':'throwaway_timing_only','resumable':False,'step':step,'next_crop':step%4,
        'config':config,'provenance':provenance,'proposal_spec':model.proposal_model.model_spec(),
        'graph_spec':model.graph_model.model_spec(),'model':cpu_snapshot(model.state_dict()),
        'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}


def restore_timing(cp,model,recipe,config,provenance):
    if (cp['schema']!=SCHEMA or cp['purpose']!='throwaway_timing_only' or cp['resumable'] is not False
            or cp['config']!=config or cp['provenance']!=provenance or type(cp['step']) is not int
            or not 0<=cp['step']<=8 or cp['next_crop']!=cp['step']%4
            or cp['proposal_spec']!=model.proposal_model.model_spec() or cp['graph_spec']!=model.graph_model.model_spec()):
        raise ValueError('Different nonresumable timing checkpoint/model/backend')
    model.load_state_dict(cp['model']);optimizer=runner.optimizer_for(model,recipe)
    optimizer.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    if stability.cursors(model,optimizer)!=expected_cursors(cp['step']):raise ValueError('Mixed optimizer cursor mismatch')
    return optimizer


def reload_check(model,optimizer,step,config,recipe,provenance,records,original,device,directory):
    predictions=runner.prediction_signature(model,records,original,device)
    before_weights=runner.state_hashes(model);before_optimizer=stability.control.optimizer_digests(optimizer)
    path=directory/'temporary_reload.pt';writing=directory/'temporary_reload.pt.writing'
    if path.exists() or writing.exists():raise FileExistsError('Preserve existing reload evidence')
    try:
        torch.save(timing_checkpoint(model,optimizer,step,config,provenance),writing);writing.replace(path)
        digest=runner.base.file_digest(path)
        optimizer=restore_timing(torch.load(path,map_location='cpu',weights_only=True),model,recipe,config,provenance)
        if before_weights!=runner.state_hashes(model) or before_optimizer!=stability.control.optimizer_digests(optimizer):
            raise ValueError('Reload changed weights or optimizer state')
        runner.require_identical_predictions(predictions,runner.prediction_signature(model,records,original,device))
        return {'prediction_max_delta':0.,'weights_and_optimizer_identical':True,'checkpoint_sha256':digest,
            'cursors':stability.cursors(model,optimizer),'temporary_checkpoint_deleted':True}
    finally:
        path.unlink(missing_ok=True);writing.unlink(missing_ok=True)


def run(args):
    config=runner.base.read(CONFIG)
    if config['schema']!=SCHEMA or config['updates']!=8 or config['seed']!=0 or config['torch_threads']!=2 or config['proposal_initial_cursor']!=4000 or config['gat_initial_cursor']!=0:
        raise ValueError('Different agreed timing contract')
    torch.set_num_threads(2);runner.configure();device=torch.device('cpu' if args.action=='check' else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    recipe=runner.recipe_at(ROOT/config['source_recipe']);original,records,oracle,protected=runner.load_inputs(recipe)
    if config['training_crops']!=recipe['training_crops']:raise ValueError('Different crop order')
    for name in ('learning_rate','weight_decay','gradient_clip'):
        if config[name]!=recipe[name]:raise ValueError('Changed training settings')
    summary_path=ROOT/config['cpu_stability_summary'];summary=runner.base.read(summary_path);cpu_path=ROOT/summary['report']
    if runner.base.file_digest(cpu_path)!=summary['report_sha256'] or not summary['gate']['passed']:raise ValueError('Expected completed CPU stability')
    cpu_report=runner.base.read(cpu_path)
    runner.base.verify_hashes(summary['sources'])
    protected.update(cpu_report['protected_files_sha256'])
    for path in (summary_path,cpu_path):protected[str(path)]=runner.base.file_digest(path)
    runner.base.verify_hashes(protected)
    sources=runner.base.source_hashes(SimpleNamespace(recipe=CONFIG))
    launcher=ROOT/'scripts/pointneuron2/run_joint_saved_adam_timing.ps1';sources[str(launcher)]=runner.base.file_digest(launcher)
    root=(args.directory or ROOT/config['output']).resolve()
    if not root.is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Outputs belong under ignored tmp')
    directory=root/args.action
    if directory.exists():raise FileExistsError('Preserve existing action outputs')
    provenance={'sources':sources,'cpu_stability_report_sha256':summary['report_sha256'],'backend':runner.backend(device)}
    if args.action=='timing':
        checked=runner.base.read(root/'check/report.json')
        if checked['status']!='completed' or not checked.get('cpu_adapter_check_passed') or checked['config']!=config or checked['provenance']['sources']!=sources or checked['provenance']['cpu_stability_report_sha256']!=summary['report_sha256']:
            raise ValueError('Matching completed zero-update CPU check required')
    directory.mkdir(parents=True);started=time.perf_counter()
    report={'schema':SCHEMA,'status':'running','action':args.action,'config':config,'provenance':provenance,
        'protected_files_sha256':protected,'completed_updates':0,'attempts':[],'snapshots':[],'evaluations':[],
        'retained_trained_checkpoint':False,'full_training_available':False}
    persist=lambda:runner.base.write(directory/'report.json',report)
    persist()
    try:
        model=runner.warm_start(recipe,device)
        cp=torch.load(ROOT/recipe['proposal_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
        optimizer=stability.mixed_optimizer(model,cp['optimizer'],recipe)
        if stability.cursors(model,optimizer)!=expected_cursors(0):raise ValueError('Different initialization')
        initial_weights=runner.state_hashes(model)
        failed=runner.base.read(ROOT/summary['config']['failed_fresh_timing'])
        if initial_weights!=failed['initial_state_sha256']:
            raise ValueError('Different warm-start state')
        print(f"ACTION={args.action} device={provenance['backend']['name']} | saved proposal Adam 4000 + fresh GAT Adam 0 | timing only",flush=True)
        report['evaluations'].append(runner.evaluate(model,records,oracle,original,recipe,device,0,directory))
        report['historical_warm_replay']=runner.historical_warm_replay(directory,recipe,device)
        probes=probe_all(model,records,original,recipe,device);report['snapshots'].append({'after_updates':0,'crops':probes});persist()
        if args.action=='check':
            prior=cpu_report['snapshots'][0]['crops']
            for actual,expected in zip(probes,prior):
                for name in ('matched_query_ids','matched_target_indices','matched_centers','matched_distances','accepted_assigned','accepted_unmatched'):
                    if actual[name]!=expected[name]:raise ValueError('Device adapter differs from verified CPU stability')
            report['reload']=reload_check(model,optimizer,0,config,recipe,provenance,records,original,device,directory)
            if initial_weights!=runner.state_hashes(model):raise ValueError('Zero-update check changed weights')
            report.update(model_parameters_unchanged=True,optimizer_updates=0,prior_cpu_probes_identical=True,cpu_adapter_check_passed=True)
        else:
            torch.cuda.reset_peak_memory_stats(device);durations=[]
            for step in range(1,9):
                attempt={'stage_update':step,'crop':config['training_crops'][(step-1)%4],'optimizer_step_returned':False}
                report['attempts'].append(attempt);persist();torch.cuda.synchronize(device);one=time.perf_counter()
                instrumented_update(model,optimizer,records[(step-1)%4],original,recipe,device,attempt,persist,directory)
                torch.cuda.synchronize(device);durations.append(time.perf_counter()-one)
                report['completed_updates']=step;attempt.update(seconds_with_incoming_diagnostics_and_io=durations[-1],cursors_after=stability.cursors(model,optimizer))
                if attempt['cursors_after']!=expected_cursors(step):raise ValueError('Mixed counters do not match completed stage')
                probes=probe_all(model,records,original,recipe,device)
                report['snapshots'].append({'after_updates':step,'crops':probes});persist()
                print(f"THROWAWAY {step}/8 loss={attempt['incoming']['loss']:.5f} trusted={[p['edge_supervision']['trusted_matches'] for p in probes]} pos_neg={[(p['edge_supervision']['positive'],p['edge_supervision']['negative']) for p in probes]} seconds={durations[-1]:.3f}",flush=True)
            report['stability_gate']=stability.stability_gate(report)
            report['evaluations'].append(runner.evaluate(model,records,oracle,original,recipe,device,8,directory));persist()
            report['peak_memory']={'allocated_bytes':torch.cuda.max_memory_allocated(device),'reserved_bytes':torch.cuda.max_memory_reserved(device),
                'device_total_bytes':torch.cuda.get_device_properties(device).total_memory,'scope':'updates and probes; PyTorch allocator only'}
            report['reload']=reload_check(model,optimizer,8,config,recipe,provenance,records,original,device,directory)
            report['timing_readiness_gate']=bool(report['stability_gate']['passed'] and report['reload']['weights_and_optimizer_identical'])
            estimate=float(np.mean(durations[4:]))*config['planning_updates_only']
            report.update(estimate_update_seconds_for_4000=estimate,planning_range_seconds=[estimate,1.5*estimate],
                estimate_excludes='post-update four-crop probes, evaluation/checkpoint IO and future candidate-density changes',
                next_step='Review stability, reload, memory and automatic outputs. Full trainer requires separate mixed-state integration.')
            print(f"STABILITY gate={report['stability_gate']} | peak allocated={report['peak_memory']['allocated_bytes']/2**30:.3f} GiB | reload delta={report['reload']['prediction_max_delta']}",flush=True)
            print(f"UPDATE-TIME ESTIMATE {estimate/60:.1f}-{1.5*estimate/60:.1f} minutes for 4000; excludes evaluation/checkpoint IO. No training action exists here.",flush=True)
        runner.base.verify_hashes(protected);runner.base.verify_hashes(sources)
        report.update(status='completed',original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-started);persist()
    except BaseException as exc:
        counts=report['attempts'][-1].get('incoming',{}).get('edge_supervision',{}) if report['attempts'] else {}
        report.update(status='failed',error=str(exc),stopped_for_missing_edge_class=bool(counts and (not counts['positive'] or not counts['negative'])))
        (directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');persist();raise
    print(f"COMPLETED {args.action}; updates={report['completed_updates']} report={directory/'report.json'}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','timing'),required=True)
    parser.add_argument('--directory',type=Path)
    run(parser.parse_args())
