"""Prepare/check, separately time, then fit a capped joint TRAINING-crop prototype."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import evaluate_gat_automatic_selection as audit
from pointneuron.models.joint_query_graph import JointQueryGraph,joint_loss,retained_graph
from pointneuron.models.pointnext_query_cropnorm import load_cropnorm_checkpoint
from pointneuron.models.gatv2_connectivity import GATv2Connectivity,ConnectivityConfig
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng,restore_rng
from pointneuron.graph.proposal_graph import ProposalGraphInput,CandidateGraph
from pointneuron.graph.connectivity_decoder import export_decoded_swc

base=audit.base
SCHEMA='pointneuron2-joint-query-graph-v1'


def recipe_at(path):
    recipe=base.read(path)
    if (recipe!=base.read(ROOT/'configs/pointneuron2/joint_query_graph.json') or recipe['schema']!=SCHEMA
            or recipe['updates']!=4000 or recipe['query_count']!=512 or recipe['node_cutoff']!=.5 or recipe['edge_cutoff']!=.5
            or recipe['minimum_arclength_precision_2']!=.99 or recipe['minimum_arclength_recall_2']!=.99):
        raise ValueError('Different joint experiment contract')
    return recipe


def sources(args):
    hashes=base.source_hashes(SimpleNamespace(recipe=args.recipe))
    path=ROOT/'scripts/pointneuron2/run_joint_query_graph.ps1';hashes[str(path)]=base.file_digest(path)
    return hashes


def load_inputs(recipe):
    source=ROOT/recipe['proposal_experiment'];gat_dir=ROOT/recipe['gat_experiment']
    protected={str(source/'model/step_4000.pt'):recipe['proposal_checkpoint_sha256'],
        str(source/'model/report.json'):recipe['proposal_report_sha256'],
        str(gat_dir/'model/step_4000.pt'):recipe['gat_checkpoint_sha256'],
        str(gat_dir/'model/report.json'):recipe['gat_report_sha256']}
    base.verify_hashes(protected)
    original=base.recipe_at(ROOT/'configs/pointneuron2/branched_query_fit.json')
    manifest,records=base.load_records(SimpleNamespace(experiment_dir=source),original)
    gat_report=base.read(gat_dir/'model/report.json')
    graph_manifest,oracle=audit.stage.load_cache(SimpleNamespace(directory=gat_dir),gat_report['recipe'])
    protected.update(manifest['protected_files_sha256']);protected.update(graph_manifest['protected_files_sha256'])
    protected.update(gat_report['provenance']['sources'])
    protected[str(source/'cache/manifest.json')]=base.file_digest(source/'cache/manifest.json')
    protected[str(gat_dir/'cache/manifest.json')]=base.file_digest(gat_dir/'cache/manifest.json')
    protected[str(gat_dir/'cache/graphs.pt')]=graph_manifest['cache_sha256']
    if original['training_crops']!=recipe['training_crops']:raise ValueError('Different crop identities')
    base.verify_hashes(protected)
    return original,records,oracle,protected


def warm_start(recipe,device):
    proposal_cp=torch.load(ROOT/recipe['proposal_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
    proposal=load_cropnorm_checkpoint(proposal_cp['inference_checkpoint'])
    gat_cp=torch.load(ROOT/recipe['gat_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
    if proposal_cp['step']!=4000 or gat_cp['step']!=4000:raise ValueError('Incomplete warm start')
    gat=GATv2Connectivity(ConnectivityConfig(**gat_cp['recipe']['model']))
    if gat.model_spec()!=gat_cp['spec']:raise ValueError('Different GAT architecture')
    gat.load_state_dict(gat_cp['model'])
    for loaded,saved in ((proposal.state_dict(),proposal_cp['inference_checkpoint']['model']),
                         (gat.state_dict(),gat_cp['model'])):
        if any(not torch.equal(loaded[k],saved[k]) for k in loaded):raise ValueError('Warm-start weight transfer differs')
    return JointQueryGraph(proposal,gat,neighbors=recipe['candidate_neighbors'],radius=recipe['candidate_radius']).to(device)


def optimizer_for(model,recipe):
    return torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])


def forward(model,record,original,device):
    return model(torch.from_numpy(record['points'])[None].to(device),
        torch.tensor([record['meta']['patch_center']],dtype=torch.float32,device=device),
        torch.tensor([record['meta']['image_scale']],dtype=torch.float32,device=device),original['patch_radius'])


def loss(output,record,original,recipe):
    return joint_loss(output,record,original['query_loss'],edge_weight=recipe['edge_weight'],match_tolerance=recipe['trusted_match_tolerance'])


def state_hashes(model):
    return {k:base.tensor_digest(v) for k,v in model.state_dict().items()}


def grad_norm(parameters):
    values=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def update(model,optimizer,record,original,recipe,device):
    model.train();optimizer.zero_grad(set_to_none=True)
    output=forward(model,record,original,device);total,proposal,edge,labels=loss(output,record,original,recipe)
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite joint loss')
    total.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Missing/nonfinite joint gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite joint parameters')
    return {'loss':float(total.detach()),'proposal_loss':float(proposal.total.detach()),'edge_loss':float(edge.total.detach()),
        'preclip_norm':float(norm),'candidate_edges':len(output.candidates.edges),'edge_supervision':labels.statistics}


def cpu_packet(p):
    return ProposalGraphInput(*(getattr(p,k).detach().cpu() for k in audit.stage.FIELDS),p.provenance)


def automatic_gate(row,total_targets):
    geometry=row['geometry']['tolerances']['2.0']
    return bool(row['assigned_nodes']==total_targets and row['unmatched_nodes']==0
        and row['decoded_missing_required_edges']==0 and row['decoded_nonreference_edges']==0
        and row['decoding']['components']==row['reference_components']
        and row['retained_trusted_junctions']==row['reference_junctions'] and row['retained_junction_incidence_errors']==0
        and geometry['precision'] is not None and geometry['precision']>=.99 and geometry['recall']>=.99
        and row.get('swc_roundtrip',{}).get('passed',False))


def evaluate(model,records,oracle,original,recipe,device,step,directory):
    model.eval();rows=[]
    with torch.no_grad():
        for i,(record,reference_record) in enumerate(zip(records,oracle)):
            output=forward(model,record,original,device)
            total,proposal,edge,labels=loss(output,record,original,recipe)
            selected,c,logits=retained_graph(output);p=cpu_packet(selected)
            c=CandidateGraph(c.edges.cpu(),c.neighbors,c.radius);logits=logits.cpu()
            pi,ti=map(np.asarray,proposal.assignments[0])
            saved={'pred_indices':pi,'target_indices':ti}
            row,decoded,evaluation_labels=audit.assess(p,c,logits,saved,record,audit.stage.reference_graph(reference_record))
            q2t=dict(zip(pi.tolist(),ti.tolist()))
            row['retained_trusted_junctions']=sum(bool(record['junction_mask'][q2t[int(q)]])
                and np.linalg.norm(p.positions[j].numpy().astype(float)-record['nodes'][q2t[int(q)],1:4])<=2
                for j,q in enumerate(p.node_ids) if int(q) in q2t)
            row['retained_trusted_junctions']=int(row['retained_trusted_junctions'])
            if decoded.status=='decoded':
                path=directory/f'step_{step:04d}_crop_{i}.swc'
                row['swc_roundtrip']=export_decoded_swc(p,decoded,path,description='JOINT TRAINING-CROP DIAGNOSTIC: all 512 queries scored by GAT; original confidence filters final nodes; no oracle selection.')
                row['swc_sha256']=base.file_digest(path)
            else:row['empty_reconstruction']=True
            row.update(sample_index=record['identity']['sample_index'],patch_index=record['identity']['patch_index'],
                loss=float(total),proposal_loss=float(proposal.total),edge_loss=float(edge.total),
                all_query_candidates=len(output.candidates.edges),training_edge_supervision=labels.statistics,
                radius_mae=float(np.abs(output.proposals.radius.cpu().numpy()[0,pi,0]-np.maximum(record['nodes'][ti,4],1)).mean()))
            row['automatic_gate']=automatic_gate(row,len(record['nodes']))
            path=directory/f'step_{step:04d}_crop_{i}.npz'
            np.savez_compressed(path,centers=output.proposals.center_proposals.cpu().numpy(),
                radii=output.proposals.radius.cpu().numpy(),logits=output.proposals.objectness_logits.cpu().numpy(),
                decoder_features=output.proposals.decoder_features.cpu().numpy(),pred_indices=pi,target_indices=ti,
                candidate_edges=output.candidates.edges.cpu().numpy(),edge_logits=output.edge_logits.cpu().numpy(),
                edge_labels=labels.labels.cpu().numpy(),retained_node_ids=p.node_ids.numpy(),retained_edges=c.edges.numpy(),
                retained_edge_logits=logits.numpy(),evaluation_labels=evaluation_labels)
            row['prediction_sha256']=base.file_digest(path);rows.append(row)
            g=row['geometry']['tolerances']['2.0']
            print(f"EVAL stage={step} train={row['sample_index']}/{row['patch_index']} nodes={row['assigned_nodes']}/{len(record['nodes'])}+{row['unmatched_nodes']} edges={row['decoded_required_edges']}/{row['required_reference_edges']} P/R@2={g['precision']}/{g['recall']} gate={row['automatic_gate']}",flush=True)
    return {'stage_step':step,'crops':rows,'automatic_gate':all(r['automatic_gate'] for r in rows)}


def checkpoint(model,optimizer,step,recipe,provenance,purpose='fit'):
    if type(step) is not int or not 0<=step<=recipe['updates']:raise ValueError('Invalid joint stage step')
    if {int(s['step']) for s in optimizer.state.values()}!=({step} if step else set()):raise ValueError('Optimizer cursor differs')
    return {'schema':SCHEMA,'step':step,'next_crop':step%4,'recipe':recipe,'provenance':provenance,'purpose':purpose,
        'resumable':purpose=='fit' and step<recipe['updates'],
        'proposal_spec':model.proposal_model.model_spec(),'graph_spec':model.graph_model.model_spec(),
        'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}


def restore(cp,model,recipe,provenance,*,resume):
    if (cp['schema']!=SCHEMA or cp['recipe']!=recipe or cp['provenance']!=provenance
        or type(cp['step']) is not int or not 0<=cp['step']<=recipe['updates'] or cp['next_crop']!=cp['step']%4
        or cp['proposal_spec']!=model.proposal_model.model_spec() or cp['graph_spec']!=model.graph_model.model_spec()):
        raise ValueError('Different joint experiment/model/cursor')
    if resume and (not cp['resumable'] or cp['purpose']!='fit' or cp['step']>=recipe['updates']):raise ValueError('Terminal/check-only checkpoint cannot resume')
    model.load_state_dict(cp['model']);optimizer=optimizer_for(model,recipe);optimizer.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    if {int(s['step']) for s in optimizer.state.values()}!=({cp['step']} if cp['step'] else set()):raise ValueError('Optimizer cursor differs')
    return optimizer,cp['step']


def snapshot(model,optimizer,step,recipe,provenance,report,directory,records,oracle,original,device):
    path=directory/f'step_{step:04d}.pt';temporary=path.with_suffix('.pt.writing')
    if path.exists() or temporary.exists():raise FileExistsError('Preserve checkpoints')
    torch.save(checkpoint(model,optimizer,step,recipe,provenance),temporary);temporary.replace(path)
    report.update(last_checkpoint=path.name,last_checkpoint_step=step,last_checkpoint_sha256=base.file_digest(path))
    base.write(directory/'report.json',report)
    report['evaluations'].append(evaluate(model,records,oracle,original,recipe,device,step,directory))
    base.write(directory/'report.json',report)


def prediction_signature(model,records,original,device):
    model.eval();result=[]
    with torch.no_grad():
        for record in records:
            output=forward(model,record,original,device)
            result.append(tuple(v.detach().cpu().clone() for v in (output.proposals.center_proposals,
                output.proposals.radius,output.proposals.objectness_logits,output.candidates.edges,output.edge_logits)))
    return result


def require_identical_predictions(before,after):
    if len(before)!=len(after) or any(not torch.equal(a,b) for left,right in zip(before,after) for a,b in zip(left,right)):
        raise ValueError('Checkpoint reload changed proposal/candidate/edge predictions')
    return 0.


def historical_warm_replay(directory,recipe,device):
    diagnostics=[]
    for i in range(4):
        with np.load(directory/f'step_0000_crop_{i}.npz') as current, np.load(
                ROOT/recipe['proposal_experiment']/f'model/step_4000_crop_{i}.npz') as saved:
            if device.type=='cuda':
                for key in ('centers','radii','logits','decoder_features'):
                    torch.testing.assert_close(torch.from_numpy(current[key]),torch.from_numpy(saved[key]),rtol=1e-5,atol=1e-5)
            diagnostics.append({'crop_index':i,'max_center_coordinate_delta':float(np.abs(current['centers']-saved['centers']).max()),
                'max_logit_delta':float(np.abs(current['logits']-saved['logits']).max()),
                'node_decision_changes':int(np.sum((current['logits'][0,:,1]>=current['logits'][0,:,0])!=
                    (saved['logits'][0,:,1]>=saved['logits'][0,:,0]))),
                'comparison':'same_GPU_saved_prediction_tolerance_1e-5' if device.type=='cuda' else 'cross_backend_report_only'})
    return diagnostics


def check_cpu(model,optimizer,records,oracle,original,recipe,provenance,directory,report):
    before=state_hashes(model);report['evaluations'].append(evaluate(model,records,oracle,original,recipe,'cpu',0,directory))
    report['historical_warm_replay']=historical_warm_replay(directory,recipe,torch.device('cpu'))
    diagnostics=[]
    for i,record in enumerate(records):
        model.train();model.zero_grad(set_to_none=True);output=forward(model,record,original,'cpu')
        for value in (output.proposals.center_proposals,output.proposals.objectness_logits,output.proposals.radius,output.proposals.decoder_features):value.retain_grad()
        _,_,edge,_=loss(output,record,original,recipe);edge.total.backward()
        groups={'encoder':model.proposal_model.encoder.parameters(),'transformer_proposal':model.proposal_model.proposal.parameters(),'gat':model.graph_model.parameters(),
            'centers':[output.proposals.center_proposals],'confidence':[output.proposals.objectness_logits],
            'radii':[output.proposals.radius],'decoder_features':[output.proposals.decoder_features]}
        norms={k:grad_norm(v) for k,v in groups.items()}
        if not all(np.isfinite(v) and v>0 for v in norms.values()):raise ValueError(f'Edge-only gradient path failed: {norms}')
        model.zero_grad(set_to_none=True);output=forward(model,record,original,'cpu')
        total,_,_,_=loss(output,record,original,recipe);total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('Joint gradients missing/nonfinite')
        model.eval()
        other=forward(model,record,original,'cpu')
        if not torch.equal(output.proposals.objectness_logits.detach(),other.proposals.objectness_logits) or not torch.equal(output.edge_logits.detach(),other.edge_logits):
            raise ValueError('Train/eval predictions differ')
        # Check adapter equality with identical backend, mode and autograd context.
        standalone=base.forward(model.proposal_model,record,original,'cpu')
        for name in ('center_proposals','objectness_logits','radius','local_centers','decoder_features'):
            if not torch.equal(getattr(other.proposals,name),getattr(standalone,name)):raise ValueError('Adapter changed proposal outputs')
        with torch.no_grad():inference=forward(model,record,original,'cpu')
        for name in ('center_proposals','objectness_logits','radius','local_centers','decoder_features'):
            torch.testing.assert_close(getattr(other.proposals,name),getattr(inference.proposals,name),rtol=1e-5,atol=1e-5)
        diagnostics.append({'crop_index':i,'edge_only_gradient_l2':norms,'adapter_max_delta':0.,
            'all_query_candidates':len(output.candidates.edges),'train_eval_grad_enabled_max_delta':0.,
            'no_grad_max_center_coordinate_delta':float((other.proposals.center_proposals.detach()-inference.proposals.center_proposals).abs().max()),
            'no_grad_max_confidence_logit_delta':float((other.proposals.objectness_logits.detach()-inference.proposals.objectness_logits).abs().max())})
        print(f"CPU_GRADIENT crop={i} candidates={len(output.candidates.edges)} edge_to_encoder={norms['encoder']:.6g} edge_to_confidence={norms['confidence']:.6g}",flush=True)
    model.zero_grad(set_to_none=True)
    if before!=state_hashes(model):raise ValueError('Backward-only check changed model')
    predictions=prediction_signature(model,records,original,'cpu')
    path=directory/'untrained_reload.pt';torch.save(checkpoint(model,optimizer,0,recipe,provenance,purpose='check'),path)
    cp=torch.load(path,map_location='cpu',weights_only=True);restore(cp,model,recipe,provenance,resume=False);path.unlink()
    if before!=state_hashes(model):raise ValueError('Reload changed model')
    require_identical_predictions(predictions,prediction_signature(model,records,original,'cpu'))
    report.update(edge_gradient_diagnostics=diagnostics,model_parameters_unchanged=True,reload_max_delta=0.,optimizer_updates=0)


def backend(device):
    return {'device':str(device),'torch':str(torch.__version__),'threads':torch.get_num_threads(),
        'name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','deterministic_algorithms':True,'tf32':False}


def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(0);np.random.seed(0);torch.manual_seed(0)


def run(args):
    recipe=recipe_at(args.recipe);args.directory=args.directory.resolve()
    if not args.directory.is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Generated outputs belong under ignored tmp/pointneuron2')
    if args.action in ('prepare','check') and args.device!='cpu':raise ValueError('Preparation/check are CPU only')
    torch.set_num_threads(recipe['torch_threads']);configure();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    original,records,oracle,protected=load_inputs(recipe)
    if args.action=='prepare':
        if args.directory.exists():raise FileExistsError('Preserve existing joint outputs')
        args.directory.mkdir(parents=True)
        base.write(args.directory/'manifest.json',{'schema':SCHEMA+'-inputs','recipe':recipe,'protected_files_sha256':protected,
            'input_record_digests':[r['identity']['sha256'] for r in records],'optimizer_updates':0})
        print('PREPARED verified original frozen image clouds and two warm-start checkpoints; no model forward or update',flush=True);return
    manifest=base.read(args.directory/'manifest.json')
    if manifest['schema']!=SCHEMA+'-inputs' or manifest['recipe']!=recipe or manifest['protected_files_sha256']!=protected:
        raise ValueError('Prepared input manifest differs')
    code=sources(args);provenance={'manifest_sha256':base.file_digest(args.directory/'manifest.json'),'sources':code,'backend':backend(device)}
    directory=args.directory/('model' if args.action=='train' else args.action)
    if args.action=='resume':
        if args.checkpoint is None or not args.checkpoint.resolve().is_relative_to(args.directory) or not (args.checkpoint.parent.name=='model' or args.checkpoint.parent.name.startswith('resume_')):raise ValueError('Explicit checkpoint from this joint stage required')
        directory=args.directory/f'resume_{args.checkpoint.stem}'
    if directory.exists():raise FileExistsError('Existing action outputs cannot be overwritten')
    if args.action in ('timing','train','resume'):
        checked=base.read(args.directory/'check/report.json')
        if checked['status']!='completed' or checked['recipe']!=recipe or checked['provenance']['sources']!=code or checked['provenance']['manifest_sha256']!=provenance['manifest_sha256']:raise ValueError('Matching completed CPU check required')
    if args.action in ('train','resume'):
        timed=base.read(args.directory/'timing/report.json')
        if timed['status']!='completed' or timed['recipe']!=recipe or timed['provenance']!=provenance:raise ValueError('Matching timing on this backend required')
    directory.mkdir(parents=True);start_time=time.perf_counter()
    report={'schema':SCHEMA,'status':'running','action':args.action,'recipe':recipe,'provenance':provenance,
        'optimizer_updates':0,'phase1_frozen':True,'evaluations':[],'scope':'four fixed TRAINING crops; all 512 queries; no oracle selection'}
    base.write(directory/'report.json',report)
    try:
        for path in code:
            target=directory/'sources'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        model=warm_start(recipe,device);optimizer=optimizer_for(model,recipe);start=0
        if args.action=='resume':
            optimizer,start=restore(torch.load(args.checkpoint,map_location='cpu',weights_only=True),model,recipe,provenance,resume=True)
            report.update(resumed_from=str(args.checkpoint),optimizer_updates=start)
        report.update(parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=state_hashes(model))
        print(f"ACTION={args.action} device={provenance['backend']['name']} NEW_STAGE cap=4000 | all 512 queries to GAT before final .5 filter",flush=True)
        if args.action=='check':check_cpu(model,optimizer,records,oracle,original,recipe,provenance,directory,report)
        elif args.action=='timing':
            report['evaluations'].append(evaluate(model,records,oracle,original,recipe,device,0,directory))
            report['historical_warm_replay']=historical_warm_replay(directory,recipe,device)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            durations=[];candidate_counts=[]
            for j in range(recipe['timing_updates']):
                if device.type=='cuda':torch.cuda.synchronize()
                one=time.perf_counter();metrics=update(model,optimizer,records[j%4],original,recipe,device)
                if device.type=='cuda':torch.cuda.synchronize()
                durations.append(time.perf_counter()-one);candidate_counts.append(metrics['candidate_edges'])
                print(f"THROWAWAY {j+1}/8 loss={metrics['loss']:.5f} candidates={metrics['candidate_edges']} seconds={durations[-1]:.3f}",flush=True)
            predictions=prediction_signature(model,records,original,device)
            path=directory/'throwaway.pt';torch.save(checkpoint(model,optimizer,8,recipe,provenance,purpose='timing'),path)
            before=state_hashes(model);restore(torch.load(path,map_location='cpu',weights_only=True),model,recipe,provenance,resume=False);path.unlink()
            if before!=state_hashes(model):raise ValueError('Timing reload changed weights')
            require_identical_predictions(predictions,prediction_signature(model,records,original,device))
            estimate=float(np.mean(durations[4:]))*recipe['updates']
            report.update(throwaway_optimizer_updates=8,retained_trained_checkpoint=False,reload_max_delta=0.,
                estimated_update_seconds=estimate,planning_range_seconds=[estimate,1.5*estimate+120],timing_candidates=candidate_counts,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            print(f"ESTIMATE {estimate/60:.1f}-{(1.5*estimate+120)/60:.1f} minutes plus evaluation/checkpoint IO; train is separate",flush=True)
        else:
            clipped=0;maximum_norm=0.;maximum_candidates=0
            for step in range(start,recipe['updates']+1):
                if step>start:
                    metrics=update(model,optimizer,records[(step-1)%4],original,recipe,device)
                    clipped+=metrics['preclip_norm']>recipe['gradient_clip'];maximum_norm=max(maximum_norm,metrics['preclip_norm']);maximum_candidates=max(maximum_candidates,metrics['candidate_edges'])
                    report.update(optimizer_updates=step,clipped_updates_this_invocation=clipped,maximum_preclip_norm_this_invocation=maximum_norm,maximum_candidates_this_invocation=maximum_candidates)
                    if step%50==0:print(f"UPDATE {step}/4000 crop={(step-1)%4} loss={metrics['loss']:.5f} proposal={metrics['proposal_loss']:.5f} edge={metrics['edge_loss']:.5f} candidates={metrics['candidate_edges']} elapsed={time.perf_counter()-start_time:.1f}s",flush=True)
                if step in recipe['snapshots'] and (args.action!='resume' or step>start):snapshot(model,optimizer,step,recipe,provenance,report,directory,records,oracle,original,device)
            predictions=prediction_signature(model,records,original,device)
            before=state_hashes(model);restore(torch.load(directory/'step_4000.pt',map_location='cpu',weights_only=True),model,recipe,provenance,resume=False)
            if before!=state_hashes(model):raise ValueError('Terminal reload changed weights')
            require_identical_predictions(predictions,prediction_signature(model,records,original,device))
            report.update(reload_max_delta=0.,automatic_gate=report['evaluations'][-1]['automatic_gate'])
        base.verify_hashes(protected);base.verify_hashes(code)
        report.update(status='completed',elapsed_seconds=time.perf_counter()-start_time,original_inputs_unchanged=True,
            next_step='Review automatic geometry/node/graph gates. No extension, new node head, threshold change or production promotion.')
        base.write(directory/'report.json',report);print(f"COMPLETED {args.action}; report={directory/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        base.write(directory/'report.json',report);(directory/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','check','timing','train','resume'),required=True)
    parser.add_argument('--recipe',type=Path,default=ROOT/'configs/pointneuron2/joint_query_graph.json')
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--checkpoint',type=Path)
    run(parser.parse_args())
