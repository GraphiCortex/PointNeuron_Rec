"""Matched four-update CPU control: fresh versus saved AdamW state only."""
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import audit_joint_timing_stop as diagnostic
from scripts.pointneuron2.balanced_query_experiment import update
from pointneuron.models.query_fit_state import cpu_snapshot

runner=diagnostic.runner
CONFIG=ROOT/'configs/pointneuron2/proposal_optimizer_state_control.json'


def make_optimizer(proposal,saved,*,restored,expected_step=4000):
    """Copy identical group options in both arms; vary moments and counter only."""
    parameters=list(proposal.parameters());groups=saved['param_groups']
    if len(groups)!=1 or groups[0]['params']!=list(range(len(parameters))):
        raise ValueError('Expected original single-group constructor parameter order')
    if set(saved['state'])!=set(groups[0]['params']):raise ValueError('Incomplete source optimizer state')
    for i,p in enumerate(parameters):
        state=saved['state'][i]
        if float(state['step'])!=expected_step:raise ValueError('Different source optimizer cursor')
        for name in ('exp_avg','exp_avg_sq'):
            value=state[name]
            if value.shape!=p.shape or value.dtype!=p.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Misaligned/nonfinite optimizer moments')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative second moment')
    payload=cpu_snapshot(saved)
    if not restored:payload['state']={}
    optimizer=torch.optim.AdamW(parameters)
    optimizer.load_state_dict(payload)
    if optimizer.state_dict()['param_groups']!=groups:raise ValueError('Optimizer options changed')
    if not restored and optimizer.state:raise ValueError('Fresh arm inherited moments')
    return optimizer


def optimizer_digests(optimizer):
    return {str(i):{k:runner.base.tensor_digest(v) if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        for i,state in optimizer.state_dict()['state'].items()}


def probe_all(model,records,original,recipe,initial=None):
    probes=[diagnostic.inspect(model,r,original,recipe) for r in records]
    if initial is not None:
        for p,before in zip(probes,initial):
            old=dict(zip(before['matched_target_indices'],before['matched_query_ids']))
            p['targets_assigned_to_different_queries']=sum(old[t]!=q for t,q in zip(p['matched_target_indices'],p['matched_query_ids']))
            p['trusted_match_change_from_initial']=p['edge_supervision']['trusted_matches']-before['edge_supervision']['trusted_matches']
    return probes


def run():
    config=runner.base.read(CONFIG)
    if (config['updates_per_arm']!=4 or config['device']!='cpu' or config['arms']!=['fresh','restored']
            or config['training_crops']!=[[0,0],[0,1],[92,4],[86,3]]):raise ValueError('Different agreed control')
    torch.set_num_threads(config['torch_threads']);runner.configure();started=time.perf_counter()
    recipe=runner.recipe_at(ROOT/config['source_recipe']);original,records,_,protected=runner.load_inputs(recipe)
    source=ROOT/config['source_failed_timing'];failed=runner.base.read(source)
    if failed['status']!='failed' or failed['recipe']!=recipe:raise ValueError('Expected preserved failed timing')
    runner.base.verify_hashes(failed['provenance']['sources'])
    for name in ('learning_rate','weight_decay','gradient_clip'):
        if recipe[name]!=config[name]:raise ValueError('Changed update hyperparameters')
    cp=torch.load(ROOT/recipe['proposal_experiment']/'model/step_4000.pt',map_location='cpu',weights_only=True)
    if cp['step']!=4000 or cp['experiment_schema']!='pointneuron2-balanced-query-comparison-v1':raise ValueError('Different proposal fit')
    saved=cp['optimizer']
    if saved['param_groups'][0]['lr']!=config['learning_rate'] or saved['param_groups'][0]['weight_decay']!=config['weight_decay']:
        raise ValueError('Different source optimizer options')
    sources=runner.base.source_hashes(SimpleNamespace(recipe=CONFIG))
    output=ROOT/config['output']
    if not output.resolve().is_relative_to((ROOT/'tmp/pointneuron2').resolve()):raise ValueError('Output outside ignored tmp')
    if output.exists():raise FileExistsError('Preserve control evidence')
    output.mkdir(parents=True)
    report={'schema':config['schema'],'status':'running','config':config,'sources':sources,
        'source_failed_timing_sha256':runner.base.file_digest(source),'protected_files_sha256':protected,
        'arms':[],'retained_trained_checkpoint':False,'gpu_work':False,'primary_criterion':config['primary_criterion']}
    runner.base.write(output/'report.json',report)
    try:
        initial_hashes=None;initial_probes=None;first_options=None
        for arm in config['arms']:
            runner.configure()
            model=runner.warm_start(recipe,'cpu');hashes=runner.state_hashes(model)
            if hashes!=failed['initial_state_sha256']:raise ValueError('Different warm-start model')
            if initial_hashes is None:initial_hashes=hashes
            if hashes!=initial_hashes:raise ValueError('Arm weights differ')
            names=[n for n,_ in model.proposal_model.named_parameters()]
            if [n for n in cp['inference_checkpoint']['model'] if n in set(names)]!=names:
                raise ValueError('Source parameter order differs')
            optimizer=make_optimizer(model.proposal_model,saved,restored=arm=='restored')
            options=optimizer.state_dict()['param_groups']
            if first_options is None:first_options=options
            if options!=first_options:raise ValueError('Arms have different optimizer options')
            probes=probe_all(model,records,original,recipe)
            if initial_probes is None:initial_probes=probes
            if probes!=initial_probes:raise ValueError('Arm initial predictions differ')
            row={'arm':arm,'initial_optimizer_state':optimizer_digests(optimizer),'optimizer_options':options,
                'snapshots':[{'after_updates':0,'crops':probes}],'updates':[]}
            report['arms'].append(row);runner.base.write(output/'report.json',report)
            for i in range(4):
                metrics=update(model.proposal_model,optimizer,records[i],dict(original,gradient_clip=config['gradient_clip']),'cpu')
                row['updates'].append({'new_stage_update':i+1,'crop':config['training_crops'][i],'metrics':metrics})
                probes=probe_all(model,records,original,recipe,initial_probes)
                row['snapshots'].append({'after_updates':i+1,'crops':probes})
                print(f"CPU arm={arm} update={i+1}/4 trusted={[p['edge_supervision']['trusted_matches'] for p in probes]} max_errors={[round(p['matched_distance_max'],3) for p in probes]}",flush=True)
                runner.base.write(output/'report.json',report)
            cursor={float(s['step']) for s in optimizer.state.values()}
            if cursor!=({4004.} if arm=='restored' else {4.}):raise ValueError('Optimizer cursor did not advance exactly four times')
            if any(runner.base.tensor_digest(v)!=hashes['graph_model.'+k] for k,v in model.graph_model.state_dict().items()):raise ValueError('Frozen GAT changed')
            row.update(completed_updates=4,optimizer_cursor=sorted(cursor),graph_weights_unchanged=True,
                primary_pass=probes[0]['edge_supervision']['trusted_matches']==6,final_optimizer_state=optimizer_digests(optimizer))
            if arm=='fresh':
                prior=runner.base.read(ROOT/config['fresh_reference'])
                row['prior_fresh_cpu_probes_identical']=all(new['crops'][0]['matched_distances']==old['matched_distances']
                    for new,old in zip(row['snapshots'],prior['crop_zero_probes']))
                if not row['prior_fresh_cpu_probes_identical']:raise ValueError('Fresh reference did not reproduce')
            del model,optimizer
        runner.base.verify_hashes(protected);runner.base.verify_hashes(failed['provenance']['sources']);runner.base.verify_hashes(sources)
        report.update(status='completed',original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-started,
            same_initial_weights_and_predictions=True,same_optimizer_options=True)
        runner.base.write(output/'report.json',report)
        print(f"COMPLETED control; report={output/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc));runner.base.write(output/'report.json',report)
        (output/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':run()
