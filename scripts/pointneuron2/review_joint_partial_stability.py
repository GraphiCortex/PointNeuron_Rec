"""Read-only CPU audit of the completed paired GPU controls and saved reconstructions."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_partial_stability as control
from scripts.pointneuron2 import review_joint_saved_adam_timing as reviewer
from pointneuron.data.connectivity_targets import assign_edge_labels
from pointneuron.graph.proposal_graph import CandidateGraph


def checkpoint_audit(saved,parent,lr,steps):
    assert saved['schema']==control.SCHEMA and not saved['resumable']
    assert saved['proposal_updates']==steps and saved['graph_updates']==steps and saved['learning_rate']==lr
    names=saved['parameter_names'];group=saved['optimizer']['param_groups'][0]
    assert names==parent['parameter_names'] and len(group['params'])==len(names)
    for name,index in zip(names,group['params']):
        state=saved['optimizer']['state'][index];parameter=saved['model'][name]
        wanted=5000+steps if name.startswith('proposal_model.') else 1000+steps
        assert float(state['step'])==wanted
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].shape==parameter.shape and state[key].dtype==parameter.dtype and bool(torch.isfinite(state[key]).all())
    if steps==0:
        assert control.equal(saved['model'],parent['model'])
        assert control.equal(saved['optimizer']['state'],parent['optimizer']['state'])
        options=control.cpu_snapshot(parent['optimizer']['param_groups']);options[0]['lr']=lr
        assert saved['optimizer']['param_groups']==options
        assert control.equal(saved['rng'],parent['rng'])


def confidence_summary(saved):
    difference=saved['logits'][0].astype(float)[:,1]-saved['logits'][0].astype(float)[:,0]
    positive=np.zeros(len(difference),bool);positive[saved['pred_indices']]=True
    low=float(difference[positive].min());high=float(difference[~positive].max())
    return {'minimum_assigned_logit_margin':low,'maximum_unmatched_logit_margin':high,
        'minimum_assigned_probability':float(torch.sigmoid(torch.tensor(low,dtype=torch.float64))),
        'maximum_unmatched_probability':float(torch.sigmoid(torch.tensor(high,dtype=torch.float64))),
        'fixed_geometry_threshold_separation_possible':high<low}


def run():
    torch.set_num_threads(2);base=control.base;audit=control.runner.audit
    config=base.read(control.CONFIG);control.validate_config(config)
    directory=ROOT/config['output']/'comparison';path=directory/'report.json';report=base.read(path)
    assert report['status']=='completed' and report['config']==config and report['sources_and_parent_unchanged']
    base.verify_hashes(report['provenance']['sources']);base.verify_hashes(report['protected_files_sha256'])
    recipe,original,records,oracle,parent,parent_report,protected,parent_dir=control.load_inputs(config)
    rows=[];arms=[];initial_states=[]
    old_updates={a['stage_update']:a for a in map(json.loads,(parent_dir/'updates.jsonl').read_text(encoding='utf-8').splitlines())}
    for arm in report['arms']:
        folder=directory/arm['name'];own=base.read(folder/'report.json')
        assert own==arm and arm['status']=='completed' and arm['proposal_updates']==arm['graph_updates']==64
        assert control.qualification(arm,config)==arm['qualification']
        ledger=[json.loads(line) for line in (folder/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
        assert ledger==arm['updates'] and [a['fork_update'] for a in ledger]==list(range(1,65))
        for attempt in ledger:
            step=attempt['fork_update'];assert attempt['optimizer_step_returned']
            assert attempt['cursors_after']=={'proposal':[float(5000+step)],'gat':[float(1000+step)]}
            assert attempt['edge_active']==bool(attempt['incoming']['edge_supervision']['positive']+attempt['incoming']['edge_supervision']['negative'])
        if arm['name']=='current_lr':
            assert arm['replay_verified_steps']==list(range(1001,1014)) and arm['guard_transition_verified']
            for attempt in ledger[:13]:
                control.compare_values(attempt,{k:v for k,v in old_updates[attempt['stage_update']].items() if k!='incoming'})
                control.compare_values(attempt['incoming'],old_updates[attempt['stage_update']]['incoming'])
            control.compare_values(ledger[13]['incoming'],{k:v for k,v in base.read(parent_dir/'active_update.json')['incoming'].items()
                if k not in ('matched_centers','matched_distances','matched_query_ids','matched_target_indices')})
        for step,key in ((0,'initial_checkpoint_sha256'),(64,'terminal_checkpoint_sha256')):
            file=folder/('step_0000.pt' if step==0 else 'step_0064_terminal.pt')
            assert base.file_digest(file)==arm[key]
            saved=torch.load(file,map_location='cpu',weights_only=True);checkpoint_audit(saved,parent,arm['learning_rate'],step)
            if step==0:initial_states.append(saved)
        for evaluation in arm['evaluations']:
            step=evaluation['fork_update']
            for i,(saved_row,record,reference) in enumerate(zip(evaluation['crops'],records,oracle)):
                prediction=folder/f'step_{step:04d}_crop_{i}.npz';swc=folder/f'step_{step:04d}_crop_{i}.swc'
                assert base.file_digest(prediction)==saved_row['prediction_sha256'] and base.file_digest(swc)==saved_row['swc_sha256']
                saved=control.read_arrays(prediction);ids=reviewer.verify_selection(saved)
                packet=audit.make_packet(saved,ids,record['meta']['patch_center'],original['patch_radius'])
                candidates=CandidateGraph(torch.from_numpy(saved['retained_edges']),recipe['candidate_neighbors'],recipe['candidate_radius'])
                row,decoded,labels=audit.assess(packet,candidates,torch.from_numpy(saved['retained_edge_logits']),saved,record,audit.stage.reference_graph(reference))
                for key,value in row.items():assert value==saved_row[key],(arm['name'],step,i,key)
                assert np.array_equal(labels,saved['evaluation_labels'])
                row['swc_roundtrip']=reviewer.verify_swc(swc,packet,decoded)
                q2t=dict(zip(saved['pred_indices'].tolist(),saved['target_indices'].tolist()))
                row['retained_trusted_junctions']=int(sum(bool(record['junction_mask'][q2t[int(q)]])
                    and np.linalg.norm(packet.positions[j].numpy().astype(float)-record['nodes'][q2t[int(q)],1:4])<=2
                    for j,q in enumerate(packet.node_ids) if int(q) in q2t))
                assert control.runner.automatic_gate(row,len(record['nodes']))==saved_row['automatic_gate']
                full=audit.make_packet(saved,np.arange(512),record['meta']['patch_center'],original['patch_radius'])
                all_candidates=CandidateGraph(torch.from_numpy(saved['candidate_edges']),recipe['candidate_neighbors'],recipe['candidate_radius'])
                transferred=assign_edge_labels(full,all_candidates,record['nodes'][:,1:4],record['target_edges'],record['target_components'],
                    (saved['pred_indices'],saved['target_indices']),maximum_match_error=2.)
                assert transferred.statistics==saved_row['training_edge_supervision'] and np.array_equal(transferred.labels.numpy(),saved['edge_labels'])
                metrics={k:row[k] for k in ('assigned_nodes','unmatched_nodes','trusted_assigned_nodes','decoded_required_edges','required_reference_edges',
                    'decoded_nonreference_edges','decoded_nonreference_edge_categories','missing_edge_decomposition','reference_components',
                    'retained_reference_junctions','retained_junction_incidence_errors','retained_trusted_junctions')}
                metrics.update(components=row['decoding']['components'],geometry_at_2=row['geometry']['tolerances']['2.0'],
                    swc_verification=row['swc_roundtrip'],full_gate=saved_row['automatic_gate'])
                details=[]
                if step==64:
                    mapped={j:q2t[int(q)] for j,q in enumerate(packet.node_ids) if int(q) in q2t}
                    required=audit.sets(record['target_edges']);scores={tuple(sorted(map(int,e))):float(v) for e,v in zip(candidates.edges,saved['retained_edge_logits'])}
                    for u,v in decoded.selected_edges:
                        if u in mapped and v in mapped and tuple(sorted((mapped[u],mapped[v]))) not in required:
                            details.append({'target_pair':[mapped[u],mapped[v]],'edge_logit':scores[tuple(sorted((u,v)))],
                                'training_label':float(transferred.labels[(saved['candidate_edges']==np.sort([int(packet.node_ids[u]),int(packet.node_ids[v])])).all(1)].item())})
                rows.append({'arm':arm['name'],'fork_update':step,'crop':[record['identity']['sample_index'],record['identity']['patch_index']],
                    'metrics':metrics,'confidence':confidence_summary(saved),'selected_nonreference_assigned_pairs':details,
                    'prediction_sha256':saved_row['prediction_sha256'],'swc_sha256':saved_row['swc_sha256']})
                print(f"VERIFIED {arm['name']} step={step} crop={rows[-1]['crop']} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']} full_gate={saved_row['automatic_gate']}",flush=True)
        label_cases={name:0 for name in ('both','positive_only','negative_only','unknown_only')}
        for update in ledger:
            s=update['incoming']['edge_supervision'];kind='both' if s['positive'] and s['negative'] else 'positive_only' if s['positive'] else 'negative_only' if s['negative'] else 'unknown_only'
            label_cases[kind]+=1
        history=[]
        for i,crop in enumerate(config['training_crops']):
            probes=[p['crops'][i] for p in arm['probes']];incoming=[a['incoming'] for a in ledger if a['crop']==crop]
            history.append({'crop':crop,'cycle_trusted_min':min(p['edge_supervision']['trusted_matches'] for p in probes),
                'targets':probes[0]['edge_supervision']['targets'],'incoming_trusted_min':min(p['edge_supervision']['trusted_matches'] for p in incoming),
                'last_eight_probe_distance_mean':float(np.mean([p['matched_distance_mean'] for p in probes[-8:]])),
                'last_probe_distance_mean':probes[-1]['matched_distance_mean'],
                'maximum_cycle_distance_mean':max(p['matched_distance_mean'] for p in probes)})
        arms.append({'name':arm['name'],'learning_rate':arm['learning_rate'],'proposal_updates':64,'graph_updates':64,
            'qualification':arm['qualification'],'label_cases':label_cases,'geometry_history':history})
    assert control.equal(initial_states[0]['model'],initial_states[1]['model'])
    assert control.equal(initial_states[0]['optimizer']['state'],initial_states[1]['optimizer']['state'])
    assert control.equal(initial_states[0]['rng'],initial_states[1]['rng'])
    assert control.decision(report['arms'])==report['decision'] and len(rows)==16
    precision_start=report['arms'][1]['evaluations'][0]['crops'][1]['geometry']['tolerances']['2.0']['precision']
    precision_end=report['arms'][1]['evaluations'][-1]['crops'][1]['geometry']['tolerances']['2.0']['precision']
    base.verify_hashes(protected);base.verify_hashes(report['provenance']['sources']);assert not torch.cuda.is_initialized()
    result={'schema':'pointneuron2-partial-stability-review-v1','status':'completed','date':'2026-10-20',
        'scope':'read-only saved-output CPU audit; zero model forwards/updates/CUDA initialization',
        'source_report':str(path.relative_to(ROOT)),'source_report_sha256':base.file_digest(path),'reviewer_sha256':base.file_digest(Path(__file__)),
        'validated_output_pairs':16,'identical_parent_weights_moments_rng_verified':True,'decision':report['decision'],
        'elapsed_gpu_action_seconds':report['elapsed_seconds'],'arms':arms,'rows':rows,
        'reduced_lr_preservation_precision_decline':precision_start-precision_end,
        'reduced_lr_precision_allowance':config['maximum_arclength_decline'],
        'excess_over_declared_allowance':precision_start-precision_end-config['maximum_arclength_decline'],
        'cuda_initialized':False,'new_training_launched':False}
    output=ROOT/'tmp/pointneuron2/local_runs/joint_partial_stability_review_20261020/summary.json'
    if output.exists():raise FileExistsError('Preserve prior review evidence')
    control.write(output,result)
    print(json.dumps({'review':str(output),'arms':arms,'precision_decline':precision_start-precision_end,
        'excess_over_allowance':result['excess_over_declared_allowance'],'endpoint_confidence':[r for r in rows if r['fork_update']==64]},indent=2))


if __name__=='__main__':run()
