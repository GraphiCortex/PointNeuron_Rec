"""Saved-prediction CPU audit of joint-fit supervision stop; no model forwards/updates."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_fit as fit
from scripts.pointneuron2 import review_joint_saved_adam_timing as reviewer
from pointneuron.data.connectivity_targets import assign_edge_labels,masked_edge_loss
from pointneuron.models.balanced_query_loss import loss_from_assignments
from pointneuron.graph.proposal_graph import CandidateGraph


def arrays(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def run():
    torch.set_num_threads(2);base=fit.runner.base;audit=fit.runner.audit
    root=ROOT/'tmp/pointneuron2/local_runs/joint_saved_adam_fit_20261020'
    directory=root/'resume_step_0008';path=directory/'report.json';report=base.read(path)
    base.verify_hashes(report['provenance']['sources'])
    recipe,original,records,oracle,protected,_,_=fit.load_inputs(report['config'])
    attempt=base.read(directory/'active_update.json')
    assert report['status']=='failed' and report['completed_updates']==1013 and report['attempted_update']==1014
    assert attempt['stage_update']==1014 and attempt['crop']==[0,1] and not attempt['optimizer_step_returned']
    assert attempt['phase']=='incoming_recorded' and report['last_checkpoint_step']==1000
    assert base.file_digest(directory/'step_1000.pt')==report['last_checkpoint_sha256']
    ledger=[json.loads(s) for s in (directory/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['stage_update'] for r in ledger]==list(range(9,1014))
    assert all(r['cursors_after']==fit.timing.expected_cursors(r['stage_update']) for r in ledger)
    old=[json.loads(s) for s in (root/'model/updates.jsonl').read_text(encoding='utf-8').splitlines()]
    repeated=[r for r in ledger if r['stage_update']<=148]
    exact=sum(a==b for a,b in zip(old[8:],repeated))
    assert exact==140 and len(repeated)==140
    rows=[]
    keys=('retained_nodes','assigned_nodes','unmatched_nodes','trusted_assigned_nodes','decoded_required_edges',
        'decoded_missing_required_edges','decoded_nonreference_edges','missing_edge_decomposition',
        'retained_reference_junctions','retained_junction_incidence_errors','reference_components','geometry','decoding',
        'decoded_nonreference_edge_categories')
    for evaluation in [report['prefix_evaluation'],*report['evaluations']]:
        step=evaluation['stage_step'];folder=directory/'prefix_replay' if step==8 else directory
        for i,(saved_row,record,reference) in enumerate(zip(evaluation['crops'],records,oracle)):
            prediction=folder/f'step_{step:04d}_crop_{i}.npz';swc=folder/f'step_{step:04d}_crop_{i}.swc'
            assert base.file_digest(prediction)==saved_row['prediction_sha256'] and base.file_digest(swc)==saved_row['swc_sha256']
            saved=arrays(prediction);ids=reviewer.verify_selection(saved)
            packet=audit.make_packet(saved,ids,record['meta']['patch_center'],original['patch_radius'])
            c=CandidateGraph(torch.from_numpy(saved['retained_edges']),recipe['candidate_neighbors'],recipe['candidate_radius'])
            row,decoded,labels=audit.assess(packet,c,torch.from_numpy(saved['retained_edge_logits']),saved,record,audit.stage.reference_graph(reference))
            assert np.array_equal(labels,saved['evaluation_labels'])
            for key in keys:assert row[key]==saved_row[key],(step,i,key)
            roundtrip=reviewer.verify_swc(swc,packet,decoded)
            q2t=dict(zip(saved['pred_indices'].tolist(),saved['target_indices'].tolist()))
            row['retained_trusted_junctions']=int(sum(bool(record['junction_mask'][q2t[int(q)]])
                and np.linalg.norm(packet.positions[j].numpy().astype(float)-record['nodes'][q2t[int(q)],1:4])<=2
                for j,q in enumerate(packet.node_ids) if int(q) in q2t))
            row['swc_roundtrip']=roundtrip
            assert fit.runner.automatic_gate(row,len(record['nodes']))==saved_row['automatic_gate']
            rows.append({'stage_step':step,'crop':[record['identity']['sample_index'],record['identity']['patch_index']],
                'metrics':row,'automatic_gate':saved_row['automatic_gate'],'prediction_sha256':base.file_digest(prediction),'swc_sha256':base.file_digest(swc)})
            print(f"VERIFIED step={step} crop={rows[-1]['crop']} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']} gate={saved_row['automatic_gate']}",flush=True)
    record=records[1];raw=arrays(directory/'failure_incoming_predictions.npz');incoming=attempt['incoming']
    saved={'centers':raw['center_proposals'],'radii':raw['radius'],'logits':raw['objectness_logits'],'decoder_features':raw['decoder_features']}
    packet=audit.make_packet(saved,np.arange(512),record['meta']['patch_center'],original['patch_radius'])
    c=CandidateGraph(torch.from_numpy(raw['candidate_edges']),recipe['candidate_neighbors'],recipe['candidate_radius'])
    assignment=(incoming['matched_query_ids'],incoming['matched_target_indices'])
    labels=assign_edge_labels(packet,c,record['nodes'][:,1:4],record['target_edges'],record['target_components'],assignment,maximum_match_error=2.)
    assert labels.statistics==incoming['edge_supervision']
    proposal=loss_from_assignments(SimpleNamespace(center_proposals=torch.from_numpy(saved['centers']),radius=torch.from_numpy(saved['radii']),
        objectness_logits=torch.from_numpy(saved['logits'])),torch.from_numpy(record['nodes'])[None],torch.ones((1,12),dtype=torch.bool),[assignment],**original['query_loss'])
    edge=masked_edge_loss(torch.from_numpy(raw['edge_logits']),labels.labels)
    np.testing.assert_allclose([float(proposal.total),float(edge.total)],[incoming['proposal_loss'],incoming['edge_loss']],rtol=1e-5,atol=1e-6)
    distances=np.linalg.norm(packet.positions.numpy()[assignment[0]].astype(float)-record['nodes'][assignment[1],1:4],axis=1)
    np.testing.assert_allclose(distances,incoming['matched_distances'],rtol=0,atol=1e-8)
    baseline=arrays(directory/'step_1000_crop_1.npz')
    before={int(t):int(q) for q,t in zip(baseline['pred_indices'],baseline['target_indices'])}
    after={int(t):int(q) for q,t in zip(*assignment)}
    displacement=np.array([saved['centers'][0,after[t]]-baseline['centers'][0,before[t]] for t in range(12)])
    failed={'stage_update':1014,'crop':[0,1],'statistics':labels.statistics,'all_arrays_finite':all(np.isfinite(v).all() for v in raw.values()),
        'proposal_loss':float(proposal.total),'single_class_edge_loss':float(edge.total),'matched_distances':distances.tolist(),
        'accepted_assigned':incoming['accepted_assigned'],'accepted_unmatched':incoming['accepted_unmatched'],
        'same_target_query_assignments_as_step1000':sum(before[t]==after[t] for t in before),
        'mean_matched_displacement_from_step1000':displacement.mean(0).tolist(),
        'rms_displacement_residual':float(np.sqrt(((displacement-displacement.mean(0))**2).sum(1).mean())),
        'prediction_sha256':base.file_digest(directory/'failure_incoming_predictions.npz')}
    history=[]
    for i,record in enumerate(records):
        crop=[record['identity']['sample_index'],record['identity']['patch_index']]
        selected=[r for r in ledger if r['crop']==crop];norms=np.array([r['preclip_norm'] for r in selected])
        history.append({'crop':crop,'updates':len(selected),'trust_min':min(r['incoming']['edge_supervision']['trusted_matches'] for r in selected),
            'trust_last_ten':[{'stage_update':r['stage_update'],'trusted':r['incoming']['edge_supervision']['trusted_matches'],
                'mean_distance':r['incoming']['matched_distance_mean'],'proposal_gradient_l2':r['proposal_gradient_l2'],'gat_gradient_l2':r['gat_gradient_l2']} for r in selected[-10:]],
            'clipped_updates':int((norms>10).sum()),'preclip_median':float(np.median(norms)),'preclip_maximum':float(norms.max())})
    base.verify_hashes(protected);base.verify_hashes(report['provenance']['sources'])
    assert not torch.cuda.is_initialized()
    result={'schema':'pointneuron2-joint-saved-adam-fit-stop-review-v1','status':'completed','date':'2026-10-20',
        'scope':'saved-output CPU audit; zero model forwards, optimizer updates or CUDA initialization',
        'source_report':str(path.relative_to(ROOT)),'source_report_sha256':base.file_digest(path),'reviewer_sha256':base.file_digest(Path(__file__)),
        'completed_updates':1013,'stopped_update':1014,'last_checkpoint_stage':1000,'recovery_repeated_updates_identical':140,
        'validated_output_pairs':len(rows),'failed_packet':failed,'per_crop_history':history,'rows':rows,
        'recommendation':'Stop this control. Discuss partial-supervision and geometry-stability policy before any changed fit; no automatic resume or extension.'}
    output=ROOT/'tmp/pointneuron2/local_runs/joint_saved_adam_fit_stop_review_20261020/summary.json'
    if output.exists():raise FileExistsError('Preserve review evidence')
    fit.write(output,result)
    print(json.dumps({'report':str(output),'failure':failed,'history':history},indent=2),flush=True)


if __name__=='__main__':run()
