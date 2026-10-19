"""Four throwaway CPU updates to inspect the failed joint timing stop; no checkpoint.

Preserves the original runner, loss, cutoff, trust mask and GPU artifacts.
CPU numerical behavior is evidence about this replay, not an exact GPU replay.
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_query_graph_experiment as runner
from pointneuron.models.balanced_query_loss import balanced_query_loss
from pointneuron.data.connectivity_targets import assign_edge_labels


def inspect(model,record,original,recipe):
    model.train()
    with torch.no_grad():
        output=runner.forward(model,record,original,'cpu')
        nodes=torch.from_numpy(record['nodes'])[None]
        proposal=balanced_query_loss(output.proposals,nodes,torch.ones(nodes.shape[:2],dtype=torch.bool),**original['query_loss'])
        labels=assign_edge_labels(output.packet,output.candidates,record['nodes'][:,1:4],record['target_edges'],
            record['target_components'],proposal.assignments[0],maximum_match_error=recipe['trusted_match_tolerance'])
        pi,ti=map(np.asarray,proposal.assignments[0])
        xyz=output.packet.positions.numpy()[pi]
        distance=np.linalg.norm(xyz.astype(float)-record['nodes'][ti,1:4],axis=1)
        keep=(output.packet.objectness_logits[:,1]>=output.packet.objectness_logits[:,0]).numpy()
        missing=[]
        if not labels.statistics['positive']:missing.append('positive')
        if not labels.statistics['negative']:missing.append('negative')
        return {'crop':[record['identity']['sample_index'],record['identity']['patch_index']],
            'edge_supervision':labels.statistics,'guard_would_stop':bool(missing),'missing_classes':missing,
            'proposal_loss':float(proposal.total),'matched_distance_mean':float(distance.mean()),
            'matched_distance_max':float(distance.max()),'matched_distances':distance.tolist(),
            'matched_centers':xyz.tolist(),'matched_target_indices':ti.tolist(),'matched_query_ids':pi.tolist(),
            'accepted_assigned':int(keep[pi].sum()),'accepted_unmatched':int(keep.sum()-keep[pi].sum())}


def run(proposal_only=False):
    started=time.perf_counter();runner.configure();torch.set_num_threads(2)
    source=ROOT/'tmp/pointneuron2/local_runs/joint_query_graph_ready_20261019/timing/report.json'
    failed=runner.base.read(source)
    if failed['status']!='failed' or failed['action']!='timing':raise ValueError('Expected preserved failed timing')
    runner.base.verify_hashes(failed['provenance']['sources'])
    recipe=runner.recipe_at(ROOT/'configs/pointneuron2/joint_query_graph.json')
    original,records,_,protected=runner.load_inputs(recipe)
    output=ROOT/'tmp/pointneuron2/local_runs'/('joint_timing_stop_proposal_only_cpu_20261019' if proposal_only else 'joint_timing_stop_cpu_20261019')
    if output.exists():raise FileExistsError('Preserve diagnostic evidence')
    output.mkdir(parents=True)
    report={'schema':'pointneuron2-joint-timing-stop-cpu-v1','status':'running','device':'cpu',
        'source_report_sha256':runner.base.file_digest(source),'recipe':recipe,'throwaway_updates':0,
        'retained_trained_checkpoint':False,'gpu_trajectory_reproduced':False,'proposal_only_control':proposal_only,
        'updates':[],'crop_zero_probes':[]}
    runner.base.write(output/'report.json',report)
    model=runner.warm_start(recipe,'cpu');optimizer=runner.optimizer_for(model,recipe)
    if runner.state_hashes(model)!=failed['initial_state_sha256']:raise ValueError('Warm-start weights differ')
    initial=inspect(model,records[0],original,recipe);report['crop_zero_probes'].append({'after_updates':0,**initial})
    for i in range(4):
        before=inspect(model,records[i],original,recipe)
        if before['guard_would_stop'] and not proposal_only:
            report['stop_before_update']=i+1;report['stopping_inspection']=before;break
        if proposal_only:
            # Matched source/input/crop order/optimizer/clip; omit only graph loss.
            from scripts.pointneuron2.balanced_query_experiment import update
            metrics=update(model.proposal_model,optimizer,records[i],dict(original,gradient_clip=recipe['gradient_clip']),'cpu')
            metrics['preclip_norm']=metrics.pop('gradient_norm_before_clip')
        else:metrics=runner.update(model,optimizer,records[i],original,recipe,'cpu')
        report['updates'].append({'update':i+1,'before':before,'metrics':metrics});report['throwaway_updates']=i+1
        probe=inspect(model,records[0],original,recipe);report['crop_zero_probes'].append({'after_updates':i+1,**probe})
        print(f"CPU update={i+1} norm={metrics['preclip_norm']:.6g} crop0_trusted={probe['edge_supervision']['trusted_matches']}/6 pos={probe['edge_supervision']['positive']} neg={probe['edge_supervision']['negative']} distances={np.round(probe['matched_distances'],3).tolist()}",flush=True)
        runner.base.write(output/'report.json',report)
    if report['throwaway_updates']==4:
        report['fifth_update_inspection']=inspect(model,records[0],original,recipe)
    runner.base.verify_hashes(protected);runner.base.verify_hashes(failed['provenance']['sources'])
    if proposal_only:
        if any(runner.base.tensor_digest(v)!=failed['initial_state_sha256']['graph_model.'+k]
               for k,v in model.graph_model.state_dict().items()):raise ValueError('Proposal-only control changed GAT')
        report['graph_weights_unchanged']=True
    report.update(status='completed',original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    runner.base.write(output/'report.json',report)
    print(f"COMPLETED diagnostic; report={output/'report.json'}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-only',action='store_true')
    run(parser.parse_args().proposal_only)
