"""Read-only CPU verification of saved GPU timing outputs; no model forwards."""
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_timing as timing
from pointneuron.graph.proposal_graph import CandidateGraph
from pointneuron.data.swc import parse_swc

runner=timing.runner
audit=runner.audit


def verify_selection(saved):
    ids=audit.automatic_ids(saved)
    if not np.array_equal(ids,saved['retained_node_ids']):raise ValueError('Saved automatic selection differs')
    all_edges=saved['candidate_edges'];mask=np.isin(all_edges,ids).all(axis=1)
    mapping=np.full(saved['centers'].shape[1],-1,dtype=np.int64);mapping[ids]=np.arange(len(ids))
    if not np.array_equal(mapping[all_edges[mask]],saved['retained_edges']):raise ValueError('Late-filtered candidate mapping differs')
    if not np.array_equal(saved['edge_logits'][mask],saved['retained_edge_logits']):raise ValueError('Late-filtered scores differ')
    return ids


def verify_swc(path,packet,decoded):
    tree=parse_swc(path)
    if tree.validate():raise ValueError('Invalid SWC')
    order=decoded.traversal
    if len(tree.nodes)!=len(order) or tree.root_count!=decoded.statistics['components']:raise ValueError('SWC node/component count differs')
    serial_to_local={i+1:u for i,u in enumerate(order)}
    edges=audit.sets((serial_to_local[n.node_id],serial_to_local[n.parent_id]) for n in tree.nodes if n.parent_id!=-1)
    if edges!=audit.sets(decoded.selected_edges):raise ValueError('SWC topology differs')
    coordinate_delta=0.;radius_delta=0.
    for node in tree.nodes:
        u=serial_to_local[node.node_id]
        coordinate_delta=max(coordinate_delta,float(np.abs(np.array([node.x,node.y,node.z])-packet.positions[u].numpy()).max()))
        radius_delta=max(radius_delta,abs(node.radius-max(float(packet.radii[u,0]),1.)))
    if max(coordinate_delta,radius_delta)>1e-6:raise ValueError('SWC geometry round trip differs')
    return {'passed':True,'maximum_coordinate_delta':coordinate_delta,'maximum_radius_delta':radius_delta,
        'nodes':len(order),'edges':len(edges),'roots':tree.root_count}


def run():
    started=time.perf_counter();torch.set_num_threads(2)
    config=runner.base.read(timing.CONFIG)
    directory=ROOT/config['output']/'timing';path=directory/'report.json';report=runner.base.read(path)
    if report['status']!='completed' or report['completed_updates']!=8 or not report['timing_readiness_gate']:
        raise ValueError('Expected completed passing timing result')
    if report['config']!=config or not report['original_sources_unchanged']:raise ValueError('Different timing contract')
    runner.base.verify_hashes(report['provenance']['sources']);runner.base.verify_hashes(report['protected_files_sha256'])
    if report['reload']['prediction_max_delta']!=0 or not report['reload']['weights_and_optimizer_identical']:
        raise ValueError('Reload was not exact')
    if report['reload']['cursors']!=timing.expected_cursors(8):raise ValueError('Different optimizer history')
    if list(directory.glob('*.pt*')):raise ValueError('Unexpected retained timing checkpoint')
    recipe=runner.recipe_at(ROOT/config['source_recipe']);original,records,oracle,protected=runner.load_inputs(recipe)
    if len(report['attempts'])!=8 or len(report['snapshots'])!=9:raise ValueError('Incomplete diagnostic ledger')
    for i,attempt in enumerate(report['attempts'],1):
        labels=attempt['incoming']['edge_supervision']
        if attempt['stage_update']!=i or attempt['cursors_after']!=timing.expected_cursors(i) or not attempt['optimizer_step_returned']:
            raise ValueError('Attempt/completion/cursor mismatch')
        if labels['positive']<=0 or labels['negative']<=0:raise ValueError('Missing active edge-label class')
    if timing.stability.stability_gate(report)!=report['stability_gate']:raise ValueError('Stability gate differs')
    seconds=[a['seconds_with_incoming_diagnostics_and_io'] for a in report['attempts']]
    estimate=float(np.mean(seconds[4:]))*config['planning_updates_only']
    if not np.isclose(estimate,report['estimate_update_seconds_for_4000'],rtol=0,atol=1e-9):raise ValueError('Timing estimate differs')
    rows=[]
    keys=('retained_nodes','assigned_nodes','unmatched_nodes','trusted_assigned_nodes','decoded_required_edges',
        'decoded_missing_required_edges','decoded_nonreference_edges','missing_edge_decomposition',
        'retained_reference_junctions','retained_junction_incidence_errors','reference_components','geometry','decoding')
    for evaluation in report['evaluations']:
        step=evaluation['stage_step']
        for i,(saved_row,record,reference) in enumerate(zip(evaluation['crops'],records,oracle)):
            prediction=directory/f'step_{step:04d}_crop_{i}.npz';swc=directory/f'step_{step:04d}_crop_{i}.swc'
            if runner.base.file_digest(prediction)!=saved_row['prediction_sha256'] or runner.base.file_digest(swc)!=saved_row['swc_sha256']:
                raise ValueError('Saved artifact hash differs')
            with np.load(prediction,allow_pickle=False) as archive:saved={k:archive[k].copy() for k in archive.files}
            ids=verify_selection(saved)
            packet=audit.make_packet(saved,ids,record['meta']['patch_center'],original['patch_radius'])
            candidates=CandidateGraph(torch.from_numpy(saved['retained_edges']),recipe['candidate_neighbors'],recipe['candidate_radius'])
            row,decoded,labels=audit.assess(packet,candidates,torch.from_numpy(saved['retained_edge_logits']),saved,record,audit.stage.reference_graph(reference))
            if not np.array_equal(labels,saved['evaluation_labels']):raise ValueError('Evaluation-only labels differ')
            for key in keys:
                if row[key]!=saved_row[key]:raise ValueError(f'Recomputed metric differs: {key}')
            roundtrip=verify_swc(swc,packet,decoded)
            rows.append({'stage_step':step,'crop':[record['identity']['sample_index'],record['identity']['patch_index']],
                'metrics':row,'swc_verification':roundtrip,'prediction_sha256':saved_row['prediction_sha256'],'swc_sha256':saved_row['swc_sha256']})
            print(f"VERIFIED saved stage={step} crop={rows[-1]['crop']} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']} geometry_exact=True SWC=True",flush=True)
    if len(rows)!=8:raise ValueError('Expected eight saved output pairs')
    runner.base.verify_hashes(protected);runner.base.verify_hashes(report['provenance']['sources'])
    output=ROOT/'tmp/pointneuron2/local_runs/joint_saved_adam_timing_review_20261020'
    if output.exists():raise FileExistsError('Preserve prior review evidence')
    output.mkdir(parents=True)
    result={'schema':'pointneuron2-joint-saved-adam-timing-review-v1','status':'completed','date':'2026-10-20',
        'scope':'read-only CPU review of saved outputs; no model forwards, optimizer updates or GPU execution',
        'timing_report':str(path.relative_to(ROOT)),'timing_report_sha256':runner.base.file_digest(path),
        'reviewer_sha256':runner.base.file_digest(Path(__file__)),'original_sources_unchanged':True,
        'validated_output_pairs':8,'saved_selection_and_score_mapping_exact':True,'geometry_and_decode_metrics_exact':True,
        'memory_as_recorded':report['peak_memory'],'reload_as_recorded':report['reload'],
        'planning_range_seconds':report['planning_range_seconds'],'timing_readiness_gate':report['timing_readiness_gate'],
        'stability_gate':report['stability_gate'],'rows':rows,'elapsed_seconds':time.perf_counter()-started}
    runner.base.write(output/'summary.json',result)
    print(f"COMPLETED read-only review; summary={output/'summary.json'}",flush=True)


if __name__=='__main__':run()
