"""CPU inference-only oracle versus automatic selection on four TRAINING crops."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import frozen_proposal_gat as stage
from pointneuron.graph.connectivity_decoder import decode_forest, export_decoded_swc
from pointneuron.graph.proposal_graph import ProposalGraphInput, build_candidate_graph
from pointneuron.graph.representation_oracle import ReferenceGraph, geometry_report, distance_to_graph
from pointneuron.models.gatv2_connectivity import GATv2Connectivity, ConnectivityConfig

base=stage.base
SCHEMA='pointneuron2-gat-automatic-selection-evaluation-v1'


def automatic_ids(saved):
    """Reference-free inclusive .5 cutoff; no query count or target data accepted."""
    logits=np.asarray(saved['logits'])[0]
    return np.flatnonzero(logits[:,1]>=logits[:,0])


def make_packet(saved, ids, center, radius):
    xyz=torch.from_numpy(saved['centers'][0,ids].copy())
    return ProposalGraphInput(xyz,torch.from_numpy(saved['radii'][0,ids].copy()),
        torch.from_numpy(saved['logits'][0,ids].copy()),(xyz-torch.as_tensor(center,dtype=xyz.dtype))/radius,
        torch.from_numpy(saved['decoder_features'][0,ids].copy()),torch.as_tensor(ids,dtype=torch.long),
        {'scope':'frozen TRAINING inference comparison'})


def sets(edges):
    return {tuple(sorted(map(int,e))) for e in edges}


def assess(p,c,logits,saved,original,reference,*,oracle_retention=False):
    """Reference data enter scoring here, after model inference and selection."""
    p.validate(); c.validate(p)
    decoder_packet=replace(p,objectness_logits=torch.zeros_like(p.objectness_logits)) if oracle_retention else p
    decoded=decode_forest(decoder_packet,c,logits,node_cutoff=.5,edge_cutoff=.5)
    if len(decoded.accepted_nodes)!=len(p.positions): raise ValueError('Packet selection disagrees with decoder cutoff')
    query_to_target=dict(zip(saved['pred_indices'].tolist(),saved['target_indices'].tolist()))
    local_to_target={j:query_to_target[int(q)] for j,q in enumerate(p.node_ids) if int(q) in query_to_target}
    target_to_local={t:j for j,t in local_to_target.items()}
    required=sets(original['target_edges']); total=len(required)
    surviving={tuple(sorted((target_to_local[a],target_to_local[b]))) for a,b in required if a in target_to_local and b in target_to_local}
    candidate=sets(c.edges.tolist()); positive=sets(c.edges[logits>=0].tolist()); selected=sets(decoded.selected_edges)
    candidate_required=surviving&candidate; scored_required=candidate_required&positive; correct=surviving&selected
    losses={'node_filter':total-len(surviving),'candidate_graph':len(surviving)-len(candidate_required),
            'edge_score':len(candidate_required)-len(scored_required),'forest_decode':len(scored_required)-len(correct)}
    if min(losses.values())<0 or sum(losses.values())!=total-len(correct): raise ValueError('Edge loss decomposition inconsistent')
    labels=[]; kinds=[]
    for u,v in c.edges.tolist():
        a,b=local_to_target.get(u),local_to_target.get(v)
        if a is None or b is None: label,kind=-1.,'touches_unmatched_query'
        elif tuple(sorted((a,b))) in required: label,kind=1.,'required_reference_adjacency'
        elif original['target_components'][a]>=0 and original['target_components'][a]==original['target_components'][b]:
            label,kind=0.,'assigned_same_tree_nonadjacency'
        else: label,kind=-1.,'different_or_unknown_reference_tree'
        labels.append(label); kinds.append(kind)
    edge_kind={tuple(sorted(map(int,e))):kind for e,kind in zip(c.edges.tolist(),kinds)}
    extra_categories={kind:sum(edge_kind[e]==kind for e in selected-correct) for kind in
        ('touches_unmatched_query','assigned_same_tree_nonadjacency','different_or_unknown_reference_tree')}
    target_degree=np.zeros(len(original['nodes']),int)
    for a,b in required: target_degree[a]+=1; target_degree[b]+=1
    degree=np.zeros(len(p.positions),int); parent=list(range(len(p.positions)))
    def root(j):
        while parent[j]!=j: parent[j]=parent[parent[j]]; j=parent[j]
        return j
    for a,b in selected:
        degree[a]+=1; degree[b]+=1; parent[root(a)]=root(b)
    # Reference crop components, independent of full-reference tree identities.
    ref_parent=list(range(len(original['nodes'])))
    def ref_root(j):
        while ref_parent[j]!=j: ref_parent[j]=ref_parent[ref_parent[j]]; j=ref_parent[j]
        return j
    for a,b in required: ref_parent[ref_root(a)]=ref_root(b)
    memberships={}
    for j,t in local_to_target.items(): memberships.setdefault(root(j),set()).add(ref_root(t))
    unmatched=[j for j in range(len(p.positions)) if j not in local_to_target]
    matched=list(local_to_target)
    trusted=sum(np.linalg.norm(p.positions[j].numpy().astype(float)-original['nodes'][t,1:4])<=2 for j,t in local_to_target.items())
    positions=p.positions.numpy().astype(float)
    predicted=ReferenceGraph([('query',int(q)) for q in p.node_ids],positions,p.radii.numpy()[:,0],list(decoded.selected_edges))
    point_dist=distance_to_graph(positions,reference)
    geometry=geometry_report(reference,predicted)
    if not len(p.positions):
        # Preserve explicit empty failure and zero recall without non-JSON Infinity.
        geometry['max_reference_distance']=None
        geometry['reference_distance_unbounded_due_to_empty_prediction']=True
    row={'retained_nodes':len(p.positions),'assigned_nodes':len(matched),'unmatched_nodes':len(unmatched),
         'trusted_assigned_nodes':int(trusted),'missing_assigned_targets':len(original['nodes'])-len(matched),
         'retained_reference_junctions':sum(bool(original['junction_mask'][t]) for t in local_to_target.values()),
         'reference_junctions':int(original['junction_mask'].sum()),
         'retained_junction_incidence_errors':int(sum(bool(original['junction_mask'][t]) and degree[j]!=target_degree[t] for j,t in local_to_target.items())),
         'required_reference_edges':total,'required_edges_with_surviving_endpoints':len(surviving),
         'required_edges_in_candidates':len(candidate_required),'required_edges_above_cutoff':len(scored_required),
         'decoded_required_edges':len(correct),'decoded_missing_required_edges':total-len(correct),
         'missing_edge_decomposition':losses,'decoded_nonreference_edges':len(selected-correct),
         'decoded_nonreference_edge_categories':extra_categories,
         'output_components_spanning_multiple_reference_crop_components':sum(len(s)>1 for s in memberships.values()),
         'reference_components':len({ref_root(t) for t in range(len(original['nodes']))}),
         'isolated_unmatched_nodes':int(sum(degree[j]==0 for j in unmatched)),'isolated_assigned_nodes':int(sum(degree[j]==0 for j in matched)),
         'unmatched_nodes_within_2_of_reference':int((point_dist[unmatched]<=2).sum()),
         'unmatched_nodes_within_6_of_reference':int((point_dist[unmatched]<=6).sum()),
         'geometry':geometry,'decoding':decoded.statistics,
         'decoded_required_target_edge_indices':[list(e) for e in sorted(required) if e[0] in target_to_local and e[1] in target_to_local
             and tuple(sorted((target_to_local[e[0]],target_to_local[e[1]]))) in correct]}
    return row,decoded,np.asarray(labels,dtype=np.float32)


def run(args):
    recipe=base.read(args.recipe)
    if recipe!=base.read(ROOT/'configs/pointneuron2/gat_automatic_selection.json') or recipe['schema']!=SCHEMA or recipe['device']!='cpu' or recipe['optimizer_updates']!=0 or recipe['node_cutoff']!=.5 or recipe['edge_cutoff']!=.5:
        raise ValueError('Different declared evaluation')
    args.output=args.output.resolve()
    if not args.output.is_relative_to((ROOT/'tmp/pointneuron2').resolve()): raise ValueError('Generated outputs belong in ignored tmp/pointneuron2')
    if args.output.exists(): raise FileExistsError('Preserve existing evaluation evidence')
    torch.set_num_threads(recipe['torch_threads']); torch.use_deterministic_algorithms(True)
    gat_dir=ROOT/recipe['gat_experiment']; source=ROOT/recipe['balanced_experiment']
    protected={str(gat_dir/'model/report.json'):recipe['gat_report_sha256'],str(gat_dir/'model/step_4000.pt'):recipe['gat_checkpoint_sha256'],
               str(source/'model/report.json'):recipe['balanced_report_sha256']}
    base.verify_hashes(protected)
    fit=base.read(gat_dir/'model/report.json'); graph_recipe=fit['recipe']
    if not fit['graph_gate'] or fit['status']!='completed' or fit['optimizer_updates']!=4000: raise ValueError('Expected verified successful terminal GAT fit')
    protected.update(fit['provenance']['sources'])
    manifest,oracle_records=stage.load_cache(SimpleNamespace(directory=gat_dir),graph_recipe)
    protected.update(manifest['protected_files_sha256'])
    protected[str(gat_dir/'cache/manifest.json')]=base.file_digest(gat_dir/'cache/manifest.json')
    original_recipe=base.recipe_at(ROOT/'configs/pointneuron2/branched_query_fit.json')
    _,originals=base.load_records(SimpleNamespace(experiment_dir=source),original_recipe)
    if graph_recipe['training_crops']!=recipe['training_crops']: raise ValueError('Unexpected crops')
    cp=torch.load(gat_dir/'model/step_4000.pt',map_location='cpu',weights_only=True)
    if cp['schema']!=stage.SCHEMA or cp['step']!=4000 or cp['recipe']!=graph_recipe or cp['provenance']!=fit['provenance'] or cp['resumable']:
        raise ValueError('Different model checkpoint')
    model=GATv2Connectivity(ConnectivityConfig(**graph_recipe['model'])).eval().requires_grad_(False)
    if model.model_spec()!=cp['spec']: raise ValueError('Different scorer specification')
    model.load_state_dict(cp['model'])
    before={k:base.tensor_digest(v) for k,v in model.state_dict().items()}
    frozen_before=stage.fingerprints(oracle_records)
    sources=base.source_hashes(SimpleNamespace(recipe=args.recipe)); base.verify_hashes(protected)
    args.output.mkdir(parents=True)
    report={'schema':SCHEMA,'status':'running','recipe':recipe,'backend':{'device':'cpu','torch':str(torch.__version__),'threads':torch.get_num_threads()},
            'scope':'fixed TRAINING integration gap; frozen GAT; cached balanced image-derived proposals','optimizer_updates':0,
            'upstream_forwards':0,'gat_forwards':0,'protected_sources_sha256':protected,'evaluation_sources_sha256':sources,'crops':[]}
    base.write(args.output/'report.json',report); started=time.perf_counter()
    try:
        for path in sources:
            target=args.output/'sources'/Path(path).relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        for i,(r,original) in enumerate(zip(oracle_records,originals)):
            path=source/f'model/step_4000_crop_{i}.npz'
            with np.load(path,allow_pickle=False) as z: saved={k:z[k].copy() for k in z.files}
            reference=stage.reference_graph(r); arms={}
            for name,ids in (('oracle',r['node_ids'].numpy()),('automatic',automatic_ids(saved))):
                p=make_packet(saved,ids,original['meta']['patch_center'],original_recipe['patch_radius'])
                if name=='oracle':
                    for key in stage.FIELDS:
                        if not torch.equal(getattr(p,key),r[key]): raise ValueError('Oracle replay inputs differ')
                c=build_candidate_graph(p.positions,p.node_ids,neighbors=graph_recipe['candidate_neighbors'],radius=graph_recipe['candidate_radius'])
                input_before={k:base.tensor_digest(getattr(p,k)) for k in stage.FIELDS}
                with torch.inference_mode(): logits=model(p,c).edge_logits
                report['gat_forwards']+=1
                if input_before!={k:base.tensor_digest(getattr(p,k)) for k in stage.FIELDS}: raise ValueError('Scorer changed frozen inputs')
                row,decoded,labels=assess(p,c,logits,saved,original,reference,oracle_retention=name=='oracle')
                if name=='oracle':
                    with np.load(gat_dir/'model/scores_4000.npz') as z: baseline=torch.from_numpy(z[f'crop_{i}'].copy())
                    torch.testing.assert_close(logits,baseline,rtol=1e-5,atol=1e-5)
                    row['cpu_gpu_oracle_replay_max_delta']=float((logits-baseline).abs().max())
                    if row['decoded_missing_required_edges'] or row['decoded_nonreference_edges'] or row['unmatched_nodes']: raise ValueError('Oracle baseline no longer passes')
                elif (row['assigned_nodes']!=recipe['expected_automatic_assigned'][i] or row['unmatched_nodes']!=recipe['expected_automatic_unmatched'][i]):
                    raise ValueError('Automatic selection differs from saved balanced source')
                if decoded.status=='decoded':
                    swc=args.output/f'crop_{i}_{name}.swc'
                    row['swc_roundtrip']=export_decoded_swc(p,decoded,swc,description=f'TRAINING INFERENCE DIAGNOSTIC: {name} selection; frozen GAT checkpoint; no retraining or pruning.')
                    row['swc_sha256']=base.file_digest(swc)
                else: row['empty_reconstruction']=True
                arrays=args.output/f'crop_{i}_{name}.npz'
                np.savez_compressed(arrays,node_ids=p.node_ids.numpy(),positions=p.positions.numpy(),candidate_edges=c.edges.numpy(),
                                    edge_logits=logits.numpy(),evaluation_labels=labels)
                row.update(scores_sha256=base.file_digest(arrays),input_tensor_sha256=input_before)
                arms[name]=row
                print(f"CROP={r['sample_index']}/{r['patch_index']} arm={name} nodes={row['assigned_nodes']}+{row['unmatched_nodes']} edges={row['decoded_required_edges']}/{row['required_reference_edges']} nonreference={row['decoded_nonreference_edges']} components={row['decoding']['components']} losses={row['missing_edge_decomposition']}",flush=True)
            report['crops'].append({'sample_index':r['sample_index'],'patch_index':r['patch_index'],'arms':arms})
            base.write(args.output/'report.json',report)
        if before!={k:base.tensor_digest(v) for k,v in model.state_dict().items()} or frozen_before!=stage.fingerprints(oracle_records): raise ValueError('Model/cache changed')
        base.verify_hashes(protected);base.verify_hashes(sources)
        report.update(status='completed',elapsed_seconds=time.perf_counter()-started,model_and_inputs_unchanged=True,
            next_step='Discuss measured integration gap. No automatic retraining, pruning, cutoff change, node head or production promotion.')
        base.write(args.output/'report.json',report)
        print(f"COMPLETED CPU evaluation; report={args.output/'report.json'}",flush=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc));base.write(args.output/'report.json',report)
        (args.output/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,default=ROOT/'configs/pointneuron2/gat_automatic_selection.json')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
