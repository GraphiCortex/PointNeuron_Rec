"""User-local Stage0 CPU reference-assisted headroom. No model/GPU/training.

Build all image-only candidate pools first. Only then open references for the
explicit diagnostic oracle. Nothing is selected/adopted from arm comparisons.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import numpy as np
from scipy.spatial import cKDTree

import local_curve_experiment as e
from score_saved_curve_transfer import cached_record
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.candidate_pool import pool,crop_classical,EDGE_SOURCES
from pn2.evaluation.selection_oracle import edge_agreement,headroom_gate
from pn2.evaluation.probability_forest import decode
from pn2.evaluation.formulation import graph_segments,runs_graph,clip_segments
from pn2.evaluation.geometry import _CurveIndex
from pn2.evaluation.bounded_agreement import BoundedAgreement,bounded_distances
from pn2.evaluation.fixed_curve_graph import summarize
from pn2.evaluation.vote_scaffold import guide_at,export_forest
from pn2.evaluation.swc import parse_swc
from pn2.training.guarded_run import run

ROOT=e.ROOT
CONFIG=ROOT/'configs/candidate_headroom_20261105.json'
ARMS={'classical':('classical',),'sato':('sato',),'learned':('learned',),'union':('classical','sato','learned')}


def control():
    c=e.read(CONFIG)
    required={'schema':'pn2-candidate-headroom-v1','sato_top_n':512,'merge_radius':1.,'candidate_k':[6,10],
              'primary_k':10,'node_tolerance':2.,'node_threshold':.5,'edge_threshold':.5,'spur_below':2.,'wall_seconds':3600}
    if any(c.get(k)!=v for k,v in required.items()):raise ValueError('Changed Stage0 recipe')
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Changed Stage0 source/input: {name}')
    scientific=e.control();data,_,_=e.prepared(scientific)
    keys=[(cid,b['tile_id']) for cid,case in data.cases.items() if case['partition']=='development'
          for b in case['boxes'] if b['target']['count']]
    if len(keys)!=151 or len({k[0] for k in keys})!=7:raise ValueError('Changed study boxes')
    return c,scientific,data,keys


def inventory(c,keys):
    broad=e.read(ROOT/c['broad_root']/'evaluation/inference_complete.json')['packets']
    sato=e.read(ROOT/c['sato_root']/'all_inference_complete.json')['packets']
    broad={tuple(p['key']):p for p in broad if p['partition']=='development'}
    sato={tuple(p['key']):p for p in sato if p['policy']=='sato512'}
    if set(broad)!=set(keys) or set(sato)!=set(keys):raise ValueError('Saved source identities differ')
    expected=e.read(e.BASE/'prepare/report.json')['classical_packet_hashes']
    classical={}
    for path in sorted((ROOT/c['classical_root']).glob('case*/report.json')):
        receipt=e.read(path);cid=receipt['case_id']
        if cid in {k[0] for k in keys}:
            packet=path.parent/'graphs.npz'
            if receipt['status']!='completed' or digest(packet)!=expected[cid]:raise ValueError('Classical packet changed')
            classical[cid]=packet
    if set(classical)!={k[0] for k in keys}:raise ValueError('Missing preserved classical case')
    for group in [broad,sato]:
        for p in group.values():
            if digest(Path(p['folder'])/'graph.npz')!=p['inference']['graph_sha256']:
                raise ValueError('Saved learned/Sato candidate graph changed')
    return broad,sato,classical


def construct(c,data,keys,output):
    broad,sato,classical=inventory(c,keys)
    guides={};maxima={};classical_cache={};packets=[]
    for i,key in enumerate(keys):
        cid,tile=key;box=data.boxes[key]['box'];low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
        canonical=[output/'candidate_receipts'/f'{arm}_{i:04d}.json' for arm in ARMS]
        if all(p.exists() for p in canonical):
            for path in canonical:
                record=e.read(path)
                if record['binding']!=fingerprint({'config':fingerprint(c),'arm':record['packet']['arm'],'key':list(key)}):
                    raise ValueError('Candidate cache binding differs')
                packet=record['packet']
                if digest(output/packet['path'])!=packet['sha256']:raise ValueError('Cached candidate changed')
                packets.append(packet)
            print(f'CANDIDATES {i+1}/151 reused',flush=True);continue
        if cid not in classical_cache:
            classical_cache.clear()
            with np.load(classical[cid],allow_pickle=False) as a:classical_cache[cid]=(a['pruned_xyz'].copy(),a['pruned_edges'].copy())
        cp,ce,_=crop_classical(*classical_cache[cid],low,high)
        with np.load(Path(broad[key]['folder'])/'graph.npz',allow_pickle=False) as a:lp,le=a['xyz'].copy(),a['forest_edges'].copy()
        with np.load(Path(sato[key]['folder'])/'graph.npz',allow_pickle=False) as a:sp,se=a['xyz'].copy(),a['forest_edges'].copy()
        if len(sp)!=512:raise ValueError('Sato source cap differs')
        source={'classical':(cp,ce),'sato':(sp,se),'learned':(lp,le)}
        image=data.image(cid,tile,2048)['packet'];e.sato_scores(data,key,image,guides)
        if cid not in maxima:maxima[cid]=float(guides[cid].max())
        for arm,names in ARMS.items():
            path=output/'candidate_receipts'/f'{arm}_{i:04d}.json'
            binding=fingerprint({'config':fingerprint(c),'arm':arm,'key':list(key)})
            def compute():
                arrays=pool({name:source[name] for name in names})
                arrays['image_guide']=guide_at(guides[cid],arrays['xyz'],maxima[cid])
                folder=output/'candidates'/f'{arm}_{i:04d}_{uuid.uuid4().hex[:8]}'
                folder.mkdir(parents=True)
                np.savez_compressed(folder/'packet.npz',**arrays)
                p=arrays['xyz'];inside=np.all((p>=low)&(p<=high),axis=1)
                stats={'input_nodes':len(arrays['input_xyz']),'nodes':len(p),'edges':len(arrays['edges']),
                    'edges_k6_inherited':int(np.count_nonzero(arrays['edge_flags']&(7|EDGE_SOURCES['knn6']))),
                    'edges_k10_inherited':len(arrays['edges']),
                    'halo_nodes':int((~inside).sum()),'inherited_self_edges_dropped':int(arrays['inherited_self_edges_dropped']),
                    'source_node_counts':{name:int(np.count_nonzero(arrays['source_flags']&flag))
                                          for name,flag in [('classical',1),('sato',2),('learned',4)]},
                    'edge_source_counts':{name:int(np.count_nonzero(arrays['edge_flags']&flag))
                                          for name,flag in EDGE_SOURCES.items()}}
                packet={'arm':arm,'key':list(key),'path':(folder/'packet.npz').relative_to(output).as_posix(),
                    'sha256':digest(folder/'packet.npz'),'sizes':stats,'references_used_for_construction':False}
                write_once(folder/'construction.json',packet)
                return {'packet':packet}
            record,_=cached_record(path,binding,compute);packets.append(record['packet'])
        print(f'CANDIDATES {i+1}/151 built (image-only)',flush=True)
    manifest=output/'candidate_manifest.json'
    if not manifest.exists():write_once(manifest,{'packets':packets,'references_supplied_to_candidate_builder':False})
    elif e.read(manifest)['packets']!=packets:raise ValueError('Candidate manifest changed')
    return packets


def oracle(c,scientific,data,packets,output):
    refs=e.References(scientific,data);contexts={};box_cache={};rows=[]
    for i,packet in enumerate(packets):
        arm=packet['arm'];key=tuple(packet['key']);cid,tile=key
        record_path=output/'oracle_records'/f'{arm}_{i:04d}.json'
        binding=fingerprint({'config':fingerprint(c),'packet':packet})
        def compute():
            if cid not in contexts:
                morphology=refs.get(cid)['morphology'];full=graph_segments(morphology)
                contexts[cid]=(_CurveIndex(full,4.) if len(full) else None,
                               cKDTree(morphology.isolated_points) if len(morphology.isolated_points) else None)
            if key not in box_cache:
                reference=graph_segments(runs_graph(data.reference(cid,tile)['reference_runs']))
                box_cache[key]=BoundedAgreement(reference)
            agreement=box_cache[key];box=data.boxes[key]['box']
            low,high=np.asarray(box['start_xyz'])-.5,np.asarray(box['stop_xyz'])-.5
            with np.load(output/packet['path'],allow_pickle=False) as a:
                xyz,edges,flags=a['xyz'],a['edges'],a['edge_flags'];guide=a['image_guide']
                d=bounded_distances(xyz,contexts[cid][0])
                if contexts[cid][1] is not None and len(xyz):d=np.minimum(d,contexts[cid][1].query(xyz)[0])
                node_p=(d<=2.).astype(float)
                edge_p,in_length=edge_agreement(xyz,edges,low,high,agreement.index)
                coverage={}
                for k in (6,10):
                    mask=(flags & (7|EDGE_SOURCES[f'knn{k}']))!=0
                    graph=SimpleNamespace(xyz=xyz,edges=edges[mask])
                    coverage[str(k)]=agreement.score(clip_segments(graph_segments(graph),low,high))
                eligible=np.all(node_p[edges]>=.5,axis=1)
                eligible_graph=SimpleNamespace(xyz=xyz,edges=edges[eligible])
                node_eligible=agreement.score(clip_segments(graph_segments(eligible_graph),low,high))
                result=decode(xyz,edges,node_p,np.where(in_length>0,edge_p,0.))
                selected=result['selected_indices'];folder=output/'oracle_outputs'/f'{arm}_{i:04d}_{uuid.uuid4().hex[:8]}'
                folder.mkdir(parents=True)
                exported=export_forest(folder/'diagnostic_only.swc',xyz[selected],result['edges'],guide[selected])
                parsed=parse_swc(folder/'diagnostic_only.swc')
                if (not np.array_equal(parsed.xyz,xyz[selected])
                        or set(map(tuple,np.sort(parsed.edges,axis=1)))!=set(map(tuple,result['edges']))):
                    raise ValueError('Oracle SWC changed geometry/edges')
                metrics=agreement.score(clip_segments(graph_segments(parsed),low,high))
                np.savez_compressed(folder/'oracle_scores_DO_NOT_USE_IN_INFERENCE.npz',node_probability=node_p,
                    edge_probability=edge_p,edge_scored_length=in_length,selected_candidate_indices=selected,
                    decoded_edges=result['edges'])
            return {'row':{'arm':arm,'case_id':cid,'tile_id':tile,'segments':metrics,
                'candidate_coverage':coverage,'node_eligible_candidate_coverage':node_eligible,
                'candidate_sizes':packet['sizes'],'decoded':exported,
                'pruning':{k:v for k,v in result.items() if k in ['pre_prune_nodes','pre_prune_edges','pre_prune_components','pruned_nodes','pruned_length']},
                'swc':(folder/'diagnostic_only.swc').relative_to(output).as_posix(),
                'swc_sha256':digest(folder/'diagnostic_only.swc'),'uses_reference_at_selection':True}}
        record,reused=cached_record(record_path,binding,compute);rows.append(record['row'])
        print(f'ORACLE {i+1}/{len(packets)} {arm} reused={reused}',flush=True)
    summaries={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        summaries[arm]={'oracle':summarize(group),'candidate_coverage':{
            str(k):summarize([{'case_id':r['case_id'],'segments':r['candidate_coverage'][str(k)]} for r in group]) for k in (6,10)},
            'node_eligible_candidate_coverage':summarize(group,'node_eligible_candidate_coverage'),
            'sizes':{k:sum(r['candidate_sizes'][k] for r in group) for k in ['input_nodes','nodes','edges','halo_nodes']},
            'decoded_nodes':sum(r['decoded']['nodes'] for r in group),'decoded_edges':sum(r['decoded']['edges'] for r in group)}
    decision=headroom_gate(summaries['union']['oracle']['macro']['2.0']['f1'])
    return {'rows':rows,'summaries':summaries,'gate':decision}


def worker(output,resume):
    c,scientific,data,keys=control();output.mkdir(parents=True,exist_ok=resume)
    binding={'config_fingerprint':fingerprint(c),'stage':0,'scope':'151 repeatedly-inspected development boxes; reference-assisted diagnostic only'}
    if (output/'binding.json').exists():
        if e.read(output/'binding.json')!=binding:raise ValueError('Different Stage0 output binding')
    else:write_once(output/'binding.json',binding)
    for name in ['candidate_receipts','oracle_records']:(output/name).mkdir(exist_ok=True)
    packets=construct(c,data,keys,output);result=oracle(c,scientific,data,packets,output)
    report={'status':'completed',**binding,**result,'GPU_executed':False,'model_executed':False,'training_executed':False,
            'selects_policy_or_threshold':False,'whole_image_merge':False,'topology_certified':False}
    if not (output/'report.json').exists():write_once(output/'report.json',report)
    lines=['# Stage0 geometric oracle headroom (diagnostic only)','','| Arm | Oracle P2 | Oracle R2 | Oracle F1 at2 | Oracle F1 at6 | Candidate R2 k6 / k10 | Nodes / edges |','|---|---:|---:|---:|---:|---|---|']
    for arm,s in result['summaries'].items():
        a,b=s['oracle']['macro']['2.0'],s['oracle']['macro']['6.0']
        coverage=' / '.join(f'{100*s["candidate_coverage"][str(k)]["macro"]["2.0"]["recall"]:.2f}%' for k in (6,10))
        lines.append(f'| {arm} | {100*a["precision"]:.2f}% | {100*a["recall"]:.2f}% | {100*a["f1"]:.2f}% | {100*b["f1"]:.2f}% | {coverage} | {s["sizes"]["nodes"]} / {s["sizes"]["edges"]} |')
    lines+=['',json.dumps(decision:=result['gate'],indent=2),'','Reference-assisted geometric recipe; not inference, not maximum attainable F1 or a topology certificate.',
            'No arm/threshold selected. Stage1 follows only the declared gate; no automatic training or GPU launch.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if 'torch' in sys.modules:raise RuntimeError('Stage0 unexpectedly imported Torch')
    print('HEADROOM COMPLETE '+json.dumps({'gate':decision,'report':str(output/'report.json')}),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args();c,_,_,keys=control()
    if a.check:
        inventory(c,keys);print('Stage0 source/saved-candidate bindings verified; no model/oracle run');return
    if a.output and a.resume:p.error('Use --output OR --resume')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=e.output_path(a.resume or a.output or ROOT/'tmp'/f'candidate_headroom_{suffix}')
    if a.worker:worker(output,bool(a.resume));return
    if output.exists() and not a.resume:raise FileExistsError('Preserve headroom results; explicit --resume required')
    if a.resume and not (output/'binding.json').exists():raise ValueError('No bound Stage0 output to resume')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
             '--resume' if a.resume else '--output',str(output)]
    print(f'STAGE0 CPU ONLY; OUTPUT {output}; cap3600s; no training/GPU',flush=True)
    run(command,output.with_name(output.name+'_guard_'+suffix),3600,
        {'config_fingerprint':fingerprint(c),'output':str(output),'stage':0})


if __name__=='__main__':main()
