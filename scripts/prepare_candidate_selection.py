"""User-local Stage1 CPU preparation. Image-only pools first; training labels second.

No model execution, selector training, development labels or GPU work.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import json
import os
import sys
import uuid
import numpy as np

import measure_candidate_headroom as h
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.candidate_pool import pool,crop_classical
from pn2.evaluation.fixed_curve_graph import build
from pn2.evaluation.vote_scaffold import guide_at
from pn2.supervision.candidate_labels import label_candidates,label_counts
from pn2.training.guarded_run import run

e=h.e;ROOT=e.ROOT;CONFIG=ROOT/'configs/candidate_stage1_20261105.json'


def control():
    c=e.read(CONFIG)
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed Stage1 binding: '+name)
    for package,version in c['runtime_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Changed CPU runtime: '+package)
    if c['schema']!='pn2-candidate-stage1-v1' or c['path_bound']!=32. or c['wall_seconds']!=3600:
        raise ValueError('Changed Stage1 scope')
    hc,scientific,data,dev=h.control()
    report=e.read(ROOT/c['headroom_root']/'report.json')
    if report['gate']['status']!='proceed':raise ValueError('Stage0 gate did not authorize unconditional preparation')
    training=[(cid,b['tile_id']) for cid,case in data.cases.items() if case['partition']=='training'
              for b in case['boxes'] if b['target']['count']]
    if len(training)!=460 or len({k[0] for k in training})!=33:raise ValueError('Changed training cohort')
    return c,hc,scientific,data,training,dev


def checked_npz(path,sha):
    if digest(path)!=sha:raise ValueError('Changed packet: '+str(path))
    return np.load(path,allow_pickle=False)


def saved_inventory(c,keys):
    packets=e.read(ROOT/'tmp/broad_curve_head_20261105/evaluation/inference_complete.json')['packets']
    broad={tuple(p['key']):p for p in packets}
    if len(broad)!=611 or set(broad)!=set(keys):raise ValueError('Changed broad source identities')
    for packet in broad.values():
        if digest(Path(packet['folder'])/'graph.npz')!=packet['inference']['graph_sha256']:
            raise ValueError('Changed saved broad graph')
    return broad


def classical(data,c,training,output,guides):
    """Reuse six training extractions, perform only missing 27 on cached guides."""
    from pn2.evaluation.classical_skeleton import extract
    result={};receipts=[]
    for cid in dict.fromkeys(k[0] for k in training):
        key=next(k for k in training if k[0]==cid)
        image=data.image(*key,2048)['packet'];e.sato_scores(data,key,image,guides)
        folder=(ROOT/data.boxes[key]['clouds']['2048']['path']).parent
        guide=folder/e.read(folder/'case.json')['guide']['path'];guide_sha=digest(guide)
        if cid in c['preserved_classical']:
            spec=c['preserved_classical'][cid];path=ROOT/spec['path'];report=e.read(path.parent/'report.json')
            if digest(path)!=spec['sha256'] or report['guide_sha256']!=guide_sha or report['partition']!='training':
                raise ValueError('Preserved classical receipt differs')
            packet={'case_id':cid,'path':str(path),'sha256':spec['sha256'],'reused_historical':True,'guide_sha256':guide_sha}
        else:
            receipt=output/'classical_receipts'/f"alias{data.cases[cid]['alias']:03d}.json"
            binding=fingerprint({'config':fingerprint(c),'case_id':cid,'guide_sha256':guide_sha})
            def compute():
                raw,pruned,metadata=extract(guide)
                target=output/'classical'/f"alias{data.cases[cid]['alias']:03d}_{uuid.uuid4().hex[:8]}"
                target.mkdir(parents=True);path=target/'graphs.npz'
                np.savez_compressed(path,raw_xyz=raw.xyz,raw_edges=raw.edges,pruned_xyz=pruned.xyz,pruned_edges=pruned.edges)
                return {'packet':{'case_id':cid,'path':str(path),'sha256':digest(path),'reused_historical':False,
                                  'guide_sha256':guide_sha,'extraction':metadata}}
            record,reused=h.cached_record(receipt,binding,compute);packet=record['packet']
            if digest(Path(packet['path']))!=packet['sha256']:raise ValueError('Changed new classical packet')
        result[cid]=packet;receipts.append(packet)
        print(f"CLASSICAL {len(result)}/33 alias={data.cases[cid]['alias']} historical={packet['reused_historical']}",flush=True)
    write_if_missing(output/'classical_manifest.json',{'packets':receipts})
    return result


def write_if_missing(path,value):
    if path.exists():
        if e.read(path)!=value:raise ValueError('Existing manifest differs: '+str(path))
    else:write_once(path,value)


def construct(c,data,training,dev,output):
    allkeys=training+dev;broad=saved_inventory(c,allkeys);guides={}
    cp=classical(data,c,training,output,guides);maxima={};classical_cache={};packets=[]
    headroom=ROOT/c['headroom_root']
    old=e.read(headroom/'candidate_manifest.json')['packets']
    development={tuple(p['key']):p for p in old if p['arm']=='union'}
    if set(development)!=set(dev):raise ValueError('Changed Stage0 development pools')
    for key in allkeys:
        cid,tile=key;partition=data.cases[cid]['partition']
        if partition=='development':
            packet=development[key];path=headroom/packet['path']
            if digest(path)!=packet['sha256']:raise ValueError('Changed saved development pool')
            packets.append({'key':list(key),'partition':partition,'path':str(path),'sha256':packet['sha256'],
                            'sizes':packet['sizes'],'reused_stage0':True,'references_used_for_construction':False})
            continue
        receipt=output/'candidate_receipts'/f'train_{len(packets):04d}.json'
        binding=fingerprint({'config':fingerprint(c),'key':list(key)})
        def compute():
            nonlocal classical_cache
            if cid not in classical_cache:
                with checked_npz(Path(cp[cid]['path']),cp[cid]['sha256']) as a:
                    classical_cache={cid:(a['pruned_xyz'].copy(),a['pruned_edges'].copy())}
            box=data.boxes[key]['box'];low=np.asarray(box['start_xyz'])-.5;high=np.asarray(box['stop_xyz'])-.5
            xyz,edges,_=crop_classical(*classical_cache[cid],low,high)
            image=data.image(cid,tile,2048)['packet'];scores=e.sato_scores(data,key,image,guides)
            if cid not in maxima:maxima[cid]=float(guides[cid].max())
            ids=e.consolidate(image['native_xyz'],scores,radius=1.,identities=image['flat_indices'],cap=512)
            if len(ids)!=512:raise ValueError('Insufficient Sato candidates; do not silently change source cap')
            sf=output/'sato_forests'/f'box{len(packets):04d}_{uuid.uuid4().hex[:8]}'
            sato=build(image['native_xyz'][ids],guides[cid],maxima[cid],sf,key,'sato512',ids)
            with np.load(sf/'graph.npz',allow_pickle=False) as a:sp,se=a['xyz'].copy(),a['forest_edges'].copy()
            spec=broad[key]
            with checked_npz(Path(spec['folder'])/'graph.npz',spec['inference']['graph_sha256']) as a:
                lp,le=a['xyz'].copy(),a['forest_edges'].copy()
            arrays=pool({'classical':(xyz,edges),'sato':(sp,se),'learned':(lp,le)})
            arrays['image_guide']=guide_at(guides[cid],arrays['xyz'],maxima[cid])
            target=output/'candidates'/f'box{len(packets):04d}_{uuid.uuid4().hex[:8]}';target.mkdir(parents=True)
            path=target/'packet.npz';np.savez_compressed(path,**arrays)
            inside=np.all((arrays['xyz']>=low)&(arrays['xyz']<=high),axis=1)
            return {'packet':{'key':list(key),'partition':partition,'path':str(path),'sha256':digest(path),
                      'sizes':{'input_nodes':len(arrays['input_xyz']),'nodes':len(arrays['xyz']),
                               'edges':len(arrays['edges']),'halo_nodes':int((~inside).sum())},
                      'reused_stage0':False,'references_used_for_construction':False,'sato_forest':sato}}
        record,reused=h.cached_record(receipt,binding,compute);packet=record['packet']
        if digest(Path(packet['path']))!=packet['sha256']:raise ValueError('Changed completed candidate packet')
        packets.append(packet);print(f'CANDIDATES {len(packets)}/460 training reused={reused}',flush=True)
    manifest={'schema':'pn2-selector-candidates-v1','config_fingerprint':fingerprint(c),'packets':packets,
              'training_boxes':460,'development_boxes':151,'all_geometry_saved_before_training_labels':True,
              'raw_input_in_pool':False,'development_labels':False}
    write_if_missing(output/'candidate_manifest.json',manifest)
    return packets


def generate_labels(c,scientific,data,packets,output):
    # This function is called only after the complete geometry manifest exists.
    refs=e.References(scientific,data);rows=[];labels=[]
    for packet in packets:
        if packet['partition']!='training':continue
        key=tuple(packet['key']);cid=key[0];i=len(rows)
        binding=fingerprint({'config':fingerprint(c),'candidate_sha256':packet['sha256'],'key':list(key)})
        def compute():
            with checked_npz(Path(packet['path']),packet['sha256']) as a:
                arrays,stats=label_candidates(a['xyz'],a['edges'],refs.get(cid)['morphology'],partition='training')
            target=output/'labels'/f'box{i:04d}_{uuid.uuid4().hex[:8]}';target.mkdir(parents=True)
            path=target/'labels.npz';np.savez_compressed(path,**arrays)
            return {'row':{'case_id':cid,'key':list(key),'counts':label_counts(arrays),'topology':stats},
                    'packet':{'key':list(key),'partition':'training','path':str(path),'sha256':digest(path),
                              'candidate_sha256':packet['sha256']}}
        record,reused=h.cached_record(output/'label_receipts'/f'box{i:04d}.json',binding,compute)
        if digest(Path(record['packet']['path']))!=record['packet']['sha256']:raise ValueError('Changed saved labels')
        rows.append(record['row']);labels.append(record['packet'])
        if len(rows)%10==0 or len(rows)==460:print(f'LABELS {len(rows)}/460 training-only reused={reused}',flush=True)
    write_if_missing(output/'label_manifest.json',{'config_fingerprint':fingerprint(c),'packets':labels,
                                                  'development_labels_created':False})
    neurons=[]
    for cid in sorted({r['case_id'] for r in rows}):
        group=[r for r in rows if r['case_id']==cid];counts={}
        for kind in ('node','edge'):
            counts[kind]={k:sum(r['counts'][kind][k] for r in group) for k in ('total','positive','negative','unknown')}
            counts[kind]['masked_fraction']=counts[kind]['unknown']/counts[kind]['total'] if counts[kind]['total'] else None
            counts[kind]['reasons']={k:sum(r['counts'][kind]['reasons'][k] for r in group)
                                    for k in group[0]['counts'][kind]['reasons']}
        neurons.append({'case_id':cid,'alias':data.cases[cid]['alias'],'boxes':len(group),'counts':counts,
                        'junction_instances':sum(r['topology']['junctions'] for r in group),
                        'anchored_instances':sum(r['topology']['junctions_with_anchor'] for r in group)})
    return rows,neurons


def worker(output,resume):
    c,_,scientific,data,training,dev=control();output.mkdir(parents=True,exist_ok=resume)
    write_if_missing(output/'binding.json',{'config_fingerprint':fingerprint(c),'stage':1})
    for name in ('classical_receipts','candidate_receipts','label_receipts'):(output/name).mkdir(exist_ok=True)
    packets=construct(c,data,training,dev,output)
    rows,neurons=generate_labels(c,scientific,data,packets,output)
    report={'status':'completed','config_fingerprint':fingerprint(c),'rows':rows,'neurons':neurons,
            'candidate_manifest_sha256':digest(output/'candidate_manifest.json'),
            'label_manifest_sha256':digest(output/'label_manifest.json'),
            'training_boxes':460,'development_boxes':151,'development_labels_created':False,
            'GPU_executed':False,'model_executed':False,'selector_trained':False,
            'CV':'selector-only conditional on upstream training on all33 training neurons',
            'label_semantics':'trace agreement, not exhaustive biological foreground'}
    write_if_missing(output/'report.json',report)
    lines=['# Stage1 candidate/label preparation','','Training-only labels; development geometry reused, no development labels.',
           'Counts pool overlapping per-box candidate instances, not unique whole-neuron nodes.',
           'Junction counts below are full-reference junction instances tested per box, not unique cropped junction counts.','',
           '| Alias | Boxes | Nodes + / - / unknown | Node masked | Edges + / - / unknown | Edge masked |',
           '|---|---:|---|---:|---|---:|']
    for n in neurons:
        a,b=n['counts']['node'],n['counts']['edge']
        def triple(v):return ' / '.join(str(v[k]) for k in ('positive','negative','unknown'))
        def fraction(v):return f"{100*v['masked_fraction']:.2f}%" if v['masked_fraction'] is not None else 'n/a'
        lines.append(f"| {n['alias']} | {n['boxes']} | {triple(a)} | {fraction(a)} | {triple(b)} | {fraction(b)} |")
    lines+=['','Preparation verifies execution and supervision availability, not learnability or reconstruction quality.',
            'No model selection, development scoring, training or automatic next stage.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if 'torch' in sys.modules:raise RuntimeError('CPU preparation unexpectedly imported Torch')
    print('STAGE1 COMPLETE '+json.dumps({'report':str(output/'report.json'),'training_labels':460,'automatic_training':False}),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    c,_,_,_,training,dev=control()
    if a.check:
        saved_inventory(c,training+dev);print('Stage1 bindings/cohort verified:460 training,151 development; no preparation/model run');return
    if a.output and a.resume:p.error('Use output OR resume')
    output=e.output_path(a.resume or a.output or ROOT/'tmp/candidate_stage1_20261105')
    if a.worker:worker(output,bool(a.resume));return
    if output.exists() and not a.resume:raise FileExistsError('Preserve outputs; explicit --resume is required')
    if a.resume and not (output/'binding.json').exists():raise ValueError('No bound output to resume')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--resume' if a.resume else '--output',str(output)]
    print(f'STAGE1 CPU ONLY; OUTPUT {output}; cap3600s; no training or GPU; no automatic retry',flush=True)
    run(command,output.with_name(output.name+'_guard_'+suffix),c['wall_seconds'],{'stage':1,'output':str(output),'config':fingerprint(c)})


if __name__=='__main__':main()
