"""User-local inference-only PointNeXt feature export. No selector/model training.

Default CPU; CUDA requires the user's explicit --device cuda:0 invocation.
Consumes completed Stage1 candidates, never label files or oracle probabilities.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import sys
import uuid
import numpy as np

import local_curve_experiment as e
from score_saved_curve_transfer import cached_record
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.inputs.candidate_features import interpolate
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/candidate_pointnext_export_20261105.json'


def control(stage1):
    c=e.read(CONFIG)
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed feature-export binding: '+name)
    if c['schema']!='pn2-candidate-pointnext-export-v1' or c['wall_seconds']!=3600:
        raise ValueError('Changed export scope')
    scientific=e.control();data,_,_=e.prepared(scientific)
    report=e.read(stage1/'report.json');manifest=e.read(stage1/'candidate_manifest.json')
    if (report['status']!='completed' or report['config_fingerprint']!=fingerprint(e.read(ROOT/c['stage1_config']))
            or manifest['config_fingerprint']!=report['config_fingerprint']
            or digest(stage1/'candidate_manifest.json')!=report['candidate_manifest_sha256']):
        raise ValueError('Stage1 is not complete/bound')
    packets=manifest['packets']
    expected={(cid,b['tile_id']) for cid,case in data.cases.items() for b in case['boxes'] if b['target']['count']}
    if len(packets)!=611 or {tuple(p['key']) for p in packets}!=expected:raise ValueError('Changed feature cohort')
    if sum(p['partition']=='training' for p in packets)!=460 or sum(p['partition']=='development' for p in packets)!=151:
        raise ValueError('Changed export partitions')
    for packet in packets:
        if (packet['partition']!=data.cases[packet['key'][0]]['partition'] or packet['references_used_for_construction']
                or digest(Path(packet['path']))!=packet['sha256']):raise ValueError('Changed image-only candidates')
    return c,scientific,data,packets


def worker(stage1,output,device,resume):
    c,scientific,data,packets=control(stage1)
    import torch
    import sampler_node_common as common
    from pn2.training.node_runtime import image_batch
    e.runtime(scientific,device)
    checkpoint=ROOT/c['checkpoint'];payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=e.model_factory(scientific,'dense',device);model.load_state_dict(payload['model'],strict=True)
    del payload
    model.eval();before=common.state_digest(model.state_dict());rng=common.nested_digest(common.rng_state())
    output.mkdir(parents=True,exist_ok=resume);(output/'receipts').mkdir(exist_ok=True)
    binding={'config_fingerprint':fingerprint(c),'candidate_manifest_sha256':digest(stage1/'candidate_manifest.json'),
             'checkpoint_sha256':digest(checkpoint),'device':device,'stage':'PointNeXt inference-only'}
    if (output/'binding.json').exists():
        if e.read(output/'binding.json')!=binding:raise ValueError('Export resume binding differs')
    else:write_once(output/'binding.json',binding)
    records=[]
    with torch.inference_mode():
        for i,packet in enumerate(packets):
            key=tuple(packet['key']);source_spec=data.boxes[key]['clouds']['2048']
            row_binding=fingerprint({**binding,'key':list(key),'candidate_sha256':packet['sha256'],'input_sha256':source_spec['sha256']})
            def compute():
                image=data.image(*key,2048);votes=model(image_batch(image,device))
                source_features=votes.features[0].detach().cpu().numpy().copy()
                if source_features.shape!=(len(image['packet']['native_xyz']),32):raise ValueError('Changed PointNeXt channels')
                with np.load(packet['path'],allow_pickle=False) as a:xyz=a['xyz'].copy()
                arrays=interpolate(image['packet']['native_xyz'],source_features,xyz)
                arrays.update(source_features=source_features,source_xyz=image['packet']['native_xyz'],
                              source_flat_indices=image['packet']['flat_indices'],candidate_xyz=xyz)
                target=output/'features'/f'box{i:04d}_{uuid.uuid4().hex[:8]}';target.mkdir(parents=True)
                path=target/'features.npz';np.savez_compressed(path,**arrays)
                return {'packet':{'key':list(key),'partition':packet['partition'],'path':str(path),'sha256':digest(path),
                                   'candidate_sha256':packet['sha256'],'input_sha256':source_spec['sha256'],
                                   'channels':32,'max_nearest_source_distance':float(arrays['neighbor_distances'][:,0].max())
                                       if len(xyz) else None}}
            record,reused=cached_record(output/'receipts'/f'box{i:04d}.json',row_binding,compute)
            if digest(Path(record['packet']['path']))!=record['packet']['sha256']:raise ValueError('Changed feature packet')
            records.append(record['packet'])
            if (i+1)%10==0 or i==610:print(f'FEATURES {i+1}/611 device={device} reused={reused}',flush=True)
    if before!=common.state_digest(model.state_dict()) or rng!=common.nested_digest(common.rng_state()):
        raise RuntimeError('Inference changed weights/buffers or RNG')
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Unexpected gradients')
    if digest(checkpoint)!=binding['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    complete={'status':'completed',**binding,'packets':records,'optimizer_created':False,'updates':0,
              'model_state_unchanged':True,'RNG_unchanged':True,'reference_geometry_opened':False,
              'CUDA_initialized':torch.cuda.is_initialized(),'interpolation':'3NN inverse distance; exact copy at zero; native XYZ',
              'CV_limit':'upstream encoder and candidates trained on all33; conditional selector CV only'}
    if (output/'complete.json').exists():
        if e.read(output/'complete.json')!=complete:raise ValueError('Existing completion differs')
    else:write_once(output/'complete.json',complete)
    print('FEATURE EXPORT COMPLETE '+json.dumps({'packets':len(records),'report':str(output/'complete.json'),'updates':0}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,default=ROOT/'tmp/candidate_stage1_20261105')
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu');p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    stage1=e.output_path(a.stage1);c,_,_,_=control(stage1)
    if a.check:print('Completed Stage1/source/checkpoint bindings verified; no model execution');return
    if a.output and a.resume:p.error('Use output OR resume')
    output=e.output_path(a.resume or a.output or ROOT/'tmp/candidate_pointnext_20261105')
    if a.worker:worker(stage1,output,a.device,bool(a.resume));return
    if output.exists() and not a.resume:raise FileExistsError('Preserve outputs; explicit --resume required')
    if a.resume and not (output/'binding.json').exists():raise ValueError('No bound export to resume')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--stage1',str(stage1),'--device',a.device,
             '--resume' if a.resume else '--output',str(output)]
    print(f'INFERENCE ONLY; device={a.device}; OUTPUT {output}; cap3600s; no training/automatic next stage',flush=True)
    run(command,output.with_name(output.name+'_guard_'+suffix),c['wall_seconds'],{'stage':'feature-export','device':a.device,'output':str(output)})


if __name__=='__main__':main()
