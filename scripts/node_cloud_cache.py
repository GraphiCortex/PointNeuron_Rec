"""CPU-only image cache controls: synthetic check, cached replay, fresh-case timing.

This entry point consumes image-only manifests and reference-free guide receipts.
It never reads reference/target manifests, model checkpoints or reserved files.
Bulk preparation and model training are not launched by these actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pn2.inputs.cases import ImageCase,load_image_cases
from pn2.inputs.node_cache import (build_case,case_key,digest,fingerprint,verify_case,
                                   versions,write_once)

CONFIG=ROOT/'configs/node_cloud_cache.json'
SETTINGS={'tile_side_voxel':128,'points_per_box':2048,'coordinate_scale_voxel':64.,
          'sigmas_voxel':[1.,2.,4.],'uniform_mass':.1,'cloud_seed':0,'ordering_seed':0}
SOURCE_FILES=('src/pn2/inputs/node_cache.py','src/pn2/inputs/cases.py','src/pn2/inputs/image.py',
              'src/pn2/inputs/vaa3d.py','src/pn2/inputs/sampling.py','src/pn2/inputs/disk_guide.py',
              'src/pn2/evaluation/branch_evidence.py')


def load_config(path):
    control=json.loads(Path(path).read_text())
    required={'schema_version','image_manifest','settings','versions','source_sha256',
              'guide_registry','regression_packets','benchmark_case_id','cpu_preflight'}
    if (set(control)!=required or control['schema_version']!=1 or control['settings']!=SETTINGS
            or set(control['source_sha256'])!=set(SOURCE_FILES)):
        raise ValueError('Only the fixed image-only node-cache policy is supported')
    if control['versions']!=versions():raise ValueError('Use the pinned Phase 1 CPU runtime')
    for name,sha in control['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Cache source changed: {name}')
    spec=control['image_manifest'];path=ROOT/spec['path']
    if digest(path)!=spec['sha256']:raise ValueError('Approved image-only manifest changed')
    cases=load_image_cases(path)
    if (len(cases)!=40 or sum(c.partition=='training' for c in cases)!=33
            or sum(c.partition=='development' for c in cases)!=7 or any(c.units!='voxel' for c in cases)):
        raise ValueError('Expected the explicit native-voxel 33/7 cohort')
    if control['benchmark_case_id'] not in {c.case_id for c in cases if c.partition=='training'}:
        raise ValueError('Benchmark must be an explicitly listed training image')
    return control,cases


def cache_identity(control):
    return {'schema':'pn2-node-image-cache-v1','image_manifest_sha256':control['image_manifest']['sha256'],
            'settings':control['settings'],'versions':control['versions'],'source_sha256':control['source_sha256']}


def execution_identity(control):
    return {'cache_identity':cache_identity(control),'driver_sha256':digest(__file__),
            'control_fingerprint':fingerprint(control)}


def guide_candidates(control,cases):
    spec=control['guide_registry'];path=ROOT/spec['path']
    if digest(path)!=spec['sha256']:raise ValueError('Guide registry changed')
    registry=json.loads(path.read_text());allowed={c.case_id for c in cases};result={}
    sampling=digest(ROOT/'src/pn2/inputs/sampling.py')
    for entry in registry['entries']:
        if entry['case_id'] not in allowed or entry['case_id'] in result:
            raise ValueError('Guide registry has an unknown/duplicate case')
        source=ROOT/entry['report']
        if digest(source)!=entry['report_sha256']:raise ValueError('Original guide report changed')
        report=json.loads(source.read_text());item=report['cases'][entry['case_index']]
        recipe_path=ROOT/entry['recipe']
        if digest(recipe_path)!=report['frozen_recipe_sha256']:raise ValueError('Original guide recipe changed')
        recipe=json.loads(recipe_path.read_text())
        if (report['versions']!=control['versions'] or report['source_code_sha256']!=sampling
                or recipe['sigmas_voxel']!=SETTINGS['sigmas_voxel']
                or item['case_id']!=entry['case_id']):
            raise ValueError('Original guide implementation/runtime/scales differ')
        if report.get('guide_backend')=='disk' and report['disk_guide_code_sha256']!=digest(ROOT/'src/pn2/inputs/disk_guide.py'):
            raise ValueError('Original disk-guide operator differs')
        guide=ROOT/entry['guide_path']
        result[item['case_id']]={'case_id':item['case_id'],'guide_path':str(guide.resolve()),
            'guide_sha256':item['guide_sha256'],'source_sha256':item['image']['source_sha256'],
            'decoded_sha256':item['image']['decoded_sha256'],'normalization':item['normalization'],
            'sigmas_voxel':recipe['sigmas_voxel'],'versions':report['versions'],
            'source_receipt_sha256':entry['report_sha256'],
            'original_generation_script_sha256':report['generation_script_sha256']}
    return result


def require_cpu(control,path=None):
    report=json.loads((Path(path) if path else ROOT/control['cpu_preflight']).read_text())
    if (report.get('status')!='passed' or not report.get('exact_whole_vs_disk')
            or not report.get('exact_reuse_vs_fresh')
            or any(report.get(k)!=v for k,v in execution_identity(control).items())):
        raise ValueError('Image-cache CPU check is missing, failed or stale')


def cpu_check(control,output):
    from pn2.inputs.sampling import normalized_image,sato_guide
    from pn2.inputs.node_cache import point_arrays
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    # Asymmetric uint16 source with a bright tube and deterministic background.
    z,y,x=np.indices((15,17,19))
    raw=(3*x+5*y+7*z+((x-9)**2+(y-8)**2<4)*800).astype(np.uint16)
    path=folder/'synthetic.v3draw'
    header=b'raw_image_stack_by_hpeng'+b'L'+(2).to_bytes(2,'little')+b''.join(int(d).to_bytes(4,'little') for d in (*raw.shape[::-1],1))
    path.write_bytes(header+raw.astype('<u2').tobytes())
    case=ImageCase('synthetic_geometry',path,'synthetic')
    identity=cache_identity(control)
    receipt=build_case(case,SETTINGS,folder/'fresh',identity,reserve_bytes=0,progress=lambda s:None)
    normalized,transform=normalized_image(raw)
    expected=sato_guide(normalized,SETTINGS['sigmas_voxel'])
    # read_block explicitly closes the mapping before any reuse/copy on Windows.
    from pn2.inputs.disk_guide import read_block
    actual=read_block(folder/'fresh/sato_guide.npy',(slice(None),)*3)
    if not np.array_equal(actual,expected):raise ValueError('Fresh disk guide differs from pinned whole operator')
    tile=(slice(0,15),slice(0,17),slice(0,19))
    arrays,_=point_arrays(raw,expected,tile,0,SETTINGS,transform)
    with np.load(folder/'fresh/tile00000.npz',allow_pickle=False) as saved:
        if not all(np.array_equal(saved[k],v) for k,v in arrays.items()):
            raise ValueError('Fresh source-point adapter differs from the whole-guide path')
    reuse={'case_id':case.case_id,'guide_path':str(folder/'fresh/sato_guide.npy'),
        'guide_sha256':receipt['guide']['sha256'],'source_sha256':receipt['image']['source_sha256'],
        'decoded_sha256':receipt['image']['decoded_sha256'],'normalization':transform,
        'sigmas_voxel':SETTINGS['sigmas_voxel'],'versions':versions(),'source_receipt_sha256':digest(folder/'fresh/case.json')}
    second=build_case(case,SETTINGS,folder/'reused',identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    with np.load(folder/'fresh/tile00000.npz',allow_pickle=False) as a,np.load(folder/'reused/tile00000.npz',allow_pickle=False) as b:
        if not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Reused-guide points differ')
    verify_case(folder/'fresh',case,identity);verify_case(folder/'reused',case,identity)
    report={'status':'passed',**execution_identity(control),'exact_whole_vs_disk':True,
        'exact_reuse_vs_fresh':True,'synthetic_points':2048,'references_opened':0,
        'models_executed':0,'gpu_executed':False,'torch_imported':'torch' in sys.modules}
    write_once(folder/'report.json',report)
    print(f"PASSED synthetic image-cache check: {folder/'report.json'}",flush=True)


def cache_root(path,identity):
    folder=Path(path).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):
        raise ValueError('Generated cache must stay under PN2/tmp')
    folder.mkdir(parents=True,exist_ok=True)
    stamp=folder/'cache_policy.json'
    if stamp.exists():
        if json.loads(stamp.read_text())!=identity:raise ValueError('Cache root contains another scientific policy')
    else:
        if any(folder.iterdir()):raise ValueError('Nonempty cache root has no policy receipt')
        write_once(stamp,identity)
    return folder


def launch_worker(command,folder,timeout):
    with (folder/'console.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        def relay():
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
        thread=threading.Thread(target=relay,daemon=True);thread.start()
        try:
            code=process.wait(timeout=timeout);status='completed' if code==0 else 'failed_exit'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            code,status=process.returncode,'timeout'
        thread.join(timeout=10)
    return code,status


def regression_comparison(control,root,cases):
    expected_cases={c.case_id:c for c in cases};rows=[]
    for spec in control['regression_packets']:
        case=expected_cases[spec['case_id']]
        receipt=verify_case(root/case_key(case),case,cache_identity(control))
        row=next(r for r in receipt['records'] if r['tile_id']==spec['tile_id'])
        original=ROOT/spec['path']
        if digest(original)!=spec['sha256']:raise ValueError('Original regression input changed')
        with np.load(original,allow_pickle=False) as a,np.load(root/case_key(case)/row['packet'],allow_pickle=False) as b:
            exact=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) and a[k].dtype==b[k].dtype for k in a.files)
        if not exact:raise ValueError('New reference-free cloud differs from the successful four-box input')
        rows.append({'case_id':case.case_id,'tile_id':spec['tile_id'],'all_arrays_and_dtypes_exact':True})
    return rows


def index_cache(control,cases,root):
    rows=[];content={};duplicates=[];within=[]
    for case in cases:
        folder=root/case_key(case)
        if not folder.exists():continue
        if not (folder/'complete.json').exists():raise ValueError(f'Incomplete case preserved: {folder}')
        receipt=verify_case(folder,case,cache_identity(control))
        for signature in (('source',receipt['image']['source_sha256']),
                          ('decoded',tuple(receipt['image']['dimensions_xyzc']),receipt['image']['datatype'],receipt['image']['decoded_sha256'])):
            if signature in content and content[signature][0]!=case.partition:
                duplicates.append([content[signature][1],case.case_id])
            elif signature in content:
                within.append({'kind':signature[0],'cases':[content[signature][1],case.case_id]})
            else:content[signature]=(case.partition,case.case_id)
        rows.append({'case_id':case.case_id,'partition':case.partition,'folder':case_key(case),
            'case_json_sha256':digest(folder/'case.json'),'tiles':receipt['tile_count']})
    if duplicates:raise ValueError(f'Cross-split duplicate image content: {duplicates}')
    return {'completed_cases':len(rows),'expected_cases':len(cases),'cloud_cache_complete':len(rows)==len(cases),
        'training_ready':False,'records':rows,'cross_split_duplicate_images':duplicates,
        'within_partition_duplicate_images':within}


def worker(control,args,cases):
    require_cpu(control,args.cpu_preflight)
    case=next((c for c in cases if c.case_id==args.case_id),None)
    if case is None:raise ValueError('Worker case is not explicitly approved')
    root=cache_root(args.cache_root,cache_identity(control));target=root/case_key(case)
    if target.exists():raise FileExistsError(f'Preserve existing case; no automatic retry: {target}')
    candidates=guide_candidates(control,cases);entry=candidates.get(case.case_id)
    if args.fresh and entry:raise ValueError('Timing case must be uncached in the explicit guide registry')
    print(f"PREPARE CPU image={case.case_id} partition={case.partition} reuse_guide={bool(entry)} all_image_boxes=True",flush=True)
    receipt=build_case(case,SETTINGS,target,cache_identity(control),reuse=entry,progress=lambda s:print(s,flush=True))
    write_once(Path(args.output)/'worker_report.json',{'status':'completed',**execution_identity(control),
        'case_id':case.case_id,'cache_folder':case_key(case),'tiles':receipt['tile_count'],
        'case_json_sha256':digest(target/'case.json'),'guide_execution':receipt['guide']['execution'],
        'decode_seconds':receipt['decode_seconds'],'guide_seconds':receipt['guide_seconds'],
        'total_seconds':receipt['total_seconds'],'memory':receipt['memory']})


def controller(control,args,cases):
    require_cpu(control,args.cpu_preflight)
    if args.action not in ('replay','timing'):raise ValueError('Only bounded replay/timing actions are offered now')
    candidates=guide_candidates(control,cases)
    wanted=([control['benchmark_case_id']] if args.action=='timing' else list(dict.fromkeys(s['case_id'] for s in control['regression_packets'])))
    if args.action=='timing' and wanted[0] in candidates:raise ValueError('Benchmark must measure a fresh filter')
    root=cache_root(args.cache_root,cache_identity(control))
    for case in cases:
        if case.case_id in wanted and (root/case_key(case)).exists():
            raise FileExistsError('Requested case already exists; preserve it and use its completed report')
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Run outputs must stay under PN2/tmp')
    folder.mkdir(parents=True,exist_ok=False)
    reports=[];timeout=3600 if args.action=='timing' else 300
    for i,identity in enumerate(wanted):
        run=folder/f'worker_{i:02d}';run.mkdir()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker',
            '--config',str(Path(args.config).resolve()),'--case-id',identity,
            '--cache-root',str(root),'--output',str(run)]
        if args.action=='timing':command+=['--fresh']
        if args.cpu_preflight:command+=['--cpu-preflight',str(Path(args.cpu_preflight).resolve())]
        code,status=launch_worker(command,run,timeout)
        path=run/'worker_report.json'
        report=json.loads(path.read_text()) if path.exists() else {'status':'missing_worker_report'}
        if status!='completed' or report.get('status')!='completed' or any(report.get(k)!=v for k,v in execution_identity(control).items()):
            write_once(folder/'report.json',{'status':status if status!='completed' else 'failed_incomplete_worker',
                'exit_code':code,'case_id':identity,'completed_workers':reports})
            return 1
        reports.append(report)
    index=index_cache(control,cases,root)
    result={'status':'completed','action':args.action,**execution_identity(control),
        'cache_root':str(root),'worker_timeout_seconds':timeout,'workers':reports,'cache_index':index,
        'references_opened':0,'models_executed':0,'gpu_executed':False,'bulk_preparation_launched':False}
    if args.action=='replay':result['regression_comparison']=regression_comparison(control,root,cases)
    write_once(folder/'report.json',result)
    print(f"COMPLETED CPU {args.action}; cases={index['completed_cases']}/{index['expected_cases']}; report={folder/'report.json'}",flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','replay','timing','_worker'),required=True)
    parser.add_argument('--config',default=str(CONFIG))
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache-root',default=str(ROOT/'tmp/node_cloud_cache_20261031'))
    parser.add_argument('--cpu-preflight')
    parser.add_argument('--case-id')
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args();control,cases=load_config(args.config)
    if args.action=='check':cpu_check(control,args.output);return 0
    if args.action=='_worker':worker(control,args,cases);return 0
    return controller(control,args,cases)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
