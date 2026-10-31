"""One exact-output CPU guide comparison: existing block64 versus block128.

No filter/kernel/source/sampling/model change or cohort launch. The existing
case cache stays immutable; candidate work goes to a fresh ignored directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import psutil

import node_cloud_cache as cache_driver
from pn2.inputs.node_cache import digest,fingerprint,write_once,verify_case,case_key,point_arrays
from pn2.inputs.image import decode_image
from pn2.inputs.sampling import normalized_image,sato_guide,partition_tiles,probabilities,ranked_selection
from pn2.inputs.disk_guide import disk_sato_guide,read_block

ROOT=cache_driver.ROOT
CONFIG=ROOT/'configs/node_guide_block_compare.json'


def load_control(path):
    config=json.loads(Path(path).read_text())
    if (config['candidate_block_side']!=128 or config['baseline_block_side']!=64
            or config['script_sha256']!=digest(__file__)):
        raise ValueError('Expected the single declared block64/block128 comparison')
    control,cases=cache_driver.load_config(ROOT/config['cache_config'])
    cache_driver.require_cpu(control)
    baseline_path=ROOT/config['baseline_timing']['path']
    if digest(baseline_path)!=config['baseline_timing']['sha256']:raise ValueError('Baseline timing changed')
    baseline=json.loads(baseline_path.read_text())
    if (baseline['status']!='completed' or baseline['action']!='timing'
            or any(baseline.get(k)!=v for k,v in cache_driver.execution_identity(control).items())):
        raise ValueError('Baseline does not qualify the current unchanged cache worker')
    case=next(c for c in cases if c.case_id==control['benchmark_case_id'])
    root=Path(baseline['cache_root']).resolve()
    if not root.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Baseline cache is outside PN2/tmp')
    receipt=verify_case(root/case_key(case),case,cache_driver.cache_identity(control))
    if (receipt['guide']['execution']['block_side']!=64 or receipt['tile_count']!=25
            or receipt['guide']['execution']['filter_executed'] is not True):
        raise ValueError('Expected the completed fresh block64 Image5 case')
    return config,control,case,root/case_key(case),receipt


def identity(config,control):
    return {'comparison_fingerprint':fingerprint(config),'script_sha256':digest(__file__),
            'cache_identity':cache_driver.cache_identity(control)}


def compare_guide(candidate,baseline):
    if digest(candidate)!=digest(baseline):raise ValueError('Candidate guide is not byte-identical; do not adopt it')


def cpu_check(config,control,output):
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    checks=[]
    for i,shape in enumerate(((17,131,133),(131,17,133))):
        field=np.random.default_rng(53+i).random(shape,dtype=np.float32)
        field[0,0,0]=1.;field[-1,-1,-1]=.75
        p=folder/f'fixture{i}.npy';np.save(p,field)
        expected=sato_guide(field,control['settings']['sigmas_voxel'])
        results={}
        for side in (64,128):
            path,_=disk_sato_guide(p,control['settings']['sigmas_voxel'],folder/f'work_{i}_{side}',block_side=side)
            actual=read_block(path,(slice(None),)*3)
            if not np.array_equal(actual,expected):raise ValueError('Synthetic seam/edge parity failed')
            a,_=probabilities(actual,.1);b,_=probabilities(expected,.1)
            if not np.array_equal(ranked_selection(a,2048,0,0),ranked_selection(b,2048,0,0)):
                raise ValueError('Synthetic sampling differs')
            results[str(side)]={'guide_exact':True,'selected_indices_exact':True,'sha256':digest(path)}
        if results['64']['sha256']!=results['128']['sha256']:
            raise ValueError('Synthetic NPY guide bytes differ')
        checks.append({'shape_zyx':list(shape),'results':results})
        print(f'CPU BLOCK PARITY shape={shape}: whole/64/128 exact; point indices exact',flush=True)
    report={'status':'passed',**identity(config,control),'fixtures':checks,'real_image_filters_executed':0,
            'references_opened':0,'models_executed':0,'gpu_executed':False}
    write_once(folder/'report.json',report)


def require_cpu(config,control):
    report=json.loads((ROOT/config['cpu_preflight']).read_text())
    if (report.get('status')!='passed' or len(report.get('fixtures',[]))!=2
            or any(report.get(k)!=v for k,v in identity(config,control).items())):
        raise ValueError('Block-size CPU parity check is missing, failed or stale')


def worker(config,control,case,baseline_folder,baseline,args):
    require_cpu(config,control)
    folder=Path(args.output).resolve()
    if shutil.disk_usage(folder).free < np.prod(baseline['image']['dimensions_xyzc'])*48+2*1024**3:
        raise OSError('Insufficient space for the isolated comparison')
    began=time.perf_counter();decoded=decode_image(case);raw=decoded.array_zyx
    if any(decoded.provenance[k]!=baseline['image'][k] for k in ('source_sha256','decoded_sha256','dimensions_xyzc')):
        raise ValueError('Original image differs from the completed baseline')
    decode_seconds=time.perf_counter()-began
    transform={'minimum':float(raw.min()),'maximum':float(raw.max())}
    transform['constant']=transform['minimum']==transform['maximum']
    if transform!=baseline['normalization']:raise ValueError('Normalization metadata differs')
    one=time.perf_counter()
    normalized,metadata=normalized_image(raw)
    if metadata!=transform:raise ValueError('Normalization calculation differs')
    source=folder/'normalized.npy';np.save(source,normalized,allow_pickle=False);del normalized
    candidate=folder/'candidate_sato_guide.npy'
    _,execution=disk_sato_guide(source,control['settings']['sigmas_voxel'],folder/'guide_work',
        block_side=128,output_path=candidate,progress=lambda s:print(s,flush=True))
    (folder/'guide_work').rmdir()
    guide_seconds=time.perf_counter()-one
    compare_guide(candidate,baseline_folder/baseline['guide']['path'])
    checks=[]
    for i,tile in enumerate(partition_tiles(raw.shape,control['settings']['tile_side_voxel'])):
        arrays,row=point_arrays(raw,read_block(candidate,tile),tile,i,control['settings'],transform)
        original=baseline_folder/baseline['records'][i]['packet']
        with np.load(original,allow_pickle=False) as packet:
            if (set(packet.files)!=set(arrays) or not all(np.array_equal(packet[k],v) and packet[k].dtype==v.dtype for k,v in arrays.items())):
                raise ValueError('Candidate point inputs differ; do not adopt')
        checks.append({'tile_id':i,'all_arrays_and_dtypes_exact':True})
    source.unlink()  # Owned normalization scratch; original cache is untouched.
    memory=psutil.Process().memory_info()
    report={'status':'completed',**identity(config,control),'case_id':case.case_id,
        'baseline_block_side':64,'candidate_block_side':128,'guide_bytes_exact':True,
        'candidate_guide_sha256':digest(candidate),'baseline_guide_sha256':baseline['guide']['sha256'],
        'point_packets':checks,'decode_seconds':decode_seconds,'guide_seconds':guide_seconds,
        'baseline_guide_seconds':baseline['guide_seconds'],'observed_filter_speed_ratio':baseline['guide_seconds']/guide_seconds,
        'total_seconds':time.perf_counter()-began,'execution':execution,
        'peak_working_set_bytes':int(getattr(memory,'peak_wset',memory.rss)),
        'peak_pagefile_bytes':int(memory.peak_pagefile) if hasattr(memory,'peak_pagefile') else None,
        'references_opened':0,'models_executed':0,'gpu_executed':False,'original_cache_modified':False,
        'comparison_limit':'Same source/operator; baseline was measured earlier, not a contemporaneous controlled timing pair'}
    write_once(folder/'worker_report.json',report)
    print(f"BLOCK128 seconds={guide_seconds:.2f} baseline64={baseline['guide_seconds']:.2f} observed_ratio={report['observed_filter_speed_ratio']:.3f} byte_parity=True point_packets_exact={len(checks)}",flush=True)


def controller(config,control,args):
    require_cpu(config,control)
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Output must stay under PN2/tmp')
    folder.mkdir(parents=True,exist_ok=False)
    write_once(folder/'comparison.json',config)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker',
             '--config',str(Path(args.config).resolve()),'--output',str(folder)]
    code,status=cache_driver.launch_worker(command,folder,3600)
    path=folder/'worker_report.json'
    report=json.loads(path.read_text()) if path.exists() else {'status':'missing_worker_report'}
    if status!='completed':report['status']=status
    elif (report.get('status')!='completed' or not report.get('guide_bytes_exact')
            or len(report.get('point_packets',[]))!=25
            or any(report.get(k)!=v for k,v in identity(config,control).items())):
        report['status']='failed_incomplete_comparison'
    report.update(exit_code=code,wall_timeout_seconds=3600)
    write_once(folder/'report.json',report)
    print(f"FINISHED block comparison status={report['status']} report={folder/'report.json'}",flush=True)
    return 0 if report['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','timing','_worker'),required=True)
    parser.add_argument('--config',default=str(CONFIG));parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config,control,case,baseline_folder,baseline=load_control(args.config)
    if args.action=='check':cpu_check(config,control,args.output);return 0
    if args.action=='_worker':worker(config,control,case,baseline_folder,baseline,args);return 0
    return controller(config,control,args)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
