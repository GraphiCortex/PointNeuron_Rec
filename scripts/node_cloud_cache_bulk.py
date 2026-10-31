"""Case-wise CPU preparation with qualified exact-output processing blocks.

Completed cases are verified/skipped; partial cases and failures stop. No SWC,
target, model, GPU, automatic retry or experimental setting change is supported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import node_cloud_cache as driver
import node_guide_block_compare as comparison
from pn2.inputs.node_cache import build_case,verify_case,case_key,digest,fingerprint,write_once
from pn2.inputs.vaa3d import read_header

ROOT=driver.ROOT
CONFIG=ROOT/'configs/node_cloud_cache_bulk.json'


def case_timeout(voxels,reuse,profile,rate):
    if voxels<=0 or not math.isfinite(rate) or rate<=0:raise ValueError('Invalid measured work rate')
    if reuse:return profile['reuse_case_wall_seconds']
    policy=profile['fresh_case_wall']
    return int(min(policy['maximum_seconds'],max(policy['minimum_seconds'],
        rate*voxels*policy['measured_time_multiplier']+policy['extra_seconds'])))


def add_content_identity(seen,receipt):
    image=receipt['image'];role=receipt['partition'];case=receipt['case_id']
    signatures=(('source',image['source_sha256']),
        ('decoded',tuple(image['dimensions_xyzc']),image['datatype'],image['decoded_sha256']))
    for signature in signatures:
        if signature in seen and seen[signature][0]!=role:
            raise ValueError(f'Cross-split duplicate image content: {seen[signature][1]} and {case}')
        seen.setdefault(signature,(role,case))


def load_profile(path):
    profile=json.loads(Path(path).read_text())
    if (profile['schema_version']!=1 or profile['processing_block_side']!=128
            or profile['driver_sha256']!=digest(__file__)):
        raise ValueError('Expected the declared block128 bulk execution profile')
    config,control,case,folder,baseline=comparison.load_control(ROOT/profile['comparison_config'])
    comparison.require_cpu(config,control)
    p=ROOT/profile['qualified_comparison']['path']
    if digest(p)!=profile['qualified_comparison']['sha256']:raise ValueError('Qualified execution comparison changed')
    report=json.loads(p.read_text())
    if (report.get('status')!='completed' or not report.get('guide_bytes_exact')
            or len(report.get('point_packets',[]))!=25
            or not all(r['all_arrays_and_dtypes_exact'] for r in report['point_packets'])
            or any(report.get(k)!=v for k,v in comparison.identity(config,control).items())):
        raise ValueError('Block128 has not qualified exact guide and point-input parity')
    declared,cases=driver.load_config(ROOT/profile['cache_config'])
    if driver.cache_identity(declared)!=driver.cache_identity(control):
        raise ValueError('Bulk and qualified comparison use different input policies')
    driver.require_cpu(control)
    return profile,control,cases,report


def execution_identity(profile,control):
    return {'bulk_profile_fingerprint':fingerprint(profile),'bulk_driver_sha256':digest(__file__),
            'cache_identity':driver.cache_identity(control)}


def make_plan(profile,control,cases,root,measured):
    index=driver.index_cache(control,cases,root)
    complete={r['case_id'] for r in index['records']}
    candidates=driver.guide_candidates(control,cases)
    benchmark=next(c for c in cases if c.case_id==control['benchmark_case_id'])
    rate=measured['guide_seconds']/read_header(benchmark.image_path).voxel_count
    jobs=[]
    for case in cases:
        if case.case_id in complete:continue
        header=read_header(case.image_path);reuse=case.case_id in candidates
        if reuse and digest(candidates[case.case_id]['guide_path'])!=candidates[case.case_id]['guide_sha256']:
            raise ValueError('Registered guide copy is missing or changed; stop before bulk launch')
        jobs.append({'case_id':case.case_id,'partition':case.partition,'voxels':header.voxel_count,
            'reuse_guide':reuse,'estimated_filter_seconds':0. if reuse else rate*header.voxel_count,
            'wall_timeout_seconds':case_timeout(header.voxel_count,reuse,profile,rate)})
    return {'status':'planned_not_executed',**execution_identity(profile,control),
        'cache_root':str(root),'completed_before_launch':index['records'],'jobs':jobs,
        'pending_images':len(jobs),'pending_fresh_filters':sum(not j['reuse_guide'] for j in jobs),
        'pending_guide_copies':sum(j['reuse_guide'] for j in jobs),
        'estimated_remaining_filter_hours':sum(j['estimated_filter_seconds'] for j in jobs)/3600,
        'estimate_limitation':'One thin-image rate scaled by voxel count; not a bound; excludes decode/sampling/copy/verification and load/shape effects',
        'overall_wall_seconds':profile['overall_wall_seconds'],
        'free_disk_bytes_at_plan':shutil.disk_usage(root).free,'training_ready':False}


def worker(profile,control,cases,args):
    root=driver.cache_root(args.cache_root,driver.cache_identity(control))
    case=next((c for c in cases if c.case_id==args.case_id),None)
    if case is None:raise ValueError('Case is not in the explicit image-only cohort')
    target=root/case_key(case)
    if target.exists():raise FileExistsError('Preserve existing/partial case; no automatic retry')
    candidates=driver.guide_candidates(control,cases)
    print(f"BULK CPU case={case.case_id} processing_block=128 model_crop=128 reuse_guide={case.case_id in candidates}",flush=True)
    receipt=build_case(case,control['settings'],target,driver.cache_identity(control),
        reuse=candidates.get(case.case_id),block_side=128,progress=lambda s:print(s,flush=True))
    write_once(Path(args.output)/'worker_report.json',{'status':'completed',**execution_identity(profile,control),
        'case_id':case.case_id,'partition':case.partition,'case_json_sha256':digest(target/'case.json'),
        'tiles':receipt['tile_count'],'total_seconds':receipt['total_seconds'],
        'guide_seconds':receipt['guide_seconds'],'guide_execution':receipt['guide']['execution']})


def prepare(profile,control,cases,args,measured):
    plan_folder=Path(args.plan_run).resolve()
    if not plan_folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Plan must stay under PN2/tmp')
    plan=json.loads((plan_folder/'plan.json').read_text())
    if any(plan.get(k)!=v for k,v in execution_identity(profile,control).items()):
        raise ValueError('Bulk plan is stale or belongs to another policy')
    root=driver.cache_root(args.cache_root,driver.cache_identity(control))
    if Path(plan['cache_root']).resolve()!=root:raise ValueError('Plan and active cache roots differ')
    current=make_plan(profile,control,cases,root,measured)
    existing={r['case_id']:r for r in current['completed_before_launch']}
    if any(existing.get(r['case_id'])!=r for r in plan['completed_before_launch']):
        raise ValueError('A previously completed cache changed or disappeared since planning')
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Run output must stay under PN2/tmp')
    folder.mkdir(parents=True,exist_ok=False)
    write_once(folder/'launch.json',current)
    print(f"BULK remaining={current['pending_images']} fresh={current['pending_fresh_filters']} copies={current['pending_guide_copies']} rough_filter_hours={current['estimated_remaining_filter_hours']:.2f}; CPU only",flush=True)
    deadline=time.perf_counter()+profile['overall_wall_seconds'];done=[];status='completed';content={}
    for row in current['completed_before_launch']:
        receipt=json.loads((root/row['folder']/'case.json').read_text());add_content_identity(content,receipt)
    with (folder/'completed_cases.jsonl').open('x',encoding='utf-8') as journal:
        for i,job in enumerate(current['jobs']):
            remaining=deadline-time.perf_counter()
            if remaining<=0:status='overall_timeout';break
            run=folder/f'case_{i:02d}';run.mkdir()
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker',
                '--config',str(Path(args.config).resolve()),'--case-id',job['case_id'],
                '--cache-root',str(root),'--output',str(run)]
            timeout=min(job['wall_timeout_seconds'],remaining)
            code,worker_status=driver.launch_worker(command,run,timeout)
            path=run/'worker_report.json'
            report=json.loads(path.read_text()) if path.exists() else {}
            if (worker_status!='completed' or report.get('status')!='completed' or report.get('case_id')!=job['case_id']
                    or any(report.get(k)!=v for k,v in execution_identity(profile,control).items())):
                status=worker_status if worker_status!='completed' else 'failed_incomplete_worker'
                write_once(run/'failure.json',{'status':status,'exit_code':code,'case_id':job['case_id']})
                break
            case=next(c for c in cases if c.case_id==job['case_id'])
            try:
                verified=verify_case(root/case_key(case),case,driver.cache_identity(control))
                if digest(root/case_key(case)/'case.json')!=report['case_json_sha256']:
                    raise ValueError('Worker and verified cache receipts differ')
            except (ValueError,OSError) as error:
                status='failed_cache_validation'
                write_once(run/'failure.json',{'status':status,'reason':str(error)});break
            try:add_content_identity(content,verified)
            except ValueError as error:
                status='failed_cross_split_duplicate'
                write_once(run/'failure.json',{'status':status,'reason':str(error)});break
            done.append(report);journal.write(json.dumps(report)+'\n');journal.flush()
            print(f"CACHED {i+1}/{len(current['jobs'])}: {case.case_id}; seconds={verified['total_seconds']:.1f}",flush=True)
    try:index=driver.index_cache(control,cases,root)
    except ValueError as error:
        if status=='completed':status='failed_cache_validation'
        index={'cloud_cache_complete':False,'completed_cases':len(existing)+len(done),
               'expected_cases':len(cases),'validation_error':str(error),'training_ready':False}
    if status=='completed' and not index['cloud_cache_complete']:status='failed_incomplete_cohort'
    result={'status':status,**execution_identity(profile,control),'cache_index':index,
        'workers_completed_this_run':done,'references_opened':0,'models_executed':0,'gpu_executed':False,
        'training_ready':False,'automatic_retry':False}
    write_once(folder/'report.json',result)
    print(f"FINISHED bulk CPU status={status} cache={index['completed_cases']}/{index['expected_cases']} report={folder/'report.json'}",flush=True)
    return 0 if status=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('plan','prepare','_worker'),required=True)
    parser.add_argument('--config',default=str(CONFIG));parser.add_argument('--output',required=True)
    parser.add_argument('--cache-root',default=str(ROOT/'tmp/node_cloud_cache_20261031'))
    parser.add_argument('--plan-run');parser.add_argument('--case-id')
    args=parser.parse_args();profile,control,cases,measured=load_profile(args.config)
    if args.action=='_worker':worker(profile,control,cases,args);return 0
    if args.action=='prepare':
        if not args.plan_run:raise ValueError('Bulk preparation needs an explicit qualified --plan-run')
        return prepare(profile,control,cases,args,measured)
    root=driver.cache_root(args.cache_root,driver.cache_identity(control));folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Plan output must stay under PN2/tmp')
    folder.mkdir(parents=True,exist_ok=False)
    plan=make_plan(profile,control,cases,root,measured);write_once(folder/'plan.json',plan)
    print(json.dumps({k:plan[k] for k in ('pending_images','pending_fresh_filters','pending_guide_copies','estimated_remaining_filter_hours','overall_wall_seconds')},indent=2))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
