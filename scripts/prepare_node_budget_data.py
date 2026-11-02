"""CPU-only paired cloud views and shared targets. No filter/model/GPU/training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import node_cloud_cache as cache_driver
from pn2.inputs.node_cache import case_key, digest, fingerprint, write_once
from pn2.inputs.budget_views import build_view_case
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.supervision.node_targets import build_target_case
from plan_node_generalization import exposure_schedule

ROOT=cache_driver.ROOT
CONFIG=ROOT/'configs/node_budget_data.json'
SOURCES=('src/pn2/inputs/budget_views.py','src/pn2/inputs/node_dataset.py',
         'src/pn2/supervision/node_targets.py','src/pn2/supervision/curve_nodes.py',
         'src/pn2/evaluation/swc.py','scripts/plan_node_generalization.py')


def checked(spec):
    path=ROOT/spec['path']
    if digest(path)!=spec['sha256']: raise ValueError(f'Pinned input changed: {path}')
    return json.loads(path.read_text())


def load_control(path):
    control=json.loads(Path(path).read_text())
    if (control['schema_version']!=1 or control['budgets']!=[2048,8192]
            or control['epochs']!=20 or control['seed']!=0 or control['target_step_voxel']!=8.
            or control['query_capacity']!=512 or control['driver_sha256']!=digest(__file__)
            or set(control['source_sha256'])!=set(SOURCES)):
        raise ValueError('Expected the approved paired-data settings')
    for name,sha in control['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError(f'Data source changed: {name}')
    if digest(ROOT/control['cache_config']['path'])!=control['cache_config']['sha256']:
        raise ValueError('Image-cache policy changed')
    cache_control,cases=cache_driver.load_config(ROOT/control['cache_config']['path'])
    proof=checked(control['completion_review'])
    if proof['status']!='passed' or proof['counts']['packets_verified']!=2379:
        raise ValueError('The complete image-cache review is required')
    preflight=checked(control['cohort_preflight']); refs=checked(control['references'])
    observed={c['case_id']:c for c in preflight['records']}
    reference={c['case_id']:c for c in refs['cases']}
    if (preflight['status']!='completed' or preflight['reserved_files_opened']!=0
            or len(reference)!=40 or set(reference)!=set(observed)
            or set(reference)!={c.case_id for c in cases}):
        raise ValueError('Image/reference/preflight cohorts differ')
    for c in cases:
        r=reference[c.case_id]; a=observed[c.case_id]
        if (r['partition']!=c.partition or a['partition']!=c.partition
                or r['gradient_training_allowed']!=(c.partition=='training')
                or r['reference_sha256']!=a['reference_sha256']):
            raise ValueError('Reference identity or gradient role differs from preflight')
    return control,cache_control,cases,observed,reference,preflight


def identity(control):
    return {'control_fingerprint':fingerprint(control),'driver_sha256':digest(__file__),
            'source_sha256':control['source_sha256']}


def tmp_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'tmp').resolve()): raise ValueError('Data/log outputs must stay under PN2/tmp')
    return path


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def worker(control,cache_control,cases,observed,references,preflight,args):
    root=tmp_path(args.data_root)
    if root.exists(): raise FileExistsError('Preserve existing data/partial evidence; no automatic retry')
    root.mkdir(); (root/'clouds8192').mkdir(); (root/'targets').mkdir()
    baseline=ROOT/control['cache_root']; manifest_cases=[]; counts=Counter();began=time.perf_counter()
    for i,case in enumerate(cases):
        key=case_key(case); source=baseline/key
        print(f'CASE {i+1}/40 {case.partition} {case.case_id}: retained-guide clouds then separate reference targets',flush=True)
        view=build_view_case(case,source,cache_driver.cache_identity(cache_control),root/'clouds8192'/key,
                             budget=8192,progress=lambda s:print(s,flush=True))
        base=json.loads((source/'case.json').read_text()); ref=references[case.case_id]
        reference_path=(ROOT/control['references']['path']).parent/ref['reference_swc']
        target=build_target_case(ref,reference_path,base['records'],observed[case.case_id]['boxes'],
            root/'targets'/key,step=control['target_step_voxel'],capacity=control['query_capacity'])
        boxes=[]
        for a,b,t in zip(base['records'],view['records'],target['records']):
            if not a['tile_id']==b['tile_id']==t['tile_id']: raise ValueError('Cloud/target box order differs')
            clouds={}
            for budget,folder,row in ((2048,source,a),(8192,root/'clouds8192'/key,b)):
                if row['slots']<256: raise ValueError('Encoder-short image domain; no padding or exclusion')
                clouds[str(budget)]={'path':relative(folder/row['packet']),
                    'sha256':row['packet_sha256'],'count':row['slots']}
                counts[f'points_{budget}']+=row['slots']
            boxes.append({'tile_id':a['tile_id'],'box':a['box'],'center_xyz':a['center_xyz'],
                'source_voxels':b['source_voxels'],'source_constant':a['source_tile_constant'],
                'uniform_fallback':a['uniform_fallback'],'clouds':clouds,
                'target':{'path':relative(root/'targets'/key/t['packet']),'sha256':t['packet_sha256'],
                    'count':t['count'],'supervision':t['supervision'],
                    'gradient_training_allowed':t['gradient_training_allowed']}})
            counts[case.partition+'_boxes']+=1
            counts[case.partition+('_reference_present' if t['count'] else '_unknown')]+=1
        manifest_cases.append({'case_id':case.case_id,'partition':case.partition,
            'alias':ref['historical_alias'],'source_group':ref['source_group'],
            'dimensions_xyz':base['image']['dimensions_xyzc'][:3],'normalization':base['normalization'],
            'source_sha256':view['source_sha256'],'decoded_sha256':view['decoded_sha256'],
            'baseline_case_sha256':view['baseline_case_sha256'],
            'view_case_sha256':digest(root/'clouds8192'/key/'case.json'),
            'target_case_sha256':digest(root/'targets'/key/'case.json'),'boxes':boxes})
        print(f'COMPLETE CASE {i+1}/40 boxes={len(boxes)} elapsed_seconds={time.perf_counter()-began:.1f}',flush=True)
    train,schedule=exposure_schedule(preflight['records'],20,0)
    prior=checked(control['exposure_plan'])
    prior_path=(ROOT/control['exposure_plan']['path']).parent/prior['schedule']
    if digest(prior_path)!=prior['schedule_sha256']: raise ValueError('Original exposure plan packet changed')
    with np.load(prior_path,allow_pickle=False) as saved:
        if (not np.array_equal(saved['epoch_case_tile'],schedule)
                or saved['case_ids'].tolist()!=[r['case_id'] for r in train]):
            raise ValueError('Shared exposure order differs from the approved plan')
    schedule_path=root/'schedule.npz'
    np.savez_compressed(schedule_path,epoch_case_tile=schedule,case_ids=np.asarray([r['case_id'] for r in train]))
    manifest={'schema':'pn2-paired-node-data-v1',**identity(control),'budgets':[2048,8192],
        'epochs':20,'seed':0,'target_step_voxel':8.,'query_capacity':512,'cases':manifest_cases,
        'schedule':{'path':relative(schedule_path),'sha256':digest(schedule_path)},
        'counts':dict(counts),'updates_per_arm':len(schedule),'training_runner_ready':False}
    write_once(root/'manifest.json',manifest)
    write_once(root/'complete.json',{'manifest_sha256':digest(root/'manifest.json')})
    dataset=NodeBudgetDataset(ROOT,root); dataset.schedule()
    report={'status':'completed',**identity(control),'data_root':str(root),'manifest_sha256':digest(root/'manifest.json'),
        'cases':len(cases),'counts':dict(counts),'updates_per_arm':len(schedule),
        'reference_target_packets':sum(len(c['boxes']) for c in manifest_cases),
        'filter_reruns':0,'models_executed':0,'gpu_executed':False,'reserved_files_opened':0,
        'shared_schedule_exactly_matches_prior_plan':True,'training_runner_ready':False,
        'seconds':time.perf_counter()-began}
    write_once(Path(args.output)/'worker_report.json',report)
    print(f'COMPLETED paired data: {len(cases)} images; counts={dict(counts)}',flush=True)


def verify(control,args):
    dataset=NodeBudgetDataset(ROOT,tmp_path(args.data_root));dataset.schedule(); counts=Counter()
    if any(dataset.manifest.get(k)!=v for k,v in identity(control).items()):
        raise ValueError('Dataset was prepared by another data policy')
    for i,(cid,case) in enumerate(dataset.cases.items()):
        for b in case['boxes']:
            low=dataset.image(cid,b['tile_id'],2048); high=dataset.image(cid,b['tile_id'],8192)
            if not np.isin(low['packet']['flat_indices'],high['packet']['flat_indices']).all():
                raise ValueError('Larger cloud lost baseline source support')
            ref=dataset.reference(cid,b['tile_id'])
            if (ref['supervision']=='reference_present')!=bool(len(ref['target_xyz'])):
                raise ValueError('Unknown/reference-present status differs')
            counts['boxes']+=1;counts[case['partition']+'_boxes']+=1
            counts['points_2048']+=len(low['packet']['flat_indices']);counts['points_8192']+=len(high['packet']['flat_indices'])
        print(f'VERIFY {i+1}/{len(dataset.cases)} {cid}: both clouds and shared targets pass',flush=True)
    folder=tmp_path(args.output);folder.mkdir(exist_ok=False)
    write_once(folder/'report.json',{'status':'passed',**identity(control),'manifest_sha256':digest(dataset.folder/'manifest.json'),
        'counts':dict(counts),'schedule_rows':len(dataset.schedule()[1]),'models_executed':0,'gpu_executed':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('build','verify','_worker'),required=True)
    parser.add_argument('--config',default=str(CONFIG));parser.add_argument('--output',required=True)
    parser.add_argument('--data-root',default=str(ROOT/'tmp/node_budget_data_20261102'))
    args=parser.parse_args();control,cc,cases,observed,refs,preflight=load_control(args.config)
    if args.action=='verify':verify(control,args);return 0
    if args.action=='_worker':worker(control,cc,cases,observed,refs,preflight,args);return 0
    root=tmp_path(args.data_root)
    if root.exists():raise FileExistsError('Data root exists; preserve it')
    folder=tmp_path(args.output);folder.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action','_worker','--config',str(Path(args.config).resolve()),
        '--output',str(folder),'--data-root',str(root)]
    code,state=cache_driver.launch_worker(command,folder,control['worker_timeout_seconds'])
    path=folder/'worker_report.json';report=json.loads(path.read_text()) if path.exists() else {}
    if (state!='completed' or report.get('status')!='completed' or report.get('cases')!=40
            or report.get('reference_target_packets')!=2379 or report.get('updates_per_arm')!=9200
            or any(report.get(k)!=v for k,v in identity(control).items())):
        report={'status':'failed','worker_status':state,'exit_code':code,**identity(control),'automatic_retry':False}
    write_once(folder/'report.json',report)
    print(f'FINISHED paired-data CPU status={report["status"]} report={folder/"report.json"}',flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
