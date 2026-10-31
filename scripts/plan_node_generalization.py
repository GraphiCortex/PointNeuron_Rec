"""Build a deterministic exposure/resource plan, without clouds or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pn2.inputs.cases import load_image_cases
from audit_node_cohort import checked,digest


def exposure_schedule(records,epochs,seed):
    if type(epochs) is not int or epochs<1 or type(seed) is not int or seed<0:
        raise ValueError('Positive epoch count and nonnegative seed required')
    train=sorted((r for r in records if r['partition']=='training'),key=lambda r:r['case_id'])
    if not train or len({r['case_id'] for r in records})!=len(records):
        raise ValueError('Missing training cases or duplicate case IDs')
    if any(r['partition'] not in ('training','development') for r in records):
        raise ValueError('Reserved/unknown partitions cannot enter this plan')
    eligible=[]
    for i,case in enumerate(train):
        for box in sorted(case['boxes'],key=lambda b:b['tile_id']):
            if box['target_overflow'] or box['encoder_short_input']:
                raise ValueError('Unresolved capacity failure; no silent exclusion')
            if box['targets']>0:eligible.append((i,box['tile_id']))
    base=np.asarray(eligible,dtype=np.int32)
    if not len(base) or len(set(map(tuple,base)))!=len(base):
        raise ValueError('Missing/duplicate supervised boxes')
    arrays=[]
    for epoch in range(epochs):
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(base))
        arrays.append(np.column_stack((np.full(len(base),epoch,dtype=np.int32),base[order])))
    result=np.concatenate(arrays)
    counts=Counter(map(tuple,result[:,1:]))
    if set(counts.values())!={epochs}:raise ValueError('Exposure count mismatch')
    return train,result


def run(config_path,output):
    protocol=json.loads(Path(config_path).read_text())
    if protocol['status']!='prospective_protocol_not_a_training_launcher':
        raise ValueError('This utility only prepares a prospective plan')
    source=protocol['cohort_preflight']
    audit=checked(ROOT/source['path'],source['sha256'])
    if audit['status']!='completed' or len(audit['records'])!=40 or audit['reserved_files_opened']!=0:
        raise ValueError('Incomplete or incompatible preflight')
    images=load_image_cases(ROOT/protocol['image_manifest']['path'])
    if digest(ROOT/protocol['image_manifest']['path'])!=protocol['image_manifest']['sha256']:
        raise ValueError('Image-only manifest changed')
    if {c.case_id for c in images}!={r['case_id'] for r in audit['records']}:
        raise ValueError('Image and target-planning cohorts differ')
    references=checked(ROOT/protocol['reference_manifest']['path'],protocol['reference_manifest']['sha256'])
    observed={r['case_id']:r for r in audit['records']}
    if (len(references['cases'])!=len(images) or {r['case_id'] for r in references['cases']}!=set(observed)
            or any(r['partition']!=observed[r['case_id']]['partition']
                   or r['gradient_training_allowed']!=(r['partition']=='training')
                   or r['reference_sha256']!=observed[r['case_id']]['reference_sha256'] for r in references['cases'])):
        raise ValueError('Reference identity/hash or gradient eligibility differs from preflight')
    train,schedule=exposure_schedule(audit['records'],protocol['training']['epochs'],protocol['training']['seed'])
    if len(schedule)!=protocol['training']['total_updates']:
        raise ValueError('Declared update cap differs from exposure plan')
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    packet=folder/'training_schedule.npz'
    np.savez_compressed(packet,epoch_case_tile=schedule,case_ids=np.asarray([r['case_id'] for r in train]))
    report={'schema_version':1,'status':'planned_not_training_ready','protocol_sha256':digest(config_path),
        'script_sha256':digest(__file__),'schedule':packet.name,'schedule_sha256':digest(packet),
        'schedule_columns':['zero_based_epoch','sorted_training_case_index','tile_id'],
        'training_neurons':len(train),'development_neurons':sum(r['partition']=='development' for r in audit['records']),
        'supervised_training_boxes':len(schedule)//protocol['training']['epochs'],'total_updates':len(schedule),
        'exposures_per_supervised_box':protocol['training']['epochs'],
        'per_neuron_training_updates':[{'case_id':r['case_id'],'alias':r['alias'],
            'supervised_boxes':r['summary']['reference_present_boxes'],
            'updates':r['summary']['reference_present_boxes']*protocol['training']['epochs']} for r in train],
        'cohort_summary':audit['summary'],'image_payloads_read':0,'source_images_decoded':0,
        'models_executed':0,'optimizer_updates_executed':0,'reserved_files_opened':0}
    voxels=sum(int(np.prod(r['dimensions_xyz'])) for r in audit['records'])
    report['resources']={'total_source_voxels':voxels,'all_float32_guides_bytes':voxels*4,
        'largest_12_float_work_arrays_bytes':max(int(np.prod(r['dimensions_xyz'])) for r in audit['records'])*4*12,
        'point_slots_all_boxes':sum(r['summary']['boxes'] for r in audit['records'])*2048,
        'prior_four_box_mean_update_seconds':protocol['resources']['prior_mean_update_seconds'],
        'update_only_hours_at_prior_speed':len(schedule)*protocol['resources']['prior_mean_update_seconds']/3600,
        'limitation':'Arithmetic only, not broader-run timing or a process memory requirement'}
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','training_neurons','development_neurons','supervised_training_boxes','total_updates','exposures_per_supervised_box','resources']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/node_generalization_protocol.json'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.config,args.output)
