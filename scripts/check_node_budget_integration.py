"""CPU checks of real paired data, fixed model, supervision and exact next-step restore."""
import io
import json
from pathlib import Path
import sys

import numpy as np
import geometric_query_experiment as geometric
import node_budget_timing as timing
import sampler_node_common as common
import groupnorm_node_experiment as gn
from pn2.inputs.node_dataset import NodeBudgetDataset

ROOT=common.ROOT


def run(folder):
    import torch
    from pn2.models.node_objective import node_set_loss
    from pn2.training.node_runtime import image_batch,training_batch
    from pn2.training import node_fit_state as state
    from pn2.evaluation.node_cohort import score_box
    control,recipe=timing.load_control(timing.CONFIG);timing.require_cpu(control,recipe)
    common.runtime(recipe,'cpu')
    data_root=ROOT/'tmp/node_budget_data_20261102'
    data=NodeBudgetDataset(ROOT,data_root);case_ids,schedule=data.schedule()
    stress_cid=next(cid for cid,c in data.cases.items() if c['alias']==86 and c['partition']=='training')
    assert data.boxes[(stress_cid,20)]['target']['count']==419
    short_cid=next(cid for cid,c in data.cases.items() if c['alias']==94 and c['partition']=='development')
    short_boxes=[(cid,tile) for (cid,tile),b in data.boxes.items() if b['source_voxels']<8192]
    assert len(short_boxes)==1
    short_cid,short_tile=short_boxes[0]
    model=geometric.make_model(recipe,'cpu');initial=common.state_digest(model.state_dict());checks=[]
    for budget in (2048,8192):
        old=timing.load_records(control,recipe,budget)
        for row in old:
            cid=row['metadata'].get('case_id')
            if cid is None:
                cid=next(c for c,case in data.cases.items() if case['alias']==(32 if row['metadata']['alias']=='fruitfly32' else 22) and case['partition']=='training')
            tile=row['metadata']['tile_id'];current=data.training(cid,tile,budget)
            assert all(np.array_equal(v,current['packet'][k]) and v.dtype==current['packet'][k].dtype for k,v in row['packet'].items())
            assert np.array_equal(row['target_xyz'],current['target_xyz'])
        record=data.training(stress_cid,20,budget);batch=training_batch(record,'cpu')
        with torch.no_grad():
            model.train();a=model(batch);model.eval();b=model(batch)
        consistency=gn.prediction_consistency(a,b)
        assert consistency['passed']
        model.train();model.zero_grad(set_to_none=True);loss=node_set_loss(model(batch),batch['target'],**recipe['loss']);loss.total.backward()
        assert loss.supervised and len(loss.prediction_indices)==419
        assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
        untrained=score_box(stress_cid,20,b.center_proposals[0].numpy(),b.objectness_logits[0].softmax(-1)[:,1].numpy(),record)
        assert untrained['target_count']==419
        checks.append({'budget':budget,'actual_stress_targets':419,'real_stress_gradients_finite':True,
            'mode_consistency':consistency,'original_four_clouds_and_targets_exact':True,'core_metric_executes':True})
        print(f'CPU INTEGRATION budget={budget}: original four exact; actual 419-target forward/backward/mode/metric checks pass',flush=True)
    short=data.image(short_cid,short_tile,8192)
    assert len(short['packet']['flat_indices'])==7650
    with torch.no_grad():
        model.eval();prediction=model(image_batch(short,'cpu'))
    assert prediction.center_proposals.shape==(1,512,3) and bool(prediction.center_proposals.isfinite().all())
    unknown=next((cid,tile) for (cid,tile),b in data.boxes.items() if data.cases[cid]['partition']=='training' and b['target']['count']==0)
    try:data.training(*unknown,2048)
    except ValueError:pass
    else:raise AssertionError('Unknown crop entered training')
    empty=node_set_loss(prediction,torch.empty((0,3),dtype=torch.float32),**recipe['loss'])
    assert not empty.supervised and float(empty.total)==0.
    assert common.state_digest(model.state_dict())==initial
    # Synthetic-only optimizer updates exercise the new data/budget/cursor binding.
    fixture=geometric.synthetic_record(torch,recipe,256,'cpu');optimizer=common.make_optimizer(model,recipe)
    common.update(model,optimizer,fixture,recipe)
    bindings={'maximum_updates':9200,'budget':8192,'data_manifest_sha256':common.digest(data_root/'manifest.json'),
        'schedule_sha256':data.manifest['schedule']['sha256'],'model_identity':geometric.identity(recipe)}
    stream=io.BytesIO();torch.save(state.capture(model,optimizer,bindings,1),stream)
    common.update(model,optimizer,fixture,recipe);expected=gn.prefix_identity(model,optimizer)
    restored=geometric.make_model(recipe,'cpu');new_opt=common.make_optimizer(restored,recipe)
    stream.seek(0);payload=torch.load(stream,map_location='cpu',weights_only=False)
    assert state.restore(payload,restored,new_opt,bindings)==1
    common.update(restored,new_opt,fixture,recipe)
    assert gn.prefix_identity(restored,new_opt)==expected
    report={'status':'passed','data_manifest_sha256':common.digest(data_root/'manifest.json'),
        'data_schedule_sha256':data.manifest['schedule']['sha256'],'schedule_rows':len(schedule),
        'budgets':checks,'actual_short_development_input_points':7650,'short_development_gradient_updates':0,
        'empty_reference_loss_abstains':True,'unknown_loader_training_rejected':True,
        'real_data_optimizer_updates':0,'synthetic_unique_updates':2,'synthetic_replayed_updates':1,
        'next_synthetic_update_model_adam_rng_exact':True,'gpu_executed':False,
        'quality_gate':None,'model_identity':geometric.identity(recipe),
        'source_sha256':{n:common.digest(ROOT/n) for n in ('scripts/check_node_budget_integration.py',
            'src/pn2/training/node_runtime.py','src/pn2/training/node_fit_state.py','src/pn2/evaluation/node_cohort.py')}}
    common.write(folder/'report.json',report)
    print(f'PASSED real paired-data CPU integration: {folder/"report.json"}',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    folder=Path(args.output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Output must stay under PN2/tmp')
    folder.mkdir(exist_ok=False);run(folder)
