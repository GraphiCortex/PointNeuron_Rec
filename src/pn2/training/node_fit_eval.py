"""Fixed-cutoff evaluation on every selected image box, with profile diagnostics."""
import time

import numpy as np
import torch

from .node_runtime import image_batch
from pn2.inputs.node_cache import write_once,digest
from pn2.evaluation.node_cohort import score_box,summarize
from pn2.evaluation.node_profile_metrics import score_profiles,summarize_profiles


def evaluate(model,data,profiles,budget,device,case_ids,folder,step):
    folder.mkdir(exist_ok=False);model.eval();rows=[];arrays={};selected=sorted(case_ids);n=0;began=time.perf_counter()
    with torch.inference_mode():
        for ci,cid in enumerate(selected):
            for box in data.cases[cid]['boxes']:
                record=data.image(cid,box['tile_id'],budget);output=model(image_batch(record,device))
                centers=output.center_proposals[0].cpu().numpy()
                scores=output.objectness_logits[0].softmax(-1)[:,1].cpu().numpy()
                if centers.shape!=(512,3):raise ValueError('Expected all 512 proposal slots')
                reference=data.reference(cid,box['tile_id'])
                row=score_box(cid,box['tile_id'],centers,scores,reference)
                row['profiles']=score_profiles(centers,scores,reference,profiles.load(cid,box['tile_id']),profiles.manifest['cutoffs'])
                rows.append(row);arrays[f'box{n}_centers']=centers;arrays[f'box{n}_scores']=scores
                arrays[f'box{n}_identity']=np.asarray([ci,box['tile_id']],dtype=np.int32);n+=1
                if n%32==0:print(f'EVALUATE step={step} budget={budget} boxes={n} elapsed={time.perf_counter()-began:.1f}s',flush=True)
    summary=summarize(data,rows,selected);profile_summary=summarize_profiles(data,rows)
    write_once(folder/'metrics.json',{'step':step,'budget':budget,'case_order':selected,'summary':summary,
        'profiles':profile_summary,'boxes':rows,'scope':'tiled node-proposal reference agreement; no final graph/SWC verdict'})
    np.savez_compressed(folder/'predictions.npz',**arrays)
    return {'step':step,'budget':budget,'folder':str(folder),'seconds':time.perf_counter()-began,
        'metrics_sha256':digest(folder/'metrics.json'),'predictions_sha256':digest(folder/'predictions.npz'),
        'summary':summary,'profile_summary':profile_summary}
