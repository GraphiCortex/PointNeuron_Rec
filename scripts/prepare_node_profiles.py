"""Bounded CPU evaluation covariates; no cloud selection, model or optimizer."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import node_cloud_cache as driver
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.inputs.node_cache import digest,write_once
from pn2.inputs.image import decode_image
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.node_profiles import build_profiles,tangent_lookup,weighted_cutoff

ROOT=driver.ROOT


def run(output):
    folder=Path(output).resolve()
    if not folder.is_relative_to((ROOT/'tmp').resolve()):raise ValueError('Profiles must stay under PN2/tmp')
    folder.mkdir(exist_ok=False);began=time.perf_counter();deadline=began+3600
    config,cases=driver.load_config(ROOT/'configs/node_cloud_cache.json')
    data=NodeBudgetDataset(ROOT,ROOT/'tmp/node_budget_data_20261102')
    ref_path=ROOT/'configs/node_generalization_references.json'
    refs={c['case_id']:c for c in json.loads(ref_path.read_text())['cases']}
    rows=[];training={k:[] for k in ('width','contrast','width_valid','contrast_valid','weights')}
    for index,case in enumerate(cases):
        if time.perf_counter()>deadline:raise TimeoutError('CPU profile preparation reached its one-hour cap')
        cid=case.case_id;image=decode_image(case);meta=data.cases[cid]
        if image.provenance['source_sha256']!=meta['source_sha256']:raise ValueError('Profile image differs from prepared inputs')
        reference=(ref_path.parent/refs[cid]['reference_swc']).resolve()
        if digest(reference)!=refs[cid]['reference_sha256']:raise ValueError('Profile reference changed')
        tangent=tangent_lookup(parse_swc(reference,type_policy='opaque_integer'))
        for b in meta['boxes']:
            if time.perf_counter()>deadline:raise TimeoutError('CPU profile preparation reached its one-hour cap')
            packet=build_profiles(image.array_zyx,data.reference(cid,b['tile_id']),b['box'],meta['normalization'],tangent)
            name=f'case{index:02d}_tile{b["tile_id"]:05d}.npz';np.savez_compressed(folder/name,**packet)
            rows.append({'case_id':cid,'partition':case.partition,'tile_id':b['tile_id'],
                'packet':name,'sha256':digest(folder/name),'windows':len(packet['window_weights']),
                'nodes':len(packet['node_width']),'curve_samples':len(packet['curve_weights'])})
            if case.partition=='training':
                for key in training:training[key].append(packet['window_'+key] if key!='weights' else packet['window_weights'])
        print(f'PROFILES case={index+1}/40 windows={sum(r["windows"] for r in rows)} elapsed={time.perf_counter()-began:.1f}s',flush=True)
    values={k:np.concatenate(v) for k,v in training.items()}
    cutoffs={'width_q25':weighted_cutoff(values['width'],values['weights'],values['width_valid']),
             'contrast_q25':weighted_cutoff(values['contrast'],values['weights'],values['contrast_valid'])}
    report={'status':'completed','schema':'pn2-evaluation-profiles-v1','data_manifest_sha256':digest(data.folder/'manifest.json'),
        'reference_manifest_sha256':digest(ref_path),'cutoffs':cutoffs,'cutoff_partition':'training_only',
        'records':rows,'windows':sum(r['windows'] for r in rows),'nodes':sum(r['nodes'] for r in rows),
        'parameters':{'arc_step_voxel':4.,'curve_quadrature_step_voxel':.5,'profile_radii_voxel':[1.,2.,4.,8.],
            'half_length_voxel':2.,'background_shell_voxel':[8.,12.],'bin_width_voxel':.5,'weighted_quantile':.25,
            'node_tangent':'reference chain; unique incident chain for structural endpoints; ambiguous/zero stays unresolved',
            'contrast_normalization':'whole-source intensity range','radius_supervision':'none'},
        'source_sha256':{p:digest(ROOT/p) for p in ('scripts/prepare_node_profiles.py','src/pn2/evaluation/node_profiles.py',
            'src/pn2/evaluation/branch_evidence.py','src/pn2/evaluation/cloud_support.py','src/pn2/supervision/curve_nodes.py')},
        'seconds':time.perf_counter()-began,'models_executed':0,'optimizer_updates':0,'gpu_executed':False,
        'inference_input':False,'biological_caliber_verified':False}
    write_once(folder/'manifest.json',report)
    write_once(folder/'complete.json',{'manifest_sha256':digest(folder/'manifest.json')})
    print(f'FINISHED profile cache windows={report["windows"]} nodes={report["nodes"]} cutoffs={cutoffs}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args()
    try:run(a.output)
    except Exception:
        traceback.print_exc();raise SystemExit(1)
