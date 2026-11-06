"""CPU-only archive identity repair and FIRST benchmark scoring;no redownload.

Original Stage2/post-Stage2 results and numerical diagnostic rules are read-only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import re
import sys
import tarfile
import numpy as np

import post_stage2_diagnostics as prior
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.evaluation.post_stage2 import graph,scores
from pn2.evaluation.resampled_skeleton import clip_graph
from pn2.evaluation.swc import parse_swc
from pn2.training.guarded_run import run

ROOT=prior.ROOT;CONFIG=ROOT/'configs/bigneuron_benchmark_recovery_20261106.json'


def archive_identity(name,config):
    """Six unique published dataset IDs;Cell1 requires exact normalized group/leaf.

    Do not assign256 to the current Cell1 annotation from a shared image basename:
    the published256 SWC has a different annotation timestamp.
    """
    parts=PurePosixPath(name).parts;filename=parts[-1]
    match=re.match(r'^\d+_(\d+)\.v3d(?:pbd|raw)[_.]',filename,re.I)
    found=[cid for cid,ids in config['published_id_map'].items() if match and match.group(1) in ids]
    def normalized_group(text):return re.sub(r'^(?:[ep]_)?checked\d+_','',text.lower())
    special=config['human_case'];group=special.split('/')[0];leaf=special.split('/')[1]
    if normalized_group(group) in {normalized_group(p) for p in parts[:-1]} and leaf in parts[:-1]:found.append(special)
    found=sorted(set(found))
    if len(found)!=1:return None,None
    return found[0],('exact source-group/case leaf' if found[0]==special else 'published dataset ID in algorithm filename')


def algorithm_variant(name):
    """Strip per-file ordinal/dataset ID;retain method/parameter suffixes.

    Filename seed coordinates are per-case inputs,not cross-case method names.
    Distinguish seed-encoded mode and preserve XYZ as provenance. Replicates
    within one case/method are all retained,not selected by score.
    """
    filename=PurePosixPath(name).name
    match=re.match(r'^\d+_(\d+)\.v3d(?:pbd|raw)[_.](.+)\.swc$',filename,re.I)
    if not match:raise ValueError('Unrecognized benchmark naming convention: '+filename)
    suffix=match.group(2);seed=re.match(r'^x(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)_z(-?\d+(?:\.\d+)?)_(.+)$',suffix,re.I)
    xyz=None
    if seed:xyz=[float(seed.group(i)) for i in (1,2,3)];suffix=seed.group(4)
    return suffix.lower()+('__seed_xyz_in_filename' if seed else ''),xyz,match.group(1)


def control():
    config=prior.e.read(CONFIG)
    if config['schema']!='pn2-bigneuron-identity-recovery-v1' or config['wall_seconds']!=7200:raise ValueError('Changed benchmark recovery scope/budget')
    for p,sha in {**config['source_sha256'],**config['artifact_sha256']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Changed benchmark recovery binding '+p)
    c,scientific,data=prior.control()
    if set(config['published_id_map'])|{config['human_case']}!=set(c['development_cases']):raise ValueError('Changed seven-case benchmark scope')
    return config,c,scientific,data


def inventory(config,c,folder):
    archive=ROOT/config['archive']['path']
    if archive.stat().st_size!=config['archive']['size'] or digest(archive)!=config['archive']['sha256']:raise ValueError('Downloaded archive changed')
    source=prior.e.read(ROOT/config['old_extraction_manifest']);known={p['archive_member']:p['case_id'] for p in source['packets']}
    packets=[];catalog=[];corroborated=0;human_headers=[]
    with tarfile.open(archive,'r:gz') as tf:
        for i,member in enumerate(tf,1):
            if i%500==0:print(f'ARCHIVE REPAIR {i} entries;{len(packets)} approved outputs',flush=True)
            if not member.isfile() or not member.name.lower().endswith('.swc'):continue
            if 'human_allen_confocal' in member.name.lower():human_headers.append(member.name)
            cid,method=archive_identity(member.name,config)
            if cid is None:continue
            if member.name in known:
                if known[member.name]!=cid:raise ValueError('Numeric ID conflicts with prior explicit named-case mapping')
                corroborated+=1
            variant,seed,dataset_id=algorithm_variant(member.name)
            # Only these seven identities' SWC contents are opened.
            if member.size>100000000:raise ValueError('Unexpected benchmark file size')
            import hashlib
            path=folder/(hashlib.sha256(member.name.encode()).hexdigest()+'.swc')
            with tf.extractfile(member) as inp,path.open('xb') as out:out.write(inp.read())
            p={'case_id':cid,'algorithm_variant':variant,'seed_xyz_from_filename':seed,'dataset_id':dataset_id,
               'identity_evidence':method,'archive_member':member.name,'path':str(path),'sha256':digest(path)}
            packets.append(p);catalog.append({k:v for k,v in p.items() if k not in ['path','sha256']})
    counts={cid:sum(p['case_id']==cid for p in packets) for cid in c['development_cases']}
    if corroborated!=len(known):raise ValueError('Recovery did not corroborate every previously named-case output;inspect naming before scoring')
    report={'packets':packets,'case_counts':counts,'prior_named_files_corroborated':corroborated,
            'archive_sha256':config['archive']['sha256'],'catalog':catalog,
            'human_family_headers_only_for_unresolved_mapping':human_headers,
            'seed_origin':'Filename-encoded seed coordinates are disclosed;whether supplied by reference soma or image detection is not established here.',
            'no_ambiguous_Cell1_basename_substitution':True,'no_reserved_SWC_contents_parsed':True}
    write_once(folder/'manifest.json',report);print('REPAIRED CASE COUNTS '+json.dumps(counts),flush=True)
    return report


def mean_replicates(rows):
    """Mean primitive counts/lengths;all replicas contribute,not best-of selection."""
    if len(rows)==1:return rows[0]
    n=len(rows);base=rows[0];rl=base['segments']['reference_length'];pl=float(np.mean([r['segments']['predicted_length'] for r in rows]))
    seg={'reference_length':rl,'predicted_length':pl,'tolerances':{},'reference_has_curve':rl>0,'predicted_has_curve':pl>0}
    rs={'spacing':2.,'units':'native_voxel','tolerances':{}}
    for t in ['2.0','6.0']:
        pc=float(np.mean([r['segments']['tolerances'][t]['supported_prediction_length'] for r in rows]));rc=float(np.mean([r['segments']['tolerances'][t]['supported_reference_length'] for r in rows]))
        p,q=pc/pl if pl else 0.,rc/rl if rl else 0.
        seg['tolerances'][t]={'precision':p,'recall':q,'f1':2*p*q/(p+q) if p+q else 0.,'supported_reference_length':rc,'supported_prediction_length':pc}
        values=[r['resampled']['tolerances'][t] for r in rows];m=float(np.mean([v['matched'] for v in values]));p=float(np.mean([v['predicted'] for v in values]));q=float(np.mean([v['reference'] for v in values]))
        rs['tolerances'][t]={'matched':m,'predicted':p,'reference':q,'precision':m/p if p else 0.,'recall':m/q if q else 0.,'f1':2*m/(p+q) if p+q else 0.,'extras':p-m,'misses':q-m}
    rs['reference_samples']=rs['tolerances']['2.0']['reference'];rs['predicted_samples']=rs['tolerances']['2.0']['predicted']
    return {'case_id':base['case_id'],'tile_id':base['tile_id'],'segments':seg,'resampled':rs,'length_ratio':pl/rl if rl else None,
            'replicate_aggregation':'mean primitive lengths/matching counts;effective counts may be fractional','replicates':n}


def score_all(config,c,scientific,data,manifest,folder):
    if any(v==0 for v in manifest['case_counts'].values()):
        report={'status':'seven_case_identity_still_incomplete','case_counts':manifest['case_counts'],
                'reason':'No case excluded,no alternate gold annotation substituted and no seven-neuron numbers fabricated. Inspect repaired manifest before scoring.'}
        write_once(folder/'report.json',report);print(json.dumps(report),flush=True);return
    refs=prior.e.References(scientific,data);groups={};rows={};whole={};replicate_rows=[];checks=[];statuses=[]
    for p in manifest['packets']:groups.setdefault((p['algorithm_variant'],p['case_id']),[]).append(p)
    methods=sorted({p['algorithm_variant'] for p in manifest['packets']})
    for ai,method in enumerate(methods):
        for cid in c['development_cases']:
            ps=groups.get((method,cid),[]);inputs=ps or [None];per_box={};full=[]
            for ri,p in enumerate(inputs):
                pred=graph([],[]);status='missing'
                if p:
                    try:
                        if digest(Path(p['path']))!=p['sha256']:raise ValueError('Changed extracted benchmark output')
                        pred=parse_swc(p['path'],type_policy='opaque_integer');shape=np.asarray(data.cases[cid]['dimensions_xyz'])
                        fraction=float(np.mean(np.all((pred.xyz>=-.5)&(pred.xyz<=shape-.5),axis=1))) if len(pred.xyz) else 1.
                        checks.append({**p,'fraction_in_native_bounds':fraction,'fixed_transform':'identity XYZ;no shift,axis swap,scale or gold-fitted alignment',
                                       'origin_verified':False,'limitation':'Native bounds do not prove subvoxel origin;comparison remains provisional until overlay confirmation.'})
                        if fraction<.99:raise ValueError('Failed unchanged99% native image-bounds check')
                        status='completed' if len(pred.xyz) else 'empty'
                    except (ValueError,OSError) as error:pred=graph([],[]);status='invalid_or_coordinate_failed:'+str(error)
                statuses.append({'algorithm_variant':method,'case_id':cid,'replicate':ri,'status':status})
                for key,b in data.boxes.items():
                    if key[0]!=cid or not b['target']['count']:continue
                    low=np.asarray(b['box']['start_xyz'])-.5;high=np.asarray(b['box']['stop_xyz'])-.5
                    row={'case_id':cid,'tile_id':key[1],**scores(prior.reference_graph(refs,data,key),clip_graph(pred.xyz,pred.edges,low,high))}
                    per_box.setdefault(key[1],[]).append(row);replicate_rows.append({'algorithm_variant':method,'replicate':ri,'status':status,**row})
                m=refs.get(cid)['morphology'];shape=np.asarray(data.cases[cid]['dimensions_xyz'])
                full.append({'case_id':cid,'tile_id':'whole_image',**scores(clip_graph(m.xyz,m.edges,[-.5]*3,shape-.5),clip_graph(pred.xyz,pred.edges,[-.5]*3,shape-.5))})
            rows.setdefault(method,[]).extend(mean_replicates(v) for v in per_box.values());whole.setdefault(method,[]).append(mean_replicates(full))
        print(f'BENCHMARK FIRST SCORING {ai+1}/{len(methods)} implementation variants',flush=True)
    summaries={k:prior.summarize(v) for k,v in rows.items()};whole_summaries={k:prior.summarize(v) for k,v in whole.items()};best={};per_neuron={}
    for t in ['2.0','6.0']:
        best[t]={}
        for label,field,macro in [('segment','segments','macro'),('resampled','new_metrics','macro_resampled')]:
            values={name:s[field][macro][t]['f1'] for name,s in summaries.items()};winner=min(values,key=lambda k:(-values[k],k))
            best[t][label]={'best_fixed_variant':winner,'best':values[winner],'median_variant_macro':float(np.median(list(values.values())))}
    for cid in c['development_cases']:
        per_neuron[cid]={}
        for t in ['2.0','6.0']:
            per_neuron[cid][t]={}
            for label,field,metric in [('segment','segments','tolerances'),('resampled','new_metrics','resampled')]:
                values={name:next(n for n in s[field]['neurons'] if n['case_id']==cid)[metric][t]['f1'] for name,s in summaries.items()};winner=min(values,key=lambda k:(-values[k],k))
                per_neuron[cid][t][label]={'best_variant':winner,'best':values[winner],'median':float(np.median(list(values.values())))}
    report={'status':'completed_provisional_native_comparison','summaries':summaries,'rows':rows,'whole_image_summaries':whole_summaries,'whole_image_rows':whole,
            'replicate_rows':replicate_rows,'coordinate_checks':checks,'case_statuses':statuses,'best_median':best,'per_neuron_best_median':per_neuron,
            'replicate_policy':'all replicas contribute equally via mean primitive counts;no best-replicate selection','declaration':'implementation suffixes normalized before FIRST scoring;dataset IDs and ordinal/seedXYZ are not method identity',
            'scope':'same151 development boxes primary,separate whole-image view;all seven included;missing/invalid outputs zero',
            'external_reference_independence':'parameter-optimized release;filename seed coordinates disclosed,their origin not established;not a certified reference-free independent baseline',
            'Stage2_stopped':True,'new_development_training_labels':False,'models_selected':False}
    write_once(folder/'report.json',report);prior.table(folder/'report.md','Recovered BigNeuron implementation-variant benchmark',summaries,
              ['Dataset/ordinal prefixes removed from method labels before scoring;all seven cases retained,failures zero.',
               'Native origin remains provisional;no fitted registration or gold-dependent coordinate choice. Seed-encoded modes are disclosed.'])
    with (folder/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## Best and median implementation variants\n\n| Metric | Tolerance | Best variant | Best F1 (%) | Median F1 (%) |\n|---|---|---|---|---|\n')
        for t,metrics in best.items():
            for label,v in metrics.items():f.write(f"| {label} | {t} | {v['best_fixed_variant']} | {100*v['best']:.2f} | {100*v['median_variant_macro']:.2f} |\n")


def worker(output):
    config,c,scientific,data=control();output.mkdir(exist_ok=False,parents=True);stop=ROOT/c['stage2_root']/'STOP.json';before=digest(stop)
    inv=output/'inventory';inv.mkdir();manifest=inventory(config,c,inv)
    target=output/'scoring';target.mkdir();score_all(config,c,scientific,data,manifest,target)
    if digest(stop)!=before:raise ValueError('Stage2 STOP changed')
    write_once(output/'finished.json',{'status':'completed','config':fingerprint(config),'original_STOP_sha256':before,'first_benchmark_scoring_only':True,'GPU':False,'training':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    config,_,_,_=control();output=prior.e.output_path(a.output or ROOT/'tmp/bigneuron_benchmark_recovery_20261106')
    if a.check:print('Benchmark repair bindings verified;no archive scan,extraction or scoring');return
    if a.worker:worker(output);return
    if output.exists():raise FileExistsError('Preserve benchmark repair output;no automatic retry')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');seconds=config['wall_seconds']
    print(f'CPU ONLY;reuse downloaded archive;FIRST benchmark scoring;cap{seconds}s;OUTPUT {output};Stage2 remains stopped',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(output)],output.with_name(output.name+'_guard_'+stamp),seconds,
        {'diagnostic':'benchmark identity repair/first scoring','config':fingerprint(config),'Stage2_stopped':True})


if __name__=='__main__':main()
