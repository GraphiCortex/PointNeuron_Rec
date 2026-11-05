"""User-local CPU diagnostics only. Stage2 STOP and all its artifacts are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile
import urllib.request
import numpy as np
from scipy.spatial import cKDTree

import local_curve_experiment as e
import audit_resampled_compact as metrics_module
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.inputs.image import decode_image
from pn2.inputs.cases import ImageCase
from pn2.evaluation.post_stage2 import graph,subset,scores,ablations,intensity_ridges,weighted_offsets,compact_prune,reference_index
from pn2.evaluation.resampled_skeleton import clip_graph
from pn2.evaluation.formulation import graph_segments
from pn2.evaluation.geometry import _sample_arclength
from pn2.evaluation.bounded_agreement import bounded_distances
from pn2.evaluation.selection_oracle import edge_agreement
from pn2.evaluation.candidate_pool import crop_classical
from pn2.evaluation.vote_scaffold import minimum_spanning_forest,export_forest
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.fixed_curve_graph import summarize as segment_summary
from pn2.evaluation.resampled_skeleton import summarize as sample_summary
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/post_stage2_diagnostics_20261106.json'


def checked_npz(spec):
    path=Path(spec['path'])
    if digest(path)!=spec['sha256']:raise ValueError('Changed diagnostic input '+str(path))
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def control():
    c=e.read(CONFIG)
    if c['schema']!='pn2-post-stage2-diagnostics-v1' or c['editing_gate_resampled_F1_2']!=.6 or c['length_cap']!=1.25:raise ValueError('Changed fixed diagnostic scope')
    for name,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed diagnostic binding '+name)
    for package,version in c['runtime_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Changed CPU diagnostic runtime '+package)
    scientific=e.control();data,_,_=e.prepared(scientific)
    if len(data.cases)!=40:raise ValueError('Changed40-neuron approved cohort')
    if set(c['images'])!=set(data.cases) or set(c['development_cases'])!={cid for cid,v in data.cases.items() if v['partition']=='development'}:raise ValueError('Changed approved image/benchmark identities')
    return c,scientific,data


def reference_graph(refs,data,key):
    b=data.boxes[key]['box'];m=refs.get(key[0])['morphology']
    return clip_graph(m.xyz,m.edges,np.asarray(b['start_xyz'])-.5,np.asarray(b['stop_xyz'])-.5)


def summarize(rows):return {'segments':segment_summary(rows),'new_metrics':sample_summary(rows)}


def table(path,title,summaries,notes=()):
    lines=['# '+title,'',*notes,'','| Arm | Segment F1 2/6 (%) | Resampled F1 2/6 (%) | Pooled length |','|---|---|---|---|']
    for arm,s in summaries.items():
        a=s['segments']['macro'];b=s['new_metrics']['macro_resampled']
        lines.append(f"| {arm} | {100*a['2.0']['f1']:.2f}/{100*a['6.0']['f1']:.2f} | {100*b['2.0']['f1']:.2f}/{100*b['6.0']['f1']:.2f} | {s['new_metrics']['pooled_length_ratio']:.4f} |")
    lines+=['','Full per-neuron/box results and provenance are in report.json. No model,threshold,label or inference policy is selected.','']
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))


def graph_rows(c):
    base=ROOT/c['stage2_root'];cv=e.read(base/'comparison/C0_C1_training_CV.json');policy=e.read(base/'policy/report.json')
    return {'C0_training_CV':cv['rows']['C0'],'C1_training_CV':cv['rows']['C1'],
            **{'policy_'+name+'_development':rows for name,rows in policy['rows'].items()}}


def decomposition(c,scientific,data,folder):
    refs=e.References(scientific,data);rows={};counts=[];original={}
    for arm,inputs in graph_rows(c).items():
        original[arm]=summarize(inputs)
        for i,row in enumerate(inputs):
            key=(row['case_id'],row['tile_id']);spec=row['graph']
            if digest(Path(spec['path']))!=spec['sha256']:raise ValueError('Changed saved decode')
            m=parse_swc(spec['path']);b=data.boxes[key]['box'];g=clip_graph(m.xyz,m.edges,np.asarray(b['start_xyz'])-.5,np.asarray(b['stop_xyz'])-.5)
            variants,stats=ablations(g);ref=reference_graph(refs,data,key)
            counts.append({'arm':arm,'case_id':key[0],'tile_id':key[1],**stats})
            for name,p in variants.items():
                measured=scores(ref,p);rows.setdefault(arm+'/'+name,[]).append({'case_id':key[0],'tile_id':key[1],**measured})
                if name=='original':
                    # Replay full saved score before attributing an ablation.
                    for t in ['2.0','6.0']:
                        if measured['resampled']['tolerances'][t]!=row['resampled']['tolerances'][t]:raise ValueError('Resampling replay changed')
                    if not np.isclose(measured['length_ratio'],row['length_ratio'],rtol=1e-10,atol=1e-10):raise ValueError('Length replay changed')
            if (i+1)%10==0:print(f'DECOMPOSE {arm} {i+1}/{len(inputs)}',flush=True)
    summaries={k:summarize(v) for k,v in rows.items()}
    effects=[]
    for name,s in summaries.items():
        arm,variant=name.rsplit('/',1);before={n['case_id']:n for n in original[arm]['new_metrics']['neurons']}
        for after in s['new_metrics']['neurons']:
            prior=before[after['case_id']]
            effects.append({'arm':arm,'variant':variant,'case_id':after['case_id'],'original_length_ratio':prior['length_ratio'],
                            'length_ratio':after['length_ratio'],'resampled_F1_delta_pp':{t:100*(after['resampled'][t]['f1']-prior['resampled'][t]['f1']) for t in ['2.0','6.0']},
                            'matched_delta':{t:after['resampled'][t]['matched']-prior['resampled'][t]['matched'] for t in ['2.0','6.0']},
                            'predicted_sample_delta':after['resampled']['2.0']['predicted']-prior['resampled']['2.0']['predicted']})
    report={'rows':rows,'summaries':summaries,'counts':counts,'original_summaries':original,'paired_neuron_ablation_effects':effects,
            'F1_attribution':'independent fixed ablations,not additive shares;all geometry changes diagnostic-only',
            'spur_definition':'terminal-to-junction path length<6;whole short components not spurs',
            'duplicate_definition':'80% .5-voxel samples within2 of earlier longer nonflagged path with abs tangent cosine>=.9;exclude endpoint neighborhoods<=2',
            'Stage2_unchanged':True,'new_development_training_labels':False}
    write_once(folder/'report.json',report)
    table(folder/'report.md','Fixed resampled-F1 decomposition',summaries,['C0/C1 are training-neuron CV;policy arms are development diagnostics. These cohorts must not be conflated.',
          'Removing categories changes matching and sampling. Deltas are independent diagnostics,not additive causal attribution or adopted postprocessing.'])


def render_review(folder,cid,image,ref,selected,graphs,flags):
    from PIL import Image,ImageDraw
    panels=[];shape=np.asarray(image.shape[::-1]);colors={'reference':'#4dff83','retained':'#00d9ff','C0':'#ffbe40','C1':'#ff5cdd'}
    for axes,axis,label in [((0,1),0,'XY'),((0,2),1,'XZ'),((1,2),2,'YZ')]:
        mip=image.max(axis=axis).astype(float);lo,hi=float(image.min()),float(image.max())
        grey=np.clip((mip-lo)/max(hi-lo,1e-12),0,1)**.5
        original=Image.fromarray(np.uint8(grey*255)).convert('RGB');scale=min(1.,1000/max(original.size));size=tuple(max(1,round(v*scale)) for v in original.size)
        base=original.resize(size);sheet=Image.new('RGB',(size[0]*5,size[1]+46),'#101014')
        layers=[('image',None),('reference',ref),('retained',None),('C0',graphs['C0']),('C1',graphs['C1'])]
        for col,(name,g) in enumerate(layers):
            panel=base.copy();draw=ImageDraw.Draw(panel)
            if g is not None:
                for u,v in g.edges:
                    p=g.xyz[[u,v]][:,axes]*scale;draw.line(tuple(map(tuple,p)),fill=colors[name],width=1)
            if name=='retained':
                for p in selected[:,axes]*scale:draw.point(tuple(p),fill=colors[name])
            if name in ['retained','C0','C1']:
                for i,flag in enumerate(flags):
                    x,y=np.asarray(flag['xyz'])[list(axes)]*scale;draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=1);draw.text((x+5,y),str(i+1),fill='white')
            sheet.paste(panel,(col*size[0],36));ImageDraw.Draw(sheet).text((col*size[0]+4,12),label+' '+name,fill='white')
        name=label+'.png';sheet.save(folder/name);panels.append(name)
    text=['<html><body style="background:#101014;color:white;font-family:sans-serif">',f'<h1>{html.escape(cid)}</h1>',
          '<p>Gamma0.5 max projections. All saved points/edges shown;no model merging or relabeling. Reference green,retained cyan,C0 orange,C1 magenta.</p>',
          '<p>Red callouts flag image-supported decoded chains far from trace,not certified untraced neurites. Orthographic projections can hide depth overlap. Human image review remains required.</p>']
    text +=[f'<p><a href="{p}"><img style="max-width:100%" src="{p}"></a></p>' for p in panels]
    text+=['<pre>'+html.escape(json.dumps(flags,indent=2))+'</pre></body></html>']
    (folder/'index.html').write_text('\n'.join(text),encoding='utf-8')


def ridge_review(c,scientific,data,folder):
    from pn2.evaluation.post_stage2 import chains
    refs=e.References(scientific,data);prepared=e.read(ROOT/c['stage2_root']/'prepare/manifest.json')['packets']
    cv=e.read(ROOT/c['stage2_root']/'comparison/C0_C1_training_CV.json');rows=[];panels=[]
    graph_by={arm:{(r['case_id'],r['tile_id']):r for r in cv['rows'][arm]} for arm in ['C0','C1']}
    for number,(cid,case) in enumerate(data.cases.items()):
        spec=c['images'][cid];image=decode_image(ImageCase(cid,Path(spec['path']),case['partition']))
        if image.provenance['source_sha256']!=case['source_sha256'] or image.provenance['decoded_sha256']!=case['decoded_sha256']:raise ValueError('Image differs from retained data')
        raw=image.array_zyx;m=refs.get(cid)['morphology'];shape=np.asarray(raw.shape[::-1]);clipped=clip_graph(m.xyz,m.edges,[-.5]*3,shape-.5)
        segments=graph_segments(m);samples,weights=_sample_arclength(segments,.5)
        unit=np.array([64,64,32]);bins=np.floor(np.clip(samples,0,shape-1)/unit).astype(int);distances=np.full(len(samples),np.inf);physical=np.full(len(samples),np.inf)
        spacing=c['spacing'].get(cid);spacing=np.asarray(spacing['spacing_xyz_um']) if spacing else None
        for bi,block in enumerate(np.unique(bins,axis=0)):
            ids=np.flatnonzero(np.all(bins==block,axis=1));start=block*unit;stop=np.minimum(start+unit,shape)
            low=np.maximum(0,start-18);high=np.minimum(shape,stop+18);crop=raw[low[2]:high[2],low[1]:high[1],low[0]:high[0]]
            inside_low=np.maximum(0,start-12)-low;inside_high=np.minimum(shape,stop+12)-low
            core=tuple(slice(int(a),int(b)) for a,b in zip(inside_low[::-1],inside_high[::-1]))
            ridge=intensity_ridges(crop,case['normalization'],low,core)
            if len(ridge):
                distances[ids]=cKDTree(ridge).query(samples[ids],distance_upper_bound=np.nextafter(12.,np.inf))[0]
                if spacing is not None:physical[ids]=cKDTree(ridge*spacing).query(samples[ids]*spacing,distance_upper_bound=np.nextafter(12.*float(spacing.min()),np.inf))[0]
            if (bi+1)%10==0:print(f'RIDGE alias{case["alias"]} blocks{bi+1}',flush=True)
        record={'case_id':cid,'alias':case['alias'],'partition':case['partition'],'native':weighted_offsets(distances,weights),
                'outside_image_reference_length':max(0.,float(m.total_length-np.linalg.norm(np.diff(graph_segments(clipped),axis=1),axis=2).sum())),
                'reference_denominator':'full supplied polyline arclength including outside-image geometry;outside queries are not discarded',
                'spacing_metadata':c['spacing'].get(cid),'microns':None}
        if spacing is not None:
            length_native=np.linalg.norm(segments[:,1]-segments[:,0],axis=1);length_um=np.linalg.norm((segments[:,1]-segments[:,0])*spacing,axis=1)
            counts=np.array([max(1,int(np.ceil(float(np.linalg.norm(b-a))/.5))) for a,b in segments]);wp=weights*np.repeat(length_um/np.maximum(length_native,1e-30),counts)
            record['microns']=weighted_offsets(physical,wp,12.*float(spacing.min()),(.5,1.,2.));record['microns']['calibration']='reported metadata;not independently acquisition-verified'
        name=f'alias{case["alias"]:03d}';target=folder/name;target.mkdir();np.savez_compressed(target/'ridge_offsets.npz',samples=samples,weights=weights,distance_voxel=distances,distance_um=physical)
        rows.append(record);write_once(target/'report.json',record)
        if cid in c['review_cases']:
            f=[p for p in (ROOT/c['stage2_root']/'comparison').glob('fold*/complete.json') if e.read(p)['heldout']==cid]
            if len(f)!=1:raise ValueError('Missing heldout review fold')
            selections=e.read(f[0].parent/'selection/manifest.json')['packets'];pool={tuple(p['key']):p for p in e.read(ROOT/c['stage1_root']/'candidate_manifest.json')['packets']};retained=[];combined={}
            for p in selections:
                key=tuple(p['key'])
                if key[0]!=cid:continue
                retained.append(checked_npz(pool[key])['xyz'][checked_npz(p)['kept']])
            for arm in ['C0','C1']:
                points=[];edges=[];offset=0
                for key,r in graph_by[arm].items():
                    if key[0]!=cid:continue
                    if digest(Path(r['graph']['path']))!=r['graph']['sha256']:raise ValueError('Changed review SWC')
                    g=parse_swc(r['graph']['path']);points.append(g.xyz);edges.append(g.edges+offset);offset+=len(g.xyz)
                combined[arm]=graph(np.concatenate(points),np.concatenate(edges))
            flags=[];paths,_=chains(combined['C1']);index=reference_index(m)
            for p in sorted(paths,key=lambda p:-p['length']):
                if p['length']<10:continue
                q,w=_sample_arclength(combined['C1'].xyz[combined['C1'].edges[p['edges']]],.5)
                if np.mean(bounded_distances(q,index)>6)<.8:continue
                values=raw[np.clip(np.rint(q[:,2]).astype(int),0,raw.shape[0]-1),np.clip(np.rint(q[:,1]).astype(int),0,raw.shape[1]-1),np.clip(np.rint(q[:,0]).astype(int),0,raw.shape[2]-1)]
                norm=case['normalization'];inside=np.all((q>=0)&(q<=shape-1),axis=1)
                support=float(np.mean(inside&((values.astype(float)-norm['minimum'])/max(norm['maximum']-norm['minimum'],1e-12)>=.1)))
                if support<.8:continue
                flags.append({'xyz':q[len(q)//2].tolist(),'decoded_chain_length':p['length'],'intensity_support_share':support,'status':'possible untraced image structure;human review required'})
                if len(flags)==5:break
            render_review(target,cid,raw,clipped,np.concatenate(retained),combined,flags);panels.append({'case_id':cid,'folder':str(target),'flags':flags,'visually_reviewed':False})
        print(f'RIDGE/REVIEW {number+1}/40 alias{case["alias"]}',flush=True)
        del image,raw
    report={'rows':rows,'review':panels,'new_labels':False,'Stage2_unchanged':True,'ridge_definition':c['ridge_definition'],'selects_nothing':True}
    write_once(folder/'report.json',report)
    with (folder/'report.md').open('x',encoding='utf-8') as f:
        f.write('# Gold-to-ridge offsets\n\nFixed operational intensity ridges,not certified anatomical centerlines. Native arclength weighted;reported-metadata micron view uses physical arclength. No offsets correct any label.\n\n| Neuron | Partition | Length >1/>2/>3 voxels off-ridge (%) |\n|---|---|---|\n')
        for r in rows:f.write('| '+r['case_id']+' | '+r['partition']+' | '+' / '.join(f'{100*r["native"]["share_over"][str(t)]:.2f}' for t in (1.,2.,3.))+' |\n')
    (folder/'index.html').write_text('<h1>Fixed five-neuron review packet</h1>'+''.join('<p><a href="'+Path(p['folder']).name+'/index.html">'+html.escape(p['case_id'])+'</a></p>' for p in panels),encoding='utf-8')


def benchmark_member_case(c,name):
    """Boundary-aware identity match;SLP must not also match WED_SLP."""
    parts=Path(name).as_posix().split('/');matches=[]
    for cid in c['development_cases']:
        if not any(part.startswith(token) for token in c['benchmark_tokens'][cid] for part in parts):continue
        if cid in c.get('benchmark_ambiguous_basename_cases',[]):
            exact_group=cid.split('/')[0] in parts
            exact_id=any(i in parts for i in c['benchmark_published_ids'].get(cid,[]))
            if not exact_group and not exact_id:continue
        matches.append(cid)
    return matches[0] if len(matches)==1 else None


def benchmark_download(c,folder):
    spec=c['benchmark_archive'];path=folder/'benchtesting.tar.gz'
    request=urllib.request.Request(spec['url'],headers={'User-Agent':'PN2-CPU-diagnostic'})
    with urllib.request.urlopen(request,timeout=30) as response,path.open('xb') as out:
        size=0
        while True:
            block=response.read(1024**2)
            if not block:break
            size+=len(block)
            if size>spec['size']:raise ValueError('Archive larger than frozen release metadata')
            out.write(block)
            if size%(64*1024**2)==0:print(f'BENCHMARK DOWNLOAD {size}/{spec["size"]} bytes',flush=True)
    if size!=spec['size']:raise ValueError('Incomplete benchmark archive')
    records=[];unmapped=[]
    with tarfile.open(path,'r:gz') as archive:
        for scanned,member in enumerate(archive,1):
            if scanned%500==0:print(f'BENCHMARK ARCHIVE {scanned} entries scanned;{len(records)} approved files extracted',flush=True)
            if not member.isfile() or not member.name.lower().endswith('.swc'):continue
            cid=benchmark_member_case(c,member.name)
            if cid is None:
                if any(token in member.name for name in c['benchmark_ambiguous_basename_cases'] for token in c['benchmark_tokens'][name]):unmapped.append(member.name)
                continue
            token=next(t for t in c['benchmark_tokens'][cid] if any(part.startswith(t) for part in member.name.split('/')))
            filename=Path(member.name).name;variant=filename.replace(token,'',1).strip('_.')
            if not variant or variant.lower()=='swc':variant=Path(member.name).parent.name
            name=hashlib.sha256(member.name.encode()).hexdigest()+'.swc';target=folder/name
            if member.size>100000000:raise ValueError('Unexpected SWC size')
            with archive.extractfile(member) as source,target.open('xb') as out:out.write(source.read())
            records.append({'case_id':cid,'algorithm_variant':variant,'archive_member':member.name,'path':str(target),'sha256':digest(target)})
    write_once(folder/'manifest.json',{'packets':records,'archive_sha256':digest(path),'release':spec,'only_seven_case_contents_extracted':True,
               'ambiguous_image_names_not_extracted':unmapped,'case_counts':{cid:sum(p['case_id']==cid for p in records) for cid in c['development_cases']}})
    print(f'BENCHMARK DOWNLOAD/EXTRACT COMPLETE: {len(records)} seven-case files;no scoring',flush=True)


def benchmark_score(c,scientific,data,folder,output):
    finished=e.read(output/'benchmark_download/finished.json')
    if finished['config']!=fingerprint(c) or digest(output/'benchmark_download/manifest.json')!=finished['manifest_sha256']:raise ValueError('Changed or incomplete benchmark inventory')
    manifest=e.read(output/'benchmark_download/manifest.json');refs=e.References(scientific,data);groups={};checks=[]
    if any(manifest['case_counts'][cid]==0 for cid in c['development_cases']):
        write_once(folder/'report.json',{'status':'needs_archive_identity_mapping','case_counts':manifest['case_counts'],
                   'reason':'At least one approved case has no unambiguous algorithm files;no gold trace substituted and no seven-neuron benchmark numbers fabricated'})
        (folder/'report.md').write_text('# Benchmark identity mapping incomplete\n\nInspect benchmark_download/manifest.json. No scoring was performed;all seven cases must map unambiguously.\n',encoding='utf-8')
        print('BENCHMARK SCORING NOT RUN;unresolved seven-case identity mapping;see report.json',flush=True)
        return
    all_alg=sorted({p['algorithm_variant'] for p in manifest['packets']})
    if not all_alg:raise ValueError('No named benchmark outputs matched;inspect archive inventory,do not substitute gold references')
    for p in manifest['packets']:groups.setdefault((p['algorithm_variant'],p['case_id']),[]).append(p)
    rows={};whole={};failures=[];statuses=[]
    for ai,algorithm in enumerate(all_alg):
        for cid in c['development_cases']:
            packets=groups.get((algorithm,cid),[]);predictions=[]
            if len(packets)>1:raise ValueError('Duplicate case/algorithm variant;preserve inventory instead of selecting a replicate')
            for p in packets:
                try:
                    if digest(Path(p['path']))!=p['sha256']:raise ValueError('Changed benchmark SWC')
                    m=parse_swc(p['path'],type_policy='opaque_integer');shape=np.asarray(data.cases[cid]['dimensions_xyz'])
                    fraction=float(np.mean(np.all((m.xyz>=-.5)&(m.xyz<=shape-.5),axis=1))) if len(m.xyz) else 1.
                    checks.append({**p,'fraction_inside_native_image':fraction,'axis_order':'SWC XYZ','translation_applied':[0,0,0],'scaling_applied':[1,1,1],
                                   'origin_status':'identity native convention fixed before scoring;bounds check does not prove subvoxel alignment','passed':fraction>=.99})
                    if fraction<.99:raise ValueError('Coordinate bounds check failed')
                    predictions.append(m)
                except (ValueError,OSError) as error:failures.append({**p,'error':str(error)})
            # Keep every variant/case;missing/invalid outputs contribute empty zero scores.
            if len(predictions)>1:raise ValueError('Duplicate case/algorithm variant;preserve inventory for review instead of selecting a replicate')
            pred=predictions[0] if predictions else graph([],[])
            status='missing' if not packets else 'invalid_or_coordinate_failed' if not predictions else 'empty' if not len(pred.xyz) else 'completed'
            statuses.append({'algorithm_variant':algorithm,'case_id':cid,'status':status,'nodes':len(pred.xyz),'edges':len(pred.edges)})
            for key,b in data.boxes.items():
                if key[0]!=cid or not b['target']['count']:continue
                low=np.asarray(b['box']['start_xyz'])-.5;high=np.asarray(b['box']['stop_xyz'])-.5
                rows.setdefault(algorithm,[]).append({'case_id':cid,'tile_id':key[1],'status':status,**scores(reference_graph(refs,data,key),clip_graph(pred.xyz,pred.edges,low,high))})
            shape=np.asarray(data.cases[cid]['dimensions_xyz']);m=refs.get(cid)['morphology']
            whole.setdefault(algorithm,[]).append({'case_id':cid,'tile_id':'whole_image',**scores(clip_graph(m.xyz,m.edges,[-.5]*3,shape-.5),clip_graph(pred.xyz,pred.edges,[-.5]*3,shape-.5))})
        print(f'BENCHMARK {ai+1}/{len(all_alg)} algorithm variants',flush=True)
    summaries={k:summarize(v) for k,v in rows.items()};wholes={k:summarize(v) for k,v in whole.items()}
    best_median={}
    for t in ['2.0','6.0']:
        best_median[t]={}
        for name,field in [('segment','segments'),('resampled','new_metrics')]:
            metric=lambda s:s[field]['macro' if field=='segments' else 'macro_resampled'][t]['f1']
            best=min(summaries,key=lambda k:(-metric(summaries[k]),k))
            best_median[t][name]={'best_fixed_algorithm_variant':best,'best':metric(summaries[best]),'median_algorithm_macro':float(np.median([metric(s) for s in summaries.values()]))}
    per_neuron={}
    for cid in c['development_cases']:
        per_neuron[cid]={}
        for t in ['2.0','6.0']:
            per_neuron[cid][t]={}
            for label,field in [('segment','segments'),('resampled','new_metrics')]:
                values={}
                for algorithm,s in summaries.items():
                    ns=s[field]['neurons'];n=next(n for n in ns if n['case_id']==cid)
                    values[algorithm]=n['tolerances' if field=='segments' else 'resampled'][t]['f1']
                best=min(values,key=lambda k:(-values[k],k));per_neuron[cid][t][label]={'best_variant':best,'best':values[best],'median':float(np.median(list(values.values())))}
    report={'summaries':summaries,'rows':rows,'whole_image_summaries':wholes,'whole_image_rows':whole,'coordinate_checks':checks,
            'failures':failures,'case_statuses':statuses,'best_median':best_median,'scope':'same151 development boxes primary;whole-image context separate',
            'per_neuron_best_median':per_neuron,'per_neuron_best_is_not_a_single_algorithm':True,
            'algorithm_optimization':'published parameter-optimized release,not an independent untuned control','selection':'benchmark description only;no PN2 model selected'}
    write_once(folder/'report.json',report);table(folder/'report.md','BigNeuron external benchmark variants',summaries,
        ['Native XYZ identity fixed;no alignment fitted to the gold. Coordinate checks and failures are disclosed.',
         'Missing or invalid case results count as zero. Whole-image scores are separate in JSON. Best/median do not select a PN2 model.'])
    with (folder/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## Best fixed variant and median variant macro\n\n| Metric | Tolerance | Best variant | Best F1 (%) | Median F1 (%) |\n|---|---|---|---|---|\n')
        for t,metrics in best_median.items():
            for name,m in metrics.items():f.write(f"| {name} | {t} | {m['best_fixed_algorithm_variant']} | {100*m['best']:.2f} | {100*m['median_algorithm_macro']:.2f} |\n")


def editing(c,scientific,data,folder,output):
    from pn2.evaluation.editing_headroom import make_forests
    refs=e.References(scientific,data);pools=e.read(ROOT/c['stage1_root']/'candidate_manifest.json')['packets'];rows={'prune_only':[],'prune_add':[]};details=[]
    (output/'metric_cache').mkdir(exist_ok=True);metrics=metrics_module.Metrics(c,scientific,data,output)
    for i,p in enumerate([p for p in pools if p['partition']=='development']):
        key=tuple(p['key']);pool=checked_npz(p);spec=c['classical'][key[0]];classical=checked_npz(spec)
        b=data.boxes[key]['box'];low=np.asarray(b['start_xyz'])-.5;high=np.asarray(b['stop_xyz'])-.5
        cp,ce,_=crop_classical(classical['pruned_xyz'],classical['pruned_edges'],low,high)
        ref=reference_graph(refs,data,key);length=float(np.linalg.norm(np.diff(graph_segments(ref),axis=1),axis=2).sum())
        a,b,provenance=make_forests(graph(cp,ce),pool,ref,low,high)
        a=clip_graph(a.xyz,a.edges,low,high);b=clip_graph(b.xyz,b.edges,low,high)
        # Clipping intersections are measurement vertices,not original anchors.
        anchors=cp
        ga,sa=compact_prune(a,ref,length);gb,sb=compact_prune(b,ref,length,anchors=anchors)
        if scores(ref,ga)['resampled']['tolerances']['2.0']['f1']>scores(ref,gb)['resampled']['tolerances']['2.0']['f1']:
            gb=ga;sb={**sb,'fallback_to_prune_only':True}
        for name,g in [('prune_only',ga),('prune_add',gb)]:
            target=folder/f'box{i:04d}'/name;target.mkdir(parents=True);path=target/'REFERENCE_ASSISTED_ONLY.swc'
            export_forest(path,g.xyz,g.edges,np.zeros(len(g.xyz)))
            spec=metrics_module.graph_spec(path,digest(path));row=metrics.measure({'key':list(key),'method':'editing_'+name,'graph':spec,'segments':None})
            if row['length_ratio']>1.25+1e-9:raise ValueError('Editing oracle violated frozen per-box length budget')
            rows[name].append(row)
        details.append({'key':list(key),'prune_only':sa,'prune_add':sb,'provenance':provenance})
        print(f'EDITING ORACLES {i+1}/151',flush=True)
    # Per-box F1 improvements need not improve a neuron's pooled matching ratio.
    # The diagnostic feasible set for(b) includes the entire(a) solution as well.
    neuron_fallbacks=[]
    for cid in c['development_cases']:
        aa=[r for r in rows['prune_only'] if r['case_id']==cid];bb=[r for r in rows['prune_add'] if r['case_id']==cid]
        va=sample_summary(aa)['macro_resampled']['2.0']['f1'];vb=sample_summary(bb)['macro_resampled']['2.0']['f1']
        if va>vb:
            rows['prune_add']=[r for r in rows['prune_add'] if r['case_id']!=cid]+[{**r,'method':'editing_prune_add_fallback_to_prune_only'} for r in aa]
            neuron_fallbacks.append({'case_id':cid,'prune_only_F1_2':va,'candidate_prune_add_F1_2':vb})
    summaries={k:summarize(v) for k,v in rows.items()};a=summaries['prune_only']['new_metrics']['macro_resampled']['2.0']['f1'];b=summaries['prune_add']['new_metrics']['macro_resampled']['2.0']['f1']
    gate={'draft_editing_contract_allowed':b>=.6,'prune_only_alone_reaches_60':a>=.6,'prune_add_resampled_F1_2':b,'threshold':.6,'length_cap':1.25,
          'automatic_draft_or_training':False,'failed_heuristic_is_not_impossibility_proof':True}
    write_once(folder/'report.json',{'rows':rows,'summaries':summaries,'details':details,'neuron_fallbacks':neuron_fallbacks,'gate':gate,'new_development_training_labels':False,
                'reference_assisted_diagnostic_only':True,'Stage2_STOP_unchanged':True})
    table(folder/'report.md','Classical-anchored editing headroom',summaries,[str(gate),'Feasible bounded oracle search,not a globally optimal ceiling. References enter this diagnostic only;no inference editing contract is adopted.'])
    print('EDITING DIAGNOSTIC GATE '+json.dumps(gate),flush=True)


def worker(action,output):
    c,scientific,data=control();output.mkdir(exist_ok=True,parents=True);lock=output/'RUNNING.json'
    binding={'config':fingerprint(c),'diagnostic_only':True,'Stage2_stopped':True}
    if (output/'binding.json').exists():
        if e.read(output/'binding.json')!=binding:raise ValueError('Different diagnostic config already owns output')
    else:write_once(output/'binding.json',binding)
    write_once(lock,{'action':action,'pid':os.getpid()});folder=output/action;folder.mkdir(exist_ok=False)
    before=digest(ROOT/c['stage2_root']/'STOP.json')
    try:
        if action=='ridge_review':ridge_review(c,scientific,data,folder)
        elif action=='decompose':decomposition(c,scientific,data,folder)
        elif action=='benchmark_download':benchmark_download(c,folder)
        elif action=='benchmark_score':benchmark_score(c,scientific,data,folder,output)
        else:editing(c,scientific,data,folder,output)
        if digest(ROOT/c['stage2_root']/'STOP.json')!=before:raise ValueError('Stage2 STOP changed')
        if 'torch' in sys.modules:raise ValueError('Diagnostic imported Torch')
        receipt={'status':'completed','config':fingerprint(c),'Stage2_STOP_sha256':before,'training':False,'GPU':False}
        if (folder/'manifest.json').exists():receipt['manifest_sha256']=digest(folder/'manifest.json')
        write_once(folder/'finished.json',receipt)
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=['ridge_review','decompose','benchmark_download','benchmark_score','editing'],required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    c,_,_=control();output=e.output_path(args.output or ROOT/'tmp/post_stage2_diagnostics_20261106')
    if args.check:print('Diagnostic metadata/source/STOP bindings verified;no timed job executed');return
    if args.worker:worker(args.action,output);return
    if (output/args.action).exists():raise FileExistsError('Preserve prior diagnostic outputs;no automatic retry')
    if (output/'RUNNING.json').exists():raise RuntimeError('Another diagnostic action owns this output')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');seconds=c['wall_seconds'][args.action]
    print(f'CPU DIAGNOSTIC ONLY {args.action};output={output};cap={seconds}s;Stage2 remains stopped',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--action',args.action,'--output',str(output)],
        output.with_name(output.name+'_'+args.action+'_guard_'+stamp),seconds,{'action':args.action,'config':fingerprint(c),'Stage2_stopped':True})


if __name__=='__main__':main()
