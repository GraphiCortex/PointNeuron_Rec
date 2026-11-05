"""User-local CPU Stage2:prepare -> policy diagnostic -> C0/C1 and feature CV.

No default action, GPU, GATv2, development tuning or automatic extension.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np

import local_curve_experiment as e
import audit_resampled_compact as metric_driver
from pn2.inputs.node_cache import digest,fingerprint,write_once
from pn2.inputs.image import decode_image
from pn2.inputs.cases import ImageCase
from pn2.selection.features import node_features,NODE_NAMES,STATIC_EDGE_NAMES
from pn2.selection.data import Store,atomic_npz
from pn2.selection.trees import TREE_PARAMETERS,fit_tree,probability,representatives,final_gate,choose_contract
from pn2.supervision.candidate_labels import TrainingPaths
from pn2.evaluation.probability_forest import decode,prune_spurs
from pn2.evaluation.vote_scaffold import export_forest,image_edge_costs,minimum_spanning_forest
from pn2.evaluation.fixed_curve_graph import summarize as segment_summary
from pn2.evaluation.resampled_skeleton import summarize as new_summary,clip_graph
from pn2.evaluation.label_audit import precision_recall_area
from pn2.evaluation.swc import parse_swc
from pn2.selection.diagnostics import reconstruction_risks
from pn2.training.guarded_run import run

ROOT=e.ROOT;CONFIG=ROOT/'configs/stage2_trees_20261105.json'


def control():
    c=e.read(CONFIG)
    if c['schema']!='pn2-stage2-trees-v1' or c['node_feature_arm']!='handcrafted_only' or c['PointNeXt_gain_tolerances']!=[2.]:raise ValueError('Changed approved scope')
    if c['tree_parameters']!=TREE_PARAMETERS or c['GPU'] or c['GATv2']:raise ValueError('Changed backend/budget')
    for path,sha in {**c['source_sha256'],**c['artifact_sha256']}.items():
        if digest(ROOT/path)!=sha:raise ValueError('Changed Stage2 binding: '+path)
    for package,version in c['runtime_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Wrong isolated CPU runtime: '+package)
    scientific=e.control();data,_,_=e.prepared(scientific)
    stage1=ROOT/c['stage1_root'];pool=e.read(stage1/'candidate_manifest.json')['packets'];labels=e.read(stage1/'label_manifest.json')['packets']
    if len(pool)!=611 or len(labels)!=460 or any(p['partition']!='training' for p in labels):raise ValueError('Changed cohort/label scope')
    by={tuple(p['key']):p for p in pool}
    expected={k for k,b in data.boxes.items() if b['target']['supervision']=='reference_present'}
    if set(by)!=expected or len(by)!=len(pool):raise ValueError('Changed candidate identities')
    for specs,training_only in [(labels,True),(e.read(ROOT/c['pointnext_root']/'complete.json')['packets'],False)]:
        if len({tuple(p['key']) for p in specs})!=len(specs):raise ValueError('Duplicate packet identities')
        if {tuple(p['key']) for p in specs}!={k for k in expected if not training_only or data.cases[k[0]]['partition']=='training'}:raise ValueError('Changed feature/label cohort')
        for spec in specs:
            key=tuple(spec['key'])
            if spec['candidate_sha256']!=by[key]['sha256'] or spec['partition']!=data.cases[key[0]]['partition']:raise ValueError('Changed packet alignment')
            if not Path(spec['path']).is_file():raise FileNotFoundError(spec['path'])
    for spec in pool:
        if spec['references_used_for_construction'] or not Path(spec['path']).is_file():raise ValueError('Invalid retained image-only pool')
    return c,scientific,data,pool


def save_model(path,model,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:pickle.dump(model,f,protocol=5)
    receipt={**metadata,'path':str(path),'sha256':digest(path)};write_once(path.with_suffix('.json'),receipt)
    return receipt


def load_model(receipt):
    path=Path(receipt['path'])
    if digest(path)!=receipt['sha256']:raise ValueError('Frozen model changed')
    with path.open('rb') as f:return pickle.load(f)


def check_manifest(path,value):
    if path.exists():
        if e.read(path)!=value:raise ValueError('Different manifest: '+str(path))
    else:write_once(path,value)


def preparation(c,scientific,data,pools,output):
    folder=output/'prepare';folder.mkdir(exist_ok=False)
    by={tuple(p['key']):p for p in pools};votes_root=ROOT/c['broad_root']/'train/terminal_predictions'
    votes={tuple(p['key']):p for p in e.read(votes_root/'complete.json')['packets']};guides={};maxima={};packets=[]
    for cid,case in data.cases.items():
        keys=[key for key in by if key[0]==cid];first=keys[0]
        cloud_folder=(ROOT/data.boxes[first]['clouds']['2048']['path']).parent;receipt=e.read(cloud_folder/'case.json')
        if receipt['image']['decoded_bytes']>c['maximum_decoded_image_bytes']:raise MemoryError('Image exceeds frozen decode resource cap')
        image=decode_image(ImageCase(cid,Path(receipt['image']['image_path']),case['partition']))
        if image.provenance['source_sha256']!=case['source_sha256'] or image.provenance['decoded_sha256']!=case['decoded_sha256']:raise ValueError('Raw image differs from retained input')
        for key in keys:
            spec=by[key]
            if digest(Path(spec['path']))!=spec['sha256'] or spec['references_used_for_construction']:raise ValueError('Invalid image-only pool')
            with np.load(spec['path'],allow_pickle=False) as a:pool={k:a[k].copy() for k in a.files}
            source=data.image(*key,2048)['packet']
            if not np.array_equal(image.array_zyx.ravel()[source['flat_indices']],source['raw_intensity']):raise ValueError('Raw/cache intensity mismatch')
            e.sato_scores(data,key,source,guides);guide=guides[cid]
            if cid not in maxima:maxima[cid]=float(guide.max())
            maximum=maxima[cid]
            box=data.boxes[key]['box'];shape=np.asarray(case['dimensions_xyz']);low=np.maximum(0,np.asarray(box['start_xyz'])-12);high=np.minimum(shape,np.asarray(box['stop_xyz'])+12)
            crop=image.array_zyx[low[2]:high[2],low[1]:high[1],low[0]:high[0]].copy()
            vp=votes[key];vpath=votes_root/vp['path']
            if digest(vpath)!=vp['sha256']:raise ValueError('Changed frozen raw votes')
            with np.load(vpath,allow_pickle=False) as a:raw_votes=a['points'].copy()
            x,tangent=node_features(pool,source,crop,low,case['normalization'],guide,maximum,raw_votes,box,shape)
            index=len(packets);target=folder/f'box{index:04d}';target.mkdir();rawpath=target/'raw_crop.npz';path=target/'node_features.npz'
            atomic_npz(rawpath,raw=crop,origin_xyz=low)
            atomic_npz(path,node_features=x,tangent=tangent)
            packets.append({'index':index,'key':list(key),'partition':case['partition'],'path':str(path),'sha256':digest(path),
              'candidate_sha256':spec['sha256'],'raw_crop_path':str(rawpath),'raw_crop_sha256':digest(rawpath),
              'guide_path':str(cloud_folder/receipt['guide']['path']),'guide_sha256':receipt['guide']['sha256'],'guide_maximum':maximum})
            if len(packets)%10==0 or len(packets)==611:print(f'PREPARE {len(packets)}/611 image-only features',flush=True)
        del image
    report={'status':'completed','config_fingerprint':fingerprint(c),'packets':packets,'node_feature_names':NODE_NAMES,
            'raw_images_redecoded_not_filtered':True,'reference_geometry_opened':False,'new_development_labels':False,
            'CUDA_executed':False,'model_executed':False,'training_executed':False}
    write_once(folder/'manifest.json',report)
    with (folder/'report.md').open('x',encoding='utf-8') as f:
        f.write('# Stage2 image-only preparation\n\nCompleted '+str(len(packets))+' feature packets. Original raw images were decoded for intensity profiles; retained Sato guides and candidate packets were reused. No filtering, reference geometry, model fit, new development labels or GPU work.\n')
    print('PREPARATION COMPLETE '+str(folder/'manifest.json'),flush=True)


def node_records(store,keys):
    records=[]
    for key in keys:
        _,a=store.get(key);y=store.training_labels(key)['node_label']
        rows=np.array([f'{key[1]}:n{i}' for i in range(len(y))]);records.append((key[0],rows,a['node_features'],y))
    return records


def select_nodes(store,model,keys,folder):
    folder.mkdir(parents=True,exist_ok=False);packets=[]
    for i,key in enumerate(keys):
        pool,a=store.get(key);p=probability(model,a['node_features'])
        kept=representatives(pool['xyz'],pool['source_flags'],a['node_features'][:,0],p)
        path=folder/f'box{i:04d}.npz';atomic_npz(path,probability=p,kept=kept)
        packets.append({'key':list(key),'path':str(path),'sha256':digest(path),'candidate_sha256':store.pools[key]['sha256']})
    write_once(folder/'manifest.json',{'packets':packets,'reference_used':False,'ordering':['source_priority','Sato_descending','selector_probability_descending','XYZ/index']})
    return {tuple(p['key']):p for p in packets}


def selected(store,selection,key):
    spec=selection[key]
    if spec['candidate_sha256']!=store.pools[key]['sha256']:raise ValueError('Selection/pool mismatch')
    return store.checked(spec)


def output_graph(pool,node_p,edge_ids,edge_p,folder):
    folder.mkdir(parents=True,exist_ok=False)
    result=decode(pool['xyz'],pool['edges'][edge_ids],node_p,edge_p);ids=result['selected_indices'];path=folder/'prediction.swc'
    export_forest(path,pool['xyz'][ids],result['edges'],pool['image_guide'][ids])
    return metric_driver.graph_spec(path,digest(path))


def metric(metrics,key,graph,method):
    return metrics.measure({'key':list(key),'method':method,'graph':graph,'segments':None})


def summarize(rows):return {'segments':segment_summary(rows),'new_metrics':new_summary(rows)}


def readable_report(path,title,summaries,notes=()):
    lines=['# '+title,'',*notes,'','| Arm | Segment F1 2 / 6 (%) | Resampled F1 2 / 6 (%) | Pooled length ratio |',
           '|---|---|---|---|']
    for name,s in summaries.items():
        a=s['segments']['macro'];b=s['new_metrics']['macro_resampled']
        lines.append(f"| {name} | {100*a['2.0']['f1']:.2f} / {100*a['6.0']['f1']:.2f} | {100*b['2.0']['f1']:.2f} / {100*b['6.0']['f1']:.2f} | {s['new_metrics']['pooled_length_ratio']:.4f} |")
    lines+=['','Per-neuron metrics, counts, graph hashes and inference receipts are in the companion JSON files.',
            'Native XYZ voxel units. Per-box reconstruction; no whole-image assembly, radius or biological topology qualification.','']
    with Path(path).open('x',encoding='utf-8') as f:f.write('\n'.join(lines))


def policy(c,scientific,data,store,keys,output,metrics):
    folder=output/'policy';folder.mkdir(exist_ok=False)
    train=[k for k in keys if data.cases[k[0]]['partition']=='training']
    model,fit=fit_tree(node_records(store,train),c['node_cap_per_neuron_class'])
    receipt=save_model(folder/'node_model.pkl',model,{'config':fingerprint(c),'training_neurons':sorted({k[0] for k in train}),'fit':fit,'feature_arm':'handcrafted_only'})
    selection=select_nodes(store,model,keys,folder/'selection')
    # All policy outputs are saved/frozen before reference-assisted diagnostics.
    write_once(folder/'frozen.json',{'node_model':receipt,'selection_manifest_sha256':digest(folder/'selection/manifest.json'),'config':fingerprint(c)})
    stage0=ROOT/c['stage0_root'];oracle={(r['case_id'],r['tile_id']):r for r in e.read(stage0/'report.json')['rows'] if r['arm']=='union'}
    rows={'image_scaffold':[],'conditional_edge_oracle':[]}
    for key in [k for k in keys if data.cases[k[0]]['partition']=='development']:
        pool,_=store.get(key);s=selected(store,selection,key);kept=s['kept'];p=np.where(kept,s['probability'],0.)
        edge_ids=np.flatnonzero(np.all(kept[pool['edges']],axis=1));edges=pool['edges'][edge_ids]
        ids=np.flatnonzero(kept);mapping=np.full(len(kept),-1,int);mapping[ids]=np.arange(len(ids));guide,maximum=store.guide(key)
        cost,_=image_edge_costs(pool['xyz'],edges,guide,maximum)
        forest,_=minimum_spanning_forest(pool['xyz'][ids],mapping[edges],cost)
        survived,pruned,_=prune_spurs(pool['xyz'][ids],forest);target=folder/'diagnostic'/f'box{store.features[key]["index"]:04d}';target.mkdir(parents=True)
        path=target/'image_only_scaffold.swc';export_forest(path,pool['xyz'][ids[survived]],pruned,pool['image_guide'][ids[survived]])
        rows['image_scaffold'].append(metric(metrics,key,metric_driver.graph_spec(path,digest(path)),'policy_image_only'))
        old=oracle[key];probs=(stage0/old['swc']).parent/'oracle_scores_DO_NOT_USE_IN_INFERENCE.npz'
        with np.load(probs,allow_pickle=False) as a:edge_p=np.where(a['edge_scored_length']>0,a['edge_probability'],0.)
        graph=output_graph(pool,p,edge_ids,edge_p[edge_ids],target/'conditional_edge_oracle')
        rows['conditional_edge_oracle'].append(metric(metrics,key,graph,'policy_image_nodes_reference_assisted_edges'))
    report={'status':'completed','config':fingerprint(c),'node_model':receipt,'summaries':{k:summarize(v) for k,v in rows.items()},
            'rows':rows,'selects_nothing':True,'quality_gate_applied':False,'benchmark':{'oracle_segment_F1_2':.8851708414517212,'oracle_resampled_F1_2':.6479597076530379,'oracle_length_ratio':1.184745927495074},
            'conditional_edge_oracle_is_not_real_inference':True,'new_development_labels':False}
    write_once(folder/'report.json',report)
    readable_report(folder/'report.md','Stage2 representative-policy diagnostic',report['summaries'],[
        'Development diagnostic selects nothing and applies no quality gate. The policy is frozen before scoring.',
        'Compact union benchmark: segment F1_2 88.52%, resampled F1_2 64.80%, length 1.18475x.',
        'The image scaffold is genuine inference; the conditional edge oracle uses reference scores for evaluation only.'])
    print('POLICY DIAGNOSTIC COMPLETE; selects nothing; '+str(folder/'report.json'),flush=True)


def edge_training_records(store,selection,keys,contract,references,pointnext):
    records=[];counts={}
    for key in keys:
        s=selected(store,selection,key);ids,x=store.edge_features(key,s['kept'],pointnext)
        y=store.targets(key,s['kept'],ids,contract,references)
        rowkeys=np.array([f'{key[1]}:e{i}' for i in ids]);records.append((key[0],rowkeys,x,y))
        n=counts.setdefault(key[0],{'positive':0,'negative':0,'unknown':0,'bounded_geodesic_missing':0})
        for name,label in [('positive',1),('negative',0),('unknown',-1)]:n[name]+=int((y==label).sum())
        n['bounded_geodesic_missing']+=int(np.isnan(x[:,11]).sum())
    return records,counts


def predict_edges(store,selection,keys,model,pointnext,folder,metrics,method):
    folder.mkdir(parents=True,exist_ok=False);rows=[];prediction_specs=[]
    for key in keys:
        pool,_=store.get(key);s=selected(store,selection,key);ids,x=store.edge_features(key,s['kept'],pointnext)
        ep=probability(model,x);node=np.where(s['kept'],s['probability'],0.)
        target=folder/f'box{store.features[key]["index"]:04d}'
        graph=output_graph(pool,node,ids,ep,target)
        scores=target/'image_only_edge_scores.npz';atomic_npz(scores,edge_indices=ids,probability=ep)
        prediction_specs.append({'key':list(key),'graph':graph,'edge_scores':str(scores),'edge_scores_sha256':digest(scores)})
    # Prediction artifacts,not references,define every decoded graph.
    write_once(folder/'inference_complete.json',{'predictions':prediction_specs,'reference_edges_used':False,'reference_roots_used':False})
    for p in prediction_specs:rows.append(metric(metrics,tuple(p['key']),p['graph'],method))
    return rows


def feature_cv_AP(store,folds,folder,train,contract,references):
    """Only SAME chosen-contract labels are compared; no development labels."""
    neurons=[]
    for fi,fold in enumerate(folds):
        f=folder/f'fold{fi:02d}';heldout=fold['heldout'];selection={tuple(p['key']):p for p in e.read(f/'selection/manifest.json')['packets']}
        arms={}
        for arm,where in [('handcrafted',f/(contract+'_heldout')),('PointNeXt',f/'PointNeXt_heldout')]:
            yy=[];pp=[]
            for p in e.read(where/'inference_complete.json')['predictions']:
                key=tuple(p['key'])
                if key[0]!=heldout:raise ValueError('Wrong held-out edge AP scope')
                if digest(Path(p['edge_scores']))!=p['edge_scores_sha256']:raise ValueError('Changed saved edge predictions')
                with np.load(p['edge_scores'],allow_pickle=False) as a:ids,score=a['edge_indices'].copy(),a['probability'].copy()
                kept=selected(store,selection,key)['kept']
                yy.append(store.targets(key,kept,ids,contract,references));pp.append(score)
            arms[arm]=precision_recall_area(np.concatenate(yy),np.concatenate(pp))
        neurons.append({'case_id':heldout,'same_contract':contract,'AP':arms})
    return {'neurons':neurons,'compared_across_contracts':False,'development_AP':'not generated;no new development labels'}


def risk_report(rows,data,references):
    result=[];indices={}
    for row in rows:
        key=(row['case_id'],row['tile_id']);graph=parse_swc(row['graph']['path'])
        box=data.boxes[key]['box']
        graph=clip_graph(graph.xyz,graph.edges,np.asarray(box['start_xyz'])-.5,np.asarray(box['stop_xyz'])-.5)
        morphology=references.get(key[0])['morphology']
        if key[0] not in indices:indices[key[0]]=TrainingPaths(morphology)
        result.append({'case_id':key[0],'tile_id':key[1],**reconstruction_risks(graph,morphology,data.reference(*key)['reference_runs'],indices[key[0]])})
    return result


def stop(output,reason,details):
    write_once(output/'STOP.json',{'status':'stopped','reason':reason,'details':details,'automatic_extension':False})
    with (output/'STOP.md').open('x',encoding='utf-8') as f:
        f.write('# Stage2 stopped\n\n'+reason+'\n\nNo tuning, additional development scoring or automatic extension follows. See STOP.json and the completed stage reports for measurements. A completed computation does not imply that reconstruction criteria passed.\n')
    print('STAGE2 STOP '+json.dumps({'reason':reason,'details':details}),flush=True)


def compare(c,scientific,data,store,keys,output,metrics):
    folder=output/'comparison';folder.mkdir(exist_ok=False)
    train=[k for k in keys if data.cases[k[0]]['partition']=='training'];neurons=sorted({k[0] for k in train})
    if len(train)!=460 or len(neurons)!=33:raise ValueError('Changed33-neuron training folds')
    references=e.References(scientific,data);cv={'C0':[],'C1':[]};folds=[];began=time.perf_counter()
    for fi,heldout in enumerate(neurons):
        f=folder/f'fold{fi:02d}';f.mkdir();training=[k for k in train if k[0]!=heldout];testing=[k for k in train if k[0]==heldout]
        node,fit=fit_tree(node_records(store,training),c['node_cap_per_neuron_class'])
        nr=save_model(f/'node.pkl',node,{'heldout':heldout,'training_neurons':sorted({k[0] for k in training}),'fit':fit})
        selection=select_nodes(store,node,train,f/'selection');fold={'heldout':heldout,'node_model':nr,'selection':str(f/'selection/manifest.json'),'contracts':{}}
        for contract in ('C0','C1'):
            records,counts=edge_training_records(store,selection,training,contract,references,False)
            edge,fit=fit_tree(records,c['edge_cap_per_neuron_class']);del records
            er=save_model(f/(contract+'_edge.pkl'),edge,{'heldout':heldout,'contract':contract,'fit':fit,'label_counts':counts})
            rows=predict_edges(store,selection,testing,edge,False,f/(contract+'_heldout'),metrics,'CV_'+contract)
            fold['contracts'][contract]={'model':er,'rows':rows,'summary':summarize(rows)};cv[contract]+=rows
        write_once(f/'complete.json',fold);folds.append(fold)
        print(f'C0/C1 CV {fi+1}/33 complete;elapsed={time.perf_counter()-began:.1f}s',flush=True)
        if fi==0:
            elapsed=time.perf_counter()-began;cold=store.cold_feature_seconds
            projected=max(0.,elapsed-cold)*c['first_fold_projection_multiplier']+cold*c['cold_feature_projection_multiplier']+c['final_reserve_seconds']
            write_once(folder/'first_fold_resource_projection.json',{'seconds':elapsed,'cold_image_feature_seconds':cold,'projected_total_seconds':projected,'fixed_cap':c['wall_seconds']['compare']})
            if projected>c['wall_seconds']['compare']:
                stop(output,'first-fold CPU projection exceeds fixed comparison cap;no extension',{'projected_seconds':projected});return
    summaries={k:summarize(v) for k,v in cv.items()};decision=choose_contract(summaries)
    write_once(folder/'C0_C1_training_CV.json',{'summaries':summaries,'rows':cv,'decision':decision,'AP_compared':False,
               'CV':'selector-only conditional on broad upstream trained on all33;outer node/edge fits exclude heldout neuron'})
    readable_report(folder/'C0_C1_training_CV.md','C0 versus C1 training-neuron selector CV',summaries,
                    ['Conditional selector CV: upstream encoder and learned candidates already trained on all33 neurons.',
                     'Contract selection uses training-fold decoded geometry and length only; AP is not compared across contracts.',str(decision)])
    if decision['status']=='stop':stop(output,decision['reason'],summaries);return
    contract=decision['contract'];pn_rows=[]
    for fi,fold in enumerate(folds):
        f=folder/f'fold{fi:02d}';heldout=fold['heldout'];selection={tuple(p['key']):p for p in e.read(f/'selection/manifest.json')['packets']}
        training=[k for k in train if k[0]!=heldout];testing=[k for k in train if k[0]==heldout]
        records,counts=edge_training_records(store,selection,training,contract,references,True)
        edge,fit=fit_tree(records,c['edge_cap_per_neuron_class']);del records
        er=save_model(f/'PointNeXt_edge.pkl',edge,{'heldout':heldout,'contract':contract,'fit':fit,'label_counts':counts,'PointNeXt':'edge features only'})
        rows=predict_edges(store,selection,testing,edge,True,f/'PointNeXt_heldout',metrics,'CV_PointNeXt')
        pn_rows+=rows;write_once(f/'PointNeXt_complete.json',{'model':er,'rows':rows})
        print(f'POINTNEXT EDGE CV {fi+1}/33 complete',flush=True)
    feature_cv={'handcrafted':summaries[contract],'PointNeXt':summarize(pn_rows)}
    ap=feature_cv_AP(store,folds,folder,train,contract,references)
    write_once(folder/'feature_CV.json',{'contract':contract,'summaries':feature_cv,'PointNeXt_rows':pn_rows,'node_selector_shared':True,'same_contract_AP':ap})
    readable_report(folder/'feature_CV.md','Fixed-contract feature comparison in training-neuron selector CV',feature_cv,
                    ['Shared handcrafted node selector and representatives; PointNeXt enters edge features only.',
                     'Conditional selector CV, not full-pipeline held-out performance. Per-neuron same-contract AP is in feature_CV.json.'])
    # Fit BOTH final edge models and freeze all hashes before ANY final dev score.
    policy_frozen=e.read(output/'policy/frozen.json');selection={tuple(p['key']):p for p in e.read(output/'policy/selection/manifest.json')['packets']}
    final=output/'final';final.mkdir();models={}
    for arm,pn in [('handcrafted',False),('PointNeXt',True)]:
        records,counts=edge_training_records(store,selection,train,contract,references,pn)
        edge,fit=fit_tree(records,c['edge_cap_per_neuron_class']);del records
        models[arm]=save_model(final/(arm+'_edge.pkl'),edge,{'contract':contract,'feature_arm':arm,'fit':fit,'label_counts':counts,'training_neurons':neurons})
    frozen={'config':fingerprint(c),'contract':contract,'node_model':policy_frozen['node_model'],'edge_models':models,
            'development_scored':False,'shared_node_representatives':True,'thresholds':{'node':.5,'edge':.5}}
    write_once(final/'FROZEN_BEFORE_DEVELOPMENT.json',frozen)
    train_scores={}
    for arm,pn in [('handcrafted',False),('PointNeXt',True)]:
        fitted=load_model(models[arm]);rows=predict_edges(store,selection,train,fitted,pn,final/(arm+'_training_fit'),metrics,'TRAINING_FIT_'+arm)
        train_scores[arm]=summarize(rows)
        write_once(final/(arm+'_training_report.json'),{'rows':rows,'summary':train_scores[arm],
                   'risk_diagnostics':risk_report(rows,data,references),'training_metric_gate':False})
    baseline=e.read(ROOT/c['baseline_metrics'])['summaries']['classical_pruned/development'];dev=[k for k in keys if data.cases[k[0]]['partition']=='development']
    scores={};gates={}
    for arm,pn in [('handcrafted',False),('PointNeXt',True)]:
        started=final/(arm+'_development_STARTED.json')
        write_once(started,{'frozen_manifest_sha256':digest(final/'FROZEN_BEFORE_DEVELOPMENT.json'),'model_sha256':models[arm]['sha256'],'scoring_number':1,'thresholds_fixed':True})
        edge=load_model(models[arm]);rows=predict_edges(store,selection,dev,edge,pn,final/(arm+'_development'),metrics,'DEVELOPMENT_'+arm)
        s=summarize(rows);gate=final_gate(s,baseline,scores.get('handcrafted') if pn else None);scores[arm]=s;gates[arm]=gate
        gap={'segment_F1':{t:s['segments']['macro'][t]['f1']-feature_cv[arm]['segments']['macro'][t]['f1'] for t in ('2.0','6.0')},
             'resampled_F1':{t:s['new_metrics']['macro_resampled'][t]['f1']-feature_cv[arm]['new_metrics']['macro_resampled'][t]['f1'] for t in ('2.0','6.0')},
             'pooled_length_ratio':s['new_metrics']['pooled_length_ratio']-feature_cv[arm]['new_metrics']['pooled_length_ratio']}
        headroom={t:{'fraction_above_classical_captured':(s['segments']['macro'][t]['f1']-baseline['segments']['macro'][t]['f1'])/(ceiling-baseline['segments']['macro'][t]['f1']),
                        'model_divided_by_compact_oracle':s['segments']['macro'][t]['f1']/ceiling}
                  for t,ceiling in [('2.0',.8851708414517212),('6.0',.9510306)]}
        report={'status':'completed','arm':arm,'rows':rows,'summary':s,'gate':gate,'development_minus_CV':gap,
                'CV_summary':feature_cv[arm],'training_fit_summary':train_scores[arm],
                'compact_oracle_headroom_diagnostic':headroom,
                'risk_diagnostics':risk_report(rows,data,references),
                'repeatedly_inspected_development':True,'scoring_number':1,'new_development_labels':False}
        write_once(final/(arm+'_development_report.json'),report)
        readable_report(final/(arm+'_development_report.md'),'Frozen '+arm+' development result',
                        {'training_fit':train_scores[arm],'conditional_selector_CV':feature_cv[arm],'development':s,'matched_classical':baseline},
                        ['Development has been repeatedly inspected; this frozen model receives one scoring.',
                         'Training fit is reported without a gate. '+str(gate),
                         'Reference shortcut and break diagnostics, headroom captured and CV/development gaps are in the companion JSON.'])
        print('FINAL DEVELOPMENT '+json.dumps({'arm':arm,'gate':gate,'dev_minus_CV':gap}),flush=True)
        if not gate['passed']:
            stop(output,'failed '+arm+' development criterion;no further scoring/tuning/extension',gate);return
    write_once(output/'complete.json',{'status':'all_predeclared_criteria_passed','contract':contract,'scores':scores,'gates':gates,
               'CV_summaries':feature_cv,'PointNeXt_retained':True,'automatic_next_stage':False,'GATv2_executed':False,'GPU_executed':False})
    print('STAGE2 COMPLETE; no automatic next stage',flush=True)


def worker(action,output):
    c,scientific,data,pools=control();output.mkdir(parents=True,exist_ok=True)
    if (output/'STOP.json').exists():raise RuntimeError('Stage2 already stopped;review STOP.json,do not extend')
    check_manifest(output/'binding.json',{'config':fingerprint(c),'CPU_only':True})
    lock=output/'RUNNING.json'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'action':action,'config':fingerprint(c)},f)
    try:
        if action=='prepare':preparation(c,scientific,data,pools,output)
        else:
            prepared=e.read(output/'prepare/manifest.json')
            if prepared['status']!='completed' or prepared['config_fingerprint']!=fingerprint(c):raise ValueError('Incomplete or changed preparation')
            store=Store(ROOT,output,c,data);keys=[tuple(p['key']) for p in prepared['packets']]
            (output/'metric_cache').mkdir(exist_ok=True);metrics=metric_driver.Metrics(c,scientific,data,output)
            if action=='policy':policy(c,scientific,data,store,keys,output,metrics)
            else:
                if e.read(output/'policy/report.json')['status']!='completed':raise ValueError('Policy diagnostic must finish first')
                compare(c,scientific,data,store,keys,output,metrics)
        if 'torch' in sys.modules:raise RuntimeError('CPU Stage2 imported Torch')
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','policy','compare'),required=True)
    p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    c,_,_,_=control();output=e.output_path(a.output or ROOT/'tmp/stage2_trees_20261105')
    if a.check:print('Stage2 CPU source/runtime/input bindings verified; no data run/model fit');return
    if a.worker:worker(a.action,output);return
    if (output/'STOP.json').exists():raise RuntimeError('Prior STOP.json prohibits automatic extension')
    stage={'prepare':'prepare','policy':'policy','compare':'comparison'}[a.action]
    if (output/stage).exists():raise FileExistsError('Preserve prior stage outputs; no automatic retry')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');seconds=c['wall_seconds'][a.action]
    print(f'STAGE2 CPU ONLY;action={a.action};OUTPUT {output};cap={seconds}s;no GPU/GAT/automatic extension',flush=True)
    run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--action',a.action,'--output',str(output)],
        output.with_name(output.name+'_'+a.action+'_guard_'+suffix),seconds,{'stage':'Stage2 '+a.action,'config':fingerprint(c),'output':str(output)})


if __name__=='__main__':main()
