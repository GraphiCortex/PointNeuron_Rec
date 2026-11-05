"""CPU-only fixed gradient boosting, representative ordering and final gates."""
import hashlib
import numpy as np
from scipy.spatial import cKDTree


TREE_PARAMETERS={'learning_rate':.05,'max_iter':128,'max_leaf_nodes':15,
                 'min_samples_leaf':32,'l2_regularization':1.,'max_bins':127,
                 'early_stopping':False,'random_state':0,'categorical_features':None}


def representatives(xyz,source_flags,sato,probability):
    """Eligibility p>=.5,then SOURCE,SATO,SELECTOR order;inclusive radius2."""
    p=np.asarray(xyz,float);flags=np.asarray(source_flags);sato=np.asarray(sato,float);prob=np.asarray(probability,float)
    if (p.shape!=(len(prob),3) or sato.shape!=prob.shape or flags.shape!=prob.shape
            or not np.isfinite(prob).all() or not np.isfinite(sato).all()
            or np.any((prob<0)|(prob>1))):raise ValueError('Unaligned representative inputs')
    priority=np.where(flags&1,0,np.where(flags&2,1,np.where(flags&4,2,3)))
    ids=np.flatnonzero(prob>=.5)
    order=ids[np.lexsort((ids,p[ids,2],p[ids,1],p[ids,0],-prob[ids],-sato[ids],priority[ids]))]
    removed=np.zeros(len(p),bool);keep=np.zeros(len(p),bool);tree=cKDTree(p)
    for i in order:
        if removed[i]:continue
        keep[i]=True;near=np.asarray(tree.query_ball_point(p[i],2.+1e-10),int)
        near=near[np.linalg.norm(p[near]-p[i],axis=1)<=2.];removed[near]=True
    return keep


def sample_neuron_classes(records,cap,seed=0):
    """records: (case_id,stable row keys,X,y);equal mass per neuron/class."""
    by={}
    for cid,keys,x,y in records:
        y=np.asarray(y);known=y>=0
        if not np.isin(y,[-1,0,1]).all() or len(keys)!=len(y) or len(x)!=len(y) or np.isinf(x).any():raise ValueError('Invalid training rows')
        if np.any(known):by.setdefault(cid,[]).append((np.asarray(keys,dtype=str)[known],x[known],y[known]))
    xx=[];yy=[];ww=[];counts={}
    for cid,parts in sorted(by.items()):
        keys=np.concatenate([a for a,_,_ in parts]);x=np.concatenate([b for _,b,_ in parts]);y=np.concatenate([c for _,_,c in parts])
        groups=[];counts[cid]={}
        for label in (0,1):
            ids=np.flatnonzero(y==label)
            if not len(ids):counts[cid][str(label)]=0;continue
            ranks=np.array([int.from_bytes(hashlib.sha256(f'{seed}|{cid}|{keys[i]}'.encode()).digest()[:8],'little') for i in ids],dtype=np.uint64)
            ids=ids[np.argsort(ranks,kind='stable')[:cap]];groups.append(ids);counts[cid][str(label)]=len(ids)
        for ids in groups:
            xx.append(x[ids]);yy.append(y[ids]);ww.append(np.full(len(ids),1./(len(ids)*len(groups))))
    if not xx:raise ValueError('No known training labels')
    x=np.concatenate(xx).astype(np.float32);y=np.concatenate(yy).astype(np.int8);w=np.concatenate(ww)
    if set(y)!={0,1}:raise ValueError('No two-class training problem remains after masking')
    w*=len(w)/w.sum()
    return x,y,w,counts


def fit_tree(records,cap,parameters=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    x,y,w,counts=sample_neuron_classes(records,cap)
    model=HistGradientBoostingClassifier(**(parameters or TREE_PARAMETERS))
    with threadpool_limits(limits=4):model.fit(x,y,sample_weight=w)
    if model.n_iter_!=model.max_iter:raise RuntimeError('Unexpected early stopping')
    return model,{'sampled_rows':len(y),'sampled_counts_per_neuron':counts,'iterations':model.n_iter_,
                  'balanced_by':'equal neuron/class weight after deterministic class-capped sampling'}


def probability(model,x):
    from threadpoolctl import threadpool_limits
    if not len(x):return np.empty(0)
    with threadpool_limits(limits=4):return model.predict_proba(x)[:,list(model.classes_).index(1)]


def final_gate(candidate,baseline,pointnext_reference=None):
    """User's exact stated floors;wins strictly against matched classical."""
    segment=candidate['segments'];new=candidate['new_metrics'];checks={}
    a={n['case_id']:n for n in segment['neurons']};b={n['case_id']:n for n in baseline['segments']['neurons']}
    if set(a)!=set(b) or len(a)!=7:raise ValueError('Unmatched seven development neurons')
    for t,floor,rfloor in [('2.0',.5933,.5243),('6.0',.7414,.5972)]:
        wins={cid:a[cid]['tolerances'][t]['f1']>b[cid]['tolerances'][t]['f1'] for cid in a}
        checks['segment_'+t]=segment['macro'][t]['f1']>=floor
        checks['wins_'+t]=sum(wins.values())>=5
        checks['resampled_'+t]=new['macro_resampled'][t]['f1']>=max(rfloor,baseline['new_metrics']['macro_resampled'][t]['f1']+.05)
        checks['win_count_'+t]=sum(wins.values())
    checks['length_ratio']=new['pooled_length_ratio']<=1.25
    boolean=[v for k,v in checks.items() if not k.startswith('win_count')]
    if pointnext_reference is not None:
        checks['PointNeXt_gain_2']=segment['macro']['2.0']['f1']>=pointnext_reference['segments']['macro']['2.0']['f1']+.02
        boolean.append(checks['PointNeXt_gain_2'])
    return {'passed':bool(all(boolean)),'checks':checks,'actual_segment':segment['macro'],
            'actual_resampled':new['macro_resampled'],'pooled_length_ratio':new['pooled_length_ratio'],
            'no_automatic_extension':True}


def choose_contract(summaries):
    eligible=[k for k,v in summaries.items() if v['new_metrics']['pooled_length_ratio']<=1.25]
    if not eligible:return {'status':'stop','reason':'neither training-CV contract satisfies length<=1.25;no development model scored'}
    selected=min(eligible,key=lambda k:(-summaries[k]['segments']['macro']['2.0']['f1'],
                 -summaries[k]['segments']['macro']['6.0']['f1'],
                 -summaries[k]['new_metrics']['macro_resampled']['2.0']['f1'],k))
    return {'status':'selected_training_only','contract':selected,'criterion':'length<=1.25 then decoded segment F1_2,F1_6,resampled F1_2;stable contract-ID tie',
            'AP_compared_across_contracts':False}
