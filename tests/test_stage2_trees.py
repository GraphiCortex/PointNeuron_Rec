"""Synthetic CPU checks only: no retained neuron packet, image or model is opened."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.evaluation.swc import Morphology,parse_swc
from pn2.evaluation.bounded_agreement import BoundedAgreement
from pn2.evaluation.formulation import graph_segments
from pn2.evaluation.resampled_skeleton import resampled_scores,uniform_samples
from pn2.inputs.node_cache import digest
from pn2.selection.data import Store,atomic_npz
from pn2.selection.features import (NODE_NAMES,STATIC_EDGE_NAMES,local_geometry,
    intervening_counts,intensity_at,node_features,bounded_geodesic,static_edge_features,edge_matrix)
from pn2.selection.trees import (TREE_PARAMETERS,representatives,sample_neuron_classes,
    fit_tree,probability,final_gate,choose_contract)
from pn2.selection.diagnostics import reconstruction_risks
from pn2.supervision.candidate_labels import label_candidates


def morphology(xyz,parents):
    p=np.asarray(xyz,float).reshape(-1,3);n=len(p)
    return Morphology(None,np.arange(1,n+1),np.zeros(n,int),p,np.ones(n),np.array(parents))


@pytest.fixture
def runner():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('stage2_test_runner',scripts/'run_stage2_trees.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_representatives_source_sato_probability_and_inclusive_radius():
    p=np.array([[0,0,0],[1,0,0],[10,0,0],[11,0,0],[20,0,0],[22,0,0],[30,0,0]],float)
    keep=representatives(p,np.array([1,4,2,2,1,1,1]),
        np.array([.1,1,.8,.9,.5,.5,1]),np.array([.51,.99,.99,.51,.6,.9,.49]))
    assert np.flatnonzero(keep).tolist()==[0,3,5]
    assert np.all(np.linalg.norm(p[keep,None]-p[None,keep],axis=2)+np.eye(3)*99>2)


def test_geometry_and_all_candidate_counts():
    p=np.array([[0,0,0],[2,0,0],[4,0,0],[2,1,0],[0,1,0]],float)
    assert intervening_counts(p,[[0,2]]).tolist()==[2]
    g,t=local_geometry(p)
    assert g.shape==(5,8) and t.shape==(5,3)
    assert g[0,0]==1 and g[0,1]>0
    assert local_geometry(np.empty((0,3)))[0].shape==(0,8)


def test_intensity_interpolation_preserves_fractions_and_geodesic():
    raw=np.zeros((5,5,5),np.uint8);raw[:,:,1:]=1
    assert intensity_at(raw,[0,0,0],{'minimum':0,'maximum':1},[[.5,0,0]])[0]==.5
    guide=np.ones((8,8,8))
    value,flag=bounded_geodesic([1,1,1],[4,1,1],guide,1)
    assert value==pytest.approx(3) and flag==0
    value,flag=bounded_geodesic([1,1,1],[4,1,1],guide,1,max_cells=1)
    assert np.isnan(value) and flag==1
    assert bounded_geodesic([-1,0,0],[4,1,1],guide,1)[1]==1


def feature_fixture():
    p=np.array([[1,1,1],[4,1,1],[7,1,1]],float)
    pool={'xyz':p,'edges':np.array([[0,1],[1,2],[0,2]]),'source_flags':np.ones(3,int),'source_counts':np.ones((3,3),int)}
    raw=np.ones((10,10,10),np.uint8);source={'native_xyz':p,'attributes':np.column_stack((p,np.ones(3)))}
    node,tangent=node_features(pool,source,raw,[0,0,0],{'minimum':0,'maximum':1},raw,1,p,
            {'start_xyz':[0,0,0],'stop_xyz':[10,10,10]},[10,10,10])
    static=static_edge_features(pool,node,tangent,np.arange(3),raw,[0,0,0],{'minimum':0,'maximum':1},raw,1)
    return pool,node,static


def test_pointnext_only_adds_edge_channels_and_does_not_change_geometry():
    pool,node,static=feature_fixture();kept=np.array([True,False,True]);ids=np.array([2])
    hc=edge_matrix(pool,node,static[ids],kept,ids)
    pn=edge_matrix(pool,node,static[ids],kept,ids,np.arange(96).reshape(3,32))
    assert node.shape==(3,len(NODE_NAMES)) and static.shape==(3,len(STATIC_EDGE_NAMES))
    assert hc.shape==(1,63) and pn.shape==(1,127)
    np.testing.assert_array_equal(hc,pn[:,:63])
    assert hc[0,-2]==0 and static[2,14]==1 # retained versus full intervening count
    with pytest.raises(ValueError,match='suppressed'):edge_matrix(pool,node,static[:1],kept,np.array([0]))


def test_balancing_masks_caps_determinism_and_equal_neuron_weight():
    records=[('a',np.arange(8).astype(str),np.arange(16).reshape(8,2),np.array([0,0,0,0,0,1,1,-1])),
             ('b',np.arange(2).astype(str),np.ones((2,2)),np.array([1,1]))]
    x,y,w,counts=sample_neuron_classes(records,2)
    assert counts=={'a':{'0':2,'1':2},'b':{'0':0,'1':2}}
    assert set(y)=={0,1} and w[:4].sum()==pytest.approx(w[4:].sum())
    assert w[:2].sum()==pytest.approx(w[2:4].sum())
    for a,b in zip((x,y,w),sample_neuron_classes(records,2)[:3]):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='two-class'):sample_neuron_classes(records[1:],2)


def test_real_cpu_boosting_serialization_and_prediction(runner,tmp_path):
    x=np.linspace(-2,2,80,dtype=np.float32)[:,None];y=(x[:,0]>0).astype(np.int8)
    records=[('toy',np.arange(80).astype(str),x,y)]
    params={**TREE_PARAMETERS,'max_iter':4,'min_samples_leaf':2}
    model,receipt=fit_tree(records,100,params);p=probability(model,x)
    assert receipt['iterations']==4 and p[-1]>p[0]
    saved=runner.save_model(tmp_path/'toy.pkl',model,{'toy_only':True})
    np.testing.assert_array_equal(probability(runner.load_model(saved),x),p)
    (tmp_path/'toy.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):runner.load_model(saved)
    assert 'torch' not in sys.modules


def store_fixture(tmp_path):
    m=morphology([[0,0,0],[12,0,0]],[-1,0]);p=np.array([[0,0,0],[3,0,0],[6,0,0],[9,0,0]],float)
    edges=np.array([[0,1],[1,2],[0,2],[0,3]]);original,_=label_candidates(p,edges,m,partition='training')
    original['edge_label'][3]=-1
    store=Store.__new__(Store);store.output=tmp_path;store.config={'toy':True}
    store.data=SimpleNamespace(cases={'train':{'partition':'training'},'dev':{'partition':'development'}})
    store.labels={('train',0):{'sha256':'toy'}}
    store.get=lambda key:({'xyz':p,'edges':edges},{})
    store.training_labels=lambda key:original
    refs=SimpleNamespace(get=lambda key:{'morphology':m})
    return store,refs


def test_C1_recomputes_adjacency_only_on_training_and_preserves_unknown(tmp_path):
    store,refs=store_fixture(tmp_path);kept=np.array([True,False,True,True]);ids=np.array([2,3])
    assert store.targets(('train',0),kept,ids,'C0',refs).tolist()==[0,-1]
    assert store.targets(('train',0),kept,ids,'C1',refs).tolist()==[1,-1]
    # Cache replay gives identical labels without touching the original packets.
    assert store.targets(('train',0),kept,ids,'C1',refs).tolist()==[1,-1]
    with pytest.raises(ValueError,match='training-only'):store.targets(('dev',0),kept,ids,'C1',refs)


def test_store_development_label_guard_precedes_packet_access():
    store=Store.__new__(Store);store.data=SimpleNamespace(cases={'dev':{'partition':'development'}})
    with pytest.raises(ValueError,match='prohibited'):store.training_labels(('dev',0))


def result(segment=(.7,.8),resampled=(.6,.7),ratio=1.):
    return {'segments':{'macro':{t:{'f1':v} for t,v in zip(('2.0','6.0'),segment)},
            'neurons':[{'case_id':str(i),'tolerances':{t:{'f1':v} for t,v in zip(('2.0','6.0'),segment)}} for i in range(7)]},
            'new_metrics':{'pooled_length_ratio':ratio,'macro_resampled':{t:{'f1':v} for t,v in zip(('2.0','6.0'),resampled)}}}


@pytest.mark.parametrize('change',["segment2","segment6","resampled2","resampled6","length","wins2","wins6"])
def test_every_development_floor_can_stop(change):
    baseline=result((.5433,.6914),(.4743,.5472));candidate=result()
    assert final_gate(candidate,baseline)['passed']
    if change.startswith('segment'):candidate['segments']['macro']['2.0' if change.endswith('2') else '6.0']['f1']=0
    elif change.startswith('resampled'):candidate['new_metrics']['macro_resampled']['2.0' if change.endswith('2') else '6.0']['f1']=0
    elif change=='length':candidate['new_metrics']['pooled_length_ratio']=1.250001
    else:
        for n in candidate['segments']['neurons'][:3]:n['tolerances']['2.0' if change.endswith('2') else '6.0']['f1']=0
    assert not final_gate(candidate,baseline)['passed']


def test_pointnext_two_point_requirement_only_at_two_voxels():
    baseline=result((.5433,.6914),(.4743,.5472));hc=result();pn=result((.721,.8))
    assert final_gate(pn,baseline,hc)['passed']
    assert not final_gate(result((.719,.85)),baseline,hc)['passed']
    assert not final_gate(result((.75,.74)),baseline,hc)['passed']


def test_training_only_contract_selection_prioritizes_length():
    assert choose_contract({'C0':result((.9,.99),ratio=1.26),'C1':result()})['contract']=='C1'
    assert choose_contract({'C0':result(ratio=1.26),'C1':result(ratio=1.26)})['status']=='stop'
    assert choose_contract({'C1':result(),'C0':result()})['contract']=='C0'


def test_image_only_decode_and_matched_metrics(runner,tmp_path):
    pool={'xyz':np.array([[0,0,0],[3,0,0],[6,0,0]],float),'edges':np.array([[0,1],[1,2],[0,2]]),'image_guide':np.ones(3)}
    spec=runner.output_graph(pool,np.array([.9,.9,.9]),np.arange(3),np.array([.9,.9,.1]),tmp_path/'graph')
    graph=parse_swc(spec['path']);reference=morphology([[0,0,0],[6,0,0]],[-1,0])
    segments=BoundedAgreement(reference.segments).score(graph_segments(graph))
    assert segments['tolerances']['2.0']['f1']==1 and graph.total_length/reference.total_length==1
    assert resampled_scores(uniform_samples(reference),graph)['tolerances']['2.0']['f1']==1
    assert spec['sha256']==digest(Path(spec['path']))
    risks=reconstruction_risks(graph,reference,[reference.xyz])
    assert risks['edge_relation_counts']['same_chain']==2
    assert risks['breaks']['2.0']['uncovered_intervals']==0


def test_empty_feature_graph_and_risk_diagnostics(runner,tmp_path):
    p=np.empty((0,3));pool={'xyz':p,'edges':np.empty((0,2),int),'image_guide':np.empty(0)}
    spec=runner.output_graph(pool,np.empty(0),np.empty(0,int),np.empty(0),tmp_path/'empty')
    graph=parse_swc(spec['path']);reference=morphology([[0,0,0],[6,0,0]],[-1,0])
    risks=reconstruction_risks(graph,reference,[reference.xyz])
    assert risks['breaks']['2.0']['uncovered_reference_length']==pytest.approx(6)
    assert risks['edge_relation_counts']['shortcut_fraction_of_scorable'] is None


def test_stop_receipt_blocks_worker_before_any_fit(runner,tmp_path,monkeypatch):
    runner.stop(tmp_path,'toy failed criterion',{'passed':False})
    monkeypatch.setattr(runner,'control',lambda:({},None,None,None))
    with pytest.raises(RuntimeError,match='already stopped'):runner.worker('compare',tmp_path)
    assert not (tmp_path/'RUNNING.json').exists()


@pytest.mark.parametrize('fail_handcrafted',[True,False])
def test_orchestration_freezes_both_models_and_honors_development_stop(runner,tmp_path,monkeypatch,fail_handcrafted):
    """Exercise all33 mocked folds, not real data/models, including stop order."""
    import json
    keys=[(f'train{i%33}',i) for i in range(460)]+[(f'dev{i%7}',i) for i in range(151)]
    data=SimpleNamespace(cases={k[0]:{'partition':'training' if k[0].startswith('train') else 'development'} for k in keys})
    store=SimpleNamespace(cold_feature_seconds=0.)
    c={'node_cap_per_neuron_class':1,'edge_cap_per_neuron_class':1,
       'first_fold_projection_multiplier':50,'cold_feature_projection_multiplier':3,
       'final_reserve_seconds':1800,'wall_seconds':{'compare':21600},'baseline_metrics':'baseline.json'}
    (tmp_path/'policy/selection').mkdir(parents=True)
    runner.write_once(tmp_path/'policy/frozen.json',{'node_model':{'toy':True}})
    runner.write_once(tmp_path/'policy/selection/manifest.json',{'packets':[{'key':list(k)} for k in keys]})
    runner.write_once(tmp_path/'baseline.json',{'summaries':{'classical_pruned/development':result((.5433,.6914),(.4743,.5472))}})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.e,'References',lambda *a:None)
    monkeypatch.setattr(runner,'node_records',lambda store,ks:list(ks))
    def select(store,model,ks,folder):
        folder.mkdir(parents=True);packets=[{'key':list(k)} for k in ks]
        runner.write_once(folder/'manifest.json',{'packets':packets})
        return {tuple(p['key']):p for p in packets}
    monkeypatch.setattr(runner,'select_nodes',select)
    monkeypatch.setattr(runner,'edge_training_records',lambda store,selection,ks,contract,refs,pn:(list(ks),{}))
    monkeypatch.setattr(runner,'fit_tree',lambda records,cap:(SimpleNamespace(toy=True),{'toy':True}))
    monkeypatch.setattr(runner,'feature_cv_AP',lambda *args:{'toy':True})
    monkeypatch.setattr(runner,'risk_report',lambda *args:[])
    dev_calls=[]
    def predict(store,selection,ks,model,pn,folder,metrics,method):
        if method.startswith('DEVELOPMENT_'):
            assert (tmp_path/'final/FROZEN_BEFORE_DEVELOPMENT.json').is_file()
            assert (tmp_path/'final/handcrafted_edge.pkl').is_file() and (tmp_path/'final/PointNeXt_edge.pkl').is_file()
            dev_calls.append(method)
        return [{'method':method}]
    monkeypatch.setattr(runner,'predict_edges',predict)
    def summary(rows):
        method=rows[0]['method']
        if method=='DEVELOPMENT_handcrafted' and fail_handcrafted:return result((.5,.8))
        if method=='DEVELOPMENT_PointNeXt':return result((.73,.8))
        return result()
    monkeypatch.setattr(runner,'summarize',summary)
    runner.compare(c,None,data,store,keys,tmp_path,None)
    if fail_handcrafted:
        assert dev_calls==['DEVELOPMENT_handcrafted'] and (tmp_path/'STOP.json').exists()
        assert not (tmp_path/'final/PointNeXt_development_STARTED.json').exists()
    else:
        assert dev_calls==['DEVELOPMENT_handcrafted','DEVELOPMENT_PointNeXt']
        assert json.loads((tmp_path/'complete.json').read_text())['automatic_next_stage'] is False
