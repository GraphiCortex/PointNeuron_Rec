from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from pn2.evaluation.label_audit import precision_recall_area,label_oracle_gate
from pn2.evaluation.swc import Morphology
from pn2.inputs.node_cache import digest


def test_ap_ties_unknowns_perfect_reverse_and_no_positive():
    assert precision_recall_area([1,0],[1,1])['average_precision']==.5
    assert precision_recall_area([1,0],[1,1])['trapezoidal_auprc']==.75
    assert precision_recall_area([1,0,-1],[1,0,10])['average_precision']==1.
    assert precision_recall_area([1,0],[0,1])['average_precision']==.5
    assert precision_recall_area([0,0],[1,0])['average_precision'] is None


def test_gate_has_exact_three_percentage_point_rule():
    stage0=.9042320394860465;floor=stage0-.03
    assert label_oracle_gate(floor,stage0)['passed']
    assert not label_oracle_gate(floor-1e-8,stage0)['passed']


def test_fixed_features_are_image_only_and_use_endpoint_maximum():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import audit_candidate_labels as audit
    xyz=np.array([[0.,0,0],[4,0,0],[4,3,0]])
    f=audit.single_feature_scores(xyz,np.array([[0,1],[1,2]]),xyz[:2],np.ones((8,8,8),np.float32),1.)
    np.testing.assert_array_equal(f['sato_straight_path_cost'],[4,3])
    np.testing.assert_array_equal(f['max_endpoint_nearest_input_distance'],[0,3])


def test_real_packet_loop_masking_geometry_scorer_and_cache(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import audit_candidate_labels as audit
    m=Morphology(None,np.array([1,2]),np.zeros(2,int),np.array([[1.,1,1],[19,1,1]]),np.ones(2),np.array([-1,0]))
    xyz=np.array([[1.,1,1],[7,1,1],[13,1,1],[19,1,1]]);edges=np.array([[0,1],[1,2],[2,3],[0,3]])
    candidate=tmp_path/'candidate.npz';label=tmp_path/'labels.npz'
    np.savez_compressed(candidate,xyz=xyz,edges=edges,image_guide=np.ones(4))
    np.savez_compressed(label,node_label=np.ones(4,np.int8),edge_label=np.array([1,-1,1,0],np.int8))
    p={'key':['train',0],'path':str(candidate),'sha256':digest(candidate)}
    l={('train',0):{'path':str(label),'sha256':digest(label)}}
    monkeypatch.setattr(audit.e,'References',lambda c,d:SimpleNamespace(get=lambda cid:{'morphology':m,'distances':lambda p:np.zeros(len(p))}))
    def guide(d,k,p,g):g['train']=np.ones((24,24,24),np.float32)
    monkeypatch.setattr(audit.e,'sato_scores',guide)
    data=SimpleNamespace(cases={'train':{'alias':0}},boxes={('train',0):{'box':{'start_xyz':[0,0,0],'stop_xyz':[24,24,24]}}},
         image=lambda *a:{'packet':{'native_xyz':xyz}},reference=lambda *a:{'reference_runs':[m.xyz]})
    (tmp_path/'records').mkdir()
    rows=audit.run_packets({}, {},data,[p],l,tmp_path)
    row=rows[0]
    assert row['label_oracle']['decoded']['edges']==2
    assert row['geometric_oracle']['decoded']['edges']==3
    assert row['positive_edge_union']['tolerances']['2.0']['recall']<1
    assert row['fragment_audit']['both_positive_unknown']==1
    assert row['fragment_audit']['cross_component_negative']==0
    assert audit.run_packets({}, {},data,[p],l,tmp_path)==rows
    assert digest(candidate)==p['sha256']
