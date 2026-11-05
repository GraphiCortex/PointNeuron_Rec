from types import SimpleNamespace
import numpy as np

from pn2.evaluation.candidate_pool import merge,pool,knn,crop_classical,clipped_edges
from pn2.evaluation.selection_oracle import edge_agreement,headroom_gate
from pn2.evaluation.probability_forest import decode,prune_spurs
from pn2.evaluation.bounded_agreement import BoundedAgreement
from pn2.evaluation.formulation import graph_segments,segment_agreement
from pn2.evaluation.vote_scaffold import knn_edges,export_forest
from pn2.evaluation.swc import parse_swc


def test_merge_is_nontransitive_keeps_coordinates_flags_and_bounded_displacement():
    points=np.array([[0.,0,0],[.75,0,0],[1.5,0,0],[0,.5,0]])
    r=merge(points,[1,1,1,2])
    np.testing.assert_array_equal(r['xyz'],points[[0,2]])
    assert r['source_flags'].tolist()==[3,1]
    assert r['source_counts'].tolist()==[[2,1,0],[1,0,0]]
    assert np.all(np.linalg.norm(points-r['xyz'][r['input_to_merged']],axis=1)<=1)


def test_crop_keeps_original_halo_endpoints_of_a_crossing_edge():
    p=np.array([[-2.,0,0],[2,0,0],[0,0,0],[10,0,0]])
    xyz,edges,ids=crop_classical(p,[[0,1]],[-1,-1,-1],[1,1,1])
    assert ids.tolist()==[0,1,2] and edges.tolist()==[[0,1]]
    np.testing.assert_array_equal(xyz,p[ids])
    original,segments=clipped_edges(xyz,edges,[-1,-1,-1],[1,1,1])
    np.testing.assert_array_equal(segments,[[[-1,0,0],[1,0,0]]])


def test_fast_knn_exactly_matches_original_and_nested_with_ties():
    rng=np.random.default_rng(5)
    for points in [rng.normal(size=(31,3)),np.array([[0.,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])]:
        for k in (1,6,10):np.testing.assert_array_equal(knn(points,k),knn_edges(points,k))
        assert set(map(tuple,knn(points,6)))<=set(map(tuple,knn(points,10)))
    assert knn(np.empty((0,3)),6).shape==(0,2)


def test_pool_remaps_inherited_edges_without_welding_a_chain():
    r=pool({'classical':(np.array([[0.,0,0],[1,0,0],[2,0,0]]),[[0,1],[1,2]]),
            'sato':(np.array([[0,.5,0]]),np.empty((0,2),int))})
    assert len(r['xyz'])==2 and r['inherited_self_edges_dropped']==1
    assert r['source_flags'].tolist()==[3,1]
    assert r['edges'].tolist()==[[0,1]] and r['edge_flags'][0]&1


def test_edge_oracle_matches_individual_segment_scorer_and_exact_half_cutoff():
    reference=np.array([[[0.,0,0],[10,0,0]]]);agreement=BoundedAgreement(reference)
    xyz=np.array([[0.,0,0],[10,0,0],[0,3,0],[10,3,0],[5,0,0],[5,4,0],[-10,0,0],[-5,0,0]])
    edges=np.array([[0,1],[2,3],[4,5],[6,7]])
    p,length=edge_agreement(xyz,edges,[0,-1,-1],[10,10,1],agreement.index)
    np.testing.assert_array_equal(p,[1,0,.5,0]);assert length[-1]==0
    for i in range(3):
        expected=segment_agreement(reference,xyz[edges[i:i+1]])['tolerances']['2.0']['precision']
        assert p[i]==expected


def test_probability_forest_spans_eligible_components_instead_of_empty_minimum():
    p=np.array([[0.,0,0],[3,0,0],[6,0,0]])
    edges=np.array([[0,1],[1,2],[0,2]])
    r=decode(p,edges,np.ones(3),np.full(3,.9))
    assert len(r['edges'])==2 and len(r['selected_indices'])==3
    empty=decode(p,edges,np.zeros(3),np.full(3,.9))
    assert len(empty['selected_indices'])==len(empty['edges'])==0
    low=decode(p,edges,[1,.49,1],[.9,.9,.49])
    assert len(low['selected_indices'])==2 and len(low['edges'])==0


def test_spur_pruning_is_one_pass_preserves_short_whole_components():
    p=np.array([[0.,0,0],[1.5,0,0],[-3,0,0],[0,3,0],[10,0,0],[11,0,0]])
    kept,edges,length=prune_spurs(p,[[0,1],[0,2],[0,3],[4,5]])
    assert kept.tolist()==[0,2,3,4,5] and length==1.5
    assert len(edges)==3


def test_oracle_decode_swc_integration_is_geometry_only_with_image_root(tmp_path):
    source=np.array([[0.,0,0],[3,0,0],[6,0,0],[9,0,0],[5,8,0]])
    r=pool({'classical':(source[:4],[[0,1],[1,2],[2,3]]),'learned':(source[4:],np.empty((0,2),int))})
    ref=np.array([[[0.,0,0],[9,0,0]]]);agreement=BoundedAgreement(ref)
    prob,_=edge_agreement(r['xyz'],r['edges'],[-1,-1,-1],[10,10,1],agreement.index)
    node=(np.abs(r['xyz'][:,1])<=2).astype(float)
    decoded=decode(r['xyz'],r['edges'],node,prob);selected=decoded['selected_indices']
    guide=np.zeros(len(selected));guide[np.argmax(r['xyz'][selected,0])]=1
    export_forest(tmp_path/'oracle.swc',r['xyz'][selected],decoded['edges'],guide)
    swc=parse_swc(tmp_path/'oracle.swc')
    assert agreement.score(graph_segments(swc))['tolerances']['2.0']['f1']==1
    root=np.flatnonzero(swc.parents<0)
    assert swc.xyz[root,0].tolist()==[9.]  # image-root score, not reference root at0


def test_fixed_headroom_gate_boundaries():
    assert headroom_gate(.8)['status']=='proceed'
    assert headroom_gate(.79999)['status']=='conditional_candidate_recall_work'
    assert headroom_gate(.65)['status']=='conditional_candidate_recall_work'
    assert headroom_gate(.64999)['status']=='stop'


def test_actual_oracle_packet_loop_json_keys_cache_and_source_immutability(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from pn2.evaluation.swc import Morphology
    from pn2.inputs.node_cache import digest
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('candidate_headroom_fixture',scripts/'measure_candidate_headroom.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    morphology=Morphology(None,np.array([1,2]),np.zeros(2,int),np.array([[0.,0,0],[12,0,0]]),np.ones(2),np.array([-1,0]))
    monkeypatch.setattr(module.e,'References',lambda c,d:SimpleNamespace(get=lambda cid:{'morphology':morphology}))
    data=SimpleNamespace(boxes={('case',1):{'box':{'start_xyz':[0,0,0],'stop_xyz':[24,24,24]}}},
                         reference=lambda cid,tile:{'reference_runs':[morphology.xyz]})
    source={'classical':(np.array([[0.,0,0],[4,0,0],[12,0,0]]),[[0,1],[1,2]]),
            'sato':(np.array([[3.,0,0],[6,0,0],[9,0,0]]),[[0,1],[1,2]]),
            'learned':(np.array([[5.,8,0]]),np.empty((0,2),int))}
    packets=[];(tmp_path/'oracle_records').mkdir()
    for arm,names in module.ARMS.items():
        arrays=pool({name:source[name] for name in names});arrays['image_guide']=np.ones(len(arrays['xyz']))
        p=tmp_path/(arm+'.npz');np.savez_compressed(p,**arrays)
        packets.append({'arm':arm,'key':['case',1],'path':p.name,'sha256':digest(p),
                        'sizes':{'input_nodes':len(arrays['input_xyz']),'nodes':len(arrays['xyz']),
                                 'edges':len(arrays['edges']),'halo_nodes':0}})
    result=module.oracle({'fixture':True},{},data,packets,tmp_path)
    assert len(result['rows'])==4 and result['gate']['status']=='proceed'
    for packet in packets:assert digest(tmp_path/packet['path'])==packet['sha256']
    def unexpected(cid):raise AssertionError('Complete oracle cache must not reopen geometry')
    monkeypatch.setattr(module.e,'References',lambda c,d:SimpleNamespace(get=unexpected))
    replay=module.oracle({'fixture':True},{},data,packets,tmp_path)
    assert replay==result
