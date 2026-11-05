import numpy as np
from pn2.evaluation.edge_label_geometry import skip_types,overlap_summary,metric_score,physical_edge_overlap


def test_skip_is_interior_close_and_same_reference_chain():
    xyz=np.array([[0.,0,0],[10,0,0],[5,.9,0],[0,5,0],[0,2.5,0]])
    node=np.ones(5,int);member={0:{0:0.},1:{0:10.},2:{0:5.},3:{1:5.},4:{1:2.5}}
    kind,geo,witness=skip_types(xyz,[[0,1],[0,3]],node,member)
    assert kind.tolist()==[0,2] and geo.tolist()==[True,True] and witness.tolist()==[2,-1]
    xyz[2,1]=1.001
    kind,geo,_=skip_types(xyz,[[0,1]],node,member)
    assert kind.tolist()==[1] and geo.tolist()==[False]


def test_summary_boundaries_weighting_and_no_in_box_length():
    r=overlap_summary([0,.25,.5,.9,1,0],[1,1,2,1,1,0])
    assert r['edges']==6 and r['in_box_edges']==5 and r['no_in_box_length']==1
    assert r['overlap_ge_half']==3 and r['overlap_lt_half']==2
    assert r['length_weighted_overlap']==(0+.25+1+.9+1)/6
    assert sum(r['bins'].values())==5


def test_micron_transform_changes_distances_not_candidate_geometry():
    reference=np.array([[[0.,0,0],[10,0,0]]]);prediction=reference+np.array([0,0,1])
    spacing=np.array([.1,.1,2.])
    result=metric_score(reference*spacing,prediction*spacing)
    assert result['tolerances']['1.0']['f1']==0 and result['tolerances']['2.0']['f1']==1
    assert result['length_ratio']==1 and abs(result['reference_length']-1)<1e-12
    xyz=prediction[0]
    score,length=physical_edge_overlap(xyz,[[0,1]],[-1,-1,-1],[11,1,2],reference,spacing)
    assert score['1.0'][0]==0 and score['2.0'][0]==1 and abs(length[0]-1)<1e-12
    np.testing.assert_array_equal(xyz,prediction[0])


def test_packet_audit_and_two_unadopted_drafts_preserve_original_labels(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from pn2.evaluation.swc import Morphology
    from pn2.inputs.node_cache import digest
    from pn2.evaluation.bounded_agreement import BoundedAgreement
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import audit_edge_label_geometry as audit
    xyz=np.array([[0.,0,0],[3,0,0],[6,0,0],[9,0,0]]);edges=np.array([[0,1],[1,2],[2,3],[0,3]])
    m=Morphology(None,np.array([1,2]),np.zeros(2,int),xyz[[0,3]],np.ones(2),np.array([-1,0]))
    monkeypatch.setattr(audit.e,'References',lambda c,d:SimpleNamespace(get=lambda cid:{'morphology':m}))
    packet=tmp_path/'candidate.npz';lab=tmp_path/'label.npz'
    np.savez_compressed(packet,xyz=xyz,edges=edges,image_guide=np.ones(4))
    np.savez_compressed(lab,node_label=np.ones(4,np.int8),edge_label=np.array([1,1,1,0],np.int8),
                        edge_reason=np.array([0,0,0,7],np.int8),junction_anchor_vertex=np.full(4,-1,int))
    packets=[{'key':['train',0],'path':str(packet),'sha256':digest(packet)}]
    labels={('train',0):{'path':str(lab),'sha256':digest(lab)}}
    data=SimpleNamespace(cases={'train':{'alias':0}},boxes={('train',0):{'box':{'start_xyz':[-1,-1,-1],'stop_xyz':[12,12,12]}}},
                         reference=lambda *a:{'reference_runs':[m.xyz]})
    original=audit.prior.decoded_score(xyz,edges,np.ones(4),[1,1,1,0],np.ones(4),BoundedAgreement(m.segments),[-1,-1,-1],[12,12,12],tmp_path/'original')
    old={('train',0):{'label_oracle':original,'geometric_oracle':original}}
    out=tmp_path/'out';(out/'records').mkdir(parents=True)
    rows=audit.audit_packets({'spacing_metadata':{}},{},data,packets,labels,old,out)
    summary=audit.summarize_edges(rows)
    assert summary['pooled']['negative']['edges']==1 and summary['pooled']['negative_on_trace_share_all_edges']==1
    assert summary['pooled']['negative_categories']['same_chain_skip_within_1']['edges']==1
    drafts=audit.draft_oracles({}, {},data,packets,labels,rows,old,out)
    assert not drafts['adopted'] and not drafts['training_labels_written']
    assert drafts['rows'][0]['mask_conflicts']==1
    assert drafts['summaries']['mask_only']['macro']['2.0']['f1']==1
    assert drafts['summaries']['geometric_known']['macro']['2.0']['f1']==1
    # F1 can stay perfect despite a longer tree through redundant on-curve edges.
    a=drafts['summaries']['mask_only']['neurons'][0];b=drafts['summaries']['geometric_known']['neurons'][0]
    assert b['predicted_length']>a['predicted_length']
    assert digest(lab)==labels[('train',0)]['sha256']
