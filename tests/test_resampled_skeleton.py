from types import SimpleNamespace
import numpy as np
from pn2.evaluation.resampled_skeleton import clip_graph,uniform_samples,resampled_scores,greedy_nms


def graph(points,edges):return SimpleNamespace(xyz=np.asarray(points,float).reshape(-1,3),edges=np.asarray(edges,int).reshape(-1,2))


def sorted_points(p):return p[np.lexsort((p[:,2],p[:,1],p[:,0]))]


def test_uniform_spacing_endpoint_and_subdivision_orientation_invariance():
    a=graph([[0,0,0],[9,0,0]],[[0,1]])
    b=graph([[9,0,0],[3.2,0,0],[0,0,0]],[[0,1],[1,2]])
    p=uniform_samples(a);q=uniform_samples(b)
    np.testing.assert_allclose(sorted_points(p),sorted_points(q))
    assert sorted(p[:,0])==[0,2,4,6,8,9]


def test_junction_once_and_disconnected_coincidences_stay_distinct():
    y=graph([[0,0,0],[4,0,0],[-4,0,0],[0,4,0]],[[0,1],[0,2],[0,3]])
    p=uniform_samples(y);assert len(p)==7 and np.count_nonzero(np.all(p==0,axis=1))==1
    duplicates=graph([[0,0,0],[4,0,0],[0,0,0],[4,0,0]],[[0,1],[2,3]])
    ref=uniform_samples(graph([[0,0,0],[4,0,0]],[[0,1]]))
    r=resampled_scores(ref,duplicates)
    assert r['tolerances']['2.0']['matched']==3 and r['tolerances']['2.0']['predicted']==6
    assert r['tolerances']['2.0']['f1']==2/3


def test_cycle_zero_edge_and_isolate_have_defined_sampling():
    loop=graph([[0,0,0],[2,0,0],[2,2,0],[0,2,0]],[[0,1],[1,2],[2,3],[3,0]])
    assert len(uniform_samples(loop))==4
    z=graph([[0,0,0],[0,0,0],[4,0,0],[20,0,0]],[[0,1],[1,2]])
    assert sorted(uniform_samples(z)[:,0])==[0,2,4,20]


def test_clip_preserves_shared_junction_and_does_not_weld_fragments():
    g=graph([[-2,0,0],[0,0,0],[4,0,0],[0,4,0],[0,0,0],[0,-4,0]],[[0,1],[1,2],[1,3],[4,5]])
    clipped=clip_graph(g.xyz,g.edges,[-1,-1,-1],[3,3,1])
    p=uniform_samples(clipped)
    assert np.count_nonzero(np.all(p==0,axis=1))==2
    assert np.all((clipped.xyz>=[-1,-1,-1])&(clipped.xyz<=[3,3,1]))


def test_nms_probability_then_source_priority_inclusive_radius_no_transitivity():
    xyz=np.array([[0.,0,0],[2,0,0],[4,0,0],[10,0,0],[11,0,0],[20,0,0]])
    keep=greedy_nms(xyz,[1,1,1,.8,.9,.49],[4,1,4,1,4,1])
    # Classical at2 suppresses both0 and4; higher probability beats source at10.
    assert np.flatnonzero(keep).tolist()==[1,4]
    keep=greedy_nms(xyz[:3],[1,1,1],[1,4,4])
    assert np.flatnonzero(keep).tolist()==[0,2]


def test_metric_cache_retains_geometry_and_penalizes_duplicate_skeleton(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from pn2.inputs.node_cache import digest,write_once
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import audit_resampled_compact as audit
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    reference=tmp_path/'reference.swc';reference.write_text('1 0 0 0 0 1 -1\n2 0 10 0 0 1 1\n')
    predicted=tmp_path/'prediction.swc';predicted.write_text('1 0 0 0 0 1 -1\n2 0 10 0 0 1 1\n3 0 0 1 0 1 -1\n4 0 10 1 0 1 3\n')
    write_once(tmp_path/'refs.json',{'cases':[{'case_id':'case','reference_swc':'reference.swc','reference_sha256':digest(reference)}]})
    data=SimpleNamespace(boxes={('case',0):{'box':{'start_xyz':[-1,-1,-1],'stop_xyz':[12,12,12]}}},
                         reference=lambda *a:{'reference_runs':[np.array([[0.,0,0],[10,0,0]])]})
    output=tmp_path/'output';(output/'metric_cache').mkdir(parents=True)
    metric=audit.Metrics({}, {'reference_manifest':'refs.json'},data,output)
    job={'method':'fixture','key':['case',0],'graph':audit.graph_spec(predicted,digest(predicted)),'segments':None}
    a=metric.measure(job);b=metric.measure({**job,'method':'same_geometry_alias'})
    assert a['segments']['tolerances']['2.0']['f1']==1 and a['length_ratio']==2
    assert a['resampled']['tolerances']['2.0']['f1']==2/3
    assert b['metric_cache_reused'] and a['resampled']==b['resampled']


def test_nms_does_not_relabel_consecutive_edges_or_add_missing_links():
    from pn2.evaluation.probability_forest import decode
    p=np.array([[0.,0,0],[2,0,0],[4,0,0],[6,0,0]])
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    keep=greedy_nms(p,np.ones(4),np.ones(4,int))
    assert np.flatnonzero(keep).tolist()==[0,2]
    old_labels=np.array([1,0,0,1,0,1])
    result=decode(p,edges,keep.astype(float),old_labels)
    assert len(result['selected_indices'])==2 and len(result['edges'])==0
