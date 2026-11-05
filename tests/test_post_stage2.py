"""Synthetic checks only. No retained neuron/image/benchmark packet is opened."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from pn2.evaluation.post_stage2 import graph,subset,chains,ablations,scores,intensity_ridges,weighted_offsets,compact_prune
from pn2.evaluation.editing_headroom import make_forests


def test_straight_tube_ridge_and_length_weighted_offset_shares():
    z,y,x=np.indices((17,17,17));raw=np.exp(-((y-8.)**2+(z-8.)**2)/2.)
    ridge=intensity_ridges(raw,{'minimum':0.,'maximum':1.})
    assert len(ridge)>0 and np.all(ridge[:,1:]==8)
    d=weighted_offsets([0.,1.5,2.5,4.,np.inf],[1,2,3,4,5])
    assert d['share_over']['1.0']==pytest.approx(14/15)
    assert d['share_over']['2.0']==pytest.approx(12/15)
    assert d['share_over']['3.0']==pytest.approx(9/15)
    assert d['q90'] is None and d['no_ridge_within_limit_share']==pytest.approx(5/15)
    json.dumps(d,allow_nan=False)


def test_flat_image_has_no_ridge_and_physical_search_censor_is_explicit():
    assert len(intensity_ridges(np.zeros((9,9,9)),{'minimum':0,'maximum':1}))==0
    assert len(intensity_ridges(np.ones((9,9,9)),{'minimum':0,'maximum':1}))==0
    d=weighted_offsets([.1,np.inf],[1,1],limit=.6,thresholds=(.5,1.,2.))
    assert d['share_over']['1.0'] is None and d['search_limit']==.6


def test_chains_spurs_isolates_and_nonadditive_ablations():
    g=graph([[0,0,0],[10,0,0],[20,0,0],[10,2,0],[50,50,50]],[[0,1],[1,2],[1,3]])
    paths,degree=chains(g);assert len(paths)==3 and degree[-1]==0
    variants,counts=ablations(g)
    assert counts['isolated_nodes']==1 and counts['spur_chains']==1
    assert len(variants['minus_isolates'].xyz)==4 and len(variants['minus_spurs'].xyz)==4
    assert len(variants['minus_all_three'].xyz)==3
    ref=graph([[0,0,0],[20,0,0]],[[0,1]])
    assert scores(ref,variants['minus_isolates'])['resampled']['tolerances']['2.0']['f1']>scores(ref,g)['resampled']['tolerances']['2.0']['f1']


def test_duplicate_parallel_path_flag_and_crossing_not_flagged():
    g=graph([[0,0,0],[20,0,0],[1,1,0],[19,1,0],[10,-10,0],[10,10,0]],[[0,1],[2,3],[4,5]])
    variants,c=ablations(g)
    assert c['duplicate_path_flags']==1
    assert len(variants['minus_duplicate_paths'].edges)==2


def test_compact_prune_feasible_budget_no_positions_or_edges_invented():
    ref=graph([[0,0,0],[12,0,0]],[[0,1]])
    g=graph([[0,0,0],[12,0,0],[0,1,0],[12,1,0],[0,10,0],[12,10,0]],[[0,1],[2,3],[4,5]])
    q,report=compact_prune(g,ref,12)
    assert scores(ref,q)['length_ratio']<=1.25
    assert scores(ref,q)['resampled']['tolerances']['2.0']['f1']==1
    assert report['globally_optimal'] is False
    assert set(map(tuple,q.xyz))<=set(map(tuple,g.xyz))


def test_editing_additions_are_anchored_and_never_classical_shortcuts():
    c=graph([[0,0,0],[4,0,0],[8,0,0]],[[0,1],[1,2]])
    p={'xyz':np.array([[0,0,0],[4,0,0],[8,0,0],[12,0,0],[30,0,0],[34,0,0]],float),
       'source_flags':np.array([1,1,1,2,2,4]),'edges':np.array([[0,1],[1,2],[0,2],[2,3],[4,5]])}
    ref=graph([[0,0,0],[12,0,0],[30,0,0],[34,0,0]],[[0,1],[2,3]])
    a,b,provenance=make_forests(c,p,ref,[-1,-1,-1],[40,1,1])
    assert len(a.edges)==2 and len(b.edges)==3
    assert b.xyz[:,0].max()==12 and not provenance['new_classical_classical_edges']
    assert scores(graph([[0,0,0],[12,0,0]],[[0,1]]),b)['segments']['tolerances']['2.0']['recall']==1
    detached,_=compact_prune(b,graph([[0,0,0],[12,0,0]],[[0,1]]),12,anchors=np.array([[99,99,99]]))
    assert len(detached.xyz)==0


def test_bounded_editing_zero_reference_and_no_candidate_edges():
    ref=graph([[0,0,0],[10,0,0]],[[0,1]])
    q,report=compact_prune(graph([],[]),ref,10)
    assert len(q.xyz)==0 and scores(ref,q)['length_ratio']==0


def test_zero_length_reference_edges_do_not_create_nan_or_change_matching():
    ref=graph([[0,0,0],[0,0,0],[10,0,0]],[[0,1],[1,2]])
    q,report=compact_prune(graph([[0,0,0],[10,0,0]],[[0,1]]),ref,10)
    assert scores(ref,q)['resampled']['tolerances']['2.0']['f1']==1


def test_image_review_renderer_is_only_visualization(tmp_path):
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('post_stage2_runner',scripts/'post_stage2_diagnostics.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    g=graph([[10,80,30],[150,80,30]],[[0,1]])
    raw=np.broadcast_to(np.arange(160,dtype=np.uint8)[None,None,:],(80,160,160)).copy();raw[30,80,:]=255
    runner.render_review(tmp_path,'toy',raw,g,g.xyz,{'C0':g,'C1':g},[])
    assert all((tmp_path/f).exists() for f in ['index.html','XY.png','XZ.png','YZ.png'])
    assert 'torch' not in sys.modules


def test_archive_identity_does_not_confuse_SLP_with_WED_or_duplicate_image(tmp_path):
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('post_stage2_runner',scripts/'post_stage2_diagnostics.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    c={'development_cases':['fly/SLP','human/in_house1'],'benchmark_tokens':{'fly/SLP':['SLP.v3dpbd'],'human/in_house1':['Cell1.v3dpbd']},
       'benchmark_ambiguous_basename_cases':['human/in_house1'],'benchmark_published_ids':{'human/in_house1':['71']}}
    assert runner.benchmark_member_case(c,'outputs/SLP.v3dpbd_APP2.swc')=='fly/SLP'
    assert runner.benchmark_member_case(c,'outputs/WED_SLP.v3dpbd_APP2.swc') is None
    assert runner.benchmark_member_case(c,'outputs/Cell1.v3dpbd_APP2.swc') is None
    assert runner.benchmark_member_case(c,'human/in_house1/Cell1.v3dpbd_APP2.swc')=='human/in_house1'
    assert runner.benchmark_member_case(c,'outputs/71/Cell1.v3dpbd_APP2.swc')=='human/in_house1'


def test_download_extracts_only_matched_contents_and_never_repairs_files(tmp_path,monkeypatch):
    import io,tarfile
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('post_stage2_runner',scripts/'post_stage2_diagnostics.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as tf:
        for name in ['outputs/SLP.v3dpbd_APP2.swc','outputs/WED_SLP.v3dpbd_APP2.swc','unrelated/secret.swc']:
            data=b'1 0 0 0 0 1 -1\n';info=tarfile.TarInfo(name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
    archive=payload.getvalue();monkeypatch.setattr(runner.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(archive))
    c={'benchmark_archive':{'url':'https://example.invalid/archive','size':len(archive)},'development_cases':['fly/SLP'],
       'benchmark_tokens':{'fly/SLP':['SLP.v3dpbd']},'benchmark_ambiguous_basename_cases':[],'benchmark_published_ids':{}}
    runner.benchmark_download(c,tmp_path);m=json.loads((tmp_path/'manifest.json').read_text())
    assert len(m['packets'])==1 and m['case_counts']=={'fly/SLP':1}
    assert m['packets'][0]['algorithm_variant']=='APP2.swc'
    assert len(list(tmp_path.glob('*.swc')))==1


def test_worker_keeps_original_STOP_and_uses_separate_output(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('post_stage2_runner',scripts/'post_stage2_diagnostics.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    stopped=tmp_path/'stage2';stopped.mkdir();stop=stopped/'STOP.json';stop.write_text('{"fixed":true}\n');before=stop.read_bytes()
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'control',lambda:({'stage2_root':'stage2'},None,None))
    monkeypatch.setattr(runner,'decomposition',lambda c,s,d,folder:(folder/'report.json').write_text('{}'))
    output=tmp_path/'diagnostics';runner.worker('decompose',output)
    assert stop.read_bytes()==before and not (stopped/'finished.json').exists()
    assert (output/'decompose/finished.json').exists() and not (output/'RUNNING.json').exists()
