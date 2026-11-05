from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

from pn2.inputs.candidate_features import interpolate
from pn2.inputs.node_cache import digest,write_once
from pn2.evaluation.swc import Morphology
from pn2.evaluation.candidate_pool import pool

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_candidate_selection as prep


def test_interpolation_exact_inverse_distance_ties_empty_and_extrapolation():
    p=np.array([[0.,0,0],[2,0,0],[0,2,0],[0,-2,0]])
    f=np.arange(8).reshape(4,2)
    a=interpolate(p,f,[[0,0,0],[1,0,0],[100,0,0]])
    np.testing.assert_array_equal(a['candidate_features'][0],f[0])
    assert a['neighbor_indices'][0].tolist()==[0,1,2]
    assert a['neighbor_indices'][1].tolist()==[0,1,2]
    w=np.array([1.,1.,1/np.sqrt(5)]);w/=w.sum()
    np.testing.assert_allclose(a['candidate_features'][1],w@f[:3])
    assert a['neighbor_distances'][2,0]==98
    assert interpolate(p,f,[])['candidate_features'].shape==(0,2)
    with pytest.raises(ValueError):interpolate([],[],[[0,0,0]])


def test_training_labels_loop_never_opens_development_and_replays_completed_cache(tmp_path,monkeypatch):
    (tmp_path/'label_receipts').mkdir()
    m=Morphology(None,np.array([1,2]),np.zeros(2,int),np.array([[0.,0,0],[10,0,0]]),np.ones(2),np.array([-1,0]))
    requests=[]
    def reference(cid):
        requests.append(cid)
        assert cid=='train'
        return {'morphology':m}
    monkeypatch.setattr(prep.e,'References',lambda c,d:SimpleNamespace(get=reference))
    data=SimpleNamespace(cases={'train':{'alias':1}})
    a=pool({'classical':(np.array([[0.,0,0],[5,0,0],[10,0,0]]),[[0,1],[1,2]])})
    path=tmp_path/'candidate.npz';np.savez_compressed(path,**a)
    packet={'key':['train',0],'partition':'training','path':str(path),'sha256':digest(path)}
    # A development path that does not exist: it must never be loaded here.
    development={'key':['development',0],'partition':'development','path':'does_not_exist'}
    rows,neurons=prep.generate_labels({}, {},data,[packet,development],tmp_path)
    assert requests==['train'] and len(rows)==len(neurons)==1
    assert neurons[0]['counts']['edge']['positive']==2
    assert digest(path)==packet['sha256']
    requests.clear()
    assert prep.generate_labels({}, {},data,[packet,development],tmp_path)==(rows,neurons)
    assert not requests
    manifest=prep.e.read(tmp_path/'label_manifest.json')
    assert not manifest['development_labels_created'] and len(manifest['packets'])==1
    # A changed candidate cannot silently reuse stale labels.
    with pytest.raises(ValueError,match='different source/config'):
        prep.generate_labels({}, {},data,[{**packet,'sha256':'changed'}],tmp_path)


def test_new_training_pool_constructed_without_reference_access(tmp_path,monkeypatch):
    guide=np.ones((32,32,32),np.float32)
    key=('train',0);old=tmp_path/'old';old.mkdir()
    write_once(old/'candidate_manifest.json',{'packets':[]})
    source=np.array([[0.,0,0],[3,0,0],[6,0,0]])
    classical=tmp_path/'classical.npz';np.savez_compressed(classical,pruned_xyz=source,pruned_edges=[[0,1],[1,2]])
    broad=tmp_path/'broad';broad.mkdir();np.savez_compressed(broad/'graph.npz',xyz=source,forest_edges=[[0,1],[1,2]])
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    monkeypatch.setattr(prep,'saved_inventory',lambda c,k:{key:{'folder':str(broad),'inference':{'graph_sha256':digest(broad/'graph.npz')}}})
    def reuse_classical(d,c,t,o,g):g['train']=guide;return {'train':{'path':str(classical),'sha256':digest(classical)}}
    monkeypatch.setattr(prep,'classical',reuse_classical)
    monkeypatch.setattr(prep.e,'sato_scores',lambda d,k,p,g:np.ones(512))
    monkeypatch.setattr(prep.e,'consolidate',lambda *a,**kw:np.arange(512))
    # Stub the already-tested original Sato forest builder, retaining its I/O.
    def forest(points,g,m,folder,k,p,ids):
        folder.mkdir(parents=True);np.savez_compressed(folder/'graph.npz',xyz=points,forest_edges=np.empty((0,2),int))
        return {'image_only':True}
    monkeypatch.setattr(prep,'build',forest)
    def forbidden(*a,**kw):raise AssertionError('Construction opened references')
    monkeypatch.setattr(prep.e,'References',forbidden)
    native=np.array(np.unravel_index(np.arange(512),(8,8,8))).T.astype(float)*3
    data=SimpleNamespace(cases={'train':{'partition':'training'}},boxes={key:{'box':{'start_xyz':[0,0,0],'stop_xyz':[32,32,32]}}},
                         image=lambda *a:{'packet':{'native_xyz':native,'flat_indices':np.arange(512)}})
    (tmp_path/'candidate_receipts').mkdir()
    c={'headroom_root':'old'}
    packets=prep.construct(c,data,[key],[],tmp_path)
    assert len(packets)==1 and not packets[0]['references_used_for_construction']
    assert prep.construct(c,data,[key],[],tmp_path)==packets
    with np.load(packets[0]['path']) as a:
        assert all('label' not in name and 'oracle' not in name for name in a.files)
