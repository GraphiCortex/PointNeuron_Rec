"""Synthetic CPU forward only; never executes the real encoder/checkpoint."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import export_candidate_pointnext as export
from pn2.inputs.node_cache import digest,write_once


def test_export_worker_image_only_unchanged_model_rng_and_resume(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(32))
        def forward(self,batch):
            assert not self.training and not torch.is_grad_enabled()
            return SimpleNamespace(features=batch['points'][...,:1]*self.weight)
    model=Toy();checkpoint=tmp_path/'checkpoint.pt';torch.save({'model':model.state_dict()},checkpoint)
    monkeypatch.setattr(export,'ROOT',tmp_path)
    monkeypatch.setattr(export.e,'runtime',lambda c,d:None)
    monkeypatch.setattr(export.e,'model_factory',lambda c,a,d:model)
    stage1=tmp_path/'stage1';stage1.mkdir();write_once(stage1/'candidate_manifest.json',{'fixture':True})
    source=np.array([[0.,0,0],[4,0,0],[8,0,0]])
    path=tmp_path/'candidate.npz';np.savez_compressed(path,xyz=source[[0,2]])
    packets=[{'key':['train',0],'partition':'training','path':str(path),'sha256':digest(path)},
             {'key':['dev',1],'partition':'development','path':str(path),'sha256':digest(path)}]
    requests=[]
    def image(cid,tile,budget):
        requests.append((cid,tile))
        return {'metadata':{'center_xyz':[0,0,0]},'packet':{'native_xyz':source,'local_xyz':source.astype(np.float32),
                  'attributes':np.zeros((3,4),np.float32),'raw_intensity':np.ones(3),'flat_indices':np.arange(3)}}
    def forbidden(*args):raise AssertionError('Export opened labels/references')
    data=SimpleNamespace(image=image,reference=forbidden,training=forbidden,
                         boxes={('train',0):{'clouds':{'2048':{'sha256':'source0'}}},
                                ('dev',1):{'clouds':{'2048':{'sha256':'source1'}}}})
    c={'checkpoint':'checkpoint.pt'}
    monkeypatch.setattr(export,'control',lambda stage1:(c,{},data,packets))
    output=tmp_path/'export';export.worker(stage1,output,'cpu',False)
    done=export.e.read(output/'complete.json')
    assert len(done['packets'])==2 and done['updates']==0 and done['model_state_unchanged'] and done['RNG_unchanged']
    assert not done['reference_geometry_opened'] and not done['optimizer_created']
    assert requests==[('train',0),('dev',1)]
    requests.clear();export.worker(stage1,output,'cpu',True)
    assert not requests
    for packet in done['packets']:
        with np.load(packet['path']) as a:
            np.testing.assert_array_equal(a['candidate_features'][:,0],[0,8])
