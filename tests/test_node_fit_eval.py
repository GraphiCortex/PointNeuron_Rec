from types import SimpleNamespace
import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from pn2.training.node_fit_eval import evaluate


def test_fit_evaluation_keeps_unknown_grid_boxes_and_reads_profiles_only_after_forward(tmp_path):
    events=[]
    class Data:
        cases={'dev':{'partition':'development','alias':1,'source_group':'fixture','boxes':[{'tile_id':0},{'tile_id':1}]}}
        boxes={('dev',0):{'target':{'count':1}},('dev',1):{'target':{'count':0}}}
        def image(self,cid,tile,budget):
            events.append('image')
            return {'metadata':{'center_xyz':[0.,0.,0.]},'packet':{'local_xyz':np.zeros((4,3),np.float32),
                'attributes':np.zeros((4,4),np.float32),'native_xyz':np.zeros((4,3)),
                'raw_intensity':np.zeros(4,np.uint8)}}
        def reference(self,cid,tile):
            events.append('reference');n=int(tile==0)
            return {'target_xyz':np.zeros((n,3)),'target_roles':[frozenset(('leaf',))]*n,'reference_runs':[]}
    class Profiles:
        manifest={'cutoffs':{'width_q25':None,'contrast_q25':None}}
        def load(self,cid,tile):
            events.append('profiles');n=int(tile==0)
            return {'node_width':np.zeros(n),'node_width_valid':np.zeros(n,bool),'node_contrast':np.zeros(n),'node_contrast_valid':np.zeros(n,bool),
                'window_width':np.zeros(0),'window_width_valid':np.zeros(0,bool),'window_contrast':np.zeros(0),'window_contrast_valid':np.zeros(0,bool),
                'curve_xyz':np.empty((0,3)),'curve_weights':np.empty(0),'curve_window_index':np.empty(0,np.int64)}
    class Model(torch.nn.Module):
        def forward(self,batch):
            assert 'target' not in batch;events.append('forward')
            logits=torch.tensor([[[4.,-4.]]]).expand(1,512,2).clone();logits[0,0]=torch.tensor([-4.,4.])
            return SimpleNamespace(center_proposals=torch.zeros((1,512,3)),objectness_logits=logits)
    result=evaluate(Model(),Data(),Profiles(),2048,'cpu',['dev'],tmp_path/'eval',0)
    assert events==['image','forward','reference','profiles']*2
    assert result['summary']['partitions']['development']['tolerances']['2.0']['macro_precision']==.5
    profile=result['profile_summary']['neurons'][0]['tolerances']['2.0']['unresolved']
    assert profile['reference_nodes']==1 and profile['matched_nodes']==1
    assert (tmp_path/'eval/predictions.npz').exists()
