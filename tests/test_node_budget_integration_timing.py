"""Inference cannot consume labels; incomplete/failed timing cannot become success."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import node_budget_integration_timing as timing


def test_full_grid_evaluator_forwards_before_reading_targets_and_keeps_unknown_outputs(tmp_path):
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
            events.append('reference');n=1 if tile==0 else 0
            return {'target_xyz':np.zeros((n,3)),'target_roles':[frozenset(('leaf',))]*n,'reference_runs':[]}
    class Model(torch.nn.Module):
        def forward(self,batch):
            events.append('forward');assert 'target' not in batch
            centers=torch.zeros((1,512,3));logits=torch.tensor([[[4.,-4.]]]).expand(1,512,2).clone()
            logits[0,0]=torch.tensor([-4.,4.])
            return SimpleNamespace(center_proposals=centers,objectness_logits=logits)
    result=timing.evaluate(Model(),Data(),2048,'cpu',tmp_path)
    assert result['boxes']==2 and events==['image','forward','reference']*2
    metrics=json.loads((tmp_path/'core_metrics.json').read_text())['summary']
    assert metrics['partitions']['development']['unknown_box_accepted']==1
    assert metrics['partitions']['development']['tolerances']['2.0']['macro_precision']==.5


@pytest.mark.parametrize('worker_status',['timeout','completed'])
def test_timeout_or_missing_worker_report_stops_without_second_worker(monkeypatch,tmp_path,worker_status):
    calls=[]
    def launch(*args):calls.append(args);return -1 if worker_status=='timeout' else 0,worker_status
    monkeypatch.setattr(timing,'launch_worker',launch)
    monkeypatch.setattr(timing,'identity',lambda *a:{'fixture':True})
    control={'budgets':[2048,8192],'worker_timeout_seconds':{'2048':900,'8192':1800}}
    args=SimpleNamespace(output=str(tmp_path/'run'),config='fixture.json')
    assert timing.controller(control,{},None,args)==1 and len(calls)==1
    result=json.loads((Path(args.output)/'report.json').read_text())
    assert result['automatic_retry'] is False and result['training_action_exists'] is False
