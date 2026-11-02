"""Training controls, strict resume metadata and terminal evaluation recovery."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import node_budget_fit as fit
from pn2.training import node_fit_state as state


def test_resume_rejects_missing_adam_history_even_with_valid_file_marker(tmp_path):
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters())
    binding={'maximum_updates':9200,'budget':2048}
    path=tmp_path/'bad.pt';state.save(path,model,opt,binding,460)
    with pytest.raises(ValueError,match='Adam moments'):fit.resume_payload(path,binding)


def test_quality_failure_does_not_skip_the_other_prespecified_arm(monkeypatch,tmp_path):
    monkeypatch.setattr(fit,'identity',lambda *a:{'fixture':True})
    calls=[]
    def launch(command,folder,timeout):
        budget=int(folder.name.removeprefix('points'));calls.append(budget)
        m={'macro_precision':.1,'macro_recall':.2,'macro_f1':.13,'minimum_neuron_f1':0.}
        report={'status':'completed','budget':budget,'total_updates':9200,'prefix_verified':True,'fixture':True,
            'screening':{'broader_training_gate':False,'development_pilot_gate':False},
            'evaluations':[{'step':9200,'summary':{'partitions':{'development':{'tolerances':{'2.0':m}}}}}]}
        (folder/'worker_report.json').write_text(json.dumps(report));return 0,'completed'
    monkeypatch.setattr(fit,'launch_worker',launch)
    args=SimpleNamespace(output=str(tmp_path/'run'),budget=None,checkpoint=None,config='fixture.json')
    assert fit.controller({'worker_timeout_seconds':{'2048':1,'8192':1}},{},None,None,None,args)==0
    assert calls==[2048,8192]
    report=json.loads((Path(args.output)/'report.json').read_text())
    assert not report['automatic_retry'] and report['paired_delta']['8192_minus_2048']['macro_f1']==0.


def test_complete_checkpoint_recovery_scores_terminal_state_without_any_new_update(monkeypatch,tmp_path):
    import pn2.training.node_fit_eval as evaluation
    base=torch.nn.Linear(1,1);original_opt=torch.optim.AdamW(base.parameters(),lr=.001)
    monkeypatch.setattr(fit,'identity',lambda *a:{'fixture':True})
    monkeypatch.setattr(fit.common,'runtime',lambda *a:torch)
    monkeypatch.setattr(fit.geometric,'make_model',lambda *a:copy.deepcopy(base))
    monkeypatch.setattr(fit.common,'make_optimizer',lambda m,*a:torch.optim.AdamW(m.parameters(),lr=.001))
    monkeypatch.setattr(fit.common,'update',lambda *a:pytest.fail('completed fit must not update'))
    for name in ('reset_peak_memory_stats',):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU test fixture')
    data=SimpleNamespace(cases={'train':{'partition':'training'},'dev':{'partition':'development'}},
        schedule=lambda:(['train'],np.zeros((9200,3),np.int32)))
    c={'runtime_versions':{}};expected=fit.bindings(c,{},data,2048)
    payload=state.capture(base,original_opt,expected,9200);source=tmp_path/'source.pt';source.write_bytes(b'fixture')
    monkeypatch.setattr(fit,'resume_payload',lambda *a:(payload,source))
    calls=[]
    def assess(model,data,profiles,budget,device,selected,folder,step):
        calls.append((list(selected),step))
        if selected==['train']:
            summary={'neurons':[{'tolerances':{'2.0':{'reference_present_only':{'precision':0.,'recall':0.}}}}],
                'partitions':{'training':{}}}
        else:
            summary={'partitions':{'development':{'tolerances':{'2.0':{'macro_precision':0.,'macro_recall':0.,'minimum_neuron_f1':0.}}}}}
        return {'step':step,'summary':summary}
    monkeypatch.setattr(evaluation,'evaluate',assess)
    output=tmp_path/'worker';output.mkdir()
    args=SimpleNamespace(budget=2048,output=str(output),checkpoint=str(source))
    initial=fit.common.state_digest(base.state_dict());prior={'arms':[{'budget':2048,'initial_state_sha256':initial}]}
    fit.worker(c,{},data,None,prior,args)
    report=json.loads((output/'worker_report.json').read_text())
    assert report['updates_this_run']==0 and report['total_updates']==9200
    assert calls==[(['dev'],9200),(['train'],9200)]
