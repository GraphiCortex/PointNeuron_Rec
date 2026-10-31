"""No relaxed parity or silent success for the execution-only block comparison."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import node_guide_block_compare as compare


def test_guide_comparison_requires_identical_bytes_even_for_tiny_value_change(tmp_path):
    a=tmp_path/'a.npy';b=tmp_path/'b.npy'
    values=np.ones((2,3,4),dtype=np.float32);np.save(a,values);np.save(b,values)
    compare.compare_guide(a,b)
    values[0,0,0]=np.nextafter(np.float32(1),np.float32(2));np.save(b,values)
    with pytest.raises(ValueError,match='byte-identical'):compare.compare_guide(a,b)


def test_cpu_report_from_other_execution_is_rejected(monkeypatch,tmp_path):
    path=tmp_path/'report.json';path.write_text(json.dumps({'status':'passed','fixtures':[{},{}],'comparison_fingerprint':'old'}))
    monkeypatch.setattr(compare,'ROOT',tmp_path)
    monkeypatch.setattr(compare,'identity',lambda *a:{'comparison_fingerprint':'new'})
    with pytest.raises(ValueError,match='stale'):compare.require_cpu({'cpu_preflight':'report.json'},{})


@pytest.mark.parametrize('status,code',[('timeout',-1),('completed',0)])
def test_worker_timeout_or_missing_report_is_failure_without_retry(monkeypatch,tmp_path,status,code):
    monkeypatch.setattr(compare,'require_cpu',lambda *a:None)
    monkeypatch.setattr(compare,'identity',lambda *a:{'fixture':True})
    calls=[]
    def launch(*a):calls.append(a);return code,status
    monkeypatch.setattr(compare.cache_driver,'launch_worker',launch)
    args=SimpleNamespace(output=str(tmp_path/'run'),config='fixture.json')
    assert compare.controller({}, {}, args)==1 and len(calls)==1
    report=json.loads((Path(args.output)/'report.json').read_text())
    assert report['status'] in ('timeout','failed_incomplete_comparison')
