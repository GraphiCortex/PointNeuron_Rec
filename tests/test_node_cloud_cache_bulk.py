"""Data-preparation budgets and duplicate guards; no real filters or images."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import node_cloud_cache_bulk as bulk


def profile():
    return {'reuse_case_wall_seconds':600,'fresh_case_wall':{'minimum_seconds':3600,
        'maximum_seconds':21600,'measured_time_multiplier':2.,'extra_seconds':600}}


def test_volume_scaled_wall_allowance_is_bounded_and_copy_has_separate_allowance():
    p=profile()
    assert bulk.case_timeout(1,False,p,.01)==3600
    assert bulk.case_timeout(400000,False,p,.01)==8600
    assert bulk.case_timeout(10000000,False,p,.01)==21600
    assert bulk.case_timeout(10000000,True,p,.01)==600


@pytest.mark.parametrize('rate',[0.,-1.,float('nan'),float('inf')])
def test_invalid_work_rate_does_not_create_a_wall_budget(rate):
    with pytest.raises(ValueError):bulk.case_timeout(10,False,profile(),rate)


def receipt(case,role,source='same',decoded='same'):
    return {'case_id':case,'partition':role,'image':{'source_sha256':source,'decoded_sha256':decoded,
        'dimensions_xyzc':[1,2,3,1],'datatype':1}}


def test_duplicate_source_or_decoded_pixels_cannot_cross_partitions():
    seen={};bulk.add_content_identity(seen,receipt('a','training'))
    bulk.add_content_identity(seen,receipt('b','training'))
    with pytest.raises(ValueError,match='Cross-split'):bulk.add_content_identity(seen,receipt('dev','development'))
    with pytest.raises(ValueError,match='Cross-split'):
        bulk.add_content_identity(seen,receipt('dev2','development',source='different_encoding'))


def test_distinct_content_is_kept_without_claiming_acquisition_independence():
    seen={};bulk.add_content_identity(seen,receipt('train','training'))
    bulk.add_content_identity(seen,receipt('dev','development',source='other',decoded='other'))
    assert len(seen)==4


def prepare_fixture(monkeypatch,tmp_path):
    root=tmp_path/'cache';root.mkdir()
    plan=tmp_path/'plan';plan.mkdir()
    before={'fixture':True,'cache_root':str(root),'completed_before_launch':[]}
    (plan/'plan.json').write_text(json.dumps(before))
    jobs=[{'case_id':'a','wall_timeout_seconds':600},{'case_id':'b','wall_timeout_seconds':600}]
    current={**before,'jobs':jobs,'pending_images':2,'pending_fresh_filters':2,'pending_guide_copies':0,
             'estimated_remaining_filter_hours':1.}
    monkeypatch.setattr(bulk,'execution_identity',lambda *a:{'fixture':True})
    monkeypatch.setattr(bulk.driver,'cache_identity',lambda *a:{'fixture':True})
    monkeypatch.setattr(bulk.driver,'cache_root',lambda *a:root)
    monkeypatch.setattr(bulk,'make_plan',lambda *a:current)
    cases=[SimpleNamespace(case_id='a',partition='training'),SimpleNamespace(case_id='b',partition='development')]
    args=SimpleNamespace(plan_run=str(plan),cache_root=str(root),output=str(tmp_path/'run'),config='fixture.json')
    return current,cases,args


def test_first_failed_case_stops_queue_and_records_partial_cache_failure(monkeypatch,tmp_path):
    _,cases,args=prepare_fixture(monkeypatch,tmp_path)
    calls=[]
    def timeout(*a):calls.append(a);return -1,'timeout'
    monkeypatch.setattr(bulk.driver,'launch_worker',timeout)
    def partial(*a):raise ValueError('Incomplete case preserved')
    monkeypatch.setattr(bulk.driver,'index_cache',partial)
    assert bulk.prepare({'overall_wall_seconds':600},{},cases,args,{})==1
    assert len(calls)==1
    report=json.loads((Path(args.output)/'report.json').read_text())
    assert report['status']=='timeout' and report['automatic_retry'] is False
    assert report['cache_index']['cloud_cache_complete'] is False
    assert (Path(args.output)/'case_00/failure.json').exists()


def test_changed_previously_completed_case_stops_before_launch(monkeypatch,tmp_path):
    current,cases,args=prepare_fixture(monkeypatch,tmp_path)
    path=Path(args.plan_run)/'plan.json';planned=json.loads(path.read_text())
    planned['completed_before_launch']=[{'case_id':'existing','case_json_sha256':'old'}]
    path.write_text(json.dumps(planned))
    current['completed_before_launch']=[{'case_id':'existing','case_json_sha256':'changed'}]
    monkeypatch.setattr(bulk.driver,'launch_worker',lambda *a:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='changed or disappeared'):
        bulk.prepare({'overall_wall_seconds':600},{},cases,args,{})
    assert not Path(args.output).exists()


def test_existing_partial_case_is_not_treated_as_completed(tmp_path):
    case=SimpleNamespace(case_id='partial',partition='training')
    (tmp_path/bulk.case_key(case)).mkdir()
    with pytest.raises(ValueError,match='Incomplete case preserved'):
        bulk.driver.index_cache({},[case],tmp_path)
