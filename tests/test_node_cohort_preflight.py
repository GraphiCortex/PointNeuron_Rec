"""Cohort isolation and literal target counts; no real data, model or GPU."""
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_node_cohort as audit


def write_json(path,value):
    path.write_text(json.dumps(value));return path


def metadata_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    source=tmp_path/'src/pn2/supervision';source.mkdir(parents=True)
    (source/'curve_nodes.py').write_text('frozen fixture identity')
    partition={'partitions':{'train':[{'sample_id':'a'}],'val':[{'sample_id':'b'}],'test':[{'sample_id':'reserved'}]}}
    split=write_json(tmp_path/'partition.json',partition)
    paths={}
    for role,identity in [('training','a'),('development','b')]:
        p=write_json(tmp_path/(role+'.json'),{'cases':[{'case_id':identity,'partition':role,
            'image_path':identity+'.v3draw','reference_swc':identity+'.swc','metadata':{}}]})
        paths[role]={'path':p.name,'sha256':audit.digest(p)}
    config={'inventories':paths,'partition':{'path':split.name,'sha256':audit.digest(split)},
        'expected_counts':{'training':1,'development':1,'reserved':1},
        'target_source_sha256':audit.digest(source/'curve_nodes.py')}
    return write_json(tmp_path/'config.json',config),config


def test_identity_checks_do_not_open_any_image_or_reference(tmp_path,monkeypatch):
    path,_=metadata_fixture(tmp_path,monkeypatch)
    _,cases,ids=audit.load_cohort(path)
    assert len(cases)==2 and ids['test']=={'reserved'}
    assert not (tmp_path/'a.v3draw').exists()


def test_reserved_identity_cannot_be_substituted_for_training(tmp_path,monkeypatch):
    path,config=metadata_fixture(tmp_path,monkeypatch)
    train=tmp_path/'training.json';v=json.loads(train.read_text())
    v['cases'][0]['case_id']='reserved';write_json(train,v)
    config['inventories']['training']['sha256']=audit.digest(train);write_json(path,config)
    with pytest.raises(ValueError,match='identities'):
        audit.load_cohort(path)


def test_exact_source_path_crossing_split_is_rejected(tmp_path,monkeypatch):
    path,config=metadata_fixture(tmp_path,monkeypatch)
    dev=tmp_path/'development.json';v=json.loads(dev.read_text())
    v['cases'][0]['image_path']='a.v3draw';write_json(dev,v)
    config['inventories']['development']['sha256']=audit.digest(dev);write_json(path,config)
    with pytest.raises(ValueError,match='source path'):
        audit.load_cohort(path)


def literal_case(tmp_path,swc,dimensions=(32,32,32)):
    image=tmp_path/'header_only.v3draw'
    header=b'raw_image_stack_by_hpeng'+b'L'+(1).to_bytes(2,'little')+b''.join(d.to_bytes(4,'little') for d in (*dimensions,1))
    image.write_bytes(header)  # Payload deliberately absent: this is a header audit.
    reference=tmp_path/'reference.swc';reference.write_text(swc)
    case={'case_id':'fixture','partition':'training','image_path':str(image),'reference_swc':str(reference),
          'metadata':{'historical_sample_index':0},'source_group':'fixture'}
    config={'tile_side_voxel':16,'candidate_input_points':2048,'query_capacity':512}
    return case,config


def test_reference_empty_boxes_remain_unknown_and_header_is_not_a_decode(tmp_path):
    case,config=literal_case(tmp_path,'1 3 1 1 1 1 -1\n2 3 5 1 1 1 1\n')
    before=audit.digest(case['reference_swc'])
    row=audit.inspect_case(case,config,float('inf'))
    assert row['summary']['boxes']==8
    assert row['summary']['reference_present_boxes']==1
    assert row['summary']['reference_empty_unknown_boxes']==7
    assert row['image_payload_read'] is False
    assert audit.digest(case['reference_swc'])==before


def test_overflow_and_coincident_identities_are_counted_without_pruning(tmp_path):
    case,config=literal_case(tmp_path,''.join(f'{i} 3 1 1 1 1 -1\n' for i in range(1,514)))
    row=audit.inspect_case(case,config,float('inf'))
    assert row['summary']['overflow_boxes']==1
    assert row['summary']['maximum_targets']==513
    assert row['boxes'][0]['coincident_target_identities']==512


def test_short_inputs_and_wall_budget_are_explicit(tmp_path):
    case,config=literal_case(tmp_path,'1 3 1 1 1 1 -1\n',dimensions=(4,4,4))
    row=audit.inspect_case(case,config,float('inf'))
    assert row['summary']['encoder_short_boxes']==1
    with pytest.raises(TimeoutError,match='wall cap'):
        audit.inspect_case(case,config,time.perf_counter()-1)
