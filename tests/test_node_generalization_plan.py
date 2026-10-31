"""Exposure planning must not train on development or label unknown as negative."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_node_generalization import exposure_schedule


def record(identity,partition,counts):
    return {'case_id':identity,'partition':partition,'boxes':[{'tile_id':i,'targets':m,
        'target_overflow':False,'encoder_short_input':False} for i,m in enumerate(counts)]}


def test_every_supervised_training_box_has_equal_exposure_without_dev_or_unknown():
    rows=[record('b','training',[0,7]),record('a','training',[4,0,8]),record('dev','development',[99,0])]
    train,schedule=exposure_schedule(rows,20,0)
    assert [r['case_id'] for r in train]==['a','b']
    assert len(schedule)==60
    assert Counter(map(tuple,schedule[:,1:]))=={(0,0):20,(0,2):20,(1,1):20}
    assert set(schedule[:,0])==set(range(20))


def test_identity_order_does_not_change_the_fixed_schedule():
    rows=[record('b','training',[0,7]),record('a','training',[4,0,8])]
    assert np.array_equal(exposure_schedule(rows,3,0)[1],exposure_schedule(rows[::-1],3,0)[1])


@pytest.mark.parametrize('failure',['target_overflow','encoder_short_input'])
def test_capacity_failure_is_not_silently_dropped(failure):
    rows=[record('a','training',[1,2])];rows[0]['boxes'][1][failure]=True
    with pytest.raises(ValueError,match='capacity'):
        exposure_schedule(rows,1,0)


def test_reserved_case_and_duplicate_case_identity_are_rejected():
    for rows in [[record('reserved','test',[1])],[record('a','training',[1]),record('a','development',[2])]]:
        with pytest.raises(ValueError):exposure_schedule(rows,1,0)
