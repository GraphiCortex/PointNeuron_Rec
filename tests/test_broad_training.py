import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from pn2.inputs.node_cache import digest, fingerprint


def test_preparation_binding_rejects_changed_schedule(tmp_path):
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('broad_training_fixture',scripts/'run_broad_curve_head.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    order=np.array([(epoch,0,tile) for epoch in range(20) for tile in range(460)])
    data=SimpleNamespace(schedule=lambda:(['train_case'],order))
    c={'fixture':True};keys=[['train_case',tile] for tile in range(460)]
    manifest={'config_fingerprint':fingerprint(c),'train':keys,'schedule':keys*20,
              'labels':[{'case_id':'train_case','tile_id':tile} for tile in range(460)]}
    prep=tmp_path/'prepare';prep.mkdir();p=prep/'manifest.json'
    p.write_text(json.dumps(manifest));(prep/'complete.json').write_text(json.dumps({'manifest_sha256':digest(p)}))
    assert len(module.load_prepared(c,tmp_path,data)['schedule'])==9200
    manifest['schedule'][0]=['development_case',1]
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='marker/hash'):module.load_prepared(c,tmp_path,data)
    (prep/'complete.json').write_text(json.dumps({'manifest_sha256':digest(p)}))
    with pytest.raises(ValueError,match='cohort/schedule'):module.load_prepared(c,tmp_path,data)
