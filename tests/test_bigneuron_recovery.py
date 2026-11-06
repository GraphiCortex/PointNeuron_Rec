"""Synthetic/archive-name metadata only;no benchmark SWC content or scoring."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from pn2.evaluation.post_stage2 import graph,scores


@pytest.fixture
def runner():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('benchmark_recovery_test',scripts/'recover_bigneuron_benchmark.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_archive_numeric_ids_repair_changed_case_names(runner):
    c={'published_id_map':{'frog/done_err_Recon112012no2-2':['245'],'mouse/sv_140921c4':['122'],'fly/SLP':['251']},
       'human_case':'p_checked6_human_allen_confocal/in_house1'}
    assert runner.archive_identity('root/checked3_frog/Recon/01_245.v3dpbd_app2.swc',c)[0]=='frog/done_err_Recon112012no2-2'
    assert runner.archive_identity('root/checked3_mouse/140921c4/02_122.v3dpbd_mst.swc',c)[0]=='mouse/sv_140921c4'
    assert runner.archive_identity('root/checked3_fly/WED/01_252.v3dpbd_mst.swc',c)==(None,None)
    assert runner.archive_identity('root/checked3_fly/SLP/01_251.v3dpbd_mst.swc',c)[0]=='fly/SLP'


def test_Cell1_requires_exact_source_case_not_shared_basename_or_256(runner):
    c={'published_id_map':{},'human_case':'p_checked6_human_allen_confocal/in_house1'}
    assert runner.archive_identity('root/checked3_human_allen_confocal/in_house1/01_256.v3dpbd_mst.swc',c)[0]==c['human_case']
    assert runner.archive_identity('root/checked3_human_allen_confocal_independent_trial2_haru/HO_in_house1/01_256.v3dpbd_mst.swc',c)==(None,None)
    assert runner.archive_identity('root/256/01_256.v3dpbd_mst.swc',c)==(None,None)


def test_implementation_identity_removes_dataset_and_ordinal_but_keeps_mode(runner):
    a=runner.algorithm_variant('root/01_103.v3dpbd_app2new3.swc');b=runner.algorithm_variant('root/17_251.v3dpbd_APP2NEW3.swc')
    assert a[0]==b[0]=='app2new3' and a[2]=='103' and b[2]=='251'
    name,seed,cid=runner.algorithm_variant('02_103.v3dpbd_x258_y220_z34_app2.swc')
    assert name=='app2__seed_xyz_in_filename' and seed==[258.,220.,34.]
    assert runner.algorithm_variant('05_122.v3dpbd_x10_y11_z12_app2.swc')[0]==name
    assert runner.algorithm_variant('05_122.v3dpbd_app2.swc')[0]!=name
    with pytest.raises(ValueError):runner.algorithm_variant('unknown.swc')


def test_all_replicates_contribute_and_reference_not_multiplied(runner):
    ref=graph([[0,0,0],[10,0,0]],[[0,1]])
    good={'case_id':'toy','tile_id':0,**scores(ref,ref)}
    bad={'case_id':'toy','tile_id':0,**scores(ref,graph([],[]))}
    mean=runner.mean_replicates([good,bad])
    assert mean['segments']['reference_length']==10
    assert mean['segments']['predicted_length']==5
    assert mean['segments']['tolerances']['2.0']['f1']==pytest.approx(2/3)
    assert mean['resampled']['tolerances']['2.0']['f1']==pytest.approx(2/3)
    assert mean['replicates']==2 and mean['length_ratio']==.5
    assert runner.mean_replicates([good]) is good
