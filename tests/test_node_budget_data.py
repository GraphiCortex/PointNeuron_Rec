"""Paired data provenance, unknown supervision and inference/target separation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pn2.inputs.cases import ImageCase
from pn2.inputs import node_cache as cache
from pn2.inputs.budget_views import build_view_case, distinct_count
from pn2.inputs.node_dataset import NodeBudgetDataset
from pn2.inputs.sampling import normalized_image
from pn2.inputs.image import decode_image
from pn2.supervision.node_targets import build_target_case


def source_fixture(tmp_path):
    raw=np.arange(9*10*11,dtype=np.uint16).reshape(9,10,11)
    path=tmp_path/'source.v3draw'
    header=b'raw_image_stack_by_hpeng'+b'L'+(2).to_bytes(2,'little')+b''.join(
        int(d).to_bytes(4,'little') for d in (*raw.shape[::-1],1))
    path.write_bytes(header+raw.astype('<u2').tobytes())
    case=ImageCase('training-case',path,'training');image=decode_image(case)
    settings={'tile_side_voxel':8,'points_per_box':4,'coordinate_scale_voxel':64.,
              'sigmas_voxel':[1.,2.,4.],'uniform_mass':.1,'cloud_seed':0,'ordering_seed':0}
    _,normalization=normalized_image(raw)
    guide=tmp_path/'guide.npy';np.save(guide,np.zeros(raw.shape,dtype=np.float32))
    reuse={'case_id':case.case_id,'guide_path':str(guide),'guide_sha256':cache.digest(guide),
        'source_sha256':image.provenance['source_sha256'],'decoded_sha256':image.provenance['decoded_sha256'],
        'normalization':normalization,'sigmas_voxel':settings['sigmas_voxel'],
        'versions':cache.versions(),'source_receipt_sha256':'fixture'}
    identity={'settings':settings}
    receipt=cache.build_case(case,settings,tmp_path/'base',identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    return case,identity,receipt


def test_larger_view_retains_baseline_attributes_and_clips_distinct_count(tmp_path):
    case,identity,base=source_fixture(tmp_path)
    view=build_view_case(case,tmp_path/'base',identity,tmp_path/'larger',budget=16,reserve_bytes=0,progress=lambda s:None)
    assert view['references_opened']==0 and view['filter_reruns']==0
    assert view['records'][-1]['slots']==6
    for a,b in zip(base['records'],view['records']):
        with np.load(tmp_path/'base'/a['packet']) as lower, np.load(tmp_path/'larger'/b['packet']) as higher:
            assert len(np.unique(higher['flat_indices']))==b['slots']
            assert np.isin(lower['flat_indices'],higher['flat_indices']).all()
    with pytest.raises(FileExistsError):
        build_view_case(case,tmp_path/'base',identity,tmp_path/'larger',budget=16,reserve_bytes=0)


def test_original_source_change_stops_before_view_files_are_created(tmp_path):
    case,identity,_=source_fixture(tmp_path)
    payload=bytearray(case.image_path.read_bytes());payload[-1]^=1;case.image_path.write_bytes(payload)
    with pytest.raises(ValueError,match='source differs'):
        build_view_case(case,tmp_path/'base',identity,tmp_path/'larger',budget=16,reserve_bytes=0)
    assert not (tmp_path/'larger').exists()


def target_fixture(tmp_path,partition='training',coincident=False):
    path=tmp_path/'reference.swc'
    path.write_text('1 3 1 1 1 1 -1\n2 3 1 1 1 1 -1\n' if coincident else '1 3 1 1 1 1 -1\n2 3 9 1 1 1 1\n')
    ref={'case_id':'case','partition':partition,'gradient_training_allowed':partition=='training',
         'coordinates':{'mode':'native','units':'voxel'},'reference_sha256':cache.digest(path)}
    boxes=[{'tile_id':0,'box':{'start_xyz':[0,0,0],'stop_xyz':[8,8,8]}},
           {'tile_id':1,'box':{'start_xyz':[0,8,0],'stop_xyz':[8,16,8]}}]
    expected=[{'tile_id':b['tile_id'],**b['box'],'targets':2 if b['tile_id']==0 else 0} for b in boxes]
    receipt=build_target_case(ref,path,boxes,expected,tmp_path/'targets')
    return ref,receipt


def test_shared_targets_preserve_coincident_identities_and_empty_unknown(tmp_path):
    _,receipt=target_fixture(tmp_path,coincident=True)
    positive,empty=receipt['records']
    assert positive['count']==2 and positive['coincident_identities']==1
    assert positive['gradient_training_allowed'] is True
    with np.load(tmp_path/'targets'/positive['packet']) as p:
        assert p['xyz'].shape==(2,3) and np.array_equal(p['xyz'][0],p['xyz'][1])
        assert len(set(p['identities_json']))==2
    assert empty['count']==0 and empty['supervision']=='unknown' and empty['gradient_training_allowed'] is False


def test_development_targets_never_allow_gradients(tmp_path):
    _,receipt=target_fixture(tmp_path,partition='development')
    assert all(not r['gradient_training_allowed'] for r in receipt['records'])


def dataset_fixture(tmp_path):
    root=tmp_path;folder=root/'tmp/data';folder.mkdir(parents=True)
    shape=(4,4,4);ids=np.array([0,1,2,3]);xyz=np.asarray(np.unravel_index(ids,shape))[::-1].T.astype(np.float64)
    center=np.ones(3)*1.5;local=((xyz-center)/64).astype(np.float32);raw=np.arange(4,dtype=np.uint8)
    packet=folder/'image.npz';np.savez_compressed(packet,flat_indices=ids,native_xyz=xyz,raw_intensity=raw,
        local_xyz=local,attributes=np.column_stack((local,raw.astype(np.float32)/3)))
    def cloud():return {'path':packet.relative_to(root).as_posix(),'sha256':cache.digest(packet),'count':4}
    def box(tile,role,count):return {'tile_id':tile,'box':{'start_xyz':[0,0,0],'stop_xyz':[4,4,4]},
        'source_voxels':64,'clouds':{'4':cloud()},'target':{'path':'tmp/nonexistent_target.npz','sha256':'missing',
        'count':count,'supervision':'reference_present' if count else 'unknown',
        'gradient_training_allowed':role=='training' and count>0}}
    cases=[{'case_id':'train','partition':'training','alias':0,'dimensions_xyz':[4,4,4],
        'normalization':{'minimum':0.,'maximum':3.},'boxes':[box(0,'training',2),box(1,'training',0)]},
        {'case_id':'dev','partition':'development','alias':1,'dimensions_xyz':[4,4,4],
        'normalization':{'minimum':0.,'maximum':3.},'boxes':[box(0,'development',2)]}]
    schedule=folder/'schedule.npz';np.savez_compressed(schedule,epoch_case_tile=np.array([[0,0,0],[1,0,0]],dtype=np.int32),case_ids=np.array(['train']))
    manifest={'schema':'pn2-paired-node-data-v1','cases':cases,'epochs':2,
        'schedule':{'path':schedule.relative_to(root).as_posix(),'sha256':cache.digest(schedule)}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    (folder/'complete.json').write_text(json.dumps({'manifest_sha256':cache.digest(folder/'manifest.json')}))
    return NodeBudgetDataset(root,folder),packet


def test_image_only_loading_does_not_open_nonexistent_target_packet(tmp_path):
    dataset,_=dataset_fixture(tmp_path)
    image=dataset.image('train',0,4)
    assert len(image['packet']['flat_indices'])==4 and 'target_xyz' not in image
    assert dataset.image('dev',0,4)['metadata']['partition']=='development'


def test_development_and_unknown_training_boxes_rejected_before_packet_access(tmp_path):
    dataset,_=dataset_fixture(tmp_path)
    for cid,tile in (('dev',0),('train',1)):
        with pytest.raises(ValueError,match='must not enter gradient'):
            dataset.training(cid,tile,4)
    ids,schedule=dataset.schedule()
    assert ids==['train'] and schedule.tolist()==[[0,0,0],[1,0,0]]


def test_changed_packet_is_rejected_before_numpy_decode(tmp_path):
    dataset,packet=dataset_fixture(tmp_path);packet.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='content changed'):dataset.image('train',0,4)


def test_schedule_cannot_sneak_unknown_box_into_training(tmp_path):
    dataset,_=dataset_fixture(tmp_path);path=dataset.folder/'schedule.npz'
    np.savez_compressed(path,epoch_case_tile=np.array([[0,0,1],[1,0,1]],dtype=np.int32),case_ids=np.array(['train']))
    dataset.manifest['schedule']['sha256']=cache.digest(path)
    with pytest.raises(ValueError,match='Unknown/development'):dataset.schedule()
