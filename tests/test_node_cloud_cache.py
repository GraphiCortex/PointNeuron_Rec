"""Reference-free data integrity, exact adapter behavior and fail-closed execution."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.inputs.cases import ImageCase,load_image_cases
from pn2.inputs import node_cache as cache
from pn2.inputs.sampling import normalized_image,partition_tiles

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import node_cloud_cache as driver


def fixture(tmp_path,*,constant=False,partition='training',identity='fixture'):
    z,y,x=np.indices((9,10,11))
    raw=np.full(z.shape,273,dtype=np.uint16) if constant else (101*z+13*y+x).astype(np.uint16)
    path=tmp_path/(identity+'.v3draw')
    header=b'raw_image_stack_by_hpeng'+b'L'+(2).to_bytes(2,'little')+b''.join(int(d).to_bytes(4,'little') for d in (*raw.shape[::-1],1))
    path.write_bytes(header+raw.astype('<u2').tobytes())
    settings={**driver.SETTINGS,'tile_side_voxel':8,'points_per_box':4}
    case=ImageCase(identity,path,partition)
    _,normalization=normalized_image(raw)
    from pn2.inputs.image import decode_image
    image=decode_image(case).provenance
    guide=tmp_path/(identity+'_guide.npy');np.save(guide,np.zeros(raw.shape,dtype=np.float32))
    reuse={'case_id':identity,'guide_path':str(guide),'guide_sha256':cache.digest(guide),
        'source_sha256':image['source_sha256'],'decoded_sha256':image['decoded_sha256'],
        'normalization':normalization,'sigmas_voxel':settings['sigmas_voxel'],
        'versions':cache.versions(),'source_receipt_sha256':'fixture'}
    return case,raw,settings,reuse,{'settings':settings,'test':'fixture'}


def test_adapter_preserves_native_axes_clipped_bounds_and_feature_rounding(tmp_path):
    case,raw,settings,_,_=fixture(tmp_path)
    _,normalization=normalized_image(raw)
    tiles=list(partition_tiles(raw.shape,8));all_ids=[]
    for i,tile in enumerate(tiles):
        arrays,row=cache.point_arrays(raw,np.zeros(raw[tile].shape),tile,i,settings,normalization)
        expected=np.asarray(np.unravel_index(arrays['flat_indices'],raw.shape))[::-1].T
        assert np.array_equal(expected,arrays['native_xyz'])
        assert np.array_equal(arrays['raw_intensity'],raw.ravel()[arrays['flat_indices']])
        expected_i=((arrays['raw_intensity'].astype(np.float64)-normalization['minimum'])/
                    (normalization['maximum']-normalization['minimum'])).astype(np.float32)
        assert np.array_equal(arrays['attributes'][:,3],expected_i)
        assert np.allclose(arrays['local_xyz']*64+row['center_xyz'],expected,atol=1e-5,rtol=0)
        assert row['uniform_fallback'] is True
        all_ids.extend(arrays['flat_indices'].tolist())
    assert len(set(all_ids))==len(all_ids)==8*4


def test_constant_image_is_retained_with_distinct_real_voxels(tmp_path):
    case,raw,settings,reuse,identity=fixture(tmp_path,constant=True)
    receipt=cache.build_case(case,settings,tmp_path/'cache',identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    assert receipt['tile_count']==8 and all(r['source_tile_constant'] for r in receipt['records'])
    for row in receipt['records']:
        with np.load(tmp_path/'cache'/row['packet']) as a:
            assert np.all(a['attributes'][:,3]==0) and len(np.unique(a['flat_indices']))==4
    assert receipt['references_opened']==0 and receipt['targets_created'] is False
    assert cache.verify_case(tmp_path/'cache',case,identity)['status']=='completed'


@pytest.mark.parametrize('field',['source_sha256','decoded_sha256','guide_sha256'])
def test_stale_reuse_fails_without_filter_fallback(tmp_path,monkeypatch,field):
    case,_,settings,reuse,identity=fixture(tmp_path)
    reuse[field]='wrong'
    monkeypatch.setattr(cache,'disk_sato_guide',lambda *a,**k:pytest.fail('must not silently regenerate'))
    with pytest.raises(ValueError):cache.build_case(case,settings,tmp_path/'cache',identity,reuse=reuse,reserve_bytes=0)
    assert not (tmp_path/'cache/complete.json').exists()


def test_existing_output_is_never_overwritten(tmp_path):
    case,_,settings,reuse,identity=fixture(tmp_path)
    folder=tmp_path/'cache';folder.mkdir();(folder/'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError):cache.build_case(case,settings,folder,identity,reuse=reuse,reserve_bytes=0)
    assert (folder/'keep.txt').read_text()=='keep'


def test_insufficient_space_fails_before_creating_case_or_decoding(tmp_path,monkeypatch):
    case,_,settings,_,identity=fixture(tmp_path)
    monkeypatch.setattr(cache.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    monkeypatch.setattr(cache,'decode_image',lambda *a:pytest.fail('must not decode'))
    with pytest.raises(OSError,match='space'):cache.build_case(case,settings,tmp_path/'cache',identity)
    assert not (tmp_path/'cache').exists()


def test_packet_tampering_is_detected(tmp_path):
    case,_,settings,reuse,identity=fixture(tmp_path)
    folder=tmp_path/'cache';cache.build_case(case,settings,folder,identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    with (folder/'tile00000.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='packet changed'):cache.verify_case(folder,case,identity)


def test_semantically_incomplete_receipt_fails_even_with_matching_receipt_hash(tmp_path):
    case,_,settings,reuse,identity=fixture(tmp_path)
    folder=tmp_path/'cache';cache.build_case(case,settings,folder,identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    receipt=json.loads((folder/'case.json').read_text());receipt['records'].pop()
    (folder/'case.json').write_text(json.dumps(receipt))
    (folder/'complete.json').write_text(json.dumps({'case_json_sha256':cache.digest(folder/'case.json')}))
    with pytest.raises(ValueError,match='every image-defined box'):cache.verify_case(folder,case,identity)


def test_manifest_cannot_carry_reference_keys_or_reserved_partition(tmp_path):
    row={'case_id':'x','partition':'training','image_path':'does_not_exist',
         'coordinates':{'mode':'native','units':'voxel'},'reference_swc':'forbidden.swc'}
    p=tmp_path/'images.json';p.write_text(json.dumps({'schema_version':1,'cases':[row]}))
    with pytest.raises(ValueError):load_image_cases(p)
    del row['reference_swc'];row['partition']='test'
    p.write_text(json.dumps({'schema_version':1,'cases':[row]}))
    with pytest.raises(ValueError):load_image_cases(p)


def test_partial_cache_is_not_training_ready_and_cross_split_duplicate_is_rejected(tmp_path,monkeypatch):
    a,_,settings,reuse,identity=fixture(tmp_path,identity='a')
    b=ImageCase('b',a.image_path,'development')
    monkeypatch.setattr(driver,'cache_identity',lambda c:identity)
    cache.build_case(a,settings,tmp_path/cache.case_key(a),identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    result=driver.index_cache({},[a,b],tmp_path)
    assert result['completed_cases']==1 and result['cloud_cache_complete'] is False and result['training_ready'] is False
    reuse={**reuse,'case_id':'b'}
    cache.build_case(b,settings,tmp_path/cache.case_key(b),identity,reuse=reuse,reserve_bytes=0,progress=lambda s:None)
    with pytest.raises(ValueError,match='Cross-split duplicate'):driver.index_cache({},[a,b],tmp_path)


def test_controller_timeout_does_not_retry(monkeypatch,tmp_path):
    case=ImageCase('fixture',tmp_path/'unused','training')
    control={'regression_packets':[{'case_id':'fixture'}]}
    monkeypatch.setattr(driver,'require_cpu',lambda *a:None)
    monkeypatch.setattr(driver,'guide_candidates',lambda *a:{})
    monkeypatch.setattr(driver,'cache_identity',lambda *a:{'fixture':True})
    monkeypatch.setattr(driver,'cache_root',lambda *a:tmp_path)
    calls=[]
    def timeout(*a):calls.append(a);return -1,'timeout'
    monkeypatch.setattr(driver,'launch_worker',timeout)
    args=SimpleNamespace(action='replay',cache_root=str(tmp_path),output=str(tmp_path/'run'),
        config='fixture.json',cpu_preflight=None)
    assert driver.controller(control,args,[case])==1 and len(calls)==1
    assert json.loads((tmp_path/'run/report.json').read_text())['status']=='timeout'
