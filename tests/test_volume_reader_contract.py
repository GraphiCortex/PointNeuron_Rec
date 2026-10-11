import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.vaa3d_raw import decode_pbd8, decode_pbd16
from pointneuron.models.sampling_pipeline import inference_sample
from test_sampling_training import fixture_manifest


def test_old_cache_rejected_for_training_but_available_for_historical_analysis(tmp_path):
    path, partition, manifest, record = fixture_manifest(tmp_path)
    del manifest['volume_reader_version']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Rebuild'):
        load_sampling_manifest(path, partition)
    assert load_sampling_manifest(path, partition, allow_historical_reader=True)['records']


def test_new_manifest_cannot_claim_old_record_is_corrected(tmp_path):
    path, partition, manifest, record = fixture_manifest(tmp_path)
    with np.load(record) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays['metadata'])); del metadata['volume_reader_version']
    arrays['metadata'] = np.array(json.dumps(metadata)); np.savez(record, **arrays)
    manifest['records'][0]['sha256'] = hashlib.sha256(record.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='decoder version'):
        load_sampling_manifest(path, partition)


def test_old_sampling_checkpoint_rejected_before_image_processing():
    with patch('pointneuron.models.sampling_pipeline.build_image_pools') as pools:
        with pytest.raises(ValueError, match='decoder correction'):
            inference_sample({'sampling_pipeline': {'strategy': 'random'}}, None, None, None, 1, 1, 0)
        pools.assert_not_called()


@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 94, 95])
def test_pbd8_reference_packing_and_carry_across_full_run(length):
    codes = [1, 2, 3, 0] * 24
    codes = codes[:length]
    packed = [sum(code * 4 ** j for j, code in enumerate(codes[i:i + 4])) for i in range(0, length, 4)]
    expected = [100]
    for code in codes + [2]:
        expected.append((expected[-1] + (code if code < 3 else -1)) % 256)
    assert decode_pbd8(bytes([0, 100, 32 + length] + packed + [33, 2]), length + 2) == bytes(expected)


@pytest.mark.parametrize('endian', ['little', 'big'])
@pytest.mark.parametrize('length', [1, 2, 3, 8, 47, 48])
def test_pbd16_reference_mapping_across_byte_boundaries(length, endian):
    codes = ([0, 1, 2, 3, 4, 5, 6, 7] * 6)[:length]
    bits = ''.join(format(c, '03b') for c in codes)
    bits += '0' * ((-len(bits)) % 8)
    packed = [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]
    expected = [1000]
    for code in codes:
        expected.append((expected[-1] + (code if code <= 4 else 4 - code)) % 65536)
    stream = bytes([0]) + (1000).to_bytes(2, endian) + bytes([31 + length] + packed)
    assert decode_pbd16(stream, length + 1, endian) == b''.join(v.to_bytes(2, endian) for v in expected)


@pytest.mark.parametrize('code', [80, 182, 183, 222])
def test_unverified_pbd16_extensions_fail_instead_of_silently_corrupting(code):
    with pytest.raises(NotImplementedError, match='verified decoder'):
        decode_pbd16(bytes([code, 0]), 1)
