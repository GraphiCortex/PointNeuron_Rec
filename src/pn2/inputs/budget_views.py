"""Reference-free larger cloud views of the immutable image cache. No filters."""
from pathlib import Path
import shutil

import numpy as np

from .node_cache import verify_case, point_arrays, digest, write_once
from .image import decode_image
from .disk_guide import read_block


def distinct_count(budget, voxels):
    if type(budget) is not int or type(voxels) is not int or min(budget, voxels) < 1:
        raise ValueError('Positive budget and source-domain size required')
    return min(budget, voxels)


def build_view_case(case, baseline_folder, baseline_identity, folder, *, budget=8192,
                    reserve_bytes=2*1024**3, progress=print):
    """Use raw intensities and the retained guide; never read a reference/target."""
    folder = Path(folder)
    if folder.exists(): raise FileExistsError(folder)
    base = verify_case(baseline_folder, case, baseline_identity)
    estimate = sum(min(budget, int(np.prod(np.subtract(b['box']['stop_xyz'], b['box']['start_xyz']))))
                   for b in base['records']) * 80
    if shutil.disk_usage(folder.parent).free < estimate + reserve_bytes:
        raise OSError('Insufficient space for cloud packets and reserve')
    image = decode_image(case); raw = image.array_zyx
    if any(image.provenance[k] != base['image'][k]
           for k in ('source_sha256', 'decoded_sha256', 'dimensions_xyzc')):
        raise ValueError('Original source differs from the retained guide cache')
    if (float(raw.min()), float(raw.max())) != (base['normalization']['minimum'], base['normalization']['maximum']):
        raise ValueError('Whole-source normalization changed')
    folder.mkdir(); rows = []
    for b in base['records']:
        start, stop = b['box']['start_xyz'], b['box']['stop_xyz']
        tile = tuple(slice(a, z) for a, z in zip(start[::-1], stop[::-1]))
        count = distinct_count(budget, int(raw[tile].size))
        settings = {**baseline_identity['settings'], 'points_per_box': count}
        arrays, row = point_arrays(raw, read_block(Path(baseline_folder)/base['guide']['path'], tile),
                                  tile, b['tile_id'], settings, base['normalization'])
        with np.load(Path(baseline_folder)/b['packet'], allow_pickle=False) as lower:
            common, ia, ib = np.intersect1d(lower['flat_indices'], arrays['flat_indices'], return_indices=True)
            if len(common) != len(lower['flat_indices']):
                raise ValueError('Larger cloud lost baseline locations')
            if any(not np.array_equal(lower[k][ia], arrays[k][ib]) for k in arrays):
                raise ValueError('Shared source observations have different attributes')
        path = folder/f'tile{b["tile_id"]:05d}.npz'; np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k], v) or saved[k].dtype != v.dtype for k, v in arrays.items()):
                raise ValueError('Cloud serialization round trip failed')
        rows.append({**row, 'packet': path.name, 'packet_sha256': digest(path),
                     'source_voxels': int(raw[tile].size), 'baseline_packet_sha256': b['packet_sha256']})
        progress(f'CLOUD VIEW tile={b["tile_id"]+1}/{len(base["records"])} distinct={count} baseline_subset=True')
    receipt = {'schema': 'pn2-cloud-budget-view-v1', 'case_id': case.case_id, 'partition': case.partition,
        'budget': budget, 'baseline_case_sha256': digest(Path(baseline_folder)/'case.json'),
        'baseline_identity': baseline_identity, 'source_sha256': image.provenance['source_sha256'],
        'decoded_sha256': image.provenance['decoded_sha256'], 'guide_sha256': base['guide']['sha256'],
        'records': rows, 'filter_reruns': 0, 'references_opened': 0, 'targets_read': 0}
    write_once(folder/'case.json', receipt)
    write_once(folder/'complete.json', {'case_json_sha256': digest(folder/'case.json')})
    return receipt
