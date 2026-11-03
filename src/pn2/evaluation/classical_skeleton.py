"""Image-only Otsu/Lee reference baseline; not an adopted PN2 replacement."""
from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from pn2.inputs.disk_guide import mapped


def guide_mask(path, bins=256):
    """Full-image histogram including zeros; deterministic, no reference access."""
    with mapped(path) as guide:
        if guide.ndim != 3 or guide.dtype != np.float32: raise ValueError('Expected cached float32 ZYX guide')
        lo, hi = np.inf, -np.inf
        for z in range(0, len(guide), 8):
            block = np.asarray(guide[z:z+8])
            if not np.isfinite(block).all() or np.any(block < 0): raise ValueError('Invalid guide response')
            lo = min(lo, float(block.min()));hi = max(hi, float(block.max()))
        counts = np.zeros(bins, dtype=np.int64)
        if hi > lo:
            boundaries = np.linspace(lo, hi, bins+1)
            for z in range(0, len(guide), 8):
                counts += np.histogram(guide[z:z+8], bins=boundaries)[0]
            centers = (boundaries[:-1]+boundaries[1:])/2
            threshold = float(threshold_otsu(hist=(counts, centers)))
        else: threshold = hi
        mask = np.empty(guide.shape, dtype=bool)
        for z in range(0, len(guide), 8):mask[z:z+8] = guide[z:z+8] > threshold
    return mask, {'threshold_method': 'whole_guide_otsu_including_zeros', 'histogram_bins': bins,
                  'guide_min': lo, 'guide_max': hi, 'threshold': threshold,
                  'foreground_voxels': int(mask.sum())}


def voxel_graph(mask):
    """All undirected 26-neighbor adjacencies; cycles are retained and disclosed."""
    if mask.ndim != 3 or mask.dtype != bool: raise ValueError('Expected bool ZYX skeleton')
    zyx = np.argwhere(mask);ids = np.ravel_multi_index(zyx.T, mask.shape) if len(zyx) else np.empty(0, dtype=np.int64)
    edges = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):continue
        neighbor = zyx+offset
        valid = np.all((neighbor >= 0) & (neighbor < mask.shape), axis=1)
        origins = np.flatnonzero(valid)
        flat = np.ravel_multi_index(neighbor[valid].T, mask.shape)
        loc = np.searchsorted(ids, flat)
        inside = loc < len(ids)
        if len(ids): inside &= ids[np.minimum(loc, len(ids)-1)] == flat
        if inside.any():edges.append(np.column_stack((origins[inside], loc[inside])))
    return SimpleNamespace(xyz=zyx[:, ::-1].astype(float),
        edges=np.concatenate(edges).astype(np.int64) if edges else np.empty((0, 2), dtype=np.int64))


def prune_terminal_spurs(graph, maximum_length=2.):
    """One pass; remove only short leaf-to-junction tips, never entire components."""
    xyz, edges = graph.xyz, graph.edges
    if not len(edges): return SimpleNamespace(xyz=xyz.copy(), edges=edges.copy()), 0
    adjacency = csr_matrix((np.ones(2*len(edges), dtype=np.int8),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(xyz), len(xyz)))
    degrees = np.diff(adjacency.indptr);remove = np.zeros(len(xyz), dtype=bool)
    for leaf in np.flatnonzero(degrees == 1):
        path = [leaf];previous, current, length = -1, int(leaf), 0.
        while True:
            neighbors = adjacency.indices[adjacency.indptr[current]:adjacency.indptr[current+1]]
            following = next((int(n) for n in neighbors if n != previous), None)
            if following is None:break
            length += float(np.linalg.norm(xyz[following]-xyz[current]))
            if length >= maximum_length:break
            previous, current = current, following
            if degrees[current] != 2:
                if degrees[current] >= 3:remove[path] = True
                break
            path.append(current)
    keep = ~remove;mapping = np.cumsum(keep)-1
    surviving = edges[np.all(keep[edges], axis=1)]
    return SimpleNamespace(xyz=xyz[keep], edges=mapping[surviving]), int(remove.sum())


def extract(path):
    mask, metadata = guide_mask(path)
    skeleton = skeletonize(mask, method='lee');del mask
    raw = voxel_graph(np.asarray(skeleton, dtype=bool));del skeleton
    pruned, removed = prune_terminal_spurs(raw)
    metadata.update(raw_vertices=len(raw.xyz), raw_edges=len(raw.edges),
        pruned_vertices=len(pruned.xyz), pruned_edges=len(pruned.edges), removed_spur_vertices=removed,
        pruning='one pass; leaf-to-junction path strictly shorter than two native voxels',
        adjacency='undirected 26-neighbor voxel graph; no biological topology claim',
        references_opened_for_extraction=False)
    return raw, pruned, metadata
