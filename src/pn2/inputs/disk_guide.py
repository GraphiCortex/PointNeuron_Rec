"""Disk-backed execution of the pinned Sato guide; no change to its kernels.

Match skimage 0.25.2's two first-derivative Gaussian operations, including
float32 rounding after each separable axis pass and global reflect boundaries.
Only finite scalar 3D float32 .npy inputs are accepted. Input decoding and
normalization remain separate and are not claimed to be bounded by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations_with_replacement
import math
from pathlib import Path

import numpy as np

from .sampling import partition_tiles


@contextmanager
def mapped(path, mode="r"):
    """Close each mapping explicitly on Windows, including on exceptions."""
    array = np.load(path, mmap_mode=mode, allow_pickle=False)
    try:
        yield array
    finally:
        array._mmap.close()


def read_block(path, region):
    with mapped(path) as array:
        return np.array(array[region], copy=True)


def write_block(path, region, values):
    with mapped(path, "r+") as array:
        array[region] = values


def empty_array(path, shape):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    array._mmap.close()


def axis_pass(source, destination, shape, sigma, order, axis, truncate, block_side):
    from scipy.ndimage import gaussian_filter1d

    # SciPy 1.15.3 gaussian_filter1d computes this same finite kernel radius.
    radius = int(truncate * sigma + .5)
    for core in partition_tiles(shape, block_side):
        extended = list(core)
        extended[axis] = slice(max(0, core[axis].start-radius),
                               min(shape[axis], core[axis].stop+radius))
        extended = tuple(extended)
        values = read_block(source, extended)
        result = gaussian_filter1d(values, sigma, axis=axis, order=order,
                                   mode="reflect", truncate=truncate)
        inside = [slice(None)] * 3
        offset = core[axis].start - extended[axis].start
        inside[axis] = slice(offset, offset + core[axis].stop-core[axis].start)
        write_block(destination, core, result[tuple(inside)])


def gaussian_pass(source, destination, scratch, shape, sigma, derivative_axis,
                  truncate, block_side):
    # Preserve scipy.ndimage.gaussian_filter's axis order and float32 outputs.
    previous = source
    for axis in range(3):
        target = destination if axis == 2 else scratch[axis]
        axis_pass(previous, target, shape, sigma, int(axis == derivative_axis),
                  axis, truncate, block_side)
        previous = target


def disk_sato_guide(normalized_path, sigmas, work_dir, *, block_side=64, progress=None,
                    output_path=None):
    """Return a .npy guide path and execution metadata; preserve scratch on failure.

    work_dir and optional output_path must be new. Mapping lifetimes are one block, so touching the full
    volume does not keep full-volume mapped arrays in this process's working
    set. OS file caching, decoder memory and caller allocations are separate.
    Changing block_side changes execution granularity, not a scientific setting.
    """
    import scipy
    import skimage
    from skimage.feature import hessian_matrix_eigvals

    if scipy.__version__ != "1.15.3" or skimage.__version__ != "0.25.2":
        raise RuntimeError("Disk guide requires scipy 1.15.3 and scikit-image 0.25.2")
    sigmas = tuple(float(s) for s in sigmas)
    if not sigmas or any(not math.isfinite(s) or s <= 1e-15 for s in sigmas):
        raise ValueError("Expected finite scales greater than 1e-15")
    if type(block_side) is not int or block_side <= 0:
        raise ValueError("block_side must be a positive integer")
    with mapped(normalized_path) as normalized:
        shape = normalized.shape
        if normalized.dtype != np.float32 or normalized.ndim != 3 or not normalized.size:
            raise ValueError("Expected nonempty normalized float32 3D .npy")
    folder = Path(work_dir).resolve()
    output = Path(output_path).resolve() if output_path is not None else folder / "guide.npy"
    if output.exists():
        raise FileExistsError(output)
    folder.mkdir(parents=True, exist_ok=False)
    names = ["negative", "gradient", "scratch0", "scratch1"] + [f"hessian{i}" for i in range(6)]
    buffers = {name: folder / f"{name}.npy" for name in names}
    for path in [*buffers.values(), output]:
        empty_array(path, shape)
    for core in partition_tiles(shape, block_side):
        values = read_block(normalized_path, core)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite input; preserve scratch and stop")
        write_block(buffers["negative"], core, -values)
        write_block(output, core, np.zeros(values.shape, dtype=np.float32))
    scratch = [buffers["scratch0"], buffers["scratch1"]]
    pairs = list(combinations_with_replacement(range(3), 2))
    radii = []
    for sigma in sigmas:
        truncate = 8 if sigma > 1 else 100
        scaled = (1 / math.sqrt(2)) * sigma
        radius = int(truncate * scaled + .5)
        radii.append(radius)
        if progress:
            progress(f"DISK GUIDE scale={sigma:g} radius_per_pass={radius} block_side={block_side}")
        for first_axis in range(3):
            gaussian_pass(buffers["negative"], buffers["gradient"], scratch,
                          shape, scaled, first_axis, truncate, block_side)
            for i, (a, b) in enumerate(pairs):
                if a == first_axis:
                    gaussian_pass(buffers["gradient"], buffers[f"hessian{i}"], scratch,
                                  shape, scaled, b, truncate, block_side)
        for core in partition_tiles(shape, block_side):
            hessian = [read_block(buffers[f"hessian{i}"], core) for i in range(6)]
            eigen = hessian_matrix_eigvals(hessian)[:-1]
            response = sigma**2 * np.prod(np.maximum(eigen, 0), axis=0)**(1/len(eigen))
            if not np.isfinite(response).all() or (response < 0).any():
                raise ValueError("Invalid response; preserve scratch and stop")
            write_block(output, core, np.maximum(read_block(output, core), response))
        if progress:
            progress(f"DISK GUIDE scale={sigma:g} complete")
    # These are only explicitly named files created in our fresh directory.
    for path in buffers.values():
        path.unlink()
    return output, {"backend": "disk_separable_v1", "block_side": block_side,
                    "kernel_radii_per_axis_pass": radii, "dtype": "float32",
                    "boundary_mode": "reflect", "normalization": "unchanged external whole-image transform",
                    "scratch_removed_after_success": True}
