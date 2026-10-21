"""Decode a single explicit image, preserving native XYZ and exact byte provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .cases import ImageCase
from . import vaa3d


@dataclass(frozen=True)
class DecodedImage:
    case: ImageCase
    array_zyx: np.ndarray
    provenance: dict

    @property
    def dimensions_xyz(self):
        return tuple(int(x) for x in self.array_zyx.shape[::-1])


def decode_image(case: ImageCase) -> DecodedImage:
    """Read one source byte snapshot; no reference, rescaling, transpose or crop.

    The verified decoder operators are reused unchanged. This small loader
    dispatches them on the same bytes that are hashed. XYZ voxel (x,y,z) is
    array[z,y,x]; declared spacing is metadata, not an implicit resampling.
    """
    payload = case.image_path.read_bytes()
    if len(payload) < vaa3d.HEADER_SIZE:
        raise ValueError("Image is shorter than the 43-byte Vaa3D header")
    if payload[:24] not in {vaa3d.RAW_HEADER_KEY.encode("ascii"), vaa3d.PBD_HEADER_KEY.encode("ascii")}:
        raise NotImplementedError("Image format is unsupported; this port accepts Vaa3D raw/PBD only")
    header = vaa3d.parse_header(payload[:vaa3d.HEADER_SIZE], case.image_path)
    width, height, depth, channels = header.dimensions
    if channels != 1:
        raise NotImplementedError(f"Expected a scalar image, received {channels} channels")
    bytes_per_voxel = header.bytes_per_voxel  # Reject unsupported types before decoding.
    encoded = payload[vaa3d.HEADER_SIZE:]
    if header.key == vaa3d.RAW_HEADER_KEY:
        decoded = encoded
    elif header.datatype == 1:
        decoded = vaa3d.decode_pbd8(encoded, header.voxel_count)
    else:
        decoded = vaa3d.decode_pbd16(encoded, header.voxel_count,
                                   byteorder="little" if header.endian == "L" else "big")
    expected = header.voxel_count * bytes_per_voxel
    if len(decoded) != expected:
        raise ValueError(f"Image has {len(decoded)} decoded bytes; expected {expected}")
    dtype = np.dtype("u1" if header.datatype == 1 else "<u2" if header.endian == "L" else ">u2")
    array = np.frombuffer(decoded, dtype=dtype).reshape(depth, height, width)
    array.setflags(write=False)
    return DecodedImage(case, array, {
        "image_path": str(case.image_path), "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload), "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "decoded_bytes": len(decoded), "reader_version": vaa3d.VOLUME_READER_VERSION,
        "encoding": header.key, "endian": header.endian, "datatype": header.datatype,
        "dimensions_xyzc": list(header.dimensions), "array_axes": "ZYX", "coordinate_axes": "XYZ",
        "coordinate_origin_xyz": [0, 0, 0], "voxel_centers": "integer indices",
        "native_coordinates_units": "voxel", "declared_metric_units": case.units,
        "spacing_xyz": list(case.spacing_xyz),
        "metric_transform": "metric_xyz = native_xyz * spacing_xyz; not applied to array or tile boxes",
        "intensity_transform": "none", "spatial_resampling": "none", "padding": "none",
    })
