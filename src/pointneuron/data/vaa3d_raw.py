from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RAW_HEADER_KEY = "raw_image_stack_by_hpeng"
PBD_HEADER_KEY = "v3d_volume_pkbitdf_encod"
HEADER_SIZE = 43


@dataclass(frozen=True)
class Vaa3dHeader:
    key: str
    endian: str
    datatype: int
    dimensions: tuple[int, int, int, int]
    data_offset: int = HEADER_SIZE

    @property
    def voxel_count(self) -> int:
        x, y, z, channels = self.dimensions
        return x * y * z * channels

    @property
    def bytes_per_voxel(self) -> int:
        if self.datatype not in {1, 2}:
            raise NotImplementedError(f"Vaa3D datatype {self.datatype} is not supported yet")
        return self.datatype


@dataclass(frozen=True)
class Vaa3dVolume:
    path: Path
    header: Vaa3dHeader
    data: bytes

    @property
    def dimensions(self) -> tuple[int, int, int, int]:
        return self.header.dimensions


def read_header(path: str | Path) -> Vaa3dHeader:
    volume_path = Path(path)
    with volume_path.open("rb") as file:
        header = file.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise ValueError(f"{volume_path}: file is too small for a Vaa3D header")
    return parse_header(header, volume_path)


def read_volume(path: str | Path) -> Vaa3dVolume:
    volume_path = Path(path)
    payload = volume_path.read_bytes()
    header = parse_header(payload[:HEADER_SIZE], volume_path)
    encoded = payload[HEADER_SIZE:]

    if header.key == RAW_HEADER_KEY:
        data = encoded
    elif header.key == PBD_HEADER_KEY:
        if header.datatype == 1:
            data = decode_pbd8(encoded, header.voxel_count)
        elif header.datatype == 2:
            data = decode_pbd16(encoded, header.voxel_count, byteorder=_byteorder(header.endian))
        else:
            raise NotImplementedError(f"{volume_path}: PBD datatype {header.datatype} is not supported yet")
    else:
        raise ValueError(f"{volume_path}: unsupported Vaa3D header key: {header.key!r}")

    expected_bytes = header.voxel_count * header.bytes_per_voxel
    if len(data) != expected_bytes:
        raise ValueError(f"{volume_path}: decoded {len(data)} bytes, expected {expected_bytes}")

    return Vaa3dVolume(path=volume_path, header=header, data=data)


def parse_header(header: bytes, path: Path | None = None) -> Vaa3dHeader:
    label = f"{path}: " if path is not None else ""
    try:
        key = header[:24].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}invalid Vaa3D header key") from exc

    if key not in {RAW_HEADER_KEY, PBD_HEADER_KEY}:
        raise ValueError(f"{label}unsupported Vaa3D header key: {key!r}")

    endian = chr(header[24])
    try:
        byteorder = _byteorder(endian)
    except ValueError as exc:
        raise ValueError(f"{label}unsupported endian marker: {endian!r}")
    byteorder = byteorder

    datatype = int.from_bytes(header[25:27], byteorder)
    dimensions = tuple(
        int.from_bytes(header[27 + index * 4 : 31 + index * 4], byteorder)
        for index in range(4)
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"{label}invalid Vaa3D dimensions: {dimensions}")

    return Vaa3dHeader(key=key, endian=endian, datatype=datatype, dimensions=dimensions)  # type: ignore[arg-type]


def decode_pbd8(encoded: bytes, expected_voxels: int) -> bytes:
    output = bytearray(expected_voxels)
    input_index = 0
    output_index = 0
    previous = 0

    while input_index < len(encoded) and output_index < expected_voxels:
        control = encoded[input_index]
        input_index += 1

        if control < 33:
            length = control + 1
            end = output_index + length
            if end > expected_voxels or input_index + length > len(encoded):
                raise ValueError("PBD8 literal run exceeds stream bounds")
            output[output_index:end] = encoded[input_index : input_index + length]
            previous = output[end - 1]
            input_index += length
            output_index = end
        elif control < 128:
            length = control - 32
            packed_length = (length + 3) // 4
            if output_index + length > expected_voxels or input_index + packed_length > len(encoded):
                raise ValueError("PBD8 difference run exceeds stream bounds")
            for packed in encoded[input_index : input_index + packed_length]:
                for shift in (6, 4, 2, 0):
                    if length == 0:
                        break
                    diff_code = (packed >> shift) & 0x03
                    diff = diff_code if diff_code < 3 else -1
                    previous = (previous + diff) & 0xFF
                    output[output_index] = previous
                    output_index += 1
                    length -= 1
            input_index += packed_length
        else:
            length = control - 127
            if input_index >= len(encoded) or output_index + length > expected_voxels:
                raise ValueError("PBD8 repeat run exceeds stream bounds")
            value = encoded[input_index]
            input_index += 1
            output[output_index : output_index + length] = bytes([value]) * length
            previous = value
            output_index += length

    if output_index != expected_voxels:
        raise ValueError(f"PBD8 decoded {output_index} voxels, expected {expected_voxels}")
    if input_index != len(encoded):
        raise ValueError(f"PBD8 stream has {len(encoded) - input_index} trailing bytes")

    return bytes(output)


def decode_pbd16(encoded: bytes, expected_voxels: int, byteorder: str = "little") -> bytes:
    output = bytearray(expected_voxels * 2)
    input_index = 0
    output_index = 0
    previous = 0

    def write_value(value: int) -> None:
        nonlocal output_index
        output[output_index * 2 : output_index * 2 + 2] = value.to_bytes(2, byteorder)
        output_index += 1

    while input_index < len(encoded) and output_index < expected_voxels:
        control = encoded[input_index]
        input_index += 1

        if control < 32:
            length = control + 1
            byte_length = length * 2
            if output_index + length > expected_voxels or input_index + byte_length > len(encoded):
                raise ValueError("PBD16 literal run exceeds stream bounds")
            for _ in range(length):
                previous = int.from_bytes(encoded[input_index : input_index + 2], byteorder)
                input_index += 2
                write_value(previous)
        elif control < 80:
            length = control - 31
            packed_length = (length * 3 + 7) // 8
            if output_index + length > expected_voxels or input_index + packed_length > len(encoded):
                raise ValueError("PBD16 3-bit difference run exceeds stream bounds")
            bit_buffer = 0
            available_bits = 0
            decoded = 0
            for packed in encoded[input_index : input_index + packed_length]:
                bit_buffer = (bit_buffer << 8) | packed
                available_bits += 8
                while available_bits >= 3 and decoded < length:
                    available_bits -= 3
                    diff_code = (bit_buffer >> available_bits) & 0x07
                    previous = (previous + _signed_difference(diff_code, 3)) & 0xFFFF
                    write_value(previous)
                    decoded += 1
                bit_buffer &= (1 << available_bits) - 1 if available_bits else 0
            if decoded != length:
                raise ValueError("PBD16 3-bit difference run ended early")
            input_index += packed_length
        elif control < 223:
            length = control - 79
            packed_length = (length + 1) // 2
            if output_index + length > expected_voxels or input_index + packed_length > len(encoded):
                raise ValueError("PBD16 4-bit difference run exceeds stream bounds")
            decoded = 0
            for packed in encoded[input_index : input_index + packed_length]:
                for shift in (4, 0):
                    if decoded == length:
                        break
                    diff_code = (packed >> shift) & 0x0F
                    previous = (previous + _signed_difference(diff_code, 4)) & 0xFFFF
                    write_value(previous)
                    decoded += 1
            input_index += packed_length
        else:
            length = control - 222
            if input_index + 2 > len(encoded) or output_index + length > expected_voxels:
                raise ValueError("PBD16 repeat run exceeds stream bounds")
            previous = int.from_bytes(encoded[input_index : input_index + 2], byteorder)
            input_index += 2
            for _ in range(length):
                write_value(previous)

    if output_index != expected_voxels:
        raise ValueError(f"PBD16 decoded {output_index} voxels, expected {expected_voxels}")
    if input_index != len(encoded):
        raise ValueError(f"PBD16 stream has {len(encoded) - input_index} trailing bytes")

    return bytes(output)


def _signed_difference(code: int, bits: int) -> int:
    sign_threshold = 1 << (bits - 1)
    if code < sign_threshold:
        return code
    return code - (1 << bits)


def _byteorder(endian: str) -> str:
    if endian == "L":
        return "little"
    if endian == "B":
        return "big"
    raise ValueError(endian)


def volume_stats(data: bytes, datatype: int = 1, endian: str = "L") -> dict[str, float | int]:
    if not data:
        return {"min": 0, "max": 0, "mean": 0.0, "nonzero": 0}
    if datatype == 1:
        values = np.frombuffer(data, dtype=np.uint8)
    elif datatype == 2:
        values = np.frombuffer(data, dtype=np.dtype("<u2" if endian == "L" else ">u2"))
    else:
        raise NotImplementedError(f"Vaa3D datatype {datatype} is not supported yet")
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "nonzero": int(np.count_nonzero(values)),
    }
