"""Reference-free, immutable image-point cache for the fixed node study.

This module never loads references, targets, model weights or a foreground mask.
Guide reuse changes execution only; all input locations/intensities come from
the newly decoded original image and all image-defined boxes are prepared.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .cases import ImageCase
from .image import decode_image
from .vaa3d import read_header
from .disk_guide import disk_sato_guide, mapped, read_block
from .sampling import normalized_image, partition_tiles, probabilities, ranked_selection, source_indices
from pn2.evaluation.branch_evidence import source_priority_order

PACKAGES=("numpy","scipy","scikit-image","psutil")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_once(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def versions():
    return {name:importlib.metadata.version(name) for name in PACKAGES}


def case_key(case):
    return f"{case.partition}_{hashlib.sha256(case.case_id.encode()).hexdigest()[:16]}"


def point_arrays(raw,guide_tile,tile,tile_id,settings,normalization):
    """One image-defined box; no target/count-dependent point allocation."""
    probability,fallback=probabilities(guide_tile,settings["uniform_mass"])
    count=settings["points_per_box"]
    if raw[tile].size<count:
        raise ValueError("Source box is too short for this fixed study; no padding or truncation")
    local_ids=ranked_selection(probability,count,settings["cloud_seed"],tile_id)
    flat,xyz=source_indices(local_ids,tile,raw.shape)
    order=source_priority_order(flat,settings["ordering_seed"])
    flat,xyz=flat[order],xyz[order].astype(np.float64)
    intensity=raw.ravel()[flat].copy()
    start=np.asarray([s.start for s in tile][::-1])
    stop=np.asarray([s.stop for s in tile][::-1])
    center=(start+stop-1.)/2.
    local=((xyz-center)/settings["coordinate_scale_voxel"]).astype(np.float32)
    lo,hi=normalization["minimum"],normalization["maximum"]
    # Match the successful adapter's feature rounding, independently of the
    # float32 normalized volume used to construct the structural guide.
    scaled=(np.zeros(len(flat),dtype=np.float32) if hi==lo else
            ((intensity.astype(np.float64)-lo)/(hi-lo)).astype(np.float32))
    arrays={"flat_indices":flat,"native_xyz":xyz,"raw_intensity":intensity,
            "local_xyz":local,"attributes":np.column_stack((local,scaled)).astype(np.float32)}
    if (len(np.unique(flat))!=count or not np.all((xyz>=start)&(xyz<stop))
            or not np.allclose(local*settings["coordinate_scale_voxel"]+center,xyz,atol=1e-5,rtol=0)
            or not np.isfinite(arrays["attributes"]).all() or np.any((scaled<0)|(scaled>1))):
        raise ValueError("Source coordinate/intensity/normalization contract failed")
    return arrays,{"tile_id":tile_id,"box":{"start_xyz":start.tolist(),"stop_xyz":stop.tolist()},
        "center_xyz":center.tolist(),"slots":count,"distinct_points":count,"duplicate_slots":0,
        "coordinate_scale_voxel":settings["coordinate_scale_voxel"],"uniform_fallback":bool(fallback),
        "source_tile_constant":bool(raw[tile].min()==raw[tile].max())}


def require_reusable_guide(entry,case,image,normalization,settings,current_versions):
    if (entry["case_id"]!=case.case_id or entry["source_sha256"]!=image["source_sha256"]
            or entry["decoded_sha256"]!=image["decoded_sha256"]
            or entry["normalization"]!=normalization or entry["sigmas_voxel"]!=settings["sigmas_voxel"]
            or entry["versions"]!=current_versions):
        raise ValueError("Existing guide provenance differs from the decoded source/recipe/runtime")
    if digest(entry["guide_path"])!=entry["guide_sha256"]:
        raise ValueError("Existing guide content changed; preserve it, do not silently rebuild")


def verify_case(folder,case,identity,*,verify_arrays=True):
    """Validate an explicitly named completed cache; no raw image/reference access."""
    folder=Path(folder)
    marker=json.loads((folder/"complete.json").read_text())
    if marker["case_json_sha256"]!=digest(folder/"case.json"):
        raise ValueError("Case receipt changed or was not completed")
    receipt=json.loads((folder/"case.json").read_text())
    if (receipt["status"]!="completed" or receipt["case_id"]!=case.case_id
            or receipt["partition"]!=case.partition or receipt["cache_identity"]!=identity):
        raise ValueError("Completed case belongs to another source/partition/cache policy")
    if receipt["guide"]["path"]!="sato_guide.npy":raise ValueError("Invalid cache guide member")
    settings=identity["settings"]
    xyz=receipt["image"]["dimensions_xyzc"][:3]
    tiles=list(partition_tiles(tuple(xyz[::-1]),settings["tile_side_voxel"]))
    if receipt["tile_count"]!=len(tiles) or len(receipt["records"])!=len(tiles):
        raise ValueError("Cache does not cover every image-defined box")
    for i,(row,tile) in enumerate(zip(receipt["records"],tiles)):
        if (row["tile_id"]!=i or row["packet"]!=f"tile{i:05d}.npz"
                or row["slots"]!=settings["points_per_box"] or row["distinct_points"]!=row["slots"]
                or row["box"]!={"start_xyz":[s.start for s in tile][::-1],"stop_xyz":[s.stop for s in tile][::-1]}):
            raise ValueError("Cache tile identity/point count/bounds changed")
    if digest(folder/receipt["guide"]["path"])!=receipt["guide"]["sha256"]:
        raise ValueError("Completed guide changed")
    if verify_arrays:
        for row in receipt["records"]:
            if digest(folder/row["packet"])!=row["packet_sha256"]:
                raise ValueError("Completed point packet changed")
    return receipt


def build_case(case:ImageCase,settings,folder,identity,*,reuse=None,block_side=64,reserve_bytes=2*1024**3,progress=print):
    """Create exactly one fresh case; failures preserve their owned files."""
    if case.partition not in ("training","development","synthetic"):
        raise ValueError("Reserved/unknown partitions are forbidden")
    folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError(folder)
    header=read_header(case.image_path)
    if header.dimensions[3]!=1:raise ValueError("Only scalar images are supported")
    tiles=list(partition_tiles(tuple(header.dimensions[:3][::-1]),settings["tile_side_voxel"]))
    peak_disk=header.voxel_count*4*(1 if reuse else 12)+len(tiles)*settings["points_per_box"]*64
    if shutil.disk_usage(folder.parent).free<peak_disk+reserve_bytes:
        raise OSError("Insufficient free space for this image's guide/work files and reserve")
    folder.mkdir()
    write_once(folder/"request.json",{"case_id":case.case_id,"partition":case.partition,
        "cache_identity":identity,"source_image":str(case.image_path),"settings":settings})
    began=time.perf_counter();decoded=decode_image(case);raw=decoded.array_zyx
    decode_seconds=time.perf_counter()-began
    minimum,maximum=float(raw.min()),float(raw.max())
    normalization={"minimum":minimum,"maximum":maximum,"constant":minimum==maximum}
    current_versions=versions();guide_path=folder/"sato_guide.npy";normalized_path=folder/"normalized.npy"
    one=time.perf_counter()
    if reuse:
        require_reusable_guide(reuse,case,decoded.provenance,normalization,settings,current_versions)
        shutil.copyfile(reuse["guide_path"],guide_path)
        if digest(guide_path)!=reuse["guide_sha256"]:raise ValueError("Copied guide differs from verified original")
        execution={"kind":"verified_copy","original_guide":str(reuse["guide_path"]),
            "source_receipt_sha256":reuse["source_receipt_sha256"],"filter_executed":False}
    else:
        progress(f"GUIDE fresh CPU disk filter: {case.case_id}")
        normalized,transform=normalized_image(raw)
        if transform!=normalization:raise ValueError("Normalization metadata differs")
        np.save(normalized_path,normalized,allow_pickle=False);del normalized
        _,execution=disk_sato_guide(normalized_path,settings["sigmas_voxel"],folder/"guide_work",
            block_side=block_side,progress=progress,output_path=guide_path)
        (folder/"guide_work").rmdir()  # Empty directory created by this invocation only.
        execution={"kind":"fresh_disk","filter_executed":True,**execution}
    guide_seconds=time.perf_counter()-one
    with mapped(guide_path) as guide:
        if guide.shape!=raw.shape or guide.dtype!=np.float32:
            raise ValueError("Guide dimensions/type differ from the decoded image")
    one=time.perf_counter();rows=[]
    for tile_id,tile in enumerate(tiles):
        guide_tile=read_block(guide_path,tile)
        arrays,row=point_arrays(raw,guide_tile,tile,tile_id,settings,normalization)
        packet=folder/f"tile{tile_id:05d}.npz"
        np.savez_compressed(packet,**arrays)
        with np.load(packet,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or not all(np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError("Point packet failed exact serialization round trip")
        row.update(packet=packet.name,packet_sha256=digest(packet));rows.append(row)
        progress(f"CLOUD tile={tile_id+1}/{len(tiles)} distinct={row['distinct_points']} fallback={row['uniform_fallback']}")
    if normalized_path.exists():normalized_path.unlink()  # Owned scratch file, after successful sampling only.
    import psutil
    memory=psutil.Process().memory_info()
    receipt={"schema_version":1,"status":"completed","case_id":case.case_id,"partition":case.partition,
        "cache_identity":identity,"image":decoded.provenance,"normalization":normalization,"versions":current_versions,
        "guide":{"path":guide_path.name,"sha256":digest(guide_path),"execution":execution},
        "records":rows,"tile_count":len(rows),"decode_seconds":decode_seconds,"guide_seconds":guide_seconds,
        "sampling_and_save_seconds":time.perf_counter()-one,"total_seconds":time.perf_counter()-began,
        "memory":{"peak_working_set_bytes":int(getattr(memory,"peak_wset",memory.rss)),
            "kind":"process_peak_working_set" if hasattr(memory,"peak_wset") else "current_RSS_not_peak",
            "peak_pagefile_bytes":int(memory.peak_pagefile) if hasattr(memory,"peak_pagefile") else None},
        "references_opened":0,"model_executed":False,"targets_created":False}
    write_once(folder/"case.json",receipt)
    write_once(folder/"complete.json",{"case_json_sha256":digest(folder/"case.json"),"cache_identity":identity})
    return receipt
