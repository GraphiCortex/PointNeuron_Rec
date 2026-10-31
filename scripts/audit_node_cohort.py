"""Bounded cohort/target-capacity preflight; image headers and explicit SWCs only.

No image payload decoding, guides, sampling, model, optimizer or reserved-file
access. Uses the frozen target constructor to count, never truncate, targets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pn2.inputs.vaa3d import HEADER_SIZE, parse_header
from pn2.inputs.sampling import partition_tiles
from pn2.evaluation.swc import parse_swc
from pn2.supervision.curve_nodes import rectangular_curve_nodes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Pinned metadata changed: {path}")
    return json.loads(Path(path).read_text())


def load_cohort(config_path):
    config = json.loads(Path(config_path).read_text())
    split_path = ROOT/config["partition"]["path"]
    split = checked(split_path, config["partition"]["sha256"])
    ids = {name:{c["sample_id"] for c in split["partitions"][name]} for name in ("train","val","test")}
    if any(ids[a]&ids[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Identity partitions overlap")
    rows, seen, paths = [], set(), defaultdict(set)
    for role, key in (("training","train"),("development","val")):
        source = config["inventories"][role]
        path = ROOT/source["path"]
        inventory = checked(path,source["sha256"])
        cases = inventory["cases"]
        if len(cases) != config["expected_counts"][role] or {c["case_id"] for c in cases} != ids[key]:
            raise ValueError("Inventory identities/counts differ from retained split")
        for case in cases:
            if case["partition"] != role or case["case_id"] in seen or case["case_id"] in ids["test"]:
                raise ValueError("Duplicate, reserved or misassigned case")
            if role == "development" and case.get("metadata",{}).get("new_gradient_training_eligible",False):
                raise ValueError("Development case must not enter gradient training")
            seen.add(case["case_id"])
            row = {**case,"image_path":str((path.parent/case["image_path"]).resolve()),
                "reference_swc":str((path.parent/case["reference_swc"]).resolve())}
            for field in ("image_path","reference_swc"):
                paths[row[field]].add(role)
            rows.append(row)
    if len(ids["test"]) != config["expected_counts"]["reserved"] or any(len(v)>1 for v in paths.values()):
        raise ValueError("Reserved count changed or an exact source path crosses train/development")
    if digest(ROOT/"src/pn2/supervision/curve_nodes.py") != config["target_source_sha256"]:
        raise ValueError("Frozen node target source changed")
    return config,rows,ids


def inspect_case(case, config, deadline):
    started = time.perf_counter()
    image, reference = Path(case["image_path"]),Path(case["reference_swc"])
    with image.open("rb") as stream:
        header = stream.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise ValueError("Incomplete image header")
    info = parse_header(header,image)
    if info.dimensions[3] != 1:
        raise ValueError("Only scalar images are supported")
    decoded_bytes = info.voxel_count*info.bytes_per_voxel
    source_hash = digest(reference)
    morphology = parse_swc(reference,type_policy="opaque_integer")
    dimensions = np.asarray(info.dimensions[:3])
    boxes = []
    for i,slices in enumerate(partition_tiles(tuple(dimensions[::-1]),config["tile_side_voxel"])):
        if time.perf_counter()>deadline:
            raise TimeoutError("Cohort preflight reached its declared wall cap; no automatic continuation")
        start = np.array([s.start for s in slices[::-1]])
        stop = np.array([s.stop for s in slices[::-1]])
        # Counting diagnostic only; the model's 512-query limit is not changed.
        target = rectangular_curve_nodes(morphology,start,stop,capacity=2**31-1)
        voxels = int(np.prod(stop-start))
        input_count = min(config["candidate_input_points"],voxels)
        queries = min(config["query_capacity"],input_count)
        boxes.append({"tile_id":i,"start_xyz":start.tolist(),"stop_xyz":stop.tolist(),
            "voxels":voxels,"candidate_input_points":input_count,"candidate_queries":queries,
            "targets":len(target.xyz),"reference_empty_unknown":len(target.xyz)==0,
            "target_overflow":len(target.xyz)>queries,"encoder_short_input":input_count<256,
            "coincident_target_identities":len(target.xyz)-len(np.unique(target.xyz,axis=0)),
            "structural_nodes":{role:sum(role in r for r in target.roles) for role in ("root","leaf","junction","crop_boundary")}})
    if digest(reference) != source_hash:
        raise ValueError("Reference changed during audit")
    low,high = np.full(3,-.5),dimensions-.5
    return {"case_id":case["case_id"],"partition":case["partition"],
        "alias":case["metadata"]["historical_sample_index"],"source_group":case["source_group"],
        "dimensions_xyz":dimensions.tolist(),"datatype":info.datatype,
        "image_header_sha256":hashlib.sha256(header).hexdigest(),"image_payload_read":False,
        "image_source_bytes":image.stat().st_size,"decoded_image_bytes":decoded_bytes,
        "reference_sha256":source_hash,"reference_nodes":morphology.node_count,
        "reference_components":morphology.component_count,
        "reference_vertices_outside_image":int((~((morphology.xyz>=low)&(morphology.xyz<=high)).all(1)).sum()),
        "boxes":boxes,"summary":{"boxes":len(boxes),
            "reference_present_boxes":sum(b["targets"]>0 for b in boxes),
            "reference_empty_unknown_boxes":sum(b["reference_empty_unknown"] for b in boxes),
            "overflow_boxes":sum(b["target_overflow"] for b in boxes),
            "maximum_targets":max((b["targets"] for b in boxes),default=0),
            "encoder_short_boxes":sum(b["encoder_short_input"] for b in boxes)},
        "seconds":time.perf_counter()-started,"status":"completed"}


def run(config_path, output):
    config,cases,ids = load_cohort(config_path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+config["wall_timeout_seconds"]
    report={"schema_version":1,"status":"running","config":config,
        "config_sha256":digest(config_path),"script_sha256":digest(__file__),
        "scope":"40 explicit train/development image headers and references; no image payload/filter/cloud/model/training or reserved files",
        "identity_counts":{k:len(v) for k,v in ids.items()},"reserved_files_opened":0,
        "training_ready":False,"records":[]}
    for case in cases:
        try:
            row=inspect_case(case,config,deadline)
        except TimeoutError as error:
            report.update(status="partial_timeout",active_case=case["case_id"],reason=str(error));break
        except (ValueError,OSError,NotImplementedError) as error:
            row={"case_id":case["case_id"],"partition":case["partition"],"status":"unscorable","reason":str(error)}
        report["records"].append(row)
        print(f"COHORT {case['partition']} alias={case['metadata']['historical_sample_index']} {row['status']} {row.get('summary',{})}",flush=True)
    if report["status"]=="running":
        report["status"]="completed" if all(r["status"]=="completed" for r in report["records"]) else "completed_with_issues"
    report["summary"]={}
    for role in ("training","development"):
        rows=[r for r in report["records"] if r["partition"]==role and r["status"]=="completed"]
        report["summary"][role]={"completed_cases":len(rows),**{key:sum(r["summary"][key] for r in rows) for key in
            ("boxes","reference_present_boxes","reference_empty_unknown_boxes","overflow_boxes","encoder_short_boxes")},
            "maximum_targets":max((r["summary"]["maximum_targets"] for r in rows),default=0)}
    hashes=defaultdict(list)
    for row in report["records"]:
        if row["status"]=="completed":hashes[row["reference_sha256"]].append((row["partition"],row["case_id"]))
    report["duplicate_reference_content_groups"]=[v for v in hashes.values() if len(v)>1]
    report["seconds"]=time.perf_counter()-started
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding="utf-8")
    print(json.dumps({"status":report["status"],"summary":report["summary"],"seconds":report["seconds"]},indent=2),flush=True)
    return 0 if report["status"]=="completed" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/node_cohort_preflight.json"))
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.config,args.output))
