"""Read-only inventory of explicit training SWC graphs; never repair annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.evaluation.swc import parse_swc
from pn2.inputs.cases import read_object


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(morphology):
    m = morphology
    roots = np.full(m.node_count, -1, dtype=np.int64)
    for i in range(m.node_count):
        trail, j = [], i
        while roots[j] < 0 and m.parents[j] >= 0:
            trail.append(j)
            j = int(m.parents[j])
        root = int(roots[j]) if roots[j] >= 0 else j
        roots[j] = root
        roots[trail] = root
    _, inverse = np.unique(m.xyz, axis=0, return_inverse=True)
    coordinate_roots = defaultdict(set)
    for group, root in zip(inverse, roots):
        coordinate_roots[int(group)].add(int(root))
    shared = {group for group, group_roots in coordinate_roots.items() if len(group_roots) > 1}
    shared_node = np.array([int(group) in shared for group in inverse], dtype=bool)
    degrees = m.degrees
    return {"nodes": m.node_count, "edges": len(m.edges), "components": m.component_count,
            "endpoints": int(np.count_nonzero(degrees == 1)),
            "junctions": int(np.count_nonzero(degrees >= 3)),
            "isolated_nodes": len(m.isolated_points), "curve_length_native": m.total_length,
            "zero_length_edges": int(np.count_nonzero(np.linalg.norm(m.segments[:, 1]-m.segments[:, 0], axis=1) == 0)),
            "unique_xyz": len(coordinate_roots),
            "exact_coordinate_groups_across_components": len(shared),
            "endpoints_coincident_with_node_in_other_component": int(np.count_nonzero(shared_node & (degrees == 1))),
            "negative_type_rows": int(np.count_nonzero(m.types < 0)),
            "reported_radius_range": [float(m.radii.min()), float(m.radii.max())] if m.node_count else None,
            "reported_unique_radii": len(np.unique(m.radii)),
            "reported_radius_one_fraction": float(np.mean(m.radii == 1)) if m.node_count else None,
            "reported_zero_radius_nodes": int(np.count_nonzero(m.radii == 0))}


def run(inventory_path, output):
    inventory_path = Path(inventory_path).resolve()
    inventory = read_object(inventory_path)
    cases = inventory["cases"]
    if not cases or any(c["partition"] != "training" for c in cases):
        raise ValueError("Only an explicit training-reference inventory is allowed")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate training case IDs")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case in cases:
        path = (inventory_path.parent / case["reference_swc"]).resolve()
        row = {"case_id": case["case_id"], "reference_path": str(path),
               "historical_sample_index": case.get("metadata", {}).get("historical_sample_index")}
        try:
            before = digest(path)
            row["source_sha256"] = before
            m = parse_swc(path, type_policy="opaque_integer")
            row.update(describe(m))
            if digest(path) != before:
                raise ValueError("Reference bytes changed during audit")
            row["status"] = "valid_forest" if m.node_count else "empty_reference"
        except (OSError, ValueError) as error:
            row.update({"status": "unscorable_reference", "reason": str(error)})
        rows.append(row)
        print(f"AUDIT alias={row['historical_sample_index']} status={row['status']} "
              f"components={row.get('components')} shared_endpoints={row.get('endpoints_coincident_with_node_in_other_component')}", flush=True)
    valid = [r for r in rows if r["status"] == "valid_forest"]
    report = {"schema_version": 1,
              "scope": "Training-reference structure only; no images, development/test references, geometry edits, node welding, anatomical type interpretation, target construction or training",
              "inventory_sha256": digest(inventory_path), "script_sha256": digest(__file__),
              "reader_sha256": digest(Path(__file__).resolve().parents[1] / "src/pn2/evaluation/swc.py"),
              "reference_type_policy": "opaque_integer", "cases": rows,
              "summary": {"declared": len(cases), "statuses": dict(Counter(r["status"] for r in rows)),
                  "multiple_component_sources": sum(r["components"] > 1 for r in valid),
                  "sources_with_cross_component_exact_coordinate_groups": sum(r["exact_coordinate_groups_across_components"] > 0 for r in valid),
                  "sources_without_encoded_junctions": sum(r["junctions"] == 0 for r in valid),
                  "sources_with_negative_type_metadata": sum(r["negative_type_rows"] > 0 for r in valid),
                  "sources_with_all_reported_radii_one": sum(r["reported_radius_one_fraction"] == 1 for r in valid)},
              "interpretation": "Counts describe file encoding. Coincidence is not proof of biological connectivity; constant radii are not proof of placeholders. Source annotation semantics/completeness and physical calibration remain unverified.",
              "automatic_label_repair": False, "automatic_training_authorized": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.inventory, args.output)
