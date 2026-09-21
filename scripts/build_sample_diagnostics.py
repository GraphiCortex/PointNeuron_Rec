"""Read-only diagnostic closure over saved experiments; never runs inference.

Canonical columns describe ONE selected run, not a best-of-experiments merge.
Every discovered observation remains in JSON with its source and original names.
See docs/sample_diagnostics.md for selection, missingness and triage semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

CHECKPOINT = "proposal_conservative_guarded30_cw3_oreg005_nw3.pt"
REPO = Path(__file__).resolve().parents[1]


def normalized(value):
    return str(value or "").replace("\\", "/")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numeric(value):
    if value in (None, "", "NA", "nan"):
        return None
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def sample_index(row):
    if row.get("sample_index") is not None:
        return int(row["sample_index"])
    # Adapter for legacy Gold166 CSVs, not a requirement of the output schema.
    for key in ("path", "record", "graph", "proposal"):
        match = re.search(r"sample_?(\d+)(?:_|\.|/|$)", normalized(row.get(key)))
        if match:
            return int(match[1])
    return None


def promoted(metadata):
    return (
        normalized(metadata.get("proposal_metadata", {}).get("checkpoint")).split("/")[-1] == CHECKPOINT
        and metadata.get("initializer") == "foreground_geodesic"
        and metadata.get("selection_mode") == "adaptive_connected_coverage_nms"
        and metadata.get("min_proposal_score") == 0.85
        and metadata.get("nms_mode") == "distance"
        and metadata.get("nms_distance") == 18
        and metadata.get("max_nodes") == 128
    )


class Inventory:
    def __init__(self, root, output):
        self.root, self.output = root.resolve(), output.resolve()
        self.observations = defaultdict(list)
        self.domains = defaultdict(list)
        self.sources = {}
        self.issues = []
        self.metadata = {}

    def label(self, path):
        try:
            return path.absolute().relative_to(REPO).as_posix()
        except ValueError:
            return path.absolute().as_posix()

    def resolve(self, value):
        path = Path(normalized(value))
        return path if path.is_absolute() else REPO / path

    def register(self, path, kind):
        label = self.label(path)
        if label not in self.sources:
            self.sources[label] = {"path": label, "kind": kind,
                                   "bytes": path.stat().st_size, "observations": 0}
        return label

    def add(self, path, kind, row, pointer, context=None):
        index = sample_index(row)
        if index is None:
            return
        source = self.register(path, kind)
        entry = {"source": source, "pointer": pointer, "kind": kind,
                 "values": clean(row), "context": clean(context or {})}
        target = self.domains if kind == "data_inventory" else self.observations
        target[index].append(entry)
        self.sources[source]["observations"] += 1

    def npz_metadata(self, path):
        label = self.label(path)
        if label not in self.metadata:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    self.metadata[label] = json.loads(str(archive["metadata"])) if "metadata" in archive else {}
            except (OSError, ValueError, KeyError) as error:
                self.issues.append({"source": label, "error": str(error)})
                self.metadata[label] = {}
        return self.metadata[label]

    def discover(self):
        paths = sorted(p for p in self.root.rglob("*") if p.is_file() and not p.is_relative_to(self.output))
        for path in paths:
            if path.suffix not in (".json", ".jsonl"):
                continue
            try:
                if path.suffix == ".jsonl":
                    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                        try:
                            row = json.loads(line)
                            self.add(path, "run_event", row, f"line:{number}")
                        except ValueError as error:
                            self.issues.append({"source": self.label(path), "line": number, "error": str(error)})
                    continue
                document = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(document, dict):
                    continue
                context = {k: v for k, v in document.items() if k not in
                           ("samples", "records", "failures", "edges", "domains", "source_groups", "selected", "rejected_preview", "splits")}
                if "domains" in document or document.get("dataset"):
                    kind = "data_inventory"
                elif "metric_note" in document:
                    kind = "paper_geometry"
                elif "summary" in path.stem:
                    kind = "run_summary"
                elif "topology" in path.stem:
                    kind = "topology"
                elif "selection" in path.stem:
                    kind = "selection_audit"
                else:
                    kind = "evaluation"
                for number, row in enumerate(document.get("samples", [])):
                    self.add(path, kind, row, f"samples/{number}", context)
                for number, row in enumerate(document.get("failures", [])):
                    self.add(path, "failure", row, f"failures/{number}", context)
                if "eligibility" in document:
                    for key in ("records", "rejected"):
                        for number, row in enumerate(document.get(key, [])):
                            self.add(path, "gae_eligibility", row, f"{key}/{number}", {"eligibility": document["eligibility"]})
                if "edges" in document and isinstance(document["edges"], list):
                    by_sample = defaultdict(list)
                    for row in document["edges"]:
                        if sample_index(row) is not None:
                            by_sample[sample_index(row)].append(row)
                    for index, rows in by_sample.items():
                        self.add(path, "edge_diagnostics", {"sample_index": index, "edges": rows}, "edges")
            except (OSError, ValueError, TypeError) as error:
                self.issues.append({"source": self.label(path), "error": str(error)})
        for path in paths:
            if path.suffix == ".csv" and not path.with_suffix(".json").exists():
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for number, row in enumerate(csv.DictReader(handle), 2):
                        self.add(path, "csv_evaluation", {k: numeric(v) for k, v in row.items()}, f"line:{number}")
            if path.suffix == ".npz" and ("proposals" in path.stem or "graph" in path.stem):
                metadata = self.npz_metadata(path)
                if sample_index(metadata) is not None:
                    kind = "graph_metadata" if "graph" in path.stem else "proposal_metadata"
                    self.add(path, kind, metadata, "metadata")
        # Record requested-but-unfinished sample directories. Never invent an index
        # from a range in an experiment directory name or from a count alone.
        for path in paths:
            if path.name != "summary.json":
                continue
            known = {i for i, entries in self.observations.items() for e in entries
                     if e["source"] == self.label(path)}
            for directory in sorted(path.parent.glob("sample_*")):
                match = re.fullmatch(r"sample_(\d+)", directory.name)
                if directory.is_dir() and match and int(match[1]) not in known:
                    self.add(path, "incomplete_request", {"sample_index": int(match[1]),
                             "directory": self.label(directory)}, "directory evidence")


FIELDS = """sample_index sample_id dataset_id dataset_name species brain_region cortical_layer primary_source primary_scope review_status execution_status
primary_failure_stage secondary_failure_stage likely_failure_mechanism diagnostic_evidence
diagnostic_confidence manual_review_required incomplete missing_fields
volume_width volume_height volume_depth volume_channels volume_voxels volume_shape_xyz
gt_aligned gt_out_of_bounds_nodes failure_command failure_returncode
requested_foreground_threshold foreground_threshold foreground_threshold_was_adapted foreground_voxels foreground_cap_satisfied
proposal_threshold proposal_threshold_fraction patches points_per_patch proposal_candidates proposal_nodes
proposal_coverage proposal_precision proposal_terminal_coverage proposal_branch_coverage
proposal_segment_coverage selected_segment_coverage proposal_skeleton_precision selected_skeleton_precision
proposal_endpoint_coverage selected_endpoint_coverage selected_branch_coverage segment_coverage_drop
audit_proposal_nodes audit_proposal_branch_coverage audit_proposal_score_threshold
selection_mode nms_mode nms_distance min_proposal_score max_nodes nodes
eligible_candidate_pairs reachable_edge_fraction mean_snap_distance mean_edge_geodesic_distance
mean_reachable_geodesic_distance mean_reachable_geodesic_ratio mean_edge_euclidean_distance
mean_edge_continuation mean_endpoint_score min_endpoint_score mean_endpoint_radius
edge_precision edge_recall edge_f1 bridge_edges bridge_hit_rate traced_hit_rate
swc_valid reconstruction_roots single_root reconstruction_nodes
point_distance_precision point_distance_recall point_distance_f1 approx_esa approx_dsa approx_pds
oracle_edge_f1 oracle_bridge_edges oracle_reachable_edge_fraction oracle_eligible_candidate_pairs
gae_threshold_f1 gae_topk_f1 gae_init_topk_f1 historical_failure_count source_count""".split()


def classify(row):
    """Triage evidence, not a biological correctness or benchmark pass criterion."""
    evidence, stages = [], []
    def add(stage, message):
        if stage not in stages:
            stages.append(stage)
        evidence.append(message)
    if row.get("foreground_cap_satisfied") is False:
        add("data_preprocessing", f"foreground_cap_satisfied=false; foreground={row.get('foreground_voxels')}, threshold={row.get('foreground_threshold')}")
    if row.get("execution_status") == "FAIL" and row.get("gt_aligned") is False and "aggregate_proposals.py" in (row.get("failure_command") or "") and row.get("failure_returncode") == 2:
        add("data_alignment", f"Saved data audit reports {row.get('gt_out_of_bounds_nodes')} out-of-bounds SWC nodes; aggregate_proposals.py exits 2 at its pre-inference alignment guard. Consistent with the saved failure, though subprocess output was not retained")
    oracle_reach = row.get("oracle_reachable_edge_fraction")
    if oracle_reach is not None and oracle_reach < 0.5 and (row.get("oracle_bridge_edges") or 0) > 5:
        add("connectivity_graph", f"Oracle nodes still give reachable={oracle_reach:.4f}, bridges={row['oracle_bridge_edges']}, edge_f1={row.get('oracle_edge_f1')}; proposal error alone cannot explain this")
    coverage = row.get("proposal_segment_coverage")
    if coverage is None:
        coverage = row.get("proposal_coverage")
    if coverage is not None and coverage < 0.5:
        add("proposal", f"Candidate coverage={coverage:.4f} < 0.50; substantial annotated morphology missing before graph selection")
    drop = row.get("segment_coverage_drop")
    if row.get("proposal_segment_coverage") is not None and row["proposal_segment_coverage"] >= 0.70 and drop is not None and drop >= 0.35:
        add("selection", f"Segment coverage falls from {row['proposal_segment_coverage']:.4f} to {row.get('selected_segment_coverage')}; drop={drop:.4f} >= 0.35")
    if row.get("swc_valid") is False or (row.get("reconstruction_roots") is not None and row["reconstruction_roots"] != 1):
        add("output_geometry", "Exported SWC is invalid or does not have exactly one root")
    f1 = row.get("edge_f1")
    reach = row.get("reachable_edge_fraction")
    geometry = row.get("point_distance_f1")
    if f1 is not None and f1 >= 0.80 and geometry is not None and geometry < 0.50:
        add("output_geometry", f"Edge F1={f1:.4f} >= 0.80 but point-distance F1={geometry:.4f} < 0.50; spatial output discrepancy, cause unresolved")
    flagged = (f1 is not None and f1 < 0.65) or (reach is not None and reach < 0.90) or (row.get("bridge_edges") or 0) > 5 or (geometry is not None and geometry < 0.50)
    if flagged:
        evidence.append(f"Review indicators: edge_f1={f1}, reachable={reach}, bridges={row.get('bridge_edges')}, spatial_f1={geometry}")
    if row.get("oracle_edge_f1", 0) is not None and (row.get("oracle_edge_f1") or 0) >= 0.80 and f1 is not None and f1 < 0.65:
        evidence.append("Oracle graph recovers; upstream node/selection sensitivity is supported, but proposal versus selection causality is not isolated")
    if row.get("execution_status") == "FAIL":
        evidence.append("Saved run failed; see failure observations for command/returncode. Subprocess failure alone does not establish a preprocessing cause")
    if row.get("incomplete"):
        evidence.append("Incomplete diagnostic record: " + ", ".join(row["missing_fields"]))
    failure = bool(stages or flagged or row.get("execution_status") == "FAIL" or row.get("incomplete"))
    primary = stages[0] if len(stages) == 1 else "mixed" if stages else "unknown" if failure else None
    return dict(review_status="REVIEW" if failure else "CLEAR", primary_failure_stage=primary,
                secondary_failure_stage=";".join(stages) if len(stages) > 1 else None,
                likely_failure_mechanism="; ".join(evidence) if evidence else "No trigger in recorded diagnostics; this is not proof of biological validity",
                diagnostic_evidence="; ".join(evidence) if evidence else "Completed run with no triage trigger",
                diagnostic_confidence="low" if primary == "unknown" else "moderate" if failure else "limited",
                manual_review_required=failure)


def build_rows(inventory, dataset_id, dataset_name=None):
    rows = []
    promoted_sources = {e["source"] for entries in inventory.observations.values() for e in entries
                        if e["kind"] == "run_summary" and e["values"].get("graph_path")
                        and promoted(inventory.npz_metadata(inventory.resolve(e["values"]["graph_path"])))}
    for index, observations in sorted(inventory.observations.items()):
        row = dict.fromkeys(FIELDS)
        row.update(sample_index=index, dataset_id=dataset_id, dataset_name=dataset_name or dataset_id,
                   field_sources={}, conflicts=[])
        def put(key, value, source):
            if value is None:
                return
            value = clean(value)
            if row.get(key) is not None and row[key] != value:
                row["conflicts"].append({"field": key, "kept": row[key], "other": value, "source": source})
                return
            row[key] = value
            row["field_sources"][key] = source
        def copy(values, keys, source):
            for key in keys:
                put(key, values.get(key), source + "#" + key)
        runs = [e for e in observations if e["kind"] == "run_summary" and e["values"].get("graph_path")]
        failures = [e for e in observations if e["kind"] == "failure"]
        candidates = []
        for entry in runs:
            metadata = inventory.npz_metadata(inventory.resolve(entry["values"]["graph_path"]))
            candidates.append((0 if promoted(metadata) else 2, entry, metadata))
        # Failed promoted requests lack graph metadata: identify the saved command
        # and corroborate the graph configuration from successful siblings.
        for entry in failures:
            rank = 0 if entry["source"] in promoted_sources and CHECKPOINT in entry["values"].get("command", "") else 3
            candidates.append((rank, entry, {}))
        candidates.sort(key=lambda item: (item[0], item[1]["source"], item[1]["pointer"]))
        selected = candidates[0] if candidates else None
        graph_path, proposal_path, swc_path, compare_html = "", "", "", ""
        if selected:
            rank, entry, metadata = selected
            source, values = entry["source"], entry["values"]
            row.update(primary_source=source, primary_scope="promoted" if rank == 0 else "historical",
                       execution_status="FAIL" if entry["kind"] == "failure" else "PASS")
            copy(values, FIELDS, source)
            if entry["kind"] == "failure":
                put("failure_command", values.get("command"), source + "#command")
                put("failure_returncode", values.get("returncode"), source + "#returncode")
            graph_path, proposal_path, swc_path = (normalized(values.get(k)) for k in ("graph_path", "proposal_path", "swc_path"))
            compare_html = normalized(values.get("compare_html"))
            copy(metadata, FIELDS, graph_path + "#metadata")
            proposal_metadata = metadata.get("proposal_metadata", {})
            if proposal_path:
                proposal_metadata = inventory.npz_metadata(inventory.resolve(proposal_path)) or proposal_metadata
            for key, target in (("threshold", "proposal_threshold"), ("threshold_fraction", "proposal_threshold_fraction"), ("patches", "patches")):
                put(target, proposal_metadata.get(key), proposal_path + "#metadata/" + key)
            for key, target in (("selected", "proposal_nodes"), ("coverage", "proposal_coverage"), ("precision", "proposal_precision"), ("terminal_coverage", "proposal_terminal_coverage"), ("branch_coverage", "proposal_branch_coverage")):
                put(target, proposal_metadata.get("metrics", {}).get(key), proposal_path + "#metadata/metrics/" + key)
            if proposal_path and inventory.resolve(proposal_path).exists():
                with np.load(inventory.resolve(proposal_path), allow_pickle=False) as archive:
                    if "all_centers" in archive:
                        put("proposal_candidates", len(archive["all_centers"]), proposal_path + "#len(all_centers)")
            if graph_path and inventory.resolve(graph_path).exists():
                # Derived from saved graph arrays only; no image decoding or GT inference.
                with np.load(inventory.resolve(graph_path), allow_pickle=False) as archive:
                    if "edges" in archive and len(archive["edges"]):
                        edges = archive["edges"].astype(int)
                        for array, field, operation in (("scores", "mean_endpoint_score", np.mean),
                                                       ("scores", "min_endpoint_score", np.min),
                                                       ("radii", "mean_endpoint_radius", np.mean)):
                            if array in archive:
                                put(field, float(operation(archive[array][edges])), graph_path + "#" + array + "[edges]")
                        if all(k in archive for k in ("edge_geodesic_reachable", "edge_tree_distance", "edge_euclidean_distance")):
                            reachable = archive["edge_geodesic_reachable"].astype(bool)
                            distance = archive["edge_tree_distance"]
                            euclidean = archive["edge_euclidean_distance"]
                            if reachable.any():
                                put("mean_reachable_geodesic_distance", float(np.mean(distance[reachable])), graph_path + "#edge_tree_distance[reachable]")
                                put("mean_reachable_geodesic_ratio", float(np.mean(distance[reachable] / np.maximum(euclidean[reachable], 1e-3))), graph_path + "#mean(distance/euclidean)[reachable]")
                        if all(k in archive for k in ("centers", "edge_path_points", "edge_path_offsets")):
                            # Reuse the existing diagnostic definition: maximum absolute
                            # alignment to another incident edge at each endpoint.
                            from diagnose_geodesic_edges import continuation_support
                            support = continuation_support(edges, archive["centers"], archive["edge_path_points"], archive["edge_path_offsets"])
                            put("mean_edge_continuation", float(np.mean(support[:, 0])), graph_path + "#continuation_support/mean")
        else:
            row.update(primary_scope="auxiliary_only", execution_status=None)
        for entry in observations:
            values, source, kind = entry["values"], entry["source"], entry["kind"]
            copy(values, ("species", "brain_region", "cortical_layer"), source)
            if kind == "topology" and graph_path and compare_html and normalized(values.get("compare_html")) == compare_html and normalized(Path(source).parent) == normalized(Path(graph_path).parent.parent):
                copy(values, FIELDS, source)
            if kind == "selection_audit" and graph_path and normalized(values.get("graph_path")) == graph_path:
                # Audit proposal_score_threshold is the GRAPH filter, not the local proposal cutoff.
                copy(values, [k for k in FIELDS if k not in ("proposal_coverage", "proposal_precision", "proposal_nodes", "proposal_branch_coverage")], source)
                for key in ("proposal_nodes", "proposal_branch_coverage", "proposal_score_threshold"):
                    put("audit_" + key, values.get(key), source + "#" + key)
            if kind == "paper_geometry" and swc_path and normalized(values.get("pred_swc")) == swc_path:
                for key in ("precision", "recall", "f1"):
                    put("point_distance_" + key, values.get(key), source + "#" + key)
                copy(values, ("approx_esa", "approx_dsa", "approx_pds"), source)
            if "beast_oracle_diagnosis" in source and kind in ("run_summary", "topology"):
                for key in ("edge_f1", "bridge_edges", "reachable_edge_fraction", "eligible_candidate_pairs"):
                    put("oracle_" + key, values.get(key), source + "#" + key)
            if "connectivity_guarded30_eligible/connectivity_eval_50e.csv" in source:
                for key in ("threshold_f1", "topk_f1", "init_topk_f1"):
                    put("gae_" + key, values.get(key), source + "#" + key)
        for entry in inventory.domains.get(index, []):
            values = entry["values"]
            # Raw/GT inventory swc_valid MUST NOT become reconstruction swc_valid.
            copy(values, ["sample_id"] + [k for k in FIELDS if k.startswith("volume_")], entry["source"])
            put("gt_aligned", values.get("aligned"), entry["source"] + "#aligned")
            put("gt_out_of_bounds_nodes", values.get("out_of_bounds_nodes"), entry["source"] + "#out_of_bounds_nodes")
            copy(values, ("species", "brain_region", "cortical_layer"), entry["source"])
        if row["sample_id"] is None:
            for entry in observations:
                if entry["values"].get("sample_id"):
                    put("sample_id", entry["values"]["sample_id"], entry["source"])
                    break
        if all(row[k] is not None for k in ("volume_width", "volume_height", "volume_depth")):
            row["volume_shape_xyz"] = [row[k] for k in ("volume_width", "volume_height", "volume_depth")]
        if row["reconstruction_roots"] is not None:
            row["single_root"] = row["reconstruction_roots"] == 1
        # Preserve the evaluator's empty-class convention, with explicit denominator.
        if row["bridge_edges"] == 0:
            row["bridge_hit_rate_convention"] = "1.0 in evaluator when no bridges; no empirical bridge accuracy evidence"
        row["missing_fields"] = [k for k in ("primary_source", "edge_f1", "swc_valid", "proposal_coverage", "point_distance_f1") if row.get(k) is None]
        row["incomplete"] = bool(row["missing_fields"])
        # Spatial metrics are optional for triage CLEAR, but their absence is still
        # exposed as diagnostic incompleteness. Do not turn absent geometry into 0.
        critical_missing = [k for k in row["missing_fields"] if k != "point_distance_f1"]
        decision_input = dict(row, incomplete=bool(critical_missing), missing_fields=critical_missing)
        row.update(classify(decision_input))
        row["historical_failure_count"] = len({(e["source"], json.dumps(e["values"], sort_keys=True)) for e in failures})
        row["source_count"] = len({e["source"] for e in observations})
        row["observations"] = observations
        row["data_observations"] = inventory.domains.get(index, [])
        if row["conflicts"]:
            row["manual_review_required"] = True
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=REPO / "tmp")
    parser.add_argument("--output-dir", type=Path, default=REPO / "tmp/diagnostics")
    parser.add_argument("--dataset-id", default="Gold166")
    parser.add_argument("--dataset-name", help="Display name; defaults to --dataset-id. One dataset per artifact root.")
    args = parser.parse_args()
    inventory = Inventory(args.artifact_root, args.output_dir)
    inventory.discover()
    rows = build_rows(inventory, args.dataset_id, args.dataset_name)
    counts = dict(Counter(r["primary_failure_stage"] or "no_trigger" for r in rows))
    summary = {"sample_count": len(rows), "promoted_samples": sum(r["primary_scope"] == "promoted" for r in rows),
               "incomplete_samples": sum(r["incomplete"] for r in rows),
               "promoted_execution_failures": sum(r["primary_scope"] == "promoted" and r["execution_status"] == "FAIL" for r in rows),
               "failure_stage_counts": counts, "review_status_counts": dict(Counter(r["review_status"] for r in rows)),
               "missing_field_counts": {k: sum(r.get(k) is None for r in rows) for k in FIELDS},
               "source_count": len(inventory.sources), "discovery_issues": inventory.issues}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "status_note": "CLEAR means no triage trigger; REVIEW includes incomplete/unknown. execution_status records completion separately as PASS/FAIL or null when unknown.",
               "summary": summary, "sources": sorted(inventory.sources.values(), key=lambda x: x["path"]), "samples": rows}
    (output / "pointneuron1_sample_diagnostics.json").write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "pointneuron1_sample_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if row.get(k) is None else json.dumps(row[k]) if isinstance(row[k], (list, dict)) else row[k] for k in FIELDS})
    (output / "pointneuron1_sources.json").write_text(json.dumps(payload["sources"], indent=2) + "\n", encoding="utf-8")
    compact = {k: v for k, v in summary.items() if k != "missing_field_counts"}
    lines = ["# PointNeuron1.0 diagnostic closure", "", "review_status=CLEAR means no recorded triage trigger, not independently established biological plausibility. REVIEW includes incomplete/unknown records. execution_status=PASS/FAIL records actual run completion separately (null when unknown).", "", "```json", json.dumps(compact, indent=2), "```", "", "## Sample-level review", "", "| Sample | Scope | Stage | Evidence |", "|---|---|---|---|"]
    for row in rows:
        if row["review_status"] == "REVIEW":
            lines.append(f"| {row['sample_index']} | {row['primary_scope']} | {row['primary_failure_stage']} | {row['diagnostic_evidence']} |")
    lines += ["", "## Sources", "", "Exact paths and observation counts: `pointneuron1_sources.json`. Original metric names, missing-field counts and field provenance: `pointneuron1_sample_diagnostics.json`."]
    (output / "pointneuron1_failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_field_counts"}, indent=2))


if __name__ == "__main__":
    main()
