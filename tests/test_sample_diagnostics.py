"""Regression checks for false attribution and cross-experiment contamination."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/build_sample_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("diagnostics", PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class DiagnosticsTests(unittest.TestCase):
    def test_different_initializers_and_audit_denominators_stay_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample_0001"
            sample.mkdir()
            graph = sample / "graph.npz"
            proposal = sample / "proposals.npz"
            np.savez(graph, metadata=json.dumps({"sample_index": 1, "nodes": 8}))
            np.savez(proposal, metadata=json.dumps({"sample_index": 1,
                "metrics": {"selected": 100, "coverage": 0.9, "branch_coverage": 0.0}}))
            (root / "summary.json").write_text(json.dumps({"samples": [{
                "sample_index": 1, "graph_path": str(graph), "proposal_path": str(proposal),
                "compare_html": "correct.html", "swc_valid": True}]}))
            for suffix, compare, f1 in (("", "correct.html", 0.9), ("_other", "other.html", 0.1)):
                (root / f"topology_report{suffix}.json").write_text(json.dumps({"samples": [{
                    "sample_index": 1, "compare_html": compare, "edge_f1": f1}]}))
            (root / "selection_audit.json").write_text(json.dumps({"samples": [{
                "sample_index": 1, "graph_path": str(graph), "proposal_nodes": 80,
                "proposal_branch_coverage": 1.0, "proposal_score_threshold": 0.85}]}))
            inv = diagnostics.Inventory(root, root / "out")
            inv.discover()
            row = diagnostics.build_rows(inv, "External")[0]
            self.assertEqual(row["edge_f1"], 0.9)
            self.assertEqual(row["proposal_nodes"], 100)
            self.assertEqual(row["audit_proposal_nodes"], 80)
            self.assertEqual(row["proposal_branch_coverage"], 0)
            self.assertEqual(row["audit_proposal_branch_coverage"], 1)
            self.assertFalse(row["conflicts"])

    def test_subprocess_error_does_not_invent_preprocessing_cause(self):
        result = diagnostics.classify({"execution_status": "FAIL"})
        self.assertEqual(result["primary_failure_stage"], "unknown")
        self.assertTrue(result["manual_review_required"])

    def test_oracle_and_cap_support_mixed_diagnosis(self):
        result = diagnostics.classify({"foreground_cap_satisfied": False,
            "oracle_reachable_edge_fraction": 0.02, "oracle_bridge_edges": 124})
        self.assertEqual(result["primary_failure_stage"], "mixed")
        self.assertIn("connectivity_graph", result["secondary_failure_stage"])

    def test_alignment_evidence_is_separate_from_foreground_preprocessing(self):
        result = diagnostics.classify({"execution_status": "FAIL", "gt_aligned": False,
            "gt_out_of_bounds_nodes": 17, "failure_command": "python scripts/aggregate_proposals.py",
            "failure_returncode": 2})
        self.assertEqual(result["primary_failure_stage"], "data_alignment")
        self.assertIn("alignment guard", result["diagnostic_evidence"])
        cap = diagnostics.classify({"foreground_cap_satisfied": False})
        self.assertEqual(cap["primary_failure_stage"], "data_preprocessing")

    def test_review_status_is_independent_of_execution_status(self):
        result = diagnostics.classify({"execution_status": "PASS", "foreground_cap_satisfied": False})
        self.assertEqual(result["review_status"], "REVIEW")
        self.assertNotIn("run_status", result)
        self.assertEqual(diagnostics.classify({"execution_status": "PASS"})["review_status"], "CLEAR")

    def test_optional_dataset_metadata_is_copied_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"samples": [], "failures": [
                {"sample_index": 1}, {"sample_index": 2}]}))
            (root / "data.json").write_text(json.dumps({"domains": [], "samples": [
                {"sample_index": 1, "species": "recorded species", "brain_region": "recorded region",
                 "cortical_layer": "recorded layer"},
                {"sample_index": 2, "domain_family": "must not infer species"}]}))
            inv = diagnostics.Inventory(root, root / "out")
            inv.discover()
            first, second = diagnostics.build_rows(inv, "external-id", "External dataset")
            self.assertEqual(first["dataset_name"], "External dataset")
            self.assertEqual(first["dataset_id"], "external-id")
            for key in ("species", "brain_region", "cortical_layer"):
                self.assertIsNotNone(first[key])
                self.assertIn(key, first["field_sources"])
                self.assertIsNone(second[key])

    def test_selection_requires_good_candidate_coverage(self):
        poor = diagnostics.classify({"proposal_segment_coverage": 0.4,
            "segment_coverage_drop": 0.38})
        good = diagnostics.classify({"proposal_segment_coverage": 0.9,
            "selected_segment_coverage": 0.3, "segment_coverage_drop": 0.6})
        self.assertEqual(poor["primary_failure_stage"], "proposal")
        self.assertEqual(good["primary_failure_stage"], "selection")

    def test_missing_values_do_not_become_zero(self):
        self.assertEqual(diagnostics.classify({"incomplete": True,
            "missing_fields": ["edge_f1"]})["primary_failure_stage"], "unknown")
        self.assertIsNone(diagnostics.clean(float("nan")))

    def test_failed_sample_retained_and_gt_validity_not_output_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"samples": [], "failures": [
                {"sample_index": 3, "status": "failed", "returncode": 2}]}))
            (root / "data.json").write_text(json.dumps({"domains": [], "samples": [
                {"sample_index": 3, "sample_id": "cell-a", "swc_valid": True},
                {"sample_index": 4, "sample_id": "unevaluated"}]}))
            inv = diagnostics.Inventory(root, root / "out")
            inv.discover()
            rows = diagnostics.build_rows(inv, "External")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], 3)
            self.assertIsNone(rows[0]["swc_valid"])
            self.assertIsNone(rows[0]["edge_f1"])
            self.assertEqual(rows[0]["execution_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
