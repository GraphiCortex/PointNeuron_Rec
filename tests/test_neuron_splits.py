"""Regression tests for neuron leakage and shared-cache worktree paths."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pointneuron.data.splits import SplitRatios, split_records, validate_split_payload, write_split_file


class NeuronSplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "split.json"
        self.records = [
            {"path": f"cache/neuron_{n}_patch_{p}.npz", "sample_id": f"neuron_{n}", "donor_id": f"donor_{n // 2}"}
            for n in range(12) for p in range(n % 4 + 1)
        ]

    def build(self, records=None, **kwargs):
        self.manifest.write_text(json.dumps({"records": self.records if records is None else records}), encoding="utf-8")
        write_split_file(self.manifest, self.output, record_root=self.root, **kwargs)
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_every_patch_and_sampling_variant_stays_with_its_neuron(self):
        records = self.records + [{**self.records[0], "path": "other_threshold/neuron_0_patch_0.npz"}]
        payload = self.build(records, seed=11)
        owners = {p: name for name, paths in payload["splits"].items() for p in paths}
        neurons = {}
        for record in payload["record_metadata"]:
            neurons.setdefault(record["sample_id"], set()).add(owners[record["path"]])
        self.assertTrue(all(len(names) == 1 for names in neurons.values()))
        self.assertEqual(sum(payload["counts"].values()), len(records))
        self.assertTrue(all(Path(p).is_absolute() for p in owners))
        validate_split_payload(payload, require_neuron_split=True)

    def test_assignment_is_independent_of_manifest_order(self):
        first = self.build(seed=3)
        second = self.build(list(reversed(self.records)), seed=3)
        self.assertEqual(first["group_splits"], second["group_splits"])
        self.assertEqual({k: set(v) for k, v in first["splits"].items()}, {k: set(v) for k, v in second["splits"].items()})

    def test_legacy_patch_suffixes_do_not_create_new_neurons(self):
        records = [{**r, "sample_index": int(r["sample_id"].split("_")[1])} for r in self.records]
        records += [{**records[0], "sample_id": "neuron_0#patch-36", "path": "other_threshold/patch36.npz"}]
        payload = self.build(records)
        self.assertEqual(sum(payload["group_counts"].values()), 12)
        metadata = next(r for r in payload["record_metadata"] if r["source_sample_id"] == "neuron_0#patch-36")
        self.assertEqual(metadata["sample_id"], "neuron_0")
        validate_split_payload(payload, True)

    def test_conflicting_neuron_names_for_one_index_are_rejected(self):
        records = [{**r, "sample_index": 0} for r in self.records]
        with self.assertRaisesRegex(ValueError, "conflicting neuron identities"):
            self.build(records)

    def test_donor_grouping_keeps_related_neurons_together(self):
        payload = self.build(group_by="donor_id")
        self.assertEqual(sum(payload["group_counts"].values()), 6)
        validate_split_payload(payload, require_neuron_split=True)

    def test_missing_identity_and_duplicate_paths_are_rejected(self):
        cases = [
            [{k: v for k, v in self.records[0].items() if k != "sample_id"}] + self.records[1:],
            self.records + [self.records[0]],
            [],
        ]
        for records in cases:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                self.build(records)

    def test_conflicting_specimen_metadata_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[2]["donor_id"] = "incorrect_donor"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.build(records, group_by="donor_id")

    def test_fewer_than_three_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "three distinct groups"):
            self.build(self.records[:3])

    def test_crossed_groups_and_tampered_neuron_identity_are_rejected(self):
        payload = self.build()
        broken = copy.deepcopy(payload)
        broken["group_splits"]["val"].append(broken["group_splits"]["train"][0])
        with self.assertRaises(ValueError):
            validate_split_payload(broken, True)
        broken = copy.deepcopy(payload)
        train_path = broken["splits"]["train"][0]
        val_path = broken["splits"]["val"][0]
        train_id = next(r["sample_id"] for r in broken["record_metadata"] if r["path"] == train_path)
        next(r for r in broken["record_metadata"] if r["path"] == val_path)["sample_id"] = train_id
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_split_payload(broken, True)

    def test_missing_metadata_and_duplicate_split_records_are_rejected(self):
        payload = self.build()
        broken = copy.deepcopy(payload)
        broken["record_metadata"].pop()
        with self.assertRaises(ValueError):
            validate_split_payload(broken, True)
        broken = copy.deepcopy(payload)
        broken["splits"]["val"].append(broken["splits"]["train"][0])
        with self.assertRaises(ValueError):
            validate_split_payload(broken, True)

    def test_explicit_legacy_mode_preserves_patch_assignment(self):
        payload = self.build(group_by="record", seed=11)
        expected = split_records([str(self.root / r["path"]) for r in self.records], seed=11)
        self.assertEqual(payload["splits"], expected)
        with self.assertRaisesRegex(ValueError, "neuron-group"):
            validate_split_payload(payload, True)

    def test_nonfinite_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            split_records(["a", "b", "c"], SplitRatios(float("nan"), 0.15, 0.15))


if __name__ == "__main__":
    unittest.main()
