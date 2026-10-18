"""Separate target/query reassignment from binary confidence-label changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"balanced": "06781fee1d01b8e8b19d80d82b2791b1cfc46ee56fd921c86c0f54d35e223775",
            "hard_negative": "95503d46bcfab5ecdf4d7933235ff18980bfdb402b9621be972cba54f1ff3dcc"}
DIRECTORIES = {"balanced": "balanced_query_20261018", "hard_negative": "hard_negative_query_20261018"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    if output.exists() or not output.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Choose a fresh ignored tmp/pointneuron2 folder")
    saved_review = json.loads((ROOT/"docs/pointneuron2/hard_negative_query_results_seed0/summary.json").read_text())
    rows, protected = [], {}
    for arm, folder in DIRECTORIES.items():
        directory = ROOT/"tmp/pointneuron2/local_runs"/folder
        report_path = directory/"model/report.json"
        assert sha(report_path) == EXPECTED[arm] == saved_review[arm]["fit_report_sha256"]
        fit = json.loads(report_path.read_text())
        assert fit["status"] == "completed" and fit["optimizer_updates"] == 4000
        for filename, digest in fit["provenance"]["source_files_sha256"].items():
            assert sha(Path(filename)) == digest
        previous = [None]*4
        for evaluation in fit["evaluations"]:
            step = evaluation["step"]
            for i, crop in enumerate(evaluation["crops"]):
                path = directory/f"model/step_{step:04d}_crop_{i}.npz"
                assert sha(path) == saved_review[arm]["prediction_files_sha256"][path.name]
                protected[str(path)] = sha(path)
                with np.load(path,allow_pickle=False) as prediction:
                    pi, ti = prediction["pred_indices"], prediction["target_indices"]
                p = crop["targets"]
                assert len(set(pi)) == len(pi) == p and sorted(ti.tolist()) == list(range(p))
                assert [pi.tolist(),ti.tolist()] == crop["assignments"]
                target_to_query = dict(zip(ti.tolist(),pi.tolist()))
                query_to_target = {q:t for t,q in target_to_query.items()}
                positive = set(query_to_target)
                if previous[i] is not None:
                    old_step, old_mapping = previous[i]
                    old_inverse = {q:t for t,q in old_mapping.items()}
                    old_positive = set(old_inverse)
                    entered, exited = positive-old_positive, old_positive-positive
                    assert len(entered) == len(exited)
                    changed = sum(target_to_query[t] != old_mapping[t] for t in target_to_query)
                    retained_but_reassigned = sum(query_to_target[q] != old_inverse[q] for q in positive & old_positive)
                    assert changed == len(entered)+retained_but_reassigned
                    reference_row = next(r for r in saved_review[arm]["rows"] if r["step"] == step
                        and r["sample_index"] == crop["sample_index"] and r["patch_index"] == crop["patch_index"])
                    assert changed == reference_row["assignment_changes_since_previous_snapshot"]
                    rows.append({"arm":arm,"crop":[crop["sample_index"],crop["patch_index"]],
                        "from_step":old_step,"to_step":step,"targets":p,"queries":512,
                        "targets_changing_matched_query":changed,
                        "queries_changing_negative_to_positive":len(entered),
                        "queries_changing_positive_to_negative":len(exited),
                        "binary_confidence_label_flips":len(entered)+len(exited),
                        "binary_label_flip_fraction_of_all_queries":(len(entered)+len(exited))/512,
                        "positive_queries_retained":len(positive & old_positive),
                        "positive_retention_fraction":len(positive & old_positive)/p,
                        "retained_positive_queries_reassigned_to_other_target":retained_but_reassigned})
                previous[i] = step,target_to_query
    for path,digest in protected.items():
        assert sha(Path(path)) == digest
    result = {"schema":"pointneuron2-assignment-contract-audit-v1","status":"completed",
        "scope":"saved training assignments only; no forwards, gradients, updates, GPU, relabeling, cutoff search or dev/test access",
        "fit_report_sha256":EXPECTED,"audit_source_sha256":sha(Path(__file__)),"rows":rows,
        "interpretation":"Target/query map changes and binary confidence-label changes differ. Neither sparse-snapshot count establishes per-update instability or causation."}
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps([r for r in rows if r["from_step"] == 2000],indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    run(parser.parse_args().output_dir)
