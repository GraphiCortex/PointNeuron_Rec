"""Algebraic audit of a proposed loss on saved scores; never trains a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FIT_SHA256 = "06781fee1d01b8e8b19d80d82b2791b1cfc46ee56fd921c86c0f54d35e223775"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    if output.exists() or not output.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Choose a fresh ignored tmp/pointneuron2 directory")
    experiment = ROOT/"tmp/pointneuron2/local_runs/balanced_query_20261018"
    report_path = experiment/"model/report.json"
    assert sha(report_path) == EXPECTED_FIT_SHA256
    fit = json.loads(report_path.read_text(encoding="utf-8"))
    assert fit["status"] == "completed" and fit["optimizer_updates"] == 4000
    for path, digest in fit["provenance"]["source_files_sha256"].items():
        assert sha(Path(path)) == digest
    rows = []
    for i, record in enumerate(fit["evaluations"][-1]["crops"]):
        path = experiment/f"model/step_4000_crop_{i}.npz"
        with np.load(path, allow_pickle=False) as data:
            logits = data["logits"][0].astype(float)
            assigned = data["pred_indices"]
        positive = np.zeros(len(logits), dtype=bool); positive[assigned] = True
        difference = logits[:, 1]-logits[:, 0]
        probability = expit(difference)
        negatives = np.flatnonzero(~positive)
        p, n = int(positive.sum()), len(negatives)
        k = min(p, n)
        # Ranking is decreasing used score; lower query ID resolves exact ties.
        hard = negatives[np.lexsort((negatives, -difference[negatives]))[:k]]
        positive_loss = np.logaddexp(0., -difference[positive]).mean()
        negative_loss = np.logaddexp(0., difference[negatives]).mean()
        hard_loss = np.logaddexp(0., difference[hard]).mean()
        balanced = .5*positive_loss+.5*negative_loss
        proposed = .5*positive_loss+.25*negative_loss+.25*hard_loss
        assert abs(balanced-record["balanced_classification_loss"]) < 1e-6
        old_weights, new_weights = np.zeros(len(logits)), np.zeros(len(logits))
        old_weights[positive] = new_weights[positive] = .5/p
        old_weights[negatives] = .5/n
        new_weights[negatives] = .25/n
        new_weights[hard] += .25/k
        assert np.isclose(old_weights[positive].sum(), .5) and np.isclose(new_weights[positive].sum(), .5)
        assert np.isclose(old_weights[negatives].sum(), .5) and np.isclose(new_weights[negatives].sum(), .5)
        assert hard_loss >= negative_loss-1e-12 and proposed >= balanced-1e-12
        if k == n:
            assert np.array_equal(old_weights, new_weights) and abs(proposed-balanced) < 1e-12
        # These are analytic derivatives with fixed assignments/ranking, not neural gradients.
        old_negative_gradient = old_weights[negatives]*probability[negatives]
        new_negative_gradient = new_weights[negatives]*probability[negatives]
        rows.append({"crop": [record["sample_index"], record["patch_index"]], "prediction_sha256": sha(path),
            "positives": p, "negatives": n, "hard_negative_count": k,
            "accepted_unmatched": int((difference[negatives] >= 0).sum()),
            "accepted_unmatched_in_hard_set": int((difference[hard] >= 0).sum()),
            "hard_query_indices": hard.tolist(), "balanced_classification_loss": float(balanced),
            "proposed_classification_loss": float(proposed), "positive_mean_ce": float(positive_loss),
            "negative_mean_ce": float(negative_loss), "hard_negative_mean_ce": float(hard_loss),
            "positive_per_query_coefficient": .5/p, "old_negative_per_query_coefficient": .5/n,
            "new_easy_negative_per_query_coefficient": .25/n,
            "new_hard_negative_per_query_coefficient": .25/n+.25/k,
            "hard_negative_gradient_multiplier_at_fixed_scores": (.25/n+.25/k)/(.5/n),
            "total_negative_logit_gradient_old": float(old_negative_gradient.sum()),
            "total_negative_logit_gradient_proposed": float(new_negative_gradient.sum()),
            "direct_formula_identical_on_this_crop": k == n})
    result = {"schema": "pointneuron2-hard-negative-objective-algebra-v1", "status": "completed",
        "scope": "saved training scores only; fixed assignments; no model forwards, neural gradients, updates, GPU or dev/test data",
        "fit_report_sha256": sha(report_path), "audit_source_sha256": sha(Path(__file__)),
        "proposed_formula": "0.5 positive_mean_CE + 0.25 all_negative_mean_CE + 0.25 topK_negative_mean_CE; K=min(P,N)",
        "note": "Larger retrospective loss/gradient is imposed by the formula, not evidence of better predictions or optimization.", "rows": rows}
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
