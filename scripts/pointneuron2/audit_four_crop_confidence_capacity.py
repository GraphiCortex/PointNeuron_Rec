"""CPU capacity oracle on actual balanced-fit classifier inputs, not a new fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
FIT_SHA = "06781fee1d01b8e8b19d80d82b2791b1cfc46ee56fd921c86c0f54d35e223775"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def linear_capacity(features, labels, verify_float32, *, seconds=25):
    x = np.asarray(features,dtype=np.float64)
    y = np.asarray(labels,dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or set(y.tolist()) != {0,1}:
        raise ValueError("Expected finite features and two nonempty classes")
    mean, scale = x.mean(0), x.std(0)
    active = scale > 0  # Remove only exactly constant columns, not small varying signals.
    z = np.column_stack(((x[:,active]-mean[active])/scale[active],np.ones(len(x))))
    signed = (2*y-1)[:,None]*z
    d = z.shape[1]
    options = {"time_limit":seconds,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9}
    # Any strict affine separator can be rescaled to margin >=1. Split signs
    # permit minimizing standardized L1 size without bounding the function class.
    started = time.perf_counter()
    primal = linprog(np.ones(2*d),A_ub=np.concatenate((-signed,signed),axis=1),b_ub=-np.ones(len(x)),
        bounds=(0,None),method="highs",options=options)
    result = {"samples":len(x),"positives":int(y.sum()),"negatives":int((1-y).sum()),
        "feature_columns":x.shape[1],"active_feature_columns":int(active.sum()),
        "minimum_active_feature_std":float(scale[active].min()) if active.any() else None,
        "solver_calls":1,"solver_seconds_limit_per_call":seconds,"primal_status":int(primal.status),
        "primal_message":primal.message,"conclusion":"unresolved"}
    if primal.success:
        w = primal.x[:d]-primal.x[d:]
        margin = signed@w
        raw = np.zeros(x.shape[1]); raw[active] = w[:-1]/scale[active]
        bias = w[-1]-mean@raw
        checked = verify_float32(raw,float(bias))
        result.update(minimum_standardized_float64_signed_margin=float(margin.min()),
            standardized_separator_l1=float(np.abs(w).sum()),raw_weight_l2=float(np.linalg.norm(raw)),
            raw_bias_abs=float(abs(bias)),float32_verification=checked)
        if margin.min() >= 1-1e-6 and checked["finite"] and checked["label_errors"] == 0 and checked["minimum_signed_margin"] > 0:
            result["conclusion"] = "strict_separator_verified_in_float32_classifier_arithmetic"
    elif primal.status == 2:
        # A convex combination of signed rows near zero is an approximate
        # infeasibility certificate. Its L1 residual also bounds the possible
        # minimum standardized margin when coefficients (including bias) have
        # infinity norm <=1. It is not exact-arithmetic proof of impossibility.
        dual = linprog(np.zeros(len(x)),A_eq=np.vstack((signed.T,np.ones(len(x)))),
            b_eq=np.r_[np.zeros(d),1.],bounds=(0,None),method="highs",options=options)
        result.update(solver_calls=2,certificate_status=int(dual.status),certificate_message=dual.message)
        if dual.success and dual.x.min() >= -1e-9 and abs(dual.x.sum()-1) <= 1e-7:
            alpha = np.maximum(dual.x,0); alpha /= alpha.sum()
            residual = signed.T@alpha
            result.update(certificate_max_residual=float(np.abs(residual).max()),
                norm_bounded_margin_upper_bound_numerical=float(np.abs(residual).sum()),
                certificate_weight_sum=float(alpha.sum()),
                certificate_support=[{"row":int(i),"label":int(y[i]),"weight":float(alpha[i])}
                    for i in np.flatnonzero(alpha > 0)])
            if np.abs(residual).max() <= 1e-7:
                result["conclusion"] = "numerical_infeasibility_certificate_within_tolerance"
    result["solver_wall_seconds"] = time.perf_counter()-started
    return result


def reconstruct_inputs(decoded, state):
    hidden = F.relu(F.linear(decoded,state["proposal.prediction.0.weight"],state["proposal.prediction.0.bias"]))
    raw = F.linear(hidden,state["proposal.prediction.2.weight"],state["proposal.prediction.2.bias"])
    return hidden, raw


def run(output):
    if output.exists() or not output.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Choose a fresh ignored tmp/pointneuron2 directory")
    torch.set_num_threads(2)
    experiment = ROOT/"tmp/pointneuron2/local_runs/balanced_query_20261018"
    fit_path = experiment/"model/report.json"
    assert digest(fit_path) == FIT_SHA
    fit = read(fit_path)
    reference = read(ROOT/"docs/pointneuron2/hard_negative_query_results_seed0/summary.json")["balanced"]
    assert reference["fit_report_sha256"] == FIT_SHA and fit["status"] == "completed" and fit["optimizer_updates"] == 4000
    checkpoint_path = experiment/"model/step_4000.pt"
    manifest_path = experiment/"cache/manifest.json"
    protected = {**fit["provenance"]["source_files_sha256"],str(fit_path):FIT_SHA,
        str(checkpoint_path):fit["final_checkpoint_sha256"],str(manifest_path):fit["provenance"]["cache_manifest_sha256"]}
    manifest = read(manifest_path)
    protected.update(manifest["protected_files_sha256"])
    def verify():
        for path, expected in protected.items():
            assert digest(path) == expected, path
    verify()
    cp = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    assert cp["step"] == 4000 and cp["recipe"] == fit["recipe"] and cp["provenance"] == fit["provenance"]
    state = cp["inference_checkpoint"]["model"]
    assert state["proposal.prediction.0.weight"].shape == (128,128)
    assert state["proposal.prediction.2.weight"].shape == (6,128)
    hidden_crops, labels, replays, identities = [], [], [], []
    for i,record in enumerate(manifest["records"]):
        path = experiment/f"model/step_4000_crop_{i}.npz"
        protected[str(path)] = reference["prediction_files_sha256"][path.name]
        assert digest(path) == protected[str(path)]
        assert record["partition"] == "train"
        label = torch.zeros(512,dtype=torch.long)
        with np.load(path,allow_pickle=False) as saved:
            decoded = torch.from_numpy(saved["decoder_features"].copy())
            original_logits = torch.from_numpy(saved["logits"].copy())
            assert decoded.shape == (1,512,128)
            label[saved["pred_indices"]] = 1
        with torch.no_grad():
            hidden,raw = reconstruct_inputs(decoded,state)
        torch.testing.assert_close(raw[...,:2],original_logits,rtol=1e-5,atol=1e-5)
        assert torch.equal(raw[0,:,1] >= raw[0,:,0],original_logits[0,:,1] >= original_logits[0,:,0])
        assert int(label.sum()) == record["target_count"]
        hidden_crops.append(hidden[0]); labels.append(label)
        identities.append([record["sample_index"],record["patch_index"]])
        replays.append({"crop":identities[-1],"labels":len(label),"positives":int(label.sum()),
            "maximum_logit_absolute_difference":float((raw[...,:2]-original_logits).abs().max()),
            "original_acceptance_reproduced_exactly":True})
    features = torch.cat(hidden_crops); y = torch.cat(labels)
    assert features.shape == (2048,128) and [int(v.sum()) for v in labels] == [6,12,16,418]
    def verify_float32(weights,bias):
        candidate_w = state["proposal.prediction.2.weight"].clone()
        candidate_b = state["proposal.prediction.2.bias"].clone()
        candidate_w[:2] = 0; candidate_b[:2] = 0
        with np.errstate(over="ignore",invalid="ignore"):
            candidate_w[1] = torch.from_numpy(weights.astype(np.float32))
            candidate_b[1] = torch.from_numpy(np.asarray(bias,dtype=np.float32))
        if not bool(torch.isfinite(candidate_w).all() and torch.isfinite(candidate_b).all()):
            return {"finite":False,"label_errors":None,"minimum_signed_margin":None,
                "reason":"Separator weights cannot be represented as finite float32"}
        margins,counts = [],[]
        with torch.no_grad():
            for hidden,label,identity in zip(hidden_crops,labels,identities):
                raw = F.linear(hidden,candidate_w,candidate_b)
                difference = raw[:,1]-raw[:,0]
                accepted = difference >= 0  # Exactly the .5 inference decision.
                signed = (2*label-1)*difference
                margins.append(signed)
                counts.append({"crop":identity,"positives":int(label.sum()),
                    "accepted_matched":int((accepted & (label == 1)).sum()),
                    "accepted_unmatched":int((accepted & (label == 0)).sum()),
                    "label_errors":int((accepted != (label == 1)).sum()),
                    "minimum_signed_margin":float(signed.min())})
        all_margins = torch.cat(margins)
        finite = bool(torch.isfinite(all_margins).all())
        return {"finite":finite,
            "label_errors":sum(r["label_errors"] for r in counts),
            "minimum_signed_margin":float(all_margins.min()) if finite else None,
            "crops":counts if finite else None}
    print("VERIFIED actual hidden inputs reproduce saved logits and acceptance on all four crops",flush=True)
    capacity = linear_capacity(features.numpy(),y.numpy(),verify_float32,seconds=25)
    for witness in capacity.get("certificate_support",[]):
        witness.update(crop=identities[witness["row"]//512],query=witness["row"]%512)
    verify()
    result = {"schema":"pointneuron2-four-crop-confidence-capacity-v1","status":"completed",
        "scope":"fixed TRAINING features/labels; CPU linear-program oracle; no image/encoder/Transformer inference, neural updates, threshold search or dev/test access",
        "source_fit_sha256":FIT_SHA,"source_checkpoint_sha256":digest(checkpoint_path),
        "feature_tensor_sha256":hashlib.sha256(features.numpy().tobytes()).hexdigest(),
        "audit_source_sha256":digest(Path(__file__)),"protected_sources_unchanged":True,
        "classifier_checkpoint_writes":0,"neural_optimizer_updates":0,"feature_replays":replays,"capacity":capacity,
        "limitations":["A witness uses known training labels, not an automatic inference rule.",
            "Representability on frozen inputs does not establish optimizer success, joint learning, generalization or reconstruction.",
            "An approximate infeasibility certificate is numerical evidence within tolerance, not exact-arithmetic proof that no extremely fragile separator exists."]}
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(capacity,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    run(parser.parse_args().output_dir)
