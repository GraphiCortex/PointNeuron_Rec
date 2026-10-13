"""Numerical linear-separability oracle on fixed training features; no model fit.

LP parameters are temporary mathematical witnesses, never saved as a classifier.
Only a diagnostic report is written. A solver failure is not a capacity result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def separability(features, labels, *, seconds=15):
    x=np.asarray(features,dtype=np.float64)
    y=np.asarray(labels,dtype=np.int64)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(y.tolist())!={0,1}:
        raise ValueError("Expected finite features and both classes")
    mean=x.mean(0);scale=x.std(0);active=scale>1e-10
    z=np.column_stack(((x[:,active]-mean[active])/scale[active],np.ones(len(x))))
    signed=(2*y-1)[:,None]*z
    d=z.shape[1]
    # A strict separating boundary can be rescaled to signed margin >= 1.
    # Positive/negative parts minimize L1 size, avoiding an arbitrary huge witness.
    options={"time_limit":seconds,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9}
    primal=linprog(np.ones(2*d),A_ub=np.concatenate((-signed,signed),1),b_ub=-np.ones(len(x)),
                   bounds=(0,None),method="highs",options=options)
    result={"samples":len(x),"features":x.shape[1],"active_feature_columns":int(active.sum()),
            "solver_status":int(primal.status),"solver_message":primal.message,
            "linear_solver_calls":1,"numerical_conclusion":"unresolved"}
    if primal.success:
        w=primal.x[:d]-primal.x[d:]
        margin=signed@w
        raw=np.zeros(x.shape[1]);raw[active]=w[:-1]/scale[active]
        bias=w[-1]-mean@raw
        score32=x.astype(np.float32)@raw.astype(np.float32)+np.float32(bias)
        result.update(minimum_float64_signed_margin=float(margin.min()),
                      minimum_float32_signed_margin=float(((2*y-1)*score32).min()),
                      standardized_separator_l1=float(np.abs(w).sum()),
                      raw_weight_l2=float(np.linalg.norm(raw)),raw_bias_abs=float(abs(bias)),
                      float32_label_errors=int(((score32>0).astype(int)!=y).sum()))
        if margin.min()>=1-1e-6:
            result["numerical_conclusion"]="strict_linear_separator_verified"
        return result
    if primal.status!=2:
        return result
    # Farkas witness: nonnegative combination of signed rows sums to zero,
    # while weights sum to one. Any margin>=1 separator would contradict it.
    equality=np.vstack((signed.T,np.ones(len(x))))
    dual=linprog(np.zeros(len(x)),A_eq=equality,b_eq=np.r_[np.zeros(d),1.],bounds=(0,None),
                 method="highs",options=options)
    result.update(linear_solver_calls=2,certificate_status=int(dual.status),certificate_message=dual.message)
    if dual.success:
        weights=dual.x
        residual=float(np.abs(signed.T@weights).max())
        result.update(certificate_max_residual=residual,certificate_weight_sum=float(weights.sum()),
                      certificate_min_weight=float(weights.min()),
                      witness_rows=[{"row":int(i),"label":int(y[i]),"weight":float(weights[i])}
                                    for i in np.flatnonzero(weights>1e-8)])
        if residual<=1e-7 and abs(weights.sum()-1)<=1e-7 and weights.min()>=-1e-9:
            result["numerical_conclusion"]="infeasibility_certificate_verified_within_tolerance"
    return result


def main(directory):
    output=directory/"diagnostics/linear_separability.json"
    if output.exists():raise FileExistsError("Preserve existing diagnostic results")
    read=lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
    fit_path=directory/"model/fit_report.json"
    manifest_path=directory/"cache/manifest.json"
    fit,manifest=read(fit_path),read(manifest_path)
    cache_path=directory/"cache/features.pt"
    checkpoint_path=directory/"model/step_2000.pt"
    if (fit["status"]!="completed" or fit["optimizer_updates"]!=2000 or not fit["cached_geometry_features_labels_unchanged"]
            or digest(manifest_path)!=fit["feature_manifest_sha256"] or digest(cache_path)!=manifest["features_sha256"]
            or digest(checkpoint_path)!=fit["final_checkpoint_sha256"]):
        raise ValueError("Expected the completed fixed-feature fit and immutable source cache")
    protected={**manifest["protected_files_sha256"],**fit["source_files_sha256"],
               str(fit_path.resolve()):digest(fit_path),str(manifest_path.resolve()):digest(manifest_path),
               str(cache_path.resolve()):digest(cache_path),str(checkpoint_path.resolve()):digest(checkpoint_path)}
    verify=lambda:all(digest(p)==h for p,h in protected.items())
    if not verify():raise ValueError("Changed source files")
    cache=torch.load(cache_path,map_location="cpu",weights_only=True)
    features,labels=cache["features"].numpy(),cache["labels"].numpy()
    if features.shape!=(2,512,128) or labels.shape!=(2,512) or labels.sum(1).tolist()!=[6,12]:
        raise ValueError("Unexpected frozen training feature/label contract")
    results=[]
    for scope,index in [("both_training_crops",None),("crop_0",0),("crop_1",1)]:
        x=features.reshape(-1,128) if index is None else features[index]
        y=labels.reshape(-1) if index is None else labels[index]
        result=separability(x,y);result["scope"]=scope
        for witness in result.get("witness_rows",[]):
            witness.update(crop=witness["row"]//512 if index is None else index,query=witness["row"]%512)
        results.append(result)
        print(f"{scope}: {result['numerical_conclusion']}",flush=True)
    if not verify():raise ValueError("Source changed during audit")
    report={"status":"completed","device":"CPU","neural_optimizer_updates":0,
            "classifier_checkpoint_writes":0,"scope":"training-label linear-feasibility oracle; no deployment claim",
            "protected_sources_unchanged":True,"results":results,
            "source_fit_sha256":digest(fit_path),"feature_cache_sha256":digest(cache_path),
            "audit_source_sha256":digest(__file__)}
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(f"report: {output}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-directory",type=Path,default=ROOT/"tmp/pointneuron2/local_runs/query_confidence_20261013")
    main(parser.parse_args().experiment_directory)
