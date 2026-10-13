"""LR-only matching, source preservation and synthetic replay lifecycle."""
from argparse import Namespace
import numpy as np
import pytest
import torch

from scripts.pointneuron2 import confidence_lr_experiment as replay
from scripts.pointneuron2.reference_gaps import segment_probe


def test_recipe_rejects_any_change_besides_declared_learning_rate():
    old=replay.fixed.load_recipe(replay.BASE_RECIPE)
    new=replay.fixed.base.read_json(replay.REPLAY_RECIPE)
    assert replay.recipe_contract(new,old)=={"learning_rate":{"baseline":.0001,"replay":.001}}
    for key,value in [("updates",3000),("score_threshold",.4),("weight_decay",.001),("initialization","fresh")]:
        with pytest.raises(ValueError,match="only learning_rate"):
            replay.recipe_contract({**new,key:value},old)


def test_missing_crop_cannot_pass_initial_comparison_by_empty_zip():
    with pytest.raises(ValueError,match="crop set differs"):
        replay.check_initial_metrics({"evaluations":[{"step":0,"crops":[]}]},{"step":0,"crops":[]})


@pytest.fixture
def synthetic_baseline(tmp_path):
    """Contract fixture only; its snapshot copies do not claim a learned baseline."""
    torch.set_num_threads(2)
    old=replay.fixed.load_recipe(replay.BASE_RECIPE)
    directory=tmp_path/"baseline";(directory/"cache").mkdir(parents=True);(directory/"model").mkdir()
    labels=torch.zeros(2,512,dtype=torch.long);labels[0,:6]=1;labels[1,:12]=1
    features=torch.zeros(2,512,128);features[:,:,0]=labels.float()*3-1
    centers=torch.full((2,512,3),1000.);records=[]
    for i,n in enumerate((6,12)):
        centers[i,:n]=torch.stack((torch.arange(n),torch.zeros(n),torch.zeros(n)),-1)
        nodes=np.column_stack((np.arange(1,n+1),centers[i,:n].numpy(),np.ones(n),np.full(n,-1)))
        segments=np.array([[[0.,0.,0.],[n-1.,0.,0.]]])
        geometry={"full_segments":segments,"critical_points":np.array([[0.,0.,0.]]),
                  "chains":[{"length_voxels":float(n-1),"probe":segment_probe(segments)}]}
        records.append({"sample_index":0,"patch_index":i,"partition":"train","assigned_queries":list(range(n)),
                        "assigned_targets":list(range(n)),"nodes":nodes.tolist(),"geometry":replay.fixed.base.pack_geometry(geometry)})
    cache={"features":features,"labels":labels,"centers":centers,"radii":torch.ones(2,512,1),
           "baseline_logits":torch.zeros(2,512,2),"initial_weight":torch.zeros(2,128),"initial_bias":torch.zeros(2)}
    features_path=directory/"cache/features.pt"
    torch.save({"schema":replay.fixed.CACHE_SCHEMA,**cache},features_path)
    manifest={"schema":replay.fixed.CACHE_SCHEMA,"status":"completed","recipe":old,"records":records,
              "optimizer_updates":0,"features_sha256":replay.fixed.base.file_digest(features_path),"protected_files_sha256":{}}
    manifest_path=directory/"cache/manifest.json";replay.fixed.base.write_json(manifest_path,manifest)
    head=replay.FrozenQueryConfidence(cache["initial_weight"],cache["initial_bias"])
    evaluations=[]
    for step in old["snapshots"]:
        evaluations.append(replay.fixed.evaluate(head,cache,manifest,old,step))
        torch.save(replay.confidence_checkpoint(head,step=step,recipe=old,provenance={"synthetic_contract":True}),directory/f"model/step_{step:04d}.pt")
    baseline={"status":"completed","optimizer_updates":2000,"recipe":old,"trainable_parameters":258,
              "cached_geometry_features_labels_unchanged":True,"source_files_sha256":{},"evaluations":evaluations,
              "feature_manifest_sha256":replay.fixed.base.file_digest(manifest_path),
              "final_checkpoint_sha256":replay.fixed.base.file_digest(directory/"model/step_2000.pt"),"gate_passed":False,
              "backend":{"device":"cpu","torch":str(torch.__version__),"cuda":torch.version.cuda,"device_name":"CPU","matmul_tf32":False}}
    replay.fixed.base.write_json(directory/"model/fit_report.json",baseline)
    return Namespace(action="preflight",device="cpu",recipe=replay.REPLAY_RECIPE,baseline_directory=directory,
                     experiment_directory=tmp_path/"replay")


def test_replay_timing_and_synthetic_fit_preserve_baseline_and_inputs(synthetic_baseline):
    args=synthetic_baseline
    before={str(p):replay.fixed.base.file_digest(p) for p in args.baseline_directory.rglob("*") if p.is_file()}
    checked=replay.source_contract(args)
    assert checked[-1]=={"learning_rate":{"baseline":.0001,"replay":.001}}
    timing=replay.run(args)
    assert timing["optimizer_updates"]==0 and timing["throwaway_optimizer_updates"]==4
    assert timing["initial_paired_metrics_passed"] and not list(args.experiment_directory.rglob("*.pt"))
    with pytest.raises(FileExistsError):replay.run(args)
    args.action="train";result=replay.run(args)
    assert result["status"]=="completed" and result["optimizer_updates"]==2000
    assert result["initial_paired_metrics_passed"] and result["reload_max_logit_delta"]==0
    assert result["frozen_cache_and_baseline_unchanged"] and result["pointnext_transformer_forward_passes"]==0
    paired=replay.fixed.base.read_json(args.experiment_directory/"comparison.json")
    assert paired["paired_checks_passed"] and len(paired["rows"])==10
    assert {str(p):replay.fixed.base.file_digest(p) for p in args.baseline_directory.rglob("*") if p.is_file()}==before
    with pytest.raises(FileExistsError):replay.run(args)


def test_baseline_initialization_tampering_fails_before_output(synthetic_baseline):
    args=synthetic_baseline;path=args.baseline_directory/"model/step_0000.pt"
    cp=torch.load(path,map_location="cpu",weights_only=True);cp["model"]["linear.bias"]+=1.
    torch.save(cp,path)
    with pytest.raises(ValueError,match="initialization differs"):
        replay.run(args)
    assert not args.experiment_directory.exists()
