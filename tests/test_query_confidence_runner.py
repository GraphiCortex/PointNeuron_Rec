"""Synthetic confidence runner checks; research GPU fitting is never launched."""
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.pointneuron2 import query_confidence_experiment as runner
from scripts.pointneuron2.reference_gaps import segment_probe


def test_synthetic_timing_then_bounded_fit_preserves_geometry_and_sources(tmp_path):
    torch.set_num_threads(2)
    recipe_path=runner.ROOT/"configs/pointneuron2/query_confidence.json"
    recipe=runner.load_recipe(recipe_path)
    cache_dir=tmp_path/"cache";cache_dir.mkdir()
    labels=torch.zeros(2,512,dtype=torch.long);labels[0,:6]=1;labels[1,:12]=1
    features=torch.zeros(2,512,128);features[:,:,0]=labels.float()*3-1
    centers=torch.full((2,512,3),1000.)
    rows=[]
    for i,n in enumerate((6,12)):
        centers[i,:n]=torch.stack((torch.arange(n),torch.zeros(n),torch.zeros(n)),-1)
        nodes=np.concatenate((np.arange(1,n+1)[:,None],centers[i,:n].numpy(),np.ones((n,1)),np.full((n,1),-1)),1)
        segments=np.array([[[0.,0.,0.],[n-1.,0.,0.]]])
        geometry={"full_segments":segments,"critical_points":np.array([[0.,0.,0.]]),
                  "chains":[{"length_voxels":float(n-1),"probe":segment_probe(segments)}]}
        rows.append({"sample_index":0,"patch_index":i,"partition":"train","assigned_queries":list(range(n)),
                     "assigned_targets":list(range(n)),"nodes":nodes.tolist(),"geometry":runner.base.pack_geometry(geometry)})
    tensors={"features":features,"labels":labels,"centers":centers,"radii":torch.ones(2,512,1),
             "baseline_logits":torch.zeros(2,512,2),"initial_weight":torch.zeros(2,128),"initial_bias":torch.zeros(2)}
    cache_path=cache_dir/"features.pt";torch.save({"schema":runner.CACHE_SCHEMA,**tensors},cache_path)
    runner.base.write_json(cache_dir/"manifest.json",{"schema":runner.CACHE_SCHEMA,"status":"completed","recipe":recipe,
        "records":rows,"optimizer_updates":0,"features_sha256":runner.base.file_digest(cache_path),"protected_files_sha256":{}})
    args=Namespace(action="preflight",device="cpu",recipe=recipe_path,experiment_directory=tmp_path)
    before=runner.base.file_digest(cache_path)
    timing=runner.run(args)
    assert timing["optimizer_updates"]==0 and timing["throwaway_optimizer_updates"]==4
    assert timing["checkpoint_retained"] is False and list(tmp_path.rglob("*.pt"))==[cache_path]
    assert not (tmp_path/"model").exists()
    with pytest.raises(FileExistsError):runner.run(args)
    args.action="train";result=runner.run(args)
    assert result["status"]=="completed" and result["optimizer_updates"]==2000
    assert result["pointnext_transformer_forward_passes"]==0 and result["trainable_parameters"]==258
    assert result["cached_geometry_features_labels_unchanged"] and result["reload_max_logit_delta"]==0
    assert [r["step"] for r in result["evaluations"]]==[0,100,500,1000,2000]
    assert runner.base.file_digest(cache_path)==before
    assert len(list((tmp_path/"model").glob("step_*.pt")))==5
    with pytest.raises(FileExistsError):runner.run(args)
