"""Synthetic CPU preparation/timing contracts; no research fit."""
from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pointneuron.models.centerline_unet import CenterlineUNet
from pointneuron.data.field_pilot import CACHE_SCHEMA
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.data.query_cache import file_digest
from scripts.pointneuron2 import pointnext_query_experiment as experiment

ROOT=Path(__file__).resolve().parents[1]
RECIPE=ROOT/"configs/pointneuron2/pointnext_query.json"
FIELD_CONFIG=ROOT/"configs/pointneuron2/field_pilot.json"


@pytest.fixture
def source_fixture(tmp_path):
    torch.set_num_threads(2)
    native,models,data=tmp_path/"native",tmp_path/"field_model",tmp_path/"data"
    native.mkdir();(models/"checkpoints").mkdir(parents=True);(data/"train").mkdir(parents=True)
    swc=data/"train/trace.swc"
    swc.write_text("1 3 0 0 0 1 -1\n2 3 40 0 0 1 1\n3 3 0 160 0 1 -1\n4 3 88 160 0 1 3\n")
    partition=tmp_path/"partition.json"
    experiment.write_json(partition,{"partitions":{"train":[{"sample_index":0,"sample_id":"train"}],
        "val":[{"sample_index":1,"sample_id":"dev"}],"test":[{"sample_index":2,"sample_id":"reserved"}]}})
    cfg=json.loads(FIELD_CONFIG.read_text());records=[]
    for sample,patch,role,center in [(0,0,"train",[20.,0.,0.]),(0,1,"train",[44.,160.,0.]),(1,0,"val",[20.,0.,0.])]:
        raw=np.zeros((97,97,97),dtype=np.uint8);raw.ravel()[:256]=255
        meta={"sample_index":sample,"patch_index":patch,"partition":role,"sample_id":"train" if role=="train" else "dev",
            "patch_center":center,"origin_xyz":(np.asarray(center)-48).tolist(),"threshold":50.,"image_scale":255.,
            "foreground_count":256,"swc_sha256":file_digest(swc)}
        path=native/f"sample_{sample}_{patch}.npz"
        np.savez_compressed(path,raw_image=raw,image=raw.astype(np.float32)/255.,valid=np.ones(raw.shape,dtype=bool),metadata=np.array(json.dumps(meta)))
        records.append({k:meta[k] for k in ["sample_index","patch_index","sample_id","partition"]} |
                       {"path":path.name,"sha256":file_digest(path),"supervised":True})
    manifest=native/"manifest.json"
    experiment.write_json(manifest,{"schema":CACHE_SCHEMA,"status":"completed","volume_reader_version":VOLUME_READER_VERSION,
        "config":cfg,"config_sha256":file_digest(FIELD_CONFIG),"partition_sha256":file_digest(partition),"records":records})
    spec={"kind":"centerline_unet_v1","shared_initialization_sha256":"synthetic"}
    cp=models/"checkpoints/field_epoch_030.pt"
    torch.save({"epoch":30,"config":cfg,"model_spec":spec,"manifest_sha256":file_digest(manifest),"model":CenterlineUNet().state_dict()},cp)
    experiment.write_json(models/"pilot_report.json",{"status":"completed","config":cfg,"model_spec":spec,
        "manifest_sha256":file_digest(manifest),"partition_sha256":file_digest(partition),"final_checkpoint_sha256":file_digest(cp),"checkpoint_reload_max_logit_delta":0.})
    return Namespace(action="prepare",recipe=RECIPE,partition=partition,field_config=FIELD_CONFIG,native_manifest=manifest,
        field_models=models,data_root=data,device="cpu",experiment_dir=tmp_path/"prototype")


def test_prepare_then_timing_preserves_sources_and_never_starts_fit(source_fixture,monkeypatch):
    args=source_fixture;calls=[]
    monkeypatch.setattr(CenterlineUNet,"forward",lambda self,x:torch.full_like(x,-10.))
    original=experiment.matching_source
    def after_selection(*a,**k):
        assert len(list((args.experiment_dir/"cache").glob("*_points.npz")))==2
        calls.append(1);return original(*a,**k)
    monkeypatch.setattr(experiment,"matching_source",after_selection)
    before=file_digest(args.field_models/"checkpoints/field_epoch_030.pt")
    report=experiment.prepare(args)
    assert report["status"]=="completed" and report["optimizer_updates"]==report["pointnext_inference_passes"]==0
    assert report["field_inference_passes"]==2 and len(calls)==2
    assert [r["target_count"] for r in report["records"]]==[6,12]
    assert [r["actual_unique_count"] for r in report["records"]]==[256,256]
    assert file_digest(args.field_models/"checkpoints/field_epoch_030.pt")==before
    with pytest.raises(FileExistsError):experiment.prepare(args)
    recipe,manifest,records,protected=experiment.load_cache(args)
    assert records[0]["points"].shape==(1,256,4) and records[1]["nodes"].shape==(1,12,6)
    evaluation=experiment.evaluate(experiment.PointNextQueryModel(),records,None,0)
    assert not evaluation["learnability_gate_passed"]
    experiment.write_json(args.experiment_dir/"initial_check.json",evaluation)
    args.action="preflight";timed=experiment.run_model_action(args)
    assert timed["status"]=="completed" and timed["throwaway_optimizer_updates"]==4
    assert timed["optimizer_updates"]==0 and not timed["checkpoint_retained"]
    assert timed["checkpoint_reload_max_delta"]==0 and not list(args.experiment_dir.rglob("*.pt"))
    with pytest.raises(FileExistsError):experiment.run_model_action(args)
    timing=experiment.read_json(args.experiment_dir/"timing.json");timing["cache_manifest_sha256"]="changed"
    experiment.write_json(args.experiment_dir/"timing.json",timing);args.action="train"
    with pytest.raises(ValueError,match="reviewed GPU timing"):experiment.run_model_action(args)
    assert not (args.experiment_dir/"model").exists()


def test_reserved_source_identity_fails_before_output(source_fixture):
    args=source_fixture;manifest=experiment.read_json(args.native_manifest)
    manifest["records"][0].update(partition="test",sample_index=2,sample_id="reserved")
    experiment.write_json(args.native_manifest,manifest)
    with pytest.raises(ValueError,match="Reserved/test"):experiment.prepare(args)
    assert not args.experiment_dir.exists()


def test_wrong_field_checkpoint_identity_fails_before_output(source_fixture):
    args=source_fixture;report=experiment.read_json(args.field_models/"pilot_report.json")
    report["model_spec"]["kind"]="attention_centerline_unet_v1"
    experiment.write_json(args.field_models/"pilot_report.json",report)
    with pytest.raises(ValueError,match="fresh-baseline"):experiment.prepare(args)
    assert not args.experiment_dir.exists()
