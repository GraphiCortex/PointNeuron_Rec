"""Explicit batch-one PointNeXt/query control with current-crop BN in both modes.

This changes reference running-statistics inference. It is a bounded research
control, not a production normalization policy or an official reproduction.
"""
from __future__ import annotations

import torch
from torch import nn

from pointneuron.data.curve_targets import TARGET_SCHEMA
from pointneuron.models.pointnext_query import PointNextQueryModel, load_model_checkpoint

CHECKPOINT_SCHEMA = "pointneuron2-pointnext-query-cropnorm-v1"
POLICY = "batch_one_current_crop_statistics_train_and_eval"


class PointNextCropNormQueryModel(PointNextQueryModel):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.normalizer_names = []
        for name, module in self.encoder.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
                for buffer in ("running_mean", "running_var", "num_batches_tracked"):
                    module.register_buffer(buffer, None)
                self.normalizer_names.append("encoder." + name)
        if len(self.normalizer_names) != 16:
            raise ValueError("Expected exactly the declared sixteen encoder BN layers")

    def forward(self, points, patch_centers, image_scales, patch_radius=48.):
        if points.ndim != 3 or len(points) != 1:
            raise ValueError("Current-crop normalization control requires batch one; no mixed-crop statistics")
        return super().forward(points, patch_centers, image_scales, patch_radius)

    def model_spec(self):
        spec = super().model_spec()
        spec.update(kind=CHECKPOINT_SCHEMA, normalization=POLICY, batch_size_contract=1,
                    batchnorm_layers=len(self.normalizer_names))
        return spec


def convert_checkpoint(source, *, source_sha256):
    """Explicitly discard only running-statistic buffers; preserve all parameters."""
    original = load_model_checkpoint(source)
    model = PointNextCropNormQueryModel(source["seed"])
    old_state = original.state_dict()
    new_state = model.state_dict()
    removed = sorted(set(old_state) - set(new_state))
    expected = sorted(name + "." + buffer for name in model.normalizer_names
                      for buffer in ("running_mean", "running_var", "num_batches_tracked"))
    if removed != expected or set(new_state) - set(old_state):
        raise ValueError("Conversion may discard only the declared BN running buffers")
    model.load_state_dict({key: old_state[key] for key in new_state}, strict=True)
    if any(not torch.equal(t, old_state[key]) for key, t in model.state_dict().items()):
        raise ValueError("Conversion changed learned parameters")
    recipe = dict(source["recipe"], normalization=POLICY)
    converted = {"schema": CHECKPOINT_SCHEMA, "target_schema": TARGET_SCHEMA,
                 "model_spec": model.model_spec(), "seed": source["seed"], "step": source["step"],
                 "recipe": recipe, "model": model.state_dict(),
                 "provenance": {"source_checkpoint_sha256": source_sha256,
                                "source_schema": source["schema"], "source_provenance": source["provenance"],
                                "discarded_running_buffers": removed, "optimizer_updates": 0}}
    return converted, model.eval()


def load_cropnorm_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("target_schema") != TARGET_SCHEMA:
        raise ValueError("Expected explicit PointNeXt/query crop-normalization checkpoint")
    model = PointNextCropNormQueryModel(checkpoint["seed"])
    if checkpoint.get("model_spec") != model.model_spec() or checkpoint["recipe"].get("normalization") != POLICY:
        raise ValueError("Crop-normalization checkpoint policy differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("Nonfinite converted parameters")
    return model.to(device).eval()
