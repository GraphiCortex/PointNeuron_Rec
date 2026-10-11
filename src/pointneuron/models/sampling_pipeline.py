"""Checkpoint contract and inference selection for the stage-1 proposal model."""
from __future__ import annotations

import numpy as np
import torch

from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.sampling_cache import SamplingPoolConfig, build_image_pools
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler


def sampler_from_checkpoint(checkpoint: dict, device="cpu"):
    pipeline = checkpoint.get("sampling_pipeline")
    if pipeline is None:
        return None
    if pipeline.get("schema_version") != 1:
        raise ValueError("Unsupported sampling checkpoint schema")
    if pipeline["strategy"] != "learned":
        return None
    sampler = AdaptiveMedialSampler(**pipeline["sampler_config"]).to(device)
    if not isinstance(checkpoint.get("sampler"), dict):
        raise ValueError("Learned checkpoint lacks sampler weights")
    sampler.load_state_dict(checkpoint["sampler"], strict=True)
    if not torch.isfinite(sampler.alpha) or sampler.alpha <= 0:
        raise ValueError("Invalid learned decay")
    return sampler.eval()


def validate_inference_reader(checkpoint: dict):
    pipeline = checkpoint.get('sampling_pipeline')
    if pipeline is not None and pipeline.get('volume_reader_version') != VOLUME_READER_VERSION:
        raise ValueError('Checkpoint was trained before the Vaa3D decoder correction. Rebuild the image cache '
                         'and retrain before reconstruction with corrected images.')


def inference_sample(checkpoint: dict, sampler, indices, data, width, height, seed: int, device="cpu"):
    validate_inference_reader(checkpoint)
    pipeline = checkpoint["sampling_pipeline"]
    pools = SamplingPoolConfig(**pipeline["pool_config"])
    medial = MedialSamplingConfig(**pipeline["medial_config"])
    inputs = build_image_pools(indices, data, width, height, pools, medial, seed)
    points = torch.from_numpy(inputs["points"])[None].to(device)
    guide = torch.from_numpy(inputs["image_guide_xyz"])[None].to(device)
    with torch.no_grad():
        if pipeline["strategy"] == "learned":
            selected = sampler(points, guide, pipeline["selected_points"], seed=seed + pipeline["evaluation_seed"]).points
        elif pipeline["strategy"] == "random":
            generator = torch.Generator(device=device).manual_seed(seed + pipeline["evaluation_seed"])
            selected_indices = torch.randperm(points.shape[1], generator=generator, device=device)[:pipeline["selected_points"]]
            selected = points[:, selected_indices]
        else:
            raise ValueError("Unknown checkpoint sampling strategy")
    return selected[0].cpu().numpy()
