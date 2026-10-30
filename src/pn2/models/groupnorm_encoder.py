"""Declared normalization-only PointNeXt variant; no trained-state conversion."""
import torch
from torch import nn


def replace_fresh_encoder_batchnorm(encoder):
    """Replace exactly sixteen fresh BN layers by GN(8,C), eps 1e-5.

    All widths in the audited prototype are divisible by eight, leaving at
    least four channels/group. No running-stat buffers are used by the variant.
    This fixed grouping is a testable starting design, not a tuned optimum.
    """
    candidates = [(name, module) for name, module in encoder.named_modules()
                  if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if len(candidates) != 16:
        raise ValueError("Expected the audited sixteen encoder BatchNorm layers")
    # Validate all layers before mutating any module.
    for _, module in candidates:
        if (not module.affine or not module.track_running_stats or module.eps != 1e-5
                or module.num_features < 32 or module.num_features % 8
                or module.num_batches_tracked is None or int(module.num_batches_tracked) != 0
                or not bool((module.running_mean == 0).all())
                or not bool((module.running_var == 1).all())):
            raise ValueError("Only the declared fresh normalizers may be replaced; no trained BN conversion")
    manifest = []
    for name, module in candidates:
        norm = nn.GroupNorm(8, module.num_features, eps=1e-5, affine=True,
                            device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            norm.weight.copy_(module.weight)
            norm.bias.copy_(module.bias)
        norm.train(module.training)
        parent_path, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent = encoder.get_submodule(parent_path) if parent_path else encoder
        parent._modules[child] = norm
        manifest.append({"name": name, "channels": module.num_features, "groups": 8,
                         "channels_per_group": module.num_features//8, "eps": 1e-5})
    if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in encoder.modules()):
        raise ValueError("A BatchNorm layer remains")
    return manifest
