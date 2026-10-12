"""Two additive skip gates; image-only Phase 1 field experiment."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from pointneuron.models.centerline_unet import CenterlineUNet


class SkipAttentionGate(nn.Module):
    """Use decoder context to attenuate skip features; no pairwise self-attention."""
    def __init__(self, skip_channels, context_channels, hidden_channels, initial_coefficient=.99):
        super().__init__()
        if not 0 < initial_coefficient < 1:
            raise ValueError("Initial gate coefficient must lie strictly between zero and one")
        self.skip_projection = nn.Conv3d(skip_channels, hidden_channels, 1, bias=False)
        self.context_projection = nn.Conv3d(context_channels, hidden_channels, 1, bias=False)
        self.score = nn.Conv3d(hidden_channels, 1, 1)
        # Start close to the ungated backbone; keep a nonzero sigmoid derivative.
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, math.log(initial_coefficient/(1-initial_coefficient)))

    def forward(self, skip, context):
        if context.shape[2:] != skip.shape[2:]:
            context = F.interpolate(context, size=skip.shape[2:], mode="trilinear", align_corners=False)
        coefficient = torch.sigmoid(self.score(F.relu(self.skip_projection(skip)+self.context_projection(context))))
        return skip*coefficient


class AttentionCenterlineUNet(CenterlineUNet):
    def __init__(self, base_channels=8, initial_coefficient=.99):
        super().__init__(base_channels)
        c = base_channels
        self.gate_b = SkipAttentionGate(c*2, c*4, c, initial_coefficient)
        self.gate_a = SkipAttentionGate(c, c*2, c//2, initial_coefficient)

    def load_shared_backbone(self, state):
        result = self.load_state_dict(state, strict=False)
        expected = {k for k in self.state_dict() if k.startswith(("gate_a.", "gate_b."))}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError("Shared initialization is not the exact ungated backbone")
        if any(not torch.equal(self.state_dict()[k].cpu(), value.cpu()) for k, value in state.items()):
            raise AssertionError("Shared initial backbone tensors differ")

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 4:
            raise ValueError("Expected B,1,Z,Y,X native image")
        a = self.enc1(x)
        b = self.enc2(F.max_pool3d(a, 2))
        h = self.bottom(F.max_pool3d(b, 2))
        up = F.interpolate(h, size=b.shape[2:], mode="trilinear", align_corners=False)
        h = self.dec2(torch.cat((up, self.gate_b(b, up)), 1))
        up = F.interpolate(h, size=a.shape[2:], mode="trilinear", align_corners=False)
        h = self.dec1(torch.cat((up, self.gate_a(a, up)), 1))
        return self.head(h)
