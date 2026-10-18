"""Four-crop confidence stage; the geometry output rows are never parameters here."""
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "pointneuron2-four-crop-frozen-confidence-v1"


class FourCropFrozenConfidence(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        if weight.shape != (2,128) or bias.shape != (2,) or weight.dtype != torch.float32 or bias.dtype != torch.float32:
            raise ValueError("Expected saved float32 confidence rows only")
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(128,2)
        self.linear.load_state_dict({"weight":weight.detach().clone(),"bias":bias.detach().clone()})
        if not all(bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise ValueError("Nonfinite initial confidence parameters")

    def forward(self, features):
        if features.shape != (4,512,128) or features.dtype != torch.float32 or features.requires_grad or not bool(torch.isfinite(features).all()):
            raise ValueError("Expected detached finite four-crop features [4,512,128]")
        return self.linear(features)


def stage_loss(logits, labels):
    if logits.shape != (4,512,2) or labels.shape != (4,512) or labels.dtype != torch.long:
        raise ValueError("Expected four fixed confidence-label vectors")
    if bool(((labels != 0) & (labels != 1)).any()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid labels or logits")
    logp = F.log_softmax(logits,dim=-1)
    losses = []
    for i in range(4):
        positive = labels[i] == 1
        if not bool(positive.any()) or not bool((~positive).any()):
            raise ValueError("This stage requires the declared supervised crops with both classes")
        losses.append(.5*(-logp[i,positive,1].mean()-logp[i,~positive,0].mean()))
    per_crop = torch.stack(losses)
    return 2*per_crop.mean(), per_crop  # Same outer confidence coefficient as balanced joint fit.
