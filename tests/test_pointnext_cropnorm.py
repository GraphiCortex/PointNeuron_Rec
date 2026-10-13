"""Train/eval/reload consistency and explicit conversion contracts."""
import copy
import pytest
import torch
from torch import nn

from pointneuron.models.pointnext_query import PointNextQueryModel, model_checkpoint, load_model_checkpoint
from pointneuron.models.pointnext_query_cropnorm import convert_checkpoint, load_cropnorm_checkpoint


def inputs(n=256):
    i = torch.arange(n)
    xyz = torch.stack((i % 16, (i // 16) % 16, i // 256), -1).float()
    center = torch.tensor([[100., 200., 300.]])
    points = torch.cat((xyz + center, torch.full((n, 1), 255.)), -1)[None]
    return points, center, torch.tensor([255.])


@pytest.mark.parametrize("n", [256, 358])
def test_conversion_is_strict_and_train_eval_reload_crop_order_agree(n):
    torch.set_num_threads(2)
    original = PointNextQueryModel(0).eval()
    cp = model_checkpoint(original, seed=0, step=2000, recipe={}, provenance={})
    converted, model = convert_checkpoint(cp, source_sha256="synthetic")
    assert converted["provenance"]["optimizer_updates"] == 0
    assert len(converted["provenance"]["discarded_running_buffers"]) == 48
    assert all(torch.equal(p, dict(original.named_parameters())[name]) for name, p in model.named_parameters())
    assert all(not m.track_running_stats and m.running_mean is None for m in model.modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    values = inputs(n)
    diagnostic = copy.deepcopy(original)
    for m in diagnostic.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()
    before = {key: t.clone() for key, t in model.state_dict().items()}
    with torch.no_grad():
        a = model(*values)
        d = diagnostic(*values)
        model.train(); b = model(*values)
        model.eval(); model(*inputs(256 if n == 358 else 358))
        repeated = model(*values)
        loaded = load_cropnorm_checkpoint(converted); c = loaded(*values)
    for name in ("center_proposals", "objectness_logits", "radius"):
        torch.testing.assert_close(getattr(a, name), getattr(d, name), rtol=0, atol=1e-5)
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=1e-4)
        torch.testing.assert_close(getattr(a, name), getattr(c, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(a, name), getattr(repeated, name), rtol=0, atol=0)
    assert all(torch.equal(t, before[key]) for key, t in model.state_dict().items())
    with pytest.raises(ValueError, match="batch one"):
        model(*(v.expand(2, *v.shape[1:]) for v in values))
    with pytest.raises(ValueError, match="explicit native"):
        load_model_checkpoint(converted)
    with pytest.raises(ValueError, match="explicit PointNeXt"):
        load_cropnorm_checkpoint(cp)
