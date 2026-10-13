"""Analytic native point-operation and small-cloud model contracts."""
import pytest
import torch

from pointneuron.models.pointnext_ops import farthest_point_indices, ball_query_indices, interpolate_features, gather_points
from pointneuron.models.pointnext_native import PointNextNativeEncoder, normalize_native_inputs
from pointneuron.models.pointnext_query import PointNextQueryModel, model_checkpoint, load_model_checkpoint
from pointneuron.models.query_pipeline import models_from_checkpoint
from pointneuron.models.query_loss import query_set_loss


@pytest.fixture(autouse=True)
def bounded_cpu():
    torch.set_num_threads(2)


def xyz(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1, 3)


def test_fps_known_geometry_ties_distinct_indices_and_batch_isolation():
    positions = xyz([[0,0,0], [1,0,0], [3,0,0], [7,0,0]])
    assert farthest_point_indices(positions, 4).tolist() == [[0,3,2,1]]
    tied = xyz([[0,0,0], [1,0,0], [-1,0,0], [0,0,0]])
    assert farthest_point_indices(tied, 4).tolist() == [[0,1,2,3]]
    paired = torch.cat((positions, positions+10), 0)
    assert farthest_point_indices(paired, 4).tolist() == [[0,3,2,1], [0,3,2,1]]
    assert farthest_point_indices(positions, 0).shape == (1, 0)
    with pytest.raises(ValueError):
        farthest_point_indices(positions, 5)


def test_ball_membership_is_strict_index_ordered_and_short_groups_repeat_first():
    support = xyz([[0,0,0], [.25,0,0], [-.25,0,0], [1,0,0]])
    query = xyz([[0,0,0], [1,0,0]])
    expected = [[[0,1,2,0,0], [1,3,1,1,1]]]
    assert ball_query_indices(support, query, 1., 5, 1).tolist() == expected
    assert ball_query_indices(support, query, 1., 5, 10).tolist() == expected
    with pytest.raises(ValueError, match="Empty ball"):
        ball_query_indices(support, xyz([[8,0,0]]), 1., 3)


def test_interpolation_known_weights_ties_chunks_and_feature_gradients():
    support = xyz([[0,0,0], [2,0,0], [4,0,0], [8,0,0]])
    query = xyz([[1,0,0], [2,0,0], [3,0,0]])
    features = torch.tensor([[[0.],[2.],[4.],[100.]]], requires_grad=True)
    result = interpolate_features(query, support, features, 1)
    torch.testing.assert_close(result[0, :, 0], torch.tensor([10/7, 2., 18/7]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(result, interpolate_features(query, support, features, 20))
    result.sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad[0,:3].abs().sum() > 0
    assert features.grad[0,3].abs().sum() == 0
    torch.testing.assert_close(interpolate_features(query, support[:,:1], features[:,:1]), torch.zeros(1,3,1))


def test_gather_keeps_requested_point_order_and_rejects_invalid_indices():
    values = torch.tensor([[[10.],[20.],[30.]], [[40.],[50.],[60.]]])
    indices = torch.tensor([[2,0],[1,2]])
    assert gather_points(values, indices).tolist() == [[[30.],[10.]], [[50.],[60.]]]
    with pytest.raises(ValueError):
        gather_points(values, torch.tensor([[3],[0]]))


def native_points(n=256):
    i = torch.arange(n)
    coords = torch.stack((i%16, (i//16)%16, i//256), -1).float()-8
    center = torch.tensor([[100.,200.,300.]])
    points = torch.cat((coords+center, torch.full((n,1),4095.)), -1)[None]
    return points, center, torch.tensor([4095.])


@pytest.mark.parametrize("n", [256,358])
def test_batch_one_short_cloud_uses_full_hierarchy_and_restores_features(n):
    torch.manual_seed(0)
    points, center, scale = native_points(n)
    local, attrs = normalize_native_inputs(points, center, scale, 48.)
    encoder = PointNextNativeEncoder().train()
    counts=[]
    handles=[layer.register_forward_hook(lambda layer, args, out: counts.append(out[0].shape[1])) for layer in encoder.abstraction]
    out=encoder(local,attrs)
    for handle in handles: handle.remove()
    assert counts==([64,16,4,1] if n==256 else [89,22,5,1])
    assert out.shape==(1,n,32) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert encoder.stem[0].weight.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    with torch.no_grad():
        encoder.eval(); assert torch.isfinite(encoder(local,attrs)).all()
    with pytest.raises(ValueError,match="N>=256"):
        encoder(local[:,:255],attrs[:,:255])


def test_explicit_scale_roundtrip_query_gradient_and_strict_reload():
    points,center,scale=native_points()
    local,attrs=normalize_native_inputs(points,center,scale,48.)
    torch.testing.assert_close(local*48+center[:,None],points[...,:3])
    torch.testing.assert_close(attrs[...,3],torch.ones(1,256))
    model=PointNextQueryModel(seed=0).train()
    prediction=model(points,center,scale)
    nodes=torch.tensor([[[1.,100.,200.,300.,1.,-1.],[2.,105.,200.,300.,1.,-1.]]])
    loss=query_set_loss(prediction,nodes,torch.ones(1,2,dtype=torch.bool))
    loss.total.backward()
    assert model.encoder.stem[0].weight.grad.abs().sum()>0
    assert model.proposal.prediction[-1].weight.grad.abs().sum()>0
    assert prediction.center_proposals.shape==(1,512,3) and prediction.decoder_features.shape==(1,512,128)
    checkpoint=model_checkpoint(model,seed=0,step=0,recipe={},provenance={})
    model.eval(); loaded=load_model_checkpoint(checkpoint)
    with torch.no_grad():
        a,b=model(points,center,scale),loaded(points,center,scale)
        torch.testing.assert_close(a.center_proposals,b.center_proposals,rtol=0,atol=0)
        shift=torch.tensor([100.,50.,20.]);moved=points.clone();moved[...,:3]+=shift
        c=model(moved,center+shift,scale)
        torch.testing.assert_close(c.center_proposals,a.center_proposals+shift,rtol=0,atol=1e-4)
        torch.testing.assert_close(c.objectness_logits,a.objectness_logits,rtol=0,atol=1e-6)
    with pytest.raises(ValueError,match="Unsupported"):
        models_from_checkpoint(checkpoint)
    with pytest.raises(ValueError,match="explicit"):
        load_model_checkpoint({"schema":"pointneuron2-query-comparison-v1"})
