"""Native formula, gradients, invariances, paired control and checkpoint identity."""
from dataclasses import replace

import pytest
import torch

from pointneuron.data.connectivity_targets import masked_edge_loss
from pointneuron.graph.proposal_graph import ProposalGraphInput, build_candidate_graph
from pointneuron.models.gatv2_connectivity import (
    ConnectivityConfig, GATv2Connectivity, NativeGATv2Block, connectivity_checkpoint,
    load_connectivity_checkpoint, symmetric_edge_features,
)


@pytest.fixture(autouse=True, scope="module")
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic(n=9, gradient=False):
    generator = torch.Generator().manual_seed(19)
    positions = torch.tensor([[i*3., (i % 3)*2., 0.] for i in range(n)]).reshape(-1, 3).requires_grad_(gradient)
    radii = (torch.rand(n, 1, generator=generator)+1).requires_grad_(gradient)
    logits = torch.randn(n, 2, generator=generator).requires_grad_(gradient)
    features = torch.randn(n, 128, generator=generator).requires_grad_(gradient)
    return ProposalGraphInput(positions, radii, logits, positions/48., features, torch.arange(n), {"source": "synthetic"})


def test_native_attention_agrees_with_independent_loop_reference_and_gradients():
    torch.manual_seed(7)
    layer = NativeGATv2Block(8, 2).double()
    features = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [3, 1], [2, 2], [3, 2], [3, 3]])
    attributes = torch.randn(len(edges), 5, dtype=torch.float64)
    actual, alpha = layer.aggregate(features, edges, attributes)
    expected_rows = []
    for i in range(len(features)):
        selected = [j for j, pair in enumerate(edges.tolist()) if pair[1] == i]
        heads = []
        for head in range(2):
            sl = slice(head*4, (head+1)*4)
            values, logits = [], []
            for e in selected:
                j = int(edges[e, 0])
                value = (features[j] @ layer.source.weight.T)[sl]
                query = (features[i] @ layer.receiver.weight.T)[sl]
                relation = (attributes[e] @ layer.edge.weight.T)[sl]
                z = query+value+relation
                activated = torch.where(z >= 0, z, .2*z)
                logits.append((activated*layer.attention[head]).sum())
                values.append(value)
            weights = torch.softmax(torch.stack(logits), dim=0)
            heads.append((weights[:, None]*torch.stack(values)).sum(0))
            assert alpha[selected, head].sum().item() == pytest.approx(1.)
        expected_rows.append(torch.cat(heads))
    expected = torch.stack(expected_rows)+layer.bias
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    a = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), features)[0]
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_attention_ranking_can_change_with_receiver_query():
    layer = NativeGATv2Block(2, 1)
    with torch.no_grad():
        layer.source.weight.copy_(torch.eye(2))
        layer.receiver.weight.copy_(torch.eye(2))
        layer.edge.weight.zero_()
        layer.attention.fill_(1.)
    features = torch.tensor([[-2., 0], [0, -2.], [1., 0], [0, 1.]])
    edges = torch.tensor([[2, 0], [3, 0], [2, 1], [3, 1]])
    _, alpha = layer.aggregate(features, edges, torch.zeros(4, 5))
    assert alpha[0] < alpha[1] and alpha[2] > alpha[3]


def test_forward_backward_reaches_input_geometry_confidence_and_features():
    p = synthetic(9, gradient=True)
    graph = build_candidate_graph(p.positions, p.node_ids)
    model = GATv2Connectivity()
    output = model(p, graph)
    assert output.node_features.shape == (9, 128)
    assert output.edge_logits.shape == (len(graph.edges),)
    labels = torch.arange(len(graph.edges)) % 2
    loss = masked_edge_loss(output.edge_logits, labels.float())
    loss.total.backward()
    for field in (p.positions, p.radii, p.objectness_logits, p.decoder_features):
        assert field.grad is not None and bool(torch.isfinite(field.grad).all()) and bool(field.grad.abs().sum() > 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def test_permutation_translation_endpoint_and_metadata_invariance():
    p = synthetic()
    graph = build_candidate_graph(p.positions, p.node_ids)
    model = GATv2Connectivity().eval()
    baseline = model(p, graph)
    permutation = torch.tensor([8, 3, 5, 0, 7, 1, 2, 6, 4])
    q = replace(p, **{key: getattr(p, key)[permutation] for key in
                     ("positions", "radii", "objectness_logits", "local_centers", "decoder_features", "node_ids")})
    other = build_candidate_graph(q.positions, q.node_ids)
    permuted = model(q, other)
    torch.testing.assert_close(permuted.node_features, baseline.node_features[permutation], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted.edge_logits, baseline.edge_logits, atol=1e-5, rtol=1e-5)
    shifted = replace(p, positions=p.positions+torch.tensor([128., -64., 32.]), provenance={"source": "translated"})
    shift_graph = build_candidate_graph(shifted.positions, shifted.node_ids)
    assert torch.equal(graph.edges, shift_graph.edges)
    torch.testing.assert_close(model(shifted, shift_graph).edge_logits, baseline.edge_logits, atol=1e-5, rtol=1e-5)
    a = symmetric_edge_features(p, baseline.node_features, graph.edges)
    b = symmetric_edge_features(p, baseline.node_features, graph.edges.flip(1))
    assert torch.equal(a, b)
    metadata_only = replace(p, provenance={"arbitrary_reference_labels": [1, 0, 1]})
    assert torch.equal(model(metadata_only, graph).edge_logits, baseline.edge_logits)


def test_control_has_identical_shared_initialization_and_constructor_preserves_rng():
    before = torch.get_rng_state().clone()
    gat = GATv2Connectivity(ConnectivityConfig(seed=10))
    control = GATv2Connectivity(ConnectivityConfig(mode="pairwise", seed=10))
    assert torch.equal(before, torch.get_rng_state())
    for name in ("stem", "edge_head"):
        assert all(torch.equal(a, b) for a, b in zip(getattr(gat, name).parameters(), getattr(control, name).parameters()))
    assert len(gat.blocks) == 2 and len(control.blocks) == 0
    p = synthetic()
    assert bool(torch.isfinite(control(p, build_candidate_graph(p.positions, p.node_ids)).edge_logits).all())


@pytest.mark.parametrize("mode", ["gatv2", "pairwise"])
def test_train_eval_strict_disk_reload_and_schema_separation(mode, tmp_path):
    p = synthetic()
    graph = build_candidate_graph(p.positions, p.node_ids)
    model = GATv2Connectivity(ConnectivityConfig(mode=mode))
    model.train()
    training = model(p, graph).edge_logits
    model.eval()
    assert torch.equal(training, model(p, graph).edge_logits)
    saved = connectivity_checkpoint(model, provenance={"source": "synthetic_execution_check"})
    assert saved["step"] == 0 and saved["optimizer"] is None and not saved["resumable_training_state"]
    path = tmp_path/f"{mode}.pt"
    torch.save(saved, path)
    loaded = load_connectivity_checkpoint(torch.load(path, map_location="cpu", weights_only=False), expected_mode=mode)
    assert torch.equal(training, loaded(p, graph).edge_logits)
    with pytest.raises(ValueError, match="identity"):
        load_connectivity_checkpoint(saved, expected_mode="pairwise" if mode == "gatv2" else "gatv2")
    with pytest.raises(ValueError, match="identity"):
        load_connectivity_checkpoint({"schema": "pointneuron2-pointnext-query-v1"}, expected_mode=mode)


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single_node_forward(n):
    p = synthetic(n)
    graph = build_candidate_graph(p.positions, p.node_ids)
    output = GATv2Connectivity()(p, graph)
    assert output.node_features.shape == (n, 128) and output.edge_logits.shape == (0,)
    assert bool(torch.isfinite(output.node_features).all())


def test_checkpoint_snapshots_model_optimizer_scheduler_and_rng_without_updates():
    model = GATv2Connectivity()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
    provenance = {"upstream": ["synthetic"]}
    saved = connectivity_checkpoint(model, provenance=provenance, optimizer=optimizer, scheduler=scheduler)
    assert saved["resumable_training_state"] and saved["step"] == 0 and saved["optimizer"]["state"] == {}
    optimizer.param_groups[0]["lr"] = 10.
    scheduler.gamma = 99.
    provenance["upstream"].append("changed")
    assert saved["optimizer"]["param_groups"][0]["lr"] == .0001
    assert saved["scheduler"]["gamma"] == .5 and saved["provenance"]["upstream"] == ["synthetic"]
    first = next(iter(saved["model"]))
    original = saved["model"][first].clone()
    with torch.no_grad():
        next(model.parameters()).zero_()
    assert torch.equal(original, saved["model"][first])
    assert set(saved["rng"]) == {"torch_cpu", "python", "numpy", "torch_cuda"}
