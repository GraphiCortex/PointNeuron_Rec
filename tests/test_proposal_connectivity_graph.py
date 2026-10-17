"""Reference boundaries, deterministic candidates, loss masks and SWC decoding."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pointneuron.data.connectivity_targets import (
    assign_edge_labels, full_reference_components, masked_edge_loss, mean_crop_edge_loss,
)
from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.graph.connectivity_decoder import decode_forest, export_decoded_swc
from pointneuron.graph.proposal_graph import (
    CandidateGraph, ProposalGraphInput, build_candidate_graph, from_query_output, message_graph, tensor_digest,
)
from pointneuron.graph.representation_oracle import clipped_reference_graph, target_graph, topology
from pointneuron.models.query_proposal import QueryProposalOutput


def packet(xyz, ids=None, logits=None):
    xyz = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3)
    n = len(xyz)
    score = torch.full((n,), 3.) if logits is None else torch.tensor(logits, dtype=xyz.dtype)
    return ProposalGraphInput(xyz, torch.ones(n, 1), torch.stack((torch.zeros(n), score), dim=1),
                              xyz/48., torch.zeros(n, 128),
                              torch.arange(n) if ids is None else torch.tensor(ids, dtype=torch.long),
                              {"source": "synthetic"})


def edge_id_pairs(p, graph):
    ids = p.node_ids.tolist()
    return [tuple(sorted((ids[u], ids[v]))) for u, v in graph.edges.tolist()]


def test_query_adapter_preserves_all_queries_and_checks_provenance():
    p = packet([[0, 0, 0], [1, 0, 0]], logits=[-10, 10])
    features = p.decoder_features.clone().requires_grad_()
    output = QueryProposalOutput(p.positions[None], p.objectness_logits[None], p.radii[None],
                                 p.local_centers[None], features[None])
    provenance = {"upstream_schema": "synthetic", "normalization_policy": "explicit_synthetic",
                  "checkpoint_sha256": "0"*64, "cloud_sha256": "1"*64,
                  "feature_sha256": tensor_digest(features), "partition": "synthetic"}
    adapted = from_query_output(output, crop_center=[0, 0, 0], patch_radius=48., provenance=provenance)
    assert len(adapted.positions) == 2 and adapted.decoder_features.requires_grad
    assert adapted.positions.data_ptr() == p.positions.data_ptr()
    assert adapted.node_features().shape == (2, 133)
    with pytest.raises(ValueError, match="normalization"):
        from_query_output(output, crop_center=[0, 0, 0], patch_radius=48., provenance={})
    with pytest.raises(ValueError, match="digest"):
        from_query_output(output, crop_center=[0, 0, 0], patch_radius=48.,
                          provenance=dict(provenance, feature_sha256="2"*64))
    with pytest.raises(ValueError, match="coordinates"):
        from_query_output(output, crop_center=[1, 0, 0], patch_radius=48., provenance=provenance)


def test_candidate_ties_are_stable_and_coincident_nodes_are_distinct():
    p = packet([[0, 0, 0]]+[[1, 0, 0]]*20, ids=[100]+list(range(20, 0, -1)))
    graph = build_candidate_graph(p.positions, p.node_ids, neighbors=1, radius=.01)
    permutation = torch.tensor([20, 0]+list(range(1, 20)))
    q = replace(p, positions=p.positions[permutation], node_ids=p.node_ids[permutation])
    other = build_candidate_graph(q.positions, q.node_ids, neighbors=1, radius=.01)
    assert edge_id_pairs(p, graph) == edge_id_pairs(q, other)
    assert (1, 100) in edge_id_pairs(p, graph)
    assert len(graph.edges) == 191  # Coincident cluster plus exactly one origin edge.
    changed_metadata = replace(p, provenance={"untrusted_reference_labels": [1, 0]})
    repeated = build_candidate_graph(changed_metadata.positions, changed_metadata.node_ids, neighbors=1, radius=.01)
    assert torch.equal(graph.edges, repeated.edges)


@pytest.mark.parametrize("n", [0, 1])
def test_empty_or_single_node_has_no_candidate_edges_but_self_messages(n):
    p = packet([[0, 0, 0]]*n)
    graph = build_candidate_graph(p.positions, p.node_ids)
    edges, attributes = message_graph(p, graph)
    assert graph.edges.shape == (0, 2) and edges.shape == (n, 2)
    assert attributes.shape == (n, 5) and not bool(attributes.any())


def test_label_masks_unknown_fragments_unmatched_and_bad_geometry():
    xyz = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [5, 5, 0], [70, 0, 0]]
    p = packet(xyz)
    graph = build_candidate_graph(p.positions, p.node_ids)
    target_xyz = np.array(xyz[:4]+[[50, 0, 0]])
    result = assign_edge_labels(p, graph, target_xyz, [(0, 1), (1, 2), (3, 4)], [10, 10, 10, 20, 20],
                                ([0, 1, 2, 3, 5], [0, 1, 2, 3, 4]))
    labels = dict(zip(edge_id_pairs(p, graph), result.labels.tolist()))
    assert labels[(0, 1)] == labels[(1, 2)] == 1
    assert labels[(0, 2)] == 0
    assert labels[(0, 3)] == labels[(0, 4)] == labels[(3, 5)] == -1
    assert result.statistics["positive"] == 2 and result.statistics["negative"] == 1
    assert result.statistics["trusted_matches"] == 4 and result.statistics["poorly_placed_matches"] == 1
    assert result.statistics["unrepresented_target_edges"] == 1
    assert result.statistics["attainable_recall_all_target_edges"] == pytest.approx(2/3)


def test_missing_positive_candidates_are_reported_without_inserting_gt_edges():
    p = packet([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    graph = CandidateGraph(torch.tensor([[0, 2]]), 1, .1)
    before = graph.edges.clone()
    result = assign_edge_labels(p, graph, p.positions, [(0, 1), (1, 2)], [5, 5, 5], ([0, 1, 2], [0, 1, 2]))
    assert result.labels.tolist() == [0.]
    assert result.statistics["missing_candidate_positive_edges"] == 2
    assert result.statistics["attainable_recall_all_target_edges"] == 0.
    assert torch.equal(before, graph.edges)


def test_crop_disconnection_is_not_full_reference_fragmentation():
    rows = [(1, -1, 0, 0, -1), (2, 5, 0, 0, 1), (3, 5, 1, 0, 2), (4, -1, 1, 0, 3)]
    tree = SwcTree(Path("synthetic.swc"), tuple(SwcNode(i, 3, x, y, z, 1., parent) for i, x, y, z, parent in rows))
    target = target_graph(tree, [(np.zeros(3), 2.)])
    assert topology(target)["components"] == 2
    components = full_reference_components(tree, target.identities)
    assert len(np.unique(components)) == 1
    p = packet(target.xyz)
    graph = build_candidate_graph(p.positions, p.node_ids)
    identity = list(range(len(target.xyz)))
    labels = assign_edge_labels(p, graph, target.xyz, target.edges, components, (identity, identity))
    assert labels.statistics["unknown"] == 0 and labels.statistics["negative"] > 0


def test_reference_empty_crop_abstains_and_loss_does_not_hide_unknowns():
    p = packet([[0, 0, 0], [1, 0, 0]])
    graph = build_candidate_graph(p.positions, p.node_ids)
    supervision = assign_edge_labels(p, graph, np.empty((0, 3)), [], np.empty(0, dtype=np.int64), ([], []))
    assert supervision.labels.tolist() == [-1.]
    logits = torch.tensor([5.], requires_grad=True)
    loss = masked_edge_loss(logits, supervision.labels)
    loss.total.backward()
    assert loss.supervised_crops == 0 and loss.total.item() == 0 and logits.grad.item() == 0
    assert supervision.statistics["attainable_recall_all_target_edges"] is None


def test_class_balanced_loss_and_equal_crop_averaging():
    logits = torch.tensor([.7, -1., .2, 999.], requires_grad=True)
    loss = masked_edge_loss(logits, torch.tensor([1., 0., 0., -1.]))
    expected = .5*torch.nn.functional.softplus(-logits[0])+.5*torch.nn.functional.softplus(logits[1:3]).mean()
    assert torch.allclose(loss.total, expected)
    loss.total.backward()
    assert logits.grad[-1].item() == 0
    other = masked_edge_loss(torch.tensor([2.]), torch.tensor([1.]))
    unknown = masked_edge_loss(torch.tensor([100.]), torch.tensor([-1.]))
    batch = mean_crop_edge_loss([loss, other, unknown])
    assert batch.supervised_crops == 2 and torch.allclose(batch.total, (loss.total+other.total)/2)
    with pytest.raises(ValueError, match="Labels"):
        masked_edge_loss(torch.ones(1), torch.tensor([2.]))


def test_decoder_rejects_cycles_keeps_isolates_and_exposes_node_cutoff(tmp_path):
    p = packet([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]], logits=[1, 2, 4, -2, 3])
    graph = CandidateGraph(torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3]]), 16, 8.)
    decoded = decode_forest(p, graph, torch.tensor([3., 2., 1., 20.]))
    assert set(decoded.selected_edges) == {(0, 1), (1, 2)}
    assert decoded.parent[2] == decoded.parent[4] == -1 and decoded.parent[3] == -2
    assert decoded.statistics["components"] == 2 and decoded.statistics["cycle_rejections"] == 1
    assert decoded.statistics["edges_rejected_by_node_cutoff"] == 1
    result = export_decoded_swc(p, decoded, tmp_path/"forest.swc", description="SYNTHETIC CHECK ONLY")
    assert result["nodes"] == 4 and result["roots"] == 2 and result["parent_before_child"]
    with pytest.raises(FileExistsError):
        export_decoded_swc(p, decoded, tmp_path/"forest.swc", description="SYNTHETIC")


def test_decoder_equal_score_ties_and_root_choice_are_permutation_consistent():
    p = packet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], ids=[30, 10, 20], logits=[0, 0, 0])
    graph = build_candidate_graph(p.positions, p.node_ids)
    decoded = decode_forest(p, graph, torch.zeros(len(graph.edges)))
    assert len(decoded.accepted_nodes) == 3  # Inclusive .5 cutoff.
    expected_edges = {tuple(sorted((int(p.node_ids[u]), int(p.node_ids[v])))) for u, v in decoded.selected_edges}
    perm = torch.tensor([2, 0, 1])
    q = replace(p, **{name: getattr(p, name)[perm] for name in
                      ("positions", "radii", "objectness_logits", "local_centers", "decoder_features", "node_ids")})
    other = build_candidate_graph(q.positions, q.node_ids)
    decoded_other = decode_forest(q, other, torch.zeros(len(other.edges)))
    actual_edges = {tuple(sorted((int(q.node_ids[u]), int(q.node_ids[v])))) for u, v in decoded_other.selected_edges}
    assert expected_edges == actual_edges == {(10, 20), (10, 30)}
    assert p.node_ids[decoded.traversal[0]] == q.node_ids[decoded_other.traversal[0]] == 10


def test_empty_decoding_is_explicit_failure_not_an_swc_success(tmp_path):
    p = packet([[0, 0, 0], [1, 0, 0]], logits=[-1, -1])
    graph = build_candidate_graph(p.positions, p.node_ids)
    decoded = decode_forest(p, graph, torch.ones(len(graph.edges))*20)
    assert decoded.status == "empty_reconstruction"
    with pytest.raises(ValueError, match="reconstruction failure"):
        export_decoded_swc(p, decoded, tmp_path/"empty.swc", description="SYNTHETIC")
    assert not (tmp_path/"empty.swc").exists()


def test_invalid_packets_candidates_and_assignments_fail_explicitly():
    p = packet([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="unique"):
        replace(p, node_ids=torch.tensor([0, 0])).validate()
    with pytest.raises(ValueError, match="Self"):
        CandidateGraph(torch.tensor([[0, 0]]), 1, 8.).validate(p)
    with pytest.raises(ValueError, match="Duplicate"):
        CandidateGraph(torch.tensor([[0, 1], [1, 0]]), 1, 8.).validate(p)
    graph = build_candidate_graph(p.positions, p.node_ids)
    with pytest.raises(ValueError, match="one-to-one"):
        assign_edge_labels(p, graph, p.positions, [(0, 1)], [0, 0], ([0, 0], [0, 1]))
    with pytest.raises(ValueError, match="crosses"):
        assign_edge_labels(p, graph, p.positions, [(0, 1)], [0, 1], ([0, 1], [0, 1]))
    with pytest.raises(ValueError, match="integers"):
        assign_edge_labels(p, graph, p.positions, [(0, 1)], [0, 0], ([0., 1.], [0, 1]))
