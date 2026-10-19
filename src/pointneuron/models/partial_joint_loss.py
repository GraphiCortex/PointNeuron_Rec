"""Isolated partial-supervision policy; the historical guarded joint loss is unchanged."""
import torch

from pointneuron.data.connectivity_targets import assign_edge_labels, masked_edge_loss
from pointneuron.models.balanced_query_loss import balanced_query_loss


POLICY = 'available_edge_class_means_or_proposal_only_v1'


def partial_joint_loss(output, record, proposal_recipe, *, edge_weight=1., match_tolerance=2.):
    device = output.packet.positions.device
    nodes = torch.as_tensor(record['nodes'], device=device)[None]
    mask = torch.ones(nodes.shape[:2], dtype=torch.bool, device=device)
    proposal = balanced_query_loss(output.proposals, nodes, mask, **proposal_recipe)
    if proposal.supervised_crops != 1:
        raise ValueError('Missing annotation remains unknown; expected supervised proposal targets')
    labels = assign_edge_labels(output.packet, output.candidates, record['nodes'][:, 1:4],
        record['target_edges'], record['target_components'], proposal.assignments[0],
        maximum_match_error=match_tolerance)
    edge = masked_edge_loss(output.edge_logits, labels.labels)
    # An attached zero graph term would give Adam zero grads rather than None.
    total = proposal.total + edge_weight*edge.total if edge.supervised_crops else proposal.total
    return total, proposal, edge, labels


def step_active_loss(model, optimizer, total, *, edge_active, gradient_clip=10.):
    """Caller clears grads before forward. Inactive graph parameters must stay None."""
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite partial joint loss')
    total.backward()
    proposal = list(model.proposal_model.parameters())
    graph = list(model.graph_model.parameters())
    for parameters, active, name in ((proposal, True, 'proposal'), (graph, edge_active, 'GAT')):
        if active:
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise ValueError(f'Missing/nonfinite active {name} gradients')
        elif any(p.grad is not None for p in parameters):
            raise ValueError('Inactive GAT must have None gradients, not zero gradients')
    def norm(parameters):
        terms = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
        return float(torch.stack(terms).sum().sqrt()) if terms else 0.
    result = {'proposal_gradient_l2': norm(proposal), 'gat_gradient_l2': norm(graph)}
    result['preclip_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True))
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite updated parameters')
    return result
