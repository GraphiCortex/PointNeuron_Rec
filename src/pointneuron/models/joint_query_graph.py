"""Joint training adapter: existing proposal and edge heads, no node filtering."""
from dataclasses import dataclass

import torch
from torch import nn

from pointneuron.models.balanced_query_loss import balanced_query_loss
from pointneuron.data.connectivity_targets import assign_edge_labels, masked_edge_loss
from pointneuron.graph.proposal_graph import ProposalGraphInput, CandidateGraph, build_candidate_graph


@dataclass(frozen=True)
class JointOutput:
    proposals: object
    packet: ProposalGraphInput
    candidates: CandidateGraph
    edge_logits: torch.Tensor


class JointQueryGraph(nn.Module):
    def __init__(self, proposal_model, graph_model, *, neighbors=16, radius=8.):
        super().__init__()
        self.proposal_model=proposal_model
        self.graph_model=graph_model
        self.neighbors,self.radius=neighbors,radius

    def forward(self,points,centers,image_scales,patch_radius=48.):
        output=self.proposal_model(points,centers,image_scales,patch_radius)
        if len(output.center_proposals)!=1: raise ValueError('Joint control requires one crop')
        q=output.center_proposals.shape[1]
        packet=ProposalGraphInput(output.center_proposals[0],output.radius[0],output.objectness_logits[0],
            output.local_centers[0],output.decoder_features[0],torch.arange(q,device=points.device),
            {'scope':'all queries before final node selection'})
        graph=build_candidate_graph(packet.positions,packet.node_ids,neighbors=self.neighbors,radius=self.radius)
        scores=self.graph_model(packet,graph)
        return JointOutput(output,packet,graph,scores.edge_logits)


def joint_loss(output,record,proposal_recipe,*,edge_weight=1.,match_tolerance=2.):
    device=output.packet.positions.device
    nodes=torch.as_tensor(record['nodes'],device=device)[None]
    mask=torch.ones(nodes.shape[:2],dtype=torch.bool,device=device)
    proposal=balanced_query_loss(output.proposals,nodes,mask,**proposal_recipe)
    if proposal.supervised_crops!=1: raise ValueError('Missing annotation is unknown, not background')
    labels=assign_edge_labels(output.packet,output.candidates,record['nodes'][:,1:4],record['target_edges'],
        record['target_components'],proposal.assignments[0],maximum_match_error=match_tolerance)
    edge=masked_edge_loss(output.edge_logits,labels.labels)
    if not edge.positive_count or not edge.negative_count:
        raise ValueError('Joint control needs known positive and negative edges; stop without relabeling unknowns')
    return proposal.total+edge_weight*edge.total,proposal,edge,labels


def retained_graph(output):
    """Final .5 node decision AFTER scoring all queries; do not rebuild candidates."""
    p=output.packet
    keep=p.objectness_logits[:,1]>=p.objectness_logits[:,0]
    ids=torch.where(keep)[0]
    fields=(p.positions,p.radii,p.objectness_logits,p.local_centers,p.decoder_features,p.node_ids)
    selected=ProposalGraphInput(*(v[ids] for v in fields),p.provenance)
    mapping=torch.full((len(keep),),-1,dtype=torch.long,device=ids.device)
    mapping[ids]=torch.arange(len(ids),device=ids.device)
    edge_keep=keep[output.candidates.edges].all(-1)
    graph=CandidateGraph(mapping[output.candidates.edges[edge_keep]],output.candidates.neighbors,output.candidates.radius)
    return selected,graph,output.edge_logits[edge_keep]
