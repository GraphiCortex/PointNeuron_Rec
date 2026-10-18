# Bounded joint proposal/connectivity prototype

19 October 2026. Following the [node/graph responsibility review](node_graph_design_decision.md),
the user agreed to prepare this joint-learning comparison with the existing
heads. GPU timing and retained training are separate user-launched actions.

## The decision this experiment supports

Can the existing PointNeXt/Transformer and GATv2 learn a clean **automatic**
node-and-edge reconstruction on the same four training crops when connected by
a shared loss and exposed to the actual proposal pool? This tests an integration
missing from the separate fits. It does not assume that joint learning fixes the
known confidence optimization problem.

Passing establishes bounded training-set learnability of this per-crop prototype.
It does not establish generalization, whole-neuron reconstruction, superiority to
PN 1.0, or superiority to a simpler pairwise edge scorer. A failure is not proof
that every joint architecture is impossible, but blocks calling this formulation
learnable and advancing it as a successful automatic system.

## Architecture and initialization

- Keep the frozen Phase 1 image-derived clouds: 848 / 2,048 / 1,219 / 2,048 distinct
  points for training crops 0/0, 0/1, 92/4, 86/3. Targets: 6 / 12 / 16 / 418;
  required edges: 5 / 10 / 15 / 404; crop forest components: 1 / 2 / 1 / 14.
- Warm-start PointNeXt/Transformer from the completed balanced 4,000-update fit,
  and GATv2 from the completed oracle-selected 4,000-update graph fit. Both
  checkpoint and report hashes are pinned in the visible
  [recipe](../../configs/pointneuron2/joint_query_graph.json).
- Create a **new** AdamW optimizer, LR 0.0001, weight decay 0.0001, joint global
  gradient clipping at 10. This is an additional stage, not exact continuation of
  either source fit. Seed zero, batch one, cyclic crop order, no augmentation,
  no schedule, maximum 4,000 new updates, snapshots 0/500/1,000/2,000/4,000.
- Keep explicit current-crop normalization in PointNeXt and zero Transformer
  dropout. GAT uses its existing LayerNorm. This normalization remains an
  experimental policy, not a validated general inference default.
- All **512 queries** enter GAT, including low-confidence and unmatched queries.
  Candidate pairs are the union of symmetrized 16-nearest neighbors and radius
  eight, rebuilt from current positions each forward. There is no hidden edge
  cap or reference-driven candidate selection.
- GAT scores edges. The original Transformer confidence still selects final
  nodes at inclusive 0.5. Keep only already-scored pairs whose endpoints survive;
  do not rebuild candidates or rerun GAT after filtering. Apply the existing
  0.5 edge cutoff and reference-free forest decoder, retaining accepted isolates.
  No graph node head, node motion head, pruning heuristic, or topology loss is added.

Delaying the node cutoff is an explicit interface-policy experiment. It lets low
confidence proposals participate in graph context; it does not let GAT directly
override the final node decision. The all-query stage-zero output is the relevant
frozen warm-start baseline. The earlier automatic integration used **early**
filtering and rebuilt candidates, so its numbers are historical context rather
than a matched baseline for an isolated training effect.

## Loss, supervision and gradients

Combined loss is the existing balanced proposal loss
`2 * class + 3 * center + 0.2 * radius`, plus `1 * balanced edge BCE`.
The class and edge terms each average positive/negative class means per crop.
This coefficient is a declared initial control, not a tuned optimum.

Recompute the existing Hungarian assignment each update. Assigned query confidence
is positive regardless of placement; unmatched query confidence is negative for
**unique target-node membership**, not an assertion of image background. Geometry
and radius losses supervise assigned queries.

For edge labels retain the ordinary two-voxel placement trust mask: known
adjacency is positive; trusted assigned same-full-tree nonadjacency is negative;
an unmatched/untrusted endpoint or different/unknown full-reference tree is
unknown and excluded from direct edge loss. Missing crop annotation is unknown,
never verified background. Unlike the prior conditional GAT fit, this experiment
does **not** bypass placement trust. Crop-disconnected pieces may belong to the
same full-reference tree; no one-component penalty is imposed on cropped forests.

Continuous positions, radii, confidence logits and decoder features stay attached
to autograd between proposal and graph models. Graph loss can reach both models.
Matching, candidate membership, edge trust membership, and final threshold/forest
decisions are discrete and receive no gradient. The final decoded graph itself
is evaluated, not differentiated through. Unknown incident edges receive no
direct target, although their nodes can influence supervised edges by attention.
Consequently, passing gradients alone cannot establish that extras will disappear.

## Declared success and stopping rules

Every crop must pass at the terminal snapshot:

1. All assigned target queries accepted, zero unmatched accepted queries.
2. Exact decoded target adjacency: no missing or nonreference edges; correct
   forest component count and junction incidence; every visible reference
   junction retained and placed within two voxels.
3. Continuous-reference arclength precision **and** recall at two voxels at least
   99%, plus valid SWC serialization and coordinate/radius round trip.

This is deliberately a strict target-representation learnability gate. Redundant
on-curve unmatched nodes can be anatomically reasonable subdivisions, so also
report geometric quality, paths, and the node/candidate/score/forest loss
decomposition. Do not call every direct-pair mismatch a broken anatomical branch.
The 99% geometry gate is stronger than the frozen learned-geometry oracle on the
dense crop; joint center learning must improve it. The reference-node/chord
representation previously reached approximately 99.73% dense recall at two
voxels, so the representation itself does not rule out this gate.

Stop at 4,000; do not extend, lower cutoffs, switch to reference selection, or
change gates after seeing results. Stop with preserved evidence on source/cache
changes, nonfinite values, unavailable deterministic kernels, OOM, or loss of
either known positive or known negative edge supervision. Do not shrink the
query pool or relabel unknown edges to rescue an execution failure. Save fit
checkpoint/optimizer/RNG **before** snapshot evaluation so a metric failure
cannot destroy the preceding fitted state. Exact resume is restricted to an
intermediate checkpoint with matching recipe, provenance, backend and total cap.

Failure interpretation: separate execution failure from optimization failure.
For a completed fit, identify automatic node errors first, then candidate
coverage, edge scoring, forest decoding and continuous geometry. Good conditional
edges with persistent node errors mean the original confidence responsibility
remains unsatisfied. This requires a design discussion; another isolated fit or
a graph node head does not follow automatically. A passing result permits
planning broader automatic evaluation; it does not promote the model into the
whole-neuron reconstruction pipeline.

This warm-start experiment has no equal-extra-compute proposal-only arm. Changes
from stage zero demonstrate what this joint stage achieves, not that edge loss
caused gains beyond another 4,000 proposal-only updates. Establishing that causal
benefit, and the value of GAT attention, would require separately agreed matched
controls. No development/test neurons are read by these actions.

## Commands and evidence

Completed readiness: **39 targeted tests pass**. The real-model CPU check took
32.60 seconds after setup, with **zero optimizer updates** to its 1,807,559
parameters. Graph-only gradients are finite and nonzero through encoder,
Transformer, GAT, positions, radii, confidence and decoder features on all four
crops. Loaded weights and the same-backend proposal adapter match exactly.
All four SWCs validate; checkpoint reload reproduces proposal coordinates,
radii, confidence, candidates and graph scores exactly. Source inputs and model
weights remain unchanged. Synthetic tests also verify an identical next update
after restoring optimizer/RNG and preserving checkpoints before metric failures.

The CPU warm start with the **new all-query context** fails all automatic gates:

| Training crop | Accepted assigned + unmatched | Direct required edges retained | Arclength precision / recall at two voxels |
|---|---:|---:|---:|
| 0/0 | 6 + 11 | 2/5 | 59.88% / 94.87% |
| 0/1 | 12 + 32 | 5/10 | 49.94% / 100% |
| 92/4 | 16 + 17 | 7/15 | 63.69% / 78.91% |
| 86/3 | 411 + 4 | 372/404 | 97.60% / 94.12% |

These are CPU baseline measurements, not joint-fit improvements. Final training
comparison uses the stage-zero baseline on the same GPU/backend as the fit.

An initial CPU check stopped because it incorrectly required tight replay of
historical **GPU** predictions on CPU. The maximum cross-backend coordinate
difference is 0.077484 voxel (maximum logit difference 0.006471), with **zero
node cutoff changes** on every crop. Its exact numerical cause is unresolved;
do not describe this as an exact historical replay. The failed check is preserved
under `tmp/pointneuron2/local_runs/joint_query_graph_20261019/check`.
The completed check uses a fresh output folder and distinguishes exact
same-backend adapter/reload verification from a reported cross-backend difference.
With autograd enabled, CPU train/eval predictions match exactly; enabling
`no_grad` changes coordinates by at most 0.0000611 voxel and logits by
0.00000120, within the declared relative/absolute 1e-5 proposal check. The GPU
timing action requires historical GPU proposal replay at relative/absolute 1e-5
before any throwaway updates, so a backend discrepancy remains visible and can
stop that action. No loss, reconstruction cutoff or success gate was relaxed.

The visible launcher exposes `prepare`, `check`, `timing`, `train`, `resume`.
Preparation and check use CPU and make no optimizer updates. Timing performs
eight throwaway updates on the RTX 3070 Ti Laptop GPU, verifies reload predictions,
then discards the temporary trained checkpoint. It measures current all-query
candidate counts and peak memory; no duration is promised from the smaller
oracle-node graph control.

Default outputs: `tmp/pointneuron2/local_runs/joint_query_graph_ready_20261019`.
Reports, copied source, predictions, SWCs, checkpoints and logs remain ignored.
CPU readiness evidence is summarized in
[the visible validation record](joint_query_graph_ready_seed0/validation.json).
The source wrapper is [joint_query_graph.py](../../src/pointneuron/models/joint_query_graph.py)
and the runner is
[joint_query_graph_experiment.py](../../scripts/pointneuron2/joint_query_graph_experiment.py).

After completed CPU preparation/check, the next user command is:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_joint_query_graph.ps1 -Action timing
```

Review its execution, candidate counts, memory and estimate before separately
launching `-Action train`. GPU timing and retained training have not been launched
as part of preparation. The terminal training report's `automatic_gate` is the
primary outcome; reference-selected success cannot pass it.
