# Prospective image-conditioned geometric queries

31 October 2026. **Discussion proposal; not implemented or authorized for a new
fit by this document.** It follows the [completed GN8 result](groupnorm_node_fit_results.md).
The next decision concerns a specified Stage 3 mechanism, not another
normalization or confidence-threshold search.

## Responsibility and motivation

The Transformer must turn image-derived features into a variable number of
well-placed, nonredundant skeleton nodes and usable features for connectivity.
Its current learned slots only partly fit four fixed training boxes. GroupNorm
removes the observed inference-mode discrepancy, yet geometry and used-slot
selection still fail. The exact cause is not established.

Current implementation: `../pointneuron2/src/pointneuron/models/query_proposal.py`,
hash-pinned by the active interface/fit configurations. It projects restored
PointNeXt features and local XYZ into memory. The 512 initial decoder tokens are
learned embeddings shared across crops. Three Transformer decoder layers can
attend to the image memory; thus these are **not image-blind final predictions**.
The shared output MLP predicts centers relative to one crop center, along with
two-class logits and radius. Radius remains unsupervised in this bounded task.

Hypothesis: tying a query's initial representation to an actual image point and
its encoded features, then predicting an offset from that location, gives the
decoder a more useful geometric starting task. It may reduce global localization
burden and help used-slot classification depend on the local image structure.
This is a testable design choice, not a diagnosis proven by score correlations.

## Architecture impact and exact candidate

```text
Frozen automatic image cloud: 2048 points + intensity
                  |
          Same GN8 PointNeXt
                  |
        Restored features at all 2048 points
                  |                         |
      All-point Transformer memory     FPS on input XYZ only
                  |                    512 starting locations
                  |                         |
                  |              Gather local encoded features
                  |              + position + learned token
                  |                         |
                  +----------> Same Transformer decoder
                                            |
                              Position = start + learned offset
                              Existing confidence/features/radius
                                            |
                            Future consolidation -> GATv2 -> SWC
```

Proposed equations, with the existing crop center `c` and scale `s = 64` voxels:

```text
memory_i = feature_projection(F_i) + position((x_i - c) / s)
indices = deterministic_FPS((x - c) / s, K=512)
start_q = x[indices[q]]
query_q = learned_token_q + memory[indices[q]]
decoded = existing_Transformer(query, memory_all_2048)
raw = existing_prediction_MLP(decoded)
center_q = start_q + s * raw_offset_q
public_local_center_q = (center_q - c) / s
```

This revises **query initialization and center parameterization within Stage 3**.
It neither bypasses the Transformer nor replaces PointNeXt/GATv2. Both changes
form one geometric-query candidate; a positive result would not identify their
individual causal contributions. Reuse of existing projections, token embeddings
and prediction MLP is intended to preserve the trainable parameter count; that
must be verified before implementing a comparison.

All 2048 points remain in encoder/attention memory. The 512 selected coordinates
are starting locations, not an input-cloud reduction, accepted-node count or
GT-selected skeleton. There is no snapping, new displacement cutoff, reference
access during selection, target-count oracle or manual mask. Offsets remain
unbounded; output local-center semantics remain crop-relative for the existing
loss and future downstream interface. No trained BN/GN checkpoint is converted.

For this comparison N is exactly 2048. A later variable-size interface could
use `K=min(512,N)` without reference-derived counts; native encoder minimum size,
duplicate-coordinate clouds, empty inputs and target overflow need explicit
policies before product use. The present check establishes only distinct source
indices in these four fixed packets, not universal spatial coverage.

## Research basis and limits

The official [3DETR implementation](https://github.com/facebookresearch/3detr/blob/main/models/model_3detr.py)
samples query coordinates by farthest-point sampling and predicts centers as
query locations plus offsets. Its decoder separately injects coordinate-derived
query position embeddings. This supports considering spatial query structure
for 3D data; it does not establish success for neuronal skeletons.

The proposed PN variant uses existing MLP position/features, learned token content
and unbounded voxel-scaled offsets. It is not a 3DETR reproduction: that model
has different encoding, positional embeddings, decoder details and bounded box
offsets. Feature-conditioned token initialization here is our hypothesis, not
an attributed 3DETR result. No denoising scheme, new loss or added head is bundled.

## CPU check: starting locations alone are insufficient

The read-only review ran the pinned native FPS on frozen hash-ordered input XYZ.
References were read only to score the already selected coordinates. No model
was instantiated or forwarded for this check and no optimizer was used.

| Training box | Targets | Full 2048 input matches at 2 voxels | 512 starts at 2 voxels | Starts at 6 voxels | Mean one-to-one target-to-start distance |
|---|---:|---:|---:|---:|---:|
| fruitfly 32/27 | 18 | 14 | 5 | 18 | 2.91 voxels |
| fruitfly 32/20 | 85 | 71 | 7 | 82 | 3.65 voxels |
| frog 22/2 | 7 | 6 | 1 | 6 | 4.10 voxels |
| frog 22/5 | 68 | 21 | 4 | 54 | 4.50 voxels |

FPS reduces fine spatial support if its output is treated as the entire cloud.
That would be a poor justification for this design. Here it only specifies
starting locations; full memory and learned movement are essential. The largest
assigned starting displacement reaches 13.59 voxels. A small movement clamp
could exclude real targets and is not proposed.

The start-coordinate prior has smaller mean assignment distances than the
current untrained model's 40.19/54.01/66.04/52.03 voxels. This is a coordinate-only
comparison: feature initialization and the actual untrained offset head have
not been executed. It is neither a learned improvement nor evidence that
selection, dim-branch retention or generalization will improve.

## Proposed comparison if agreed

1. Implement a separate, readable candidate while preserving the GN baseline.
   CPU checks cover coordinate translation/scale semantics, exact no-GT FPS
   determinism, input/parameter identity, retained full memory, finite geometry
   and class gradients, train/eval consistency, and model/Adam/RNG reload.
   No arbitrary extra constraints or new hyperparameter search are included.
2. Before fitting, report raw-start coverage separately from the actual
   untrained model's geometry and confidence. Inspect the interface and allow
   the user to steer before a timing/fit command.
3. If qualified, prepare visible timing and one fresh matched four-training-box
   fit. Keep the GN baseline's seeds/initial named parameter values, 2048 points,
   512 queries, targets, matching/loss, AdamW LR 0.0001, cutoff 0.5, evaluation
   steps and 2000-update cap. Current fit cost was about 32 minutes on the local
   RTX 3070 Ti Laptop GPU; candidate timing must establish its own cost.
4. Success remains at least **95% two-voxel one-to-one precision and recall in
   every box under ordinary inference**, with mode consistency. Report raw
   geometry, accepted nodes, extras and continuous-curve diagnostics separately.
   No reference-selected subset qualifies. Passing earns a larger training and
   held-out development study; it does not establish product quality.
5. Stop at the cap. If geometry is good but selection fails, this mechanism has
   not delivered its required node set. If geometry fails, confidence tuning
   cannot repair the missing proposal support. Neither outcome authorizes an
   automatic LR/threshold/budget/normalizer sweep, extension or GAT handoff.

On another failed candidate, stop architecture accretion and review the
supervision/formulation explicitly. Interior nodes on an eight-voxel full-chain
grid are representational choices; their phase can depend on anatomy outside
a crop. Exact used-slot labels therefore are not uniquely observable anatomical
objects. Geometric queries do not solve that ambiguity by themselves. A
curve-support or alternative discretization objective would be a separate,
substantial research decision with its own interface/metric implications.

No new GPU command is issued by this proposal. Phase 1's broad information
retention, native PointNeXt fidelity beyond the audited prototype, cross-crop
consolidation, GATv2 and final reconstruction remain separate qualifications.
