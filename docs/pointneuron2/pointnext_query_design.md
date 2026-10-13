# PointNeXt to Transformer: interface and bounded learnability plan

13 October 2026. Following the completed Phase 1 allocation studies, the user
agreed to prepare this design. **This is a reviewable plan, not an implemented
encoder or a training command.** No model code, environment, field checkpoint,
sampling recipe or production default is changed by this document.

## The next question

Can the intended PointNeXt/Transformer combination learn skeleton-node geometry
and usable confidence from a fixed field-expanded image point cloud? We have
measured an input-support benefit from expansion, but no end-to-end result for
this architecture. A two-crop fit establishes learnability only; it cannot show
generalization or superiority over PN 1.0.

The [quota comparison](field_image_reservation_results.md) is complete and is
not promoted. Retain expanded-uniform sampling B as the working research input
protocol for this next design. The frozen **fresh ungated baseline** field from
`field_attention_20261013/baseline/model` supplies P; attention and image quotas
remain comparison artifacts, not new defaults.

## Proposal alignment and current code

The original proposal's sections 2–4 specify per-point PointNeXt features,
Transformer skeleton proposals and later GATv2 connectivity. Those stage
boundaries are retained. Optional encoder Transformer fusion, explicit
multi-scale query banks and joint connectivity training are later experiments;
they are not silently implemented by the current single-query-bank decoder.
Proposal statements about expected accuracy, speed or tree structure are
hypotheses to evaluate, not observed results.

Original source: [Proposal.pdf](C:/Users/Seywe/Desktop/Research/R-Arij/Proposal.pdf),
especially pages 3–7. Its document instructions do not override the user's
subsequent agreement to pursue an automatic Phase 1 and involve them in changes.

| Component | Actual status | Planned interface |
| --- | --- | --- |
| Field guide | Trained/evaluated as an isolated Phase 1 candidate | Freeze weights during this learnability check |
| Expanded image points | Selector and comparisons exist | Freeze a new versioned two-training-crop cache |
| PointNeXt | Not implemented locally | Hierarchical encoding plus resolution restoration |
| Transformer | Existing query head, previously paired with DGCNN | Reuse body configuration and loss; explicit new memory-input width |
| GATv2 | Not implemented | Later consumer of proposal geometry/features |

The existing [query head](../../src/pointneuron/models/query_proposal.py) expects
one feature vector **aligned with each input point**, not a pooled classifier
vector. Its current [factory](../../src/pointneuron/models/query_pipeline.py)
always constructs DGCNN; loading PointNeXt into that old checkpoint format
would be wrong. A separate model identity and checkpoint schema are required.

## Proposed input contract

For each previously image-defined 97-voxel crop:

1. Freeze the field prediction from the selected epoch-30 checkpoint. These
   training-crop fields are not yet available as the new encoder input cache;
   producing them is a separate future visible preparation command, with zero
   optimizer updates.
2. Form C = F union P using the existing original intensity cutoff and field
   threshold .5. Uniformly select `N = min(2048, |C|)` without replacement,
   using the existing seed convention. This is **expanded mode**, not the
   matched audit's `min(2048, |F|)` cap. Use the already-tested B selector;
   do not invent another quota, ranking or adaptive rule.
3. Save native XYZ, raw intensity, voxel indices, field scores, validity and
   source/checkpoint hashes before opening targets for scoring.
4. For the first encoder prototype, use **local XYZ plus normalized intensity**
   as its four input feature channels. Preserve field scores in the cache for
   provenance; feeding them into the model is a separately assessed extension.

Local XYZ is `(native_xyz - patch_center) / 48`; intensity uses the full-image
`image_scale` already recorded by the native field cache. Keep the native array
unchanged for decoding, reports and later graph association. Do not recompute
intensity scale from whichever points happen to be selected, or accidentally
apply coordinate normalization twice.

Actual N is recorded. Start with batch size **1**, so crops of different counts
need no point duplication or batch padding. An empty/one-point input is reported
as insufficient for this prototype; it must not be disguised as a normal
cloud. Later batching requires explicit valid masks throughout neighborhoods,
feature propagation and Transformer attention; the current head does not yet
implement that mask contract.

## Encoder and decoder boundary

```mermaid
flowchart LR
    A[Fixed expanded image cloud] --> B[PointNeXt hierarchy]
    B --> C[Restore features to original input points]
    C --> D[Transformer learned queries]
    D --> E[Native XYZ, confidence, radius, query features]
    E -. later stage .-> F[GATv2 connectivity]
```

The proposed initial reference is the small segmentation-oriented PointNeXt-S
configuration: width 32, stage blocks `[1,1,1,1,1]`, strides `[1,4,4,4,4]`,
residual set abstraction, initial radius .1 and up to 32 neighbors. Use its
hierarchy plus generic feature propagation to restore the original resolution.
The small S configuration does not add deeper inverted-residual blocks; larger
variants are separate scaling experiments. These choices are based on the
[authors' S segmentation configuration](https://raw.githubusercontent.com/guochengqian/PointNeXt/master/cfgs/s3dis/pointnext-s.yaml)
and [reference backbone/feature-propagation code](https://raw.githubusercontent.com/guochengqian/openpoints/master/models/backbone/pointnext.py).

The proposal mentions kNN grouping; this reference configuration uses radius
ball-query grouping with relative offsets normalized by radius. Follow and
record the reference variant initially rather than combine a grouping change
with the encoder test. At our local coordinate scale, radius .1 corresponds to
4.8 native voxels. This radius convention needs an explicit unit test.

**PointNeXt feature propagation is resolution restoration, not a replacement
for the Transformer skeleton decoder.** It brings local/coarse features back
to the original points. The Transformer then predicts a separate node set.
Restoration retains fine-level skip information, but does not itself guarantee
preservation of dim branches or correct connectivity.

Expected boundary:

- Native points: `[B,N,4]`, including raw intensity.
- Encoder geometry/features: local coordinates `[B,N,3]`, features `[B,4,N]`.
- Restored descriptors: `[B,N,C]`, in original point order; C comes from the
  declared encoder configuration (normally 32 for this reference).
- Query head: memory input width C; 512 queries, model dimension 128, three
  decoder layers, four heads, feedforward width 512, dropout zero.
- Output: centers `[B,512,3]` in native voxels; two-class confidence logits;
  positive radii in existing native units; query features `[B,512,128]`.

The existing memory projection must accept C instead of DGCNN's 1,216. This is
an explicit shape adaptation, not a claim of identical weights. Decoder body
configuration, target construction and loss can be reused. A later matched
encoder comparison must verify shared compatible decoder initialization and
declare the distinct input projections; historical DGCNN runs are not a matched
control for this new input cache or batch schedule.

## Local backend decision, before coding

The [official installation](https://github.com/guochengqian/PointNeXt#installation)
states that full training/evaluation requires custom CUDA/C++ operations. The
current PointNeuron environment is Windows/PyTorch 2.12.1 + CUDA 13.0; `nvcc`
and `cl` were not found on the current PATH. This does not prove they are absent
from the machine, but the official compilation path is unverified here.

Recommended route for discussion: a documented **native PyTorch prototype** of
the reference architecture/point operations in the current environment. Specify
FPS, ball grouping, reductions and three-neighbor feature interpolation; test
analytic cases, ties, boundary inclusion, gradients and short neighbor sets.
Use chunked distance operations and native CUDA tensors where appropriate.
Label the checkpoint with the actual backend and source/config hashes. Do not
claim official benchmark reproduction, exact kernel equivalence or published
throughput without the relevant comparisons.

Alternatively, investigate the official backend and toolchain first, reporting
compatibility before any environment changes. The user has been asked which
route to use; no package install, compiler setup or backend implementation has
started. Resolve that choice before code is written.

The hierarchy must be tested on short actual counts, including 256/358 points.
Fixed strides can leave very few coarse points. Check both normalization and
sampling in batch-1 train/eval mode. Do not pad duplicate input points, silently
drop a stage, or replace normalization to get around a failure: report the issue
and agree the smallest explicit adjustment before fitting. Neighborhood index
padding, if needed to reproduce reference grouping, is distinct from duplicating
the input cloud and must have documented semantics.

## Proposed learnability run: two fixed training crops

Use training neuron **0**, image-defined crops **0 and 1**, without choosing
crops based on eventual encoder success. A read-only target check confirms:

| Crop | Native center | Original foreground points | Existing curve targets |
| --- | --- | ---: | ---: |
| 0/0 | (334,131,12) | 361 | 6 |
| 0/1 | (334,371,60) | 385 | 12 |

Targets use the existing [continuous-reference constructor](../../src/pointneuron/data/curve_targets.py):
8-voxel arc spacing, structural nodes and crop-boundary intersections, with
identity preserved and no target truncation. This confirms capacity 512 is
adequate for these two crops; input-cloud count and query capacity are different
budgets. Targets are never passed to selection or field inference.

Proposed fixed settings: batch 1 alternating the two crops; no augmentation;
AdamW LR .0001, weight decay .0001, gradient clip 10; no scheduler; at most
**2,000 optimizer updates total**. Record checks at 0/500/1,000/2,000. This is
a new learnability recipe, not an encoder-only ablation against the old batch-2
fits. Fix the sampling seed and full input arrays during the entire fit.

Reuse [query-set loss](../../src/pointneuron/models/query_loss.py): Hungarian
one-to-one matching, existing confidence/center/radius weights and scales,
positive class weight 8 and radius floor 1. Keep cutoff .5. Reference-empty
crops retain unknown/abstention semantics; they are not new background labels.
No NMS, suppression, global selection or graph tracing in this check.

First implement and verify the interface/operations on synthetic data. Then
prepare and inspect the two real input/target records. A separate visible local
GPU preflight should verify shapes, gradients and reload, and time a few
throwaway updates to estimate the bounded fit. **Do not launch the fit until
the user reviews the data, settings and RTX 3070 Ti Laptop GPU time estimate.**
There is no measured duration for this encoder yet.

## Proposed acceptance and stop criteria

Execution checks require finite outputs/gradients, exact source hashes and
partition identities, preserved point order, valid counts, and strict model
reload. Predictions after reload should agree within declared float32 tolerance
in the same device/mode. Check ordinary `eval()` inference, not only training
loss, including short clouds and repeated single-crop inference.

Suggested learnability gate at the fixed final update, **for each crop**:

- All reference targets have distinct accepted proposals at score .5 within
  one native voxel; report mean/max assignment distance and accepted counts.
- No unmatched query passes .5 on these fitted annotated crops; report it
  separately so duplicate/extra proposals cannot hide behind coverage.
- Report full clipped-curve coverage and per-chain longest gaps at tolerances
  2 and 6. Require at least 95% accepted-node curve coverage within 6 voxels;
  sparse 8-voxel target spacing makes 2-voxel curve coverage a different metric.
- Radii are finite/positive; report error against the existing floored-radius
  targets. This check does not establish accurate biological radius recovery.
- Training/eval/reload geometry and confidence remain consistent. Do not lower
  thresholds, remove hard crops or change quotas to turn a failed gate into a pass.

If the fit fails, use saved predictions/assignments to locate input, feature,
matching or confidence failure before an extension/change. If it passes,
consider a broader fixed training-crop check across multiple training neurons,
then a separately agreed train/development pilot. The seven development neurons
remain outside fitting and the eight reserved test neurons remain untouched.
Neither result alone authorizes a production model or final reconstruction claim.

## What remains after this component

Generalization, whole-neuron patch aggregation, GATv2 edge targets/training,
tree/topology handling and matched PN 1.0 reconstruction evaluation are still
required. The flat query target arrays do not yet supply a trained GATv2 graph.
Retain query features and reference identity/chain provenance for that future
interface, without inventing edge labels in this proposal fit.

The frozen field and hard selection keep this experiment outside the proposal's
fully joint sampling/encoder/detection/connectivity training. Joint optimization
and the differentiable-sampling choice remain explicit later design work.
The [PointNeXt paper](https://arxiv.org/abs/2206.04670) motivates the backbone,
but its classification/segmentation benchmarks do not establish neuron tracing
performance. This plan returns to the intended architecture while keeping the
unresolved input limitations measurable.
