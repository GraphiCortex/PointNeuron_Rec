# Repaired PN1: implementation plan before a fresh fit

**Archived direction, superseded 22 October 2026.** PN1 is for conceptual
comparison; its reconstruction port and singleton correction are not active
PN2 tasks. The PN1-only code is preserved at pushed commit `48519ff` and removed
from active source. The reusable decoder/tiler remain. Proceed from the
[original-image Phase 1 design](pn2_phase1_design.md), not the port below.

21 October 2026. Source inspection following the common evaluator milestone.
This is a concrete integration draft, not a runnable training recipe. No model
code, loss, threshold or checkpoint was changed. No image/reference contents,
model inference or training were used for this inspection.

## What the reference will retain

R is a fresh PN1-style reference: the working local DGCNN-style encoder,
pointwise center/radius/confidence head, conservative proposal objective and
image-geodesic reconstruction. N remains the proposed PointNeXt/Transformer/
GATv2 system. Preparing R does not replace a stage of N.

The [training inventory](../configs/training_inventory.json) lists exactly 33
training neurons by stable identity. It joins the existing partition to the
163-record source inventory; all 66 listed image/reference paths exist.
Development and reserved identities are excluded. This metadata check does
not certify annotation completeness, calibration or acquisition independence.

| Component | Retained source | Fresh integration requirement |
|---|---|---|
| Corrected image decoding | `data/vaa3d_raw.py` | Reuse verified decoder and independent format fixtures; record reader version; explicit unsupported encodings |
| Image intensity policy | `data/foreground.py` | Explicit image-only configuration restored with checkpoint; no dtype-range fallback |
| Encoder and proposal | `models/dgcnn.py`, `models/proposal.py` | Retain local-coordinate head with native-voxel offsets/radii; fresh parameters; audit ordinary evaluation mode |
| Proposal objective | `models/proposal_loss.py` | Retain conservative terms and document exact labels; resolve crop-boundary issue below before creating targets |
| Geodesic selection/path tracing | `scripts/initialize_geodesic_graph.py` | Extract image/proposal operations from dataset scanner; preserve named selection behavior and expose actual counts |
| SWC export | `scripts/generate_swc_from_graph.py` | Retain traced paths and forest components; automatic root convention; explicit empty result rather than process-only failure |

Paths in this table are relative to the historical `pointneuron2/src/pointneuron`
for library modules, or to that worktree root for scripts. The source checkout
is `pointneuron2-history` at `35bfae9`. The original H revision remains
`1719a56`; it is not silently replaced by this repaired code.

## The interfaces to implement first

```text
Explicit image case + frozen preparation config
    -> decoded volume + coordinate/intensity provenance
    -> deterministic covering tile inventory
    -> actual image-derived clouds
    -> R proposals in native global XYZ
    -> merged proposals + image-geodesic paths
    -> forest/SWC + job status, counts, time and memory

Separate reference manifest + saved SWC -> shared evaluator
```

The inference case must contain an image path and metadata, with no reference
path argument. Reference alignment/quality checking is a separate audit. A
reference failure must not decide whether the image is reconstructed.

The existing `aggregate_proposals.py` scans the dataset, selects by numerical
index, loads the reference, and can terminate on its alignment before inference.
It also combines inference, measurement and HTML output. Its proposal math is
reusable; invoking that CLI unchanged is not the fresh automatic interface.
The geodesic CLI also resolves images through a dataset scan. Explicit image
paths must replace both lookups; no source-wide scanner is needed.

Whole-volume inference must enumerate its entire declared tiling, without the
old eight-crop cache cap or the aggregation CLI's default 256-tile cap. Log
empty, sparse, processed and unsupported tiles separately. Demonstrate boundary
coverage with synthetic volumes. Training may sample a declared image-defined
subset; inference must not silently reuse that bounded training subset.

No new learned sampler or image field is required to port R. The repaired
random-selection control is a candidate starting policy, not a proven universal
foreground model. Thresholding and point selection can still lose weak branches;
their losses must remain visible in the full reconstruction result.

## A label issue that needs an explicit decision

The old cache builder crops **stored SWC vertices**, not intersecting segments.
Its proposal loss then matches input points to those cropped vertices.
A simple counterexample is a reference edge from `(-10,0,0)` to `(10,0,0)`
crossing a crop `[-5,5]^3`: neither stored endpoint is in the crop, although a
ten-unit traced segment crosses it. A point at the crop center is on the trace
but ten units from either stored endpoint.

Thus "no cropped nodes" need not mean "no local traced curve". The historical
trainer correctly skips proposal supervision on reference-empty records, but
vertex-only cropping can unnecessarily create such records. Nearest-vertex
targets also depend on how densely an SWC was saved. This is a representational
issue demonstrated by construction; its frequency in the new training set has
not been measured here.

**Proposed correction for review:** construct local supervision from clipped
reference segments, with an explicitly specified target/radius convention.
Do not implement this by silently densifying to another arbitrary vertex step.
Continuous nearest-curve centers would change the pointwise regression target
from nearest stored vertex to nearest traced location; that is a substantive
supervision change, even though the R architecture stays the same. Decide and
version it before producing the fresh cache. N's canonical nodes may still be
a different training representation; both must describe the same reference
curve and must use the common final-SWC evaluator.

Reference-empty regions remain unknown unless certified as background.
Reference-containing crops do not prove every unmatched bright object is false.
The chosen reference-agreement assumption and any negative-label mask must be
written down for both arms. A new penalty cannot resolve missing annotations.

## Normalization and inherited optimizer settings

Retain R's BatchNorm initially, with running statistics learned only from
training data and ordinary `eval()` inference. The PointNeXt per-crop policy
diagnosed in the old two-crop experiment is not automatically an R policy.
Check held parameters under inference batch sizes one/two and exact checkpoint
reload before training. The encoder's pooled BatchNorm needs more than one
training example; specify how a last singleton batch is handled rather than
silently dropping a fixed neuron. Short-cloud/padding behavior must also be
explicit; repeated points are not additional image information.

The proposed optimizer starting point is inherited from the promoted PN1:
fresh seed zero, AdamW, LR `0.0001`, weight decay `0.0001`, nominal batch size
two and conservative head initialization. Keep the existing conservative loss
weights as the reference starting point. These are historical settings proposed
for continuity, not newly established optimal settings. They do not inherit H's
weights, optimizer state or BatchNorm statistics.

The inherited radius-target enum named `physical` means the SWC radius value;
it does not certify micrometers. For the current uncalibrated inventory,
coordinates, offsets and radii retain their native convention. A calibrated
future cohort needs a separately declared coordinate/radius policy before
training or physical-accuracy claims.

The old five-epoch repaired control does not set the fresh budget. H's 30 epochs
are context, not an equivalent number of updates after rebuilding the data.
Finalize duration/update cap, image contributions, augmentation, precision and
checkpoint selection after the preparation/label contract and CPU inventory are
known, followed by an agreed GPU timing check. No training command exists yet.

## Budgets and geodesic geometry

Record four different counts: input points per tile, retained proposals per
tile, consolidated proposals and selected graph anchors. Also record final SWC
vertices/arclength. PN1's historical 128 graph anchors can expand into many SWC
vertices along image-geodesic paths. They are not equivalent to 128 direct
straight-line skeleton nodes in a query model. Compare final reconstructions
and resources; equalizing these different counts would not establish fairness.

The old geodesic implementation materializes Dijkstra distances and predecessors
from selected anchor candidates to foreground voxels. That allocation scales
with both counts, before final anchor selection. Measure/preflight that storage
before a whole-volume run. Raising the foreground cutoff to fit a memory limit
would change the image representation; the fresh runner must not do it silently.

Normal algorithmic graph/proposal pruning can yield a completed but inaccurate
reconstruction and must be disclosed and scored. An inability to process the
declared tiling because of a runtime/memory limit is an execution failure, not
a successful full-volume result. These are different failure mechanisms.

## Next bounded implementation and stopping point

**Port status:** the decoder, strict image-case interface and complete tile
inventory are now implemented. The [input interface verification](image_input_interface.md)
records 123 passing tests and one real training-image CPU check. Point clouds,
models, geodesic reconstruction and training are not part of that port.

22 October update: the [cloud/model execution interface](reference_forward_interface.md)
is also implemented with unchanged retained operators. Its real whole-case
check remains unsupported because eight tiles contain singleton clouds. An
explicit neighborhood-boundary correction is proposed there, not applied;
no threshold, loss, learned checkpoint or training change was made.

The first port should contain only the verified decoder, explicit image-case
interface, deterministic tile inventory and provenance/count reporting. Use
independent format fixtures and analytic boundary/short-input checks. Then
integrate the retained model and geodesic/export operations under that interface,
with CPU execution checks before any substantial GPU command.

Success for that port is correct decoding/units, complete tile accounting,
reference-independent output decisions, reload consistency and valid declared
SWC/status output. It is not high reconstruction F1 with random weights.
Stop on a decoder disagreement, reference access during inference, unaccounted
tiles or unsupported memory/input behavior. Fix the demonstrated interface
defect within this scope; do not respond by adding a loss, changing the model
or extending a training budget.

Before the first fresh fit, review the label correction, preparation policy and
complete run recipe together. The project-level destination remains one matched
whole-neuron R/N comparison, with structural review and prospective advancement
criteria, as specified in the [revised proposal](revised_proposal.md).
