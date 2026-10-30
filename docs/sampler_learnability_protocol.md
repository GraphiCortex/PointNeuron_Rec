# Phase 1 through PointNeXt and Transformer: bounded comparison

30 October 2026. The user agreed to a controlled learned comparison after the
[matched input evidence](pn1_input_comparison_results.md). **Input preparation,
CPU verification and the bounded learned runner are implemented. GPU timing is
the next manual step; the full fit awaits its results.**
No model is trained by this document or the preparation command.

The [real-box target audit and CPU model interface check](sampler_model_interface_results.md)
now pass on all eight inputs. The reference-only rectangular clipper and
untrained diagnostic wrapper are implemented. The subsequent user-approved
[node-only fit recipe](sampler_node_fit.md) fixes the audited target grid,
excludes radius supervision and scores ordinary saved-BN inference. The
CPU diagnostic itself did not select those policies.

## Decision this experiment must inform

Does the current Sato-selected cloud help the intended encoder/proposal model
recover reference skeleton nodes, compared with an intensity-gated control at
the same requested input budget? This is a component experiment within
image -> points -> PointNeXt -> Transformer -> GATv2 -> reconstruction. GATv2,
cross-crop merging and final SWC are outside this first comparison; their absence
here is not a replacement product architecture.

The two arms and the input adapter are fixed below. We first check training-set
learnability, then consider a separate neuron-disjoint development comparison.
Passing the small task cannot establish generalization or finish Phase 1.

## Fixed first task and controls

Reuse the four already exposed TRAINING boxes, without selecting new boxes based
on the latest scores. Both models see the same four reference regions. The
boxes were originally chosen using reference arclength; this makes them suitable
for a diagnostic fit, not a representative automatic inference benchmark.

| Training image / tile | Actual XYZ box size | Requested slots per arm | Threshold control distinct points | Sato distinct points |
|---|---|---:|---:|---:|
| Fruitfly 32 / 27 | 128 x 45 x 27 | 2048 | 2048 | 2048 |
| Fruitfly 32 / 20 | 128 x 128 x 27 | 2048 | 2048 | 2048 |
| Frog 22 / 2 | 128 x 128 x 81 | 2048 | 195 | 2048 |
| Frog 22 / 5 | 128 x 128 x 81 | 2048 | 2048 | 2048 |

- **Threshold control:** `pn1_range_random` from the completed comparison:
  verified decoder, threshold `round(0.2 * verified whole-source maximum)`, strict
  greater-than gate, then the actual fixed-count random helper. Random is chosen
  as the documented historical control, not because it wins every region. The
  spatial variant remains a known alternative; do not choose between them using
  future learned scores in this experiment.
- **Candidate:** `pn2_sato`, the existing image-only Sato sampler with its existing
  10% uniform probability-mass mixture. No new guide, threshold or mixture.
- Cloud seed zero, existing frozen 2048-slot packets, same four boxes and order.
  Larger existing budgets and the other cloud seeds remain available, but are
  not additional automatic fits.

The threshold helper already sampled with replacement in frog tile 2. Preserve
those exact slots for this control; do not add new locations, jitter coordinates,
change the threshold or drop the box. Report 195 distinct source observations and
1853 duplicate slots. This is equal tensor-slot cost, **not equal distinct evidence**.
Any effect includes repetition weighting as well as selection. It is not a pure
comparison of descriptor quality given equal unique points. Short/empty-cloud
behavior for a future production model remains a separate interface obligation.

The primary experiment uses 2048 because the existing comparison established
both benefits and counterexamples at that budget, and the small fit is cheaper.
It is not a proposed product-wide budget. A failure at 2048 cannot rule out the
dense sampler that performed better in the frog measurements. A later dense
comparison must be declared separately rather than silently increasing the cap.

## Input interface now prepared and checked

`configs/sampler_learnability_inputs.json` pins the existing raw comparison report
and training inventory. `scripts/prepare_sampler_comparison.py` opens the two
original images with the verified reader, checks source/decoded hashes, confirms
every selected source ID and raw intensity, and saves eight small model-input
packets. It does not open SWCs, rerun selection/filtering, import Torch or change
the original packets. Generated files are under `tmp/`.

Use the audited adapter:

- native XYZ voxel centers, with actual clipped-box center `(start+stop-1)/2`;
- local XYZ = `(native XYZ - actual center)/64`, without axis stretching/padding;
- whole-source intensity `(I-min)/(max-min)`, constant image -> zero;
- float32 attributes `[local_x, local_y, local_z, normalized_intensity]`;
- fixed source-ID hash ordering, independent of sampling race, preserving repeats;
- retain native XYZ, typed raw intensity, flat source IDs, box metadata and hashes.

This addresses input ordering before grouping; it does not replace first-K ball
query with KNN or change the audited PointNeXt radii. Preparation round-trips the
saved arrays exactly and checks inverse coordinate recovery. Inputs contain no
reference targets or reference-derived features.

Executed preparation: `tmp/sampler_learnability_inputs_20261030/inputs.json`.
An independent readback verified all eight packet hashes, unchanged source-ID /
intensity multisets, coordinate inversion, attribute alignment and the four
complete arm pairs. The visible [input evidence manifest](evidence/sampler_learnability_inputs_20261030.json)
preserves the packet/source hashes and actual counts. This verifies preparation,
not model execution or learnability.

## Model and supervision review before the fit is executable

Reuse audited source as an explicit starting asset, with **fresh parameters**:
native PointNeXt-S with the parity-tested compact interpolation indices, followed
by the existing Transformer proposal head. The head directly attends to aligned
point descriptors; it does not require an extra pooling/model stage. Historical
checkpoints and optimizer states are not adopted.

The following items must be resolved together in the visible recipe and CPU
checks before publishing a training command:

1. **Targets in the actual rectangular domain.** Preserve endpoints, junctions
   and true box intersections, plus one stable arclength discretization. Audit
   the existing eight-voxel canonical convention before reusing it. The old
   crop helper takes a scalar cube radius: using radius 64 around a clipped
   27-plane box would incorrectly supervise outside the image domain. Adapt
   clipping explicitly, check subdivision stability, counts, close/coincident
   identities and target-polyline approximation error. Never truncate targets
   to fit queries or remove inconvenient nodes after a failed fit.
2. **Proposal capacity and meaning.** Initially review the existing 512-query
   head; confirm all four target counts fit. The score means "used slot under
   the supplied reference representation", not certified biological foreground.
   Unmatched slots are reference-agreement no-object labels, with incompleteness
   disclosed. A reference-empty crop is unknown, not a background-only example.
3. **Geometry and radius labels.** Check coordinate inversion and evaluate the
   target discretization against the unchanged reference curve before learning.
   Radius values have uncertain anatomical provenance; decide explicitly whether
   they are auxiliary reference-agreement targets or excluded from this node task.
   Do not let an unreviewed radius convention silently determine matching.
4. **Normalization and inference.** Declare the exact train/eval behavior. The
   capacity probe used ordinary BatchNorm; it did not solve the historical
   train/eval discrepancy. Inspect the same learned parameters under the declared
   inference policy. A training-mode diagnostic cannot substitute for its score.
   Any alternative normalization policy needs discussion before implementation.
5. **One paired learning recipe.** Freeze loss terms and their normalization,
   optimizer/LR, clipping, crop contributions, initialization, evaluations and
   checkpoint policy for both arms. Do not inherit the historical LR/weights
   merely because they exist. Use identical initial model tensors and crop order;
   record optimizer/RNG state so a resume can be exact.

These are prerequisites for a coherent learned comparison, not a search over new
heads, constraints or hyperparameters. Implementation must expose this review
point rather than presenting an incomplete command as launch-ready.

## Success criteria and stopping rules to freeze with that recipe

Proposed engineering gate for review: under the declared inference policy and a
fixed 0.5 used-slot cutoff, at least 95% one-to-one node precision **and** recall
within two native voxels in **each** box. Report exact counts as well as rates,
all-query geometry separately from accepted-query geometry, and endpoint/junction
support. This is a tiny-set sanity criterion, not a biological adequacy standard;
perfect 512-slot labeling is not required to advance.

Also retain complete PR/score diagnostics without selecting a new cutoff from
these results. Report curve-to-accepted-node coverage and gaps at two/six voxels
alongside the same measures for the target nodes themselves: an eight-voxel
target grid cannot be assumed to give complete two-voxel curve coverage. Node
proposals alone do not define reconstructed edges or a reconstructed curve.

Before the user launches a fit, fix one update cap per arm and report timing
from a separate throwaway GPU check on the actual model/input/loss. Provide
expected duration, RTX 3070 Ti device, output paths and visible PowerShell commands.
No duration is inferred from the encoder-only timing. No training command is
published until target/loss/evaluation checks pass.

- Nonfinite values, OOM, target overflow or inconsistent coordinates stop the
  run explicitly. Preserve the failed case, logs and latest complete state.
- No automatic retry, extension, threshold adjustment, target pruning or
  checkpoint selection by the most favorable box. Compare the fixed terminal
  checkpoints; intermediate scores explain learning, not new experiment arms.
- If neither arm learns, investigate the recorded geometry, label, optimization
  and train/eval evidence before a broader fit. This is not yet a verdict on
  sampling or the intended architecture.
- If one learns and the other fails, report the asymmetry. Do not force the
  control to pass or assume a training fit predicts a development advantage.
- If both learn, stop the tiny task. Do not spend more updates perfecting it.

## Separate development decision

Only after the sanity task supports proceeding, prepare a prospectively fixed
broader training comparison on the existing training partition and evaluate the
seven development neurons with the same automatic tiling, target and inference
policy. Freeze the manifest, training exposure, budgets, resources, quantitative
advancement margins and handling of unknown labels before those learned results.
Do not treat these four boxes as development data or reuse them as the evidence
for selecting a general sampler. Preserve all reserved identities.

Report paired per-neuron node/reference-agreement results, macro precision/recall,
worst cases and weak-branch findings. Use image-derived width/contrast strata
without choosing a method by neuron identity or reference scores. The six current
apparent-thin/dim windows are insufficient for a general thin/dim claim.

Interpretation: a repeatable learned benefit warrants a Phase 1 policy decision;
equivalent quality favors the simpler/cheaper reliable preparation; mixed
source-specific benefits leave the general policy unresolved. A fresh hybrid is
a new hypothesis to discuss, not the automatic response. Even a development
node gain still requires later GAT/consolidation/whole-neuron reconstruction
validation before claiming a better PN2 reconstruction system.
