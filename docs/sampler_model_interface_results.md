# Real-box targets and PointNeXt-Transformer CPU integration

30 October 2026. **Execution passed on all eight prepared inputs. No training.**
This completes the CPU target/interface review promised in the
[two-sampler protocol](sampler_learnability_protocol.md). It does not choose a
sampler, establish learnability, or make the GPU fit launch-ready.

## What is now implemented

The new reference-only `pn2.supervision.curve_nodes` module constructs a candidate
node grid in actual rectangular voxel-cell bounds `[start-.5, stop-.5]`. It
retains the historical full-chain eight-voxel arclength phase, structural nodes
and actual crop intersections. It does not adopt historical optional endpoint
pruning, radius-based target selection or spatial merging. Reference radii are
interpolated as metadata, without an anatomical-accuracy claim.

Separate curve runs are retained when a branch leaves and reenters the box.
No target edge crosses a positive-length outside interval. Distinct reference
identities remain distinct even at coincident positions; zero-length structural
incidence is retained. The graph here audits the target representation, not GAT
predictions or a reconstructed SWC. This candidate schema is explicitly marked
`pn2-rectangular-curve-nodes-audit-v1`.

The CPU integration script imports the hash-pinned historical native encoder
and Transformer head read-only, applies the already parity-tested compact-index
interpolation correction, and initializes fresh seed-zero encoder / seed-101 head
weights. All eight 2048-slot clouds reach the full encoder and the 512-query head.
The input adapter is the one already verified; the head attends directly to the
aligned point descriptors. No extra pooling stage or new head is introduced.

## Measured target representation

| Training image / tile | Targets / query slots | Minimum target separation, voxels | Targets with another within 0.25 voxel | Maximum source-curve deviation from its corresponding target chord, voxels |
|---|---:|---:|---:|---:|
| Fruitfly 32 / 27 | 18 / 512 | 0.22550 | 2 | 1.4666 |
| Fruitfly 32 / 20 | 85 / 512 | 0.19027 | 2 | 3.2616 |
| Frog 22 / 2 | 7 / 512 | 1.50461 | 0 | 0.9821 |
| Frog 22 / 5 | 68 / 512 | 0.02168 | 4 | 1.8722 |

Counts are identical for both samplers because they share references and boxes.
None exceeds capacity; no target truncation is needed. The reference files for
these two images each have one encoded component; the previously audited
multi-component annotation cases are not involved.

The very close pairs all involve a regular grid sample and a required node on
the same branch: a junction, leaf or crop intersection. They are not evidence
that separate biological branches should be welded. The closest example is
frog tile 5: chain 530 -> 576 has a sample at arclength 72 and an actual crop
intersection at arclength 72.0216824921. A sample nearly coinciding with a boundary
is an artifact of cropping a fixed-phase grid, not a new structure in the image.
The 0.25-voxel count is a descriptive audit, not a suppression threshold.

Target nodes alone support only 53.3-55.6% of reference arclength within two
voxels, and 100% within six voxels. This is expected for a roughly eight-voxel
grid and explains why node proximity must not be reported as reconstructed
curve recall. Connecting targets using their *reference* edges gives measured
curve precision/recall 1.0 at both two/six voxels in these four boxes. That is a
target-geometry diagnostic with known edges, not a model result or topology proof.

The 3.2616 value above is a per-span maximum against its own corresponding
chord, bounded by checking all intervening source vertices. Global curve
proximity can still pass because another target segment is nearby and because
the score uses arclength quadrature. Do not interpret the global score as proof
that every branch is represented accurately. The stronger arclength-parametric
deviation bounds are 1.4685, 3.3690, 0.9827 and 1.9517 voxels, respectively.

## Why rectangular clipping matters

A same-policy control used a 128-wide cube around each actual box center. In
fruitfly tile 27 it generated 44 targets, **28 outside the actual input box**;
actual-domain clipping produces 18 targets. The counts do not differ simply by
28 because intersections themselves move with the domain. The other three boxes
had zero out-of-domain cube targets in this particular reference geometry.

This demonstrates a concrete label-domain mismatch that would arise from using
the old scalar-radius interface unchanged. It does not show that a historical
training run used these new boxes or that this explains its failures. No such
fit has been launched in fresh PN2. The repair is in the reference clipper, not
the image sampler or intended architecture.

## Measured model execution

All eight cases passed fresh CPU eval forward, train forward plus synthetic
backward, and eval after one BatchNorm-buffer update:

- 1,688,518 parameters; every parameter gradient present and finite;
- all model parameters bitwise unchanged, with no optimizer or checkpoint;
- centers `[1,512,3]`, logits `[1,512,2]`, radii `[1,512,1]`, decoder features
  `[1,512,128]`; finite outputs;
- exact same initial parameters and buffers reset before every input;
- the thresholded frog tile 2 control still contains only 195 distinct source
  voxels in 2048 repeated slots. The execution pass does not restore lost evidence.

Torch 2.12.1+cu130, CPU only, one thread. The eight model checks took 29.12 seconds
after imports/target audit. No GPU duration estimate follows from this.

Ordinary BatchNorm is unchanged. Maximum coordinate-component difference between
train output and eval output after just one untrained buffer update is about
0.384-0.422 voxel; maximum logit difference is about 0.0081-0.0084. These are
fresh-weight diagnostics, **not evidence that a learned normalization discrepancy
is solved**. The synthetic backward objective tests gradient connectivity, not
the proposed supervised loss. Radius gradients here do not validate radius labels.

## Verification, evidence and decision

Twelve new target tests plus the relevant existing geometry, cloud-support and
SWC-policy tests pass: **51 passed**. The target tests cover rectangular boundaries,
full-chain phase, collinear subdivision, branch reentry, coincident components,
zero-length/tiny structural spans, required roles, bend error and explicit
overflow. All four target packet hashes and eight prediction packet hashes were
checked; current script/target source hashes match the executed run. Initial
state hashes agree with the preceding retained execution check.

Executed, verified run: `tmp/sampler_model_interface_verified_20261030/`.
The earlier `tmp/sampler_model_interface_20261030/` is preserved. The verified
replay followed a diagnostic tensor-detach correction and a synthetic tiny-span
identity guard; it did not change the four real target counts or fit any data.
The visible [evidence summary](evidence/sampler_model_interface_20261030.json)
contains the complete target/model tables, source hashes and cube-domain control.

**The source-to-model interface executes.** The remaining pre-fit discussion is
about the declared target discretization, radius supervision, normalization and
one bounded learning recipe. Close targets are not proven to cause a fitting
failure, and this audit does not justify a new pruning rule or a further Phase 1
redesign. Any proposed spacing change must preserve required structural nodes
and be evaluated as a target-policy change, not credited to point sampling.

Keep the two samplers and their input packets fixed. The learned comparison must
still use the same final target/loss/inference recipe for both arms. A tiny fit
can establish learnability; a neuron-disjoint development comparison is required
before choosing a generally useful sampler. No training command is ready yet.
