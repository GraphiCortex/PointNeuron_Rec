# PN2

This is the home for all fresh PN2 development, in its own Git worktree on
branch `pointneuron2`. It is a sibling of the historical `pointneuron2` folder.
The original PN1 checkout remains on `main`.

Start with the [revised proposal](docs/revised_proposal.md) and
[evidence audit](docs/research_reset_evidence.md).

## Layout

| Folder | Purpose |
|---|---|
| `src/` | New implementation, introduced through reviewed changes |
| `scripts/` | New commands and experiment entry points |
| `configs/` | New benchmark manifests and reviewed run settings |
| `tests/` | Checks for the new implementation and evaluator |
| `docs/` | Active proposal, design decisions and readable results |
| `tmp/` | Ignored generated caches, runs, checkpoints and logs |

All new work goes here. The sibling historical worktree's sources, reports and scripts are
retained reference material; reusing them will be explicit. No old experiment
or checkpoint is adopted automatically.

## Active direction: original raw images to PN2

Scope clarified 22 October 2026: PN1 informs conceptual design. Rebuilding,
training or fixing a PN1 reference is not a prerequisite for the product.

```text
Original raw image -> automatic PN2 point preparation -> PointNeXt
-> Transformer proposals -> consolidation -> GATv2 -> graph/SWC
```

Start with the [PN2 Phase 1 design draft and original-image views](docs/pn2_phase1_design.md).
The verified decoder, coordinate/tiling interface, source inventories and common
SWC evaluator remain useful infrastructure. Two original training files have now
been inspected directly, without segmentation, reference content or model input
selection. No PN2 foreground method, guide or sampler is promoted by those views.

The PN1-only cloud/model detour has been removed from active source. Reports
and pushed commit `48519ff` preserve it. Its DGCNN singleton issue is not the
next PN2 task. No PN1 cutoff, weights or point budget become PN2 defaults.

The first bounded [Phase 1 sampling protocol](docs/phase1_sampling_protocol.md)
compares uniform, intensity-weighted and Sato-weighted source points. The user
agreed to the three controls. The CPU probe is implemented and tested, without
promoting a guide to the product. Keep execution, representation retention,
training learnability and genuine reconstruction evidence distinct.

[Initial results](docs/phase1_geometry_rescore_results.md): the frozen two-image
rescore passes the comparative input-support gate under an explicit policy
preserving unknown reference type metadata. Structural weighting improves
support over intensity weighting in both contexts; lean clouds still have
large local gaps. No model was trained or product sampling policy promoted.
The [original strict-reader result](docs/phase1_sampling_results.md) remains
preserved as an incomplete comparison.

The [four-context qualification](docs/phase1_qualification_plan.md) is now
[complete](docs/phase1_qualification_results.md), with all 120 clouds scored
and unchanged settings. Its memory preflight led to an explicit
[disk-backed guide](docs/phase1_disk_guide.md): both original guide files and all
20 ordered Sato selections replay exactly. Resident memory is substantially
smaller in these observations, with extra runtime and disk costs; whole-volume
decoding/normalization and end-to-end hardware requirements remain separate.
The disk backend completed generation with a 0.939-GiB process peak working set
and 2.22-GiB peak commit/pagefile counter. Generation and reference scoring took
about 4 hours 22 minutes; these are research-comparison costs, not model
inference requirements. The comparison did not meet all preset criteria:
Sato helps strongly on frog, but offers mixed incremental benefit over
intensity weighting across the four inputs. Local losses and sparse-image
budget use require discussion before promoting a Phase 1 policy. No model fit
followed the comparison.

The next [input-retention revision](docs/phase1_input_retention_draft.md) is a
discussion draft. It separates exact source representation from the finite
PointNeXt view, proposes explicit context and conditional detail selection, and
defines a bounded accounting check. No replacement sampler or model adapter
has been implemented from that draft.
Its [six-source accounting check](docs/input_retention_accounting_results.md)
is complete: exact repeated-value encoding helps sparse inputs, but only
13/135 nonconstant boxes fit the old dense quota before context. The main
finite-cloud/encoder-capacity decision remains. No replacement sampler or
training follows this result automatically.

The [input/PointNeXt contract audit](docs/pointnext_input_contract.md) now separates
that accounting from model capacity. A pinned historical encoder has passed a
small CPU execution check; a visible untrained GPU budget sweep is prepared.
The [branch-evidence protocol](docs/branch_retention_protocol.md) specifies the
next retention and learned-feature questions. The user-launched GPU sweep is now
[reviewed](docs/pointnext_capacity_results.md): forward/backward completed through
32k, and 64k inference completed before training-forward OOM. A CPU audit confirms
that three-neighbor index views retain full-sort backing storage in this native
backend. The original model and run are preserved; no learned fit has begun.

A [compact-index correction](docs/pointnext_compact_replay64.md) is now explicitly
implemented for the standalone diagnostic and passes exact CPU parity/storage
checks. The user-launched 64k-only GPU replay now passes both modes: inference
averages 13.55 seconds and forward/backward 14.96 seconds, with 0.58/0.86 GiB peak
allocated CUDA memory. This closes the bounded encoder execution check through
64k; it does not establish branch preservation or learned reconstruction quality.

The [first real-cloud evidence audit](docs/real_branch_evidence_results.md) is now
complete: four fixed training boxes, all three frozen selectors/five seeds, and
actual seed-zero neighborhood traces. Sato retains substantially more local
contrast in the frog boxes; intensity retains more in the fruitfly boxes.
Finite grouping restricts one route of access, while fine-level skips remain.
Only six windows fall jointly in the apparent-thin/dim stratum, so broad thin/dim
adequacy is not established. No sampler policy, model change or learned fit
follows automatically.

The user-requested [PN1 versus PN2 input comparison](docs/pn1_input_comparison_results.md)
is complete: seven preparation arms, matched 2048/dense slot budgets and five
seeds on the same four boxes. Reader repairs, source-range controls and duplicate
counts are separated. PN2 Sato preserves evidence excluded by the hard gate in
frog regions, but thresholded PN1-style controls beat it in both fruitfly boxes
at 2048 slots. Phase 1 is not established as a universal input improvement.
This is a component comparison; no PN1 model rebuild or PN2 training occurred.

The next [controlled learned comparison](docs/sampler_learnability_protocol.md)
is now specified. Eight frozen inputs (two selectors, four training boxes) have
been verified and packed into the audited PointNeXt interface on CPU. The short
thresholded pool remains visible as 195 distinct voxels in 2048 sampled slots.
The model/target recipe still needs explicit rectangular clipping, normalization
and supervision checks; no learned fit or GPU command is ready yet.

The [CPU target/model interface check](docs/sampler_model_interface_results.md)
is now complete: all eight inputs pass full PointNeXt/Transformer forward and
backward execution with fresh weights and no optimizer. Actual rectangular
targets fit 512 queries (18/85/7/68 nodes). The fixed eight-voxel grid also creates
some nearly coincident regular/required nodes; this is a target-policy finding,
not a Phase 1 failure or justification for silently adding suppression rules.
The bounded fit recipe remains to be discussed before training.

The user-approved [node-learning recipe and manual timing command](docs/sampler_node_fit.md)
are now prepared: two fresh models, 2000 updates per sampler, fixed clouds and
targets, position/used-slot loss without radius supervision, and ordinary
saved-BatchNorm inference as the scored policy. Current-crop statistics are
diagnostic only. Twenty-nine focused tests and eight real-input CPU loss/backward
checks pass; no optimizer was applied to real data. GPU timing is the next manual
action. Training requires a passed current-source paired timing run and exact
eight-update model/Adam/RNG replay; no GPU run has been launched by the agent.

The [user-launched paired GPU timing](docs/sampler_node_timing_results.md) now
passes on both methods, including exact model/Adam/RNG serialization reload.
Measured second-cycle update times are 1.025/1.057 seconds and peak allocated
memory is 0.233 GiB in this small component experiment. Estimated paired fit
update time is 55.5-83.3 minutes plus overhead. Initial node accuracy is still
unestablished; no sampler is selected. The unchanged bounded fit is ready for
the user's manual launch using the reviewed timing directory.

Both [user-launched 2000-update node fits](docs/sampler_node_fit_results.md)
now complete, but both fail learnability substantially. Saved-state inspection
confirms full optimizer/BatchNorm updates. Identical-parameter diagnostics show
a large normalization gap; current-crop statistics improve geometric placement
but leave missing nodes and many accepted extras. Neither a threshold change
nor adopting the diagnostic mode alone can satisfy the frozen gate. The read-only
review exports saved-proposal figures and preserves all failures; no further
training, normalizer/query change or Phase 1 policy selection has been made.

The user-approved [normalization-only GN8 revision](docs/groupnorm_node_control.md)
is now implemented separately. All sixteen encoder BatchNorm layers become
eight-group GroupNorm, retaining exactly the same initial parameter values and
parameter count. Queries, clouds, targets, loss and update cap are unchanged.
CPU train/eval predictions are bitwise identical on all four real Sato clouds
and synthetic 256/4096-point grids; supervised gradients are finite without an
optimizer update. Twenty-six focused tests pass. The next action is a manual
one-worker GN GPU timing check; no GPU or fresh learned fit has been launched.

The [user-launched GN8 GPU timing](docs/groupnorm_node_timing_results.md) now
passes: all eight updates complete, train/eval centers/logits/scores have zero
differences at steps 0 and 8, and model/Adam/RNG reload is exact. Second-cycle
update time is 0.948 seconds; estimated one-fit update time is 25.3-37.9 minutes
plus overhead. This closes the tested consistency check, not node learning.
The user subsequently completed the unchanged 2000-update GN fit.

The [completed GN8 fit review](docs/groupnorm_node_fit_results.md) confirms exact
train/eval consistency at all five evaluation steps, but the frozen node-learning
gate fails in all four boxes. Ordinary-inference geometry improves relative to
the saved BN arm; confidence still accepts many extras, and fruitfly proposal
geometry also misses targets. No normalization sweep, threshold change or fit
extension follows. A [prospective geometric-query design](docs/geometric_query_proposal.md)
specifies image-derived starting locations/features and learned offsets within
the retained Transformer stage. Its CPU coordinate-only check is disclosed;
the changed model and any new fit remain unimplemented pending discussion.

A separate read-only [training-reference audit](docs/training_reference_contract_audit.md)
finds three of 33 SWCs with disconnected paths and exact coordinate overlaps
across components. Their graph-label interpretation needs verification; the
audit makes no repairs and does not explain all historical fitting failures.

## Working directory and runtime

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
```

The Phase 1 CPU probe uses an isolated environment at
`tmp\phase1_cpu_env\Scripts\python.exe`, with its pinned installation and commands
in [scripts/README.md](scripts/README.md). The historical runtime is unchanged.
Original images, PN1 assets and the historical evidence archive remain outside
this fresh folder. New generated outputs belong in this folder's ignored `tmp/`.

The parent collaboration instructions still apply: discuss substantive changes,
use visible commands for agreed GPU runs, and distinguish execution from
reconstruction quality. No training or inference is launched by this setup.

To evaluate saved reconstructions, use the [evaluation command](scripts/README.md).
The command reads explicit development/synthetic SWC paths and never runs a
model. The base package uses NumPy and SciPy. Raw inspection uses the existing
Matplotlib runtime. No model library is imported by image/evaluation commands.
