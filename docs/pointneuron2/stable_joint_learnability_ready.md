# One capped learnability trial at the demonstrated stable learning rate

20 October 2026. **Historical preparation protocol; the GPU fit is now complete.**
See the [completed results](stable_joint_learnability_results.md): stable placement,
two full crop passes, original combined gate failed. The preparation and launch
instructions below document the original contract; they are not a new rerun request.

At preparation, the controller was CPU-checked and GPU fitting was unlaunched.
The user chose this direction after reviewing the residual selection errors and
the earlier failed hard-negative objective. No new objective, head, architecture,
point-cloud preparation or inference threshold is introduced.

Launch the new fit visibly:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_stable_joint_learnability.ps1 -Action train
```

The completed CPU prerequisite does not need rerunning. This is a new controller,
not the old guarded resume command or the completed 64-update comparison.

## Why finish this test before changing the loss

The [short paired result](joint_partial_stability_results.md) demonstrates local
geometry stability at LR 0.00001 and one complete automatic training-crop pass.
It does not demonstrate four-crop learnability. The stable setting received just
16 optimizer exposures per crop; recorded classification loss decreases on all
four between its first and last all-crop probes. That is not convergence evidence,
but it also does not establish that the current objective cannot learn the task.

High-confidence extras and false graph edges are real output errors, not reasons
to declare a new loss successful before testing it. The
[previous 50/25/25 hard-negative variant](hard_negative_query_results.md) already
amplified hard-negative confidence gradients and regressed badly. Its failure
remains rejected. The earlier frozen-classifier fit also failed, on different
fixed features and a different optimization recipe; neither result decides the
outcome of a stable joint fit on the present state.

I therefore revised the recommendation to change selection/ranking objectives
immediately: first test whether the existing formulation can meet its declared
reconstruction criterion under the now-demonstrated stable setting. This trial
answers one unfinished question, rather than adding a new loss and another
uncontrolled explanation. It is explicitly approved prospectively. The short
comparison remains formally **unqualified**; its precision tolerance is not
relaxed or reinterpreted as a pass.

## Fixed scope and budget

| Item | Setting |
|---|---|
| TRAINING crops | Neuron/crop 0/0, 0/1, 92/4, 86/3 |
| Input clouds / targets | 848/6, 2,048/12, 1,219/16, 2,048/418 |
| Phase 1 / native model | Frozen image-derived clouds; existing PointNeXt/Transformer/GATv2 prototypes |
| Initial state | Original saved joint stage 1,000, with model, Adam moments and RNG |
| Initial Adam clocks | Proposal 5,000; GAT 1,000 |
| LR / weight decay / clipping | 0.00001 / 0.0001 / global norm ten |
| Batch / order | One crop; fixed four-crop cycle, no augmentation or scheduler |
| Query/candidate budgets | All 512 queries; existing symmetrized 16NN union radius eight |
| Placement/selection policy | Two-voxel edge trust; original inclusive 0.5 node/edge decisions |
| Maximum new updates | **3,000**, ending at historical joint stage **4,000** |

The original parent checkpoint is imported, not the terminal nonresumable short
control. The first 64 new updates repeat that reduced-LR trajectory, so the
64-update diagnostic does not get silently counted as extra learnability budget.
This trial still consumes additional physical GPU work relative to all past
diagnostics; its own counted trajectory ends at 4,000 rather than 4,000 new
updates after the parent.

The partial-supervision objective is exactly the one used in both short arms:
mean loss over available known edge classes, no relabeling unknowns, and
proposal-only updates with GAT weights/moments/counters unchanged when no edges
are known. GAT remains an edge scorer. Proposal confidence still owns final
node selection; it is not redefined as biological foreground.

Device: **NVIDIA GeForce RTX 3070 Ti Laptop GPU**. Plan roughly an hour or longer
depending on load and I/O. This sustained controller has not been GPU-timed;
the preceding 128-update paired job took 6.65 minutes including frequent probes
and exports, and does not give a clean update-only extrapolation. The new run
prints actual elapsed time and progress. Runtime does not change the cap.

## Reproduction, checkpoints and monitoring

The first 64 incoming predictions/losses, gradients and optimizer counters must
match the preserved reduced-LR control. At 64, saved automatic predictions,
candidate/selection decisions and the **complete model, Adam and RNG state**
must reproduce its terminal checkpoint. Failure stops before update 65. This is
an integrity check, not a reconstruction success criterion.

Probe all four crops without gradients every four updates through that replay,
then every 20 new updates. Existing observation checks reject any change to
weights, buffers or RNG from probing. Keep per-update incoming geometry/labels
as well. The later probe cadence reduces overhead; it is an explicit monitoring
change, not a change to training inputs or loss.

Full-state checkpoints precede fallible graph/SWC evaluation at historical
stages **1,000, 1,064, 1,500, 2,000, 3,000 and 4,000**. Their schema records
separate proposal/GAT clocks, next crop, replay proof, source/backend identity,
configuration, Adam options/moments, model and RNG. No source checkpoint is
edited. Terminal disk reload must reproduce state and predictions exactly.

An explicit resume action can recover an intermediate checkpoint from **this**
fit after an infrastructure interruption. It preserves the 3,000-new-update
cap, filters saved diagnostic history to the checkpoint step, and uses a new
numbered output directory. A milestone is evaluated again before advancing if
its previous metric export failed. A pre-proof stage-1,064 checkpoint reruns
the exact replay proof. Terminal, CPU-check-only and geometry-stop checkpoints
cannot resume. A deterministic policy/model failure requires interpretation,
not an automatic retry.

## Success, failure interpretation and stopping

At the terminal historical stage 4,000, require **all four automatic crop gates**:
all assigned targets accepted with zero extras; exact reference adjacency and
junction incidence/placement; correct crop-forest component counts; valid SWC;
and continuous arclength precision and recall within two voxels each at least
99%. The forest may legitimately have multiple components. Reference labels
are used for supervision/evaluation, never for selecting inference nodes.

Also require the existing 90%-trusted stability diagnostic on every crop over
the final eight all-crop probes and final eight incoming occurrences per crop.
This prospective additional safeguard prevents calling a single favorable
terminal snapshot stable. It does not weaken the original reconstruction gate.
Execution completion, automatic gates and this stability check are reported
separately; `learnability_gate` requires the cap to complete and both checks.

Stop early if any crop's mean matched distance exceeds four voxels at three
consecutive probes. Save a nonresumable geometry-stop state before exporting
its automatic output. Nonfinite states, unexpected gradients, altered sources
or replay mismatch are execution failures. No automatic extension, rollback,
threshold search, LR grid or objective change follows a stop.

If terminal geometry is stable but selection/topology still fails, that is
stronger bounded evidence for revisiting their learning objective/interface.
If geometry becomes persistently unstable, the short control did not establish
long-run stability. Neither outcome proves mathematical impossibility or a
capacity limit. A pass establishes four-crop **training-set learnability**, not
whole-neuron reconstruction, development generalization, matched PN 1.0
superiority or product readiness. Do not select an intermediate pass as the
terminal result or silently promote any component to a default.

## Verified preparation and output locations

**55 targeted tests pass.** They cover exact agreement with short-controller
update arithmetic, separate inactive GAT clocks, disk resume with mixed clocks
and RNG, caps/source/options checks, checkpoint survival after failed metrics,
and model/optimizer/RNG replay proof. The real-model CPU check performs zero
updates, verifies probe noninterference, four baseline SWCs, and exact state/RNG/
prediction disk reload without CUDA initialization. It does not claim that the
new full GPU fit or GPU resume has executed.

Default ignored root: `tmp/pointneuron2/local_runs/stable_joint_learnability_20261020`.
Preparation is in `check/`; training will create `model/` with checkpoints,
reports, probes, incoming diagnostics, predictions and SWCs. Console logs are
also ignored. No training output folder exists at preparation. Existing action
folders are never overwritten.

Visible [recipe](../../configs/pointneuron2/stable_joint_learnability.json),
[controller](../../scripts/pointneuron2/stable_joint_learnability.py),
[launcher](../../scripts/pointneuron2/run_stable_joint_learnability.ps1),
[tests](../../tests/test_stable_joint_learnability.py) and
[validation/provenance](stable_joint_learnability_ready_seed0/validation.json).
