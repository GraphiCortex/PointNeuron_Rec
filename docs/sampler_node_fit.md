# Two-sampler node learnability fit: recipe and manual launch

30 October 2026. The user agreed to proceed with the node-only comparison and
the reviewed target grid. **Preparation and CPU verification complete; GPU
timing and training have not been launched.** This fixes the outstanding recipe
choices in the [comparison protocol](sampler_learnability_protocol.md).

## Question and scope

First ask whether the intended fresh PointNeXt/Transformer can fit the declared
nodes from each of two frozen preparations. This is a training-set engineering
check. It is not a final selection of a universal sampler. A sufficiently
flexible model can memorize four clouds, including a cloud that omits local
branch evidence. Success therefore needs later neuron-disjoint development
evidence; failure under one recipe does not prove the architecture cannot work.

The main architecture remains image -> point cloud -> PointNeXt -> Transformer
-> consolidation -> GATv2 -> graph/SWC. This fit covers the encoder/proposal
portion only. No learned GAT or final SWC claim is made.

## Fixed data and model

| Training box, in update order | Target nodes | Input slots per method | Threshold-control distinct voxels | Sato distinct voxels |
|---|---:|---:|---:|---:|
| Fruitfly 32 / 27 | 18 | 2048 | 2048 | 2048 |
| Fruitfly 32 / 20 | 85 | 2048 | 2048 | 2048 |
| Frog 22 / 2 | 7 | 2048 | 195 | 2048 |
| Frog 22 / 5 | 68 | 2048 | 2048 | 2048 |

Two fresh independent models: `pn1_range_random` and `pn2_sato`. The first is
the explicitly counterfactual, correctly decoded source-range control from the
input comparison, not a PN1 model rebuild or a PN2 product default. Preserve
replacement draws in frog tile 2; duplicate slots are not new observations.

Use the eight prepared cloud packets and four audited rectangular target packets,
unchanged and hash-checked. The unpruned eight-voxel target grid retains the close
regular/required pairs documented in the [target audit](sampler_model_interface_results.md).
No target pruning, new image filtering, new data, budget expansion or guide
channel is introduced. SWCs enter targets/evaluation only. Forward consumes
local XYZ, XYZI attributes and the box coordinate transform, not target counts,
target geometry, neuron identity or reference edges. The 512 queries are fixed.

The audited native PointNeXt-S prototype, compact-index interpolation correction
and three-layer 128-channel Transformer head initialize exactly as in the CPU
interface check: encoder seed 0, head seed 101; 1,688,518 parameters. Source is
reused read-only with hashes, not declared an official benchmark reproduction.
No historical weights or optimizer state transfer into either arm.

## Learning objective and resources

Versioned recipe: `configs/sampler_node_fit.json`.

| Setting | Fixed value |
|---|---|
| Per-arm allowance | 2000 updates; each box appears 500 times |
| Batch / numerical type | One box / float32 |
| Optimizer | Fresh AdamW, LR 0.0001, betas (0.9,0.999), epsilon 1e-8 |
| Weight decay | Zero for this memorization/sanity task |
| Gradient clipping | Global norm 10 |
| Dropout / augmentation / scheduler | None |
| Evaluation updates | 0, 8, 500, 1000, 2000 |
| Reported operating cutoff | Used-slot score >=0.5 |
| Per-arm process wall cap | Timing 300 seconds; fit 3600 seconds |

Hungarian matching uses `3 * distance_in_voxels / 6 - P(used)`. The loss is
`2 * classification + 3 * center`, where classification is the mean of the
positive-query mean CE and unused-query mean CE, and center is mean SmoothL1
of matched Euclidean distances, with beta four voxels. This preserves the
reviewed balanced-query position/classification convention and avoids dominance
by the many unused slots. No radius enters matching or loss. An empty reference
abstains rather than producing background labels; all four declared task boxes
must remain supervised, so an unexpected empty task stops the run.

These weights and LR are one transparent starting recipe, not proven optimum.
They are held fixed across methods. The smaller 2000-update allowance is a
bounded readiness test on 7-85 nodes per box, not an assertion of convergence.
There is no automatic extension or search if it fails. Weight decay/augmentation
are omitted to avoid regularizing against the very memorization this step tests;
they will need a separate broader-training specification.

The radius output remains in the unchanged head, but is unsupervised. Its
dedicated output row receives zero direct loss gradient; shared features can
still change its predictions. Those numbers must not be used as trained radii
or exported as evidence of radius estimation quality.

## Inference, evaluation and normalization diagnosis

**Primary:** ordinary `.eval()` inference using saved BatchNorm running statistics.
The same policy is scored for both methods. It is retained explicitly as the
reviewed prototype policy, not assumed to have solved the historical mismatch.

**Diagnostic:** the same parameters using current-crop training-mode statistics,
with no gradients. All buffers and original mode are restored after every
diagnostic; evaluation checks that neither model state nor RNG state changed.
The diagnostic cannot make the main gate pass or replace the ordinary result.
Dropout is zero, so a train/eval difference is interpretable without dropout noise.

The recipe uses deterministic algorithms, TF32 disabled and math scaled-dot-product
attention, with the optional MHA fast path disabled. These are explicit execution
choices to support comparable numerical replay, not new learned constraints.
GPU timing must check their actual CUDA execution; CPU verification cannot
guarantee every CUDA backward operator is supported. An unsupported operation
stops timing before any full fit. No kernel/backend fallback is automatic.

Accepted proposal precision/recall uses **maximum-cardinality one-to-one geometric
matching** within two and six native voxels. This differs from reusing training
assignments to decide which accepted nodes count: one accepted proposal cannot
recover two nearby targets, and duplicate accepted proposals remain extras.
Structural-role counts are geometric diagnostics under a particular matching;
close-node assignments may be ambiguous and do not establish connectivity.

The main per-box engineering gate is precision >=95% and recall >=95%, at two
voxels and the fixed 0.5 score cutoff. All four boxes must pass for an arm's
terminal learnability gate. Exact labels on every unused query are not required.
The checkpoint at update 2000 supplies the verdict; intermediate checkpoints are
learning diagnostics, not candidates from which to pick a favorable winner.

Also save all query positions/logits/scores and assignments, the all-query node
recall, fixed cutoff diagnostics (0.1/0.25/0.5/0.75/0.9), and curve-to-accepted-node
coverage/gaps. The complete saved predictions allow later PR inspection without
retuning this run's gate. Curve-to-node coverage is compared with the target-node
oracle from the audit; eight-voxel node spacing does not give full two-voxel curve
coverage even with perfect nodes. There are no predicted edges here.

Scores measure agreement with the supplied reference discretization. Incomplete
annotations are not certified biological background, and the used-slot score is
not a calibrated foreground probability. No threshold is chosen from these four
boxes to fix their last errors.

## Verification already executed

Twelve learned-component tests passed in the retained Torch runtime, with
third-party pytest plugin auto-loading disabled for that invocation only. They
check analytic balanced loss, target-order/translation behavior, zero-distance
backward, unknown-crop abstention, radius exclusion, non-greedy one-to-one
matching, duplicate/extras handling, fixed cutoffs, curve weighting, BatchNorm
buffer restoration even on exceptions, and exact next Adam/dropout update after
restoring model/optimizer/Python/NumPy/Torch RNG state on a synthetic fixture.
No real neuron was optimized by those tests.

Five controller tests and twelve target tests passed in the isolated CPU runtime:
missing/incomplete reports stop the paired run, timeouts kill only the owned
worker, logs survive, timing estimates aggregate, and training requires timing.
PowerShell launcher parsing passed. Total focused checks: **29 tests passed**.

The actual node loss was checked on all eight prepared inputs using fresh
parameters and backward **without an optimizer**. Every parameter gradient was
finite; confidence/position rows received gradients; the radius row received
zero direct loss gradient; parameters remained bitwise unchanged. Final CPU
preflight is `tmp/sampler_node_cpu_verified_20261030/report.json` and the
[visible evidence summary](evidence/sampler_node_cpu_20261030.json) records the
recipe/code/source identity. No GPU or PN2 fit was launched by the agent.

## Launch the short timing check first

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_sampler_node_fit.ps1 -Action timing
```

It runs two fresh GPU processes on the local RTX 3070 Ti Laptop GPU, one per
sampler. Each starts from the same audited parameters, evaluates the four boxes,
performs eight throwaway optimizer updates (two cycles), verifies exact in-memory
model/Adam/RNG serialization reload, and evaluates again. No trained checkpoint
is retained. A fresh timestamped directory under `PN2/tmp/sampler_node_timing_*`
contains the recipe, console logs, update journal, raw predictions and reports.

The eight-update cap is 300 seconds per worker, so the hard paired worker bound
is ten minutes plus controller/preflight overhead. This is a bound, not a runtime
estimate. GPU timing will report update-time estimates from the second four-box
cycle for the 2000-update fits. Checkpoint/evaluation overhead and laptop/system
variability remain additional costs. Peak timing memory includes the reload probe.
The displayed range is a 0.8-1.2 times heuristic, not a statistical confidence
interval. A false node-learnability gate at update eight is expected to remain
possible: timing passes on completed finite updates and exact state reload, not
on having learned the four targets in eight updates.

The already-passed CPU preflight is validated before GPU execution. If source or
recipe changes, it becomes stale. A fresh CPU check can be written to a different
directory with `-Action check -Output ...`, then supplied via `-CpuPreflight ...\report.json`.
Existing result directories are never overwritten; the completed CPU check does
not need repeating for the current source.

## Full fit, only after reviewing timing

The launcher implements `-Action train -TimingRun <passed timing directory>`.
This is the later action; **run timing now and review its output first**. The fit
rejects absent, incomplete or stale timing. Its first eight updates must match
the passed timing's model, Adam and RNG hashes exactly before proceeding.

At every evaluation update, the fit first saves a new immutable checkpoint with
model, optimizer, RNG, current mode, recipe and source identities. Completion
markers contain file hashes. No repeatedly replaced `active_update.json` is used;
the update journal is append-only. A terminal checkpoint survives a later
evaluation failure. The first launcher has no resume action, but preserves the
state needed to prepare a deliberate exact continuation if separately agreed.

Execution failure/timeout/nonfinite state stops the control and preserves logs,
checkpoints and the unattempted second arm's status. A completed arm with a
**failed learnability gate** does not prevent the other planned arm from running.
`status=completed` means execution reached its cap; the separate gate field
reports learnability. No result is silently relabeled successful.

## Interpret results before changing direction

- Both pass: close the tiny task. Do not perfect its last scores; prepare the
  agreed broader training/development comparison before choosing a sampler.
- Only one passes: a recipe-specific learnability asymmetry, not generalization
  or proof of information loss. Inspect geometry, confidence and normalization.
- Both fail: inspect those same saved components before broader training. Do not
  automatically change Phase 1, add a loss, extend the cap or lower the cutoff.
- Diagnostic passes while ordinary inference fails: the declared inference
  policy is implicated; training-mode performance cannot qualify the main run.

The seven development neurons and all reserved cases remain unopened by this
preparation. No training-set fit establishes a PN2 reconstruction improvement,
whole-neuron performance, external-domain reliability or product readiness.
