# New commands

## Phase 1 CPU sampling probe

Read the [fixed comparison protocol](../docs/phase1_sampling_protocol.md) first.
By default, `probe_phase1_sampling.py` reads the two original training images, generating
three count-matched controls at two densities and five seeds. It computes one
whole-image guide, then samples disjoint image-defined tiles. It never reads a
reference. `evaluate_phase1_sampling.py` subsequently reads the saved clouds and
the two explicit training SWCs; it verifies source coordinates/intensities and
reports spatial support, local gaps and reference disagreement, not reconstruction
quality. No training or automatic extension follows a failed gate.

The isolated CPU runtime is under ignored `tmp/`; the historical environment is
unchanged. To recreate it in a new installation (network access required):

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -m venv .\tmp\phase1_cpu_env
& .\tmp\phase1_cpu_env\Scripts\python.exe -m pip install --only-binary=:all: -r .\configs\phase1_cpu_lock.txt
```

From the PN2 root, run each command separately and inspect its result:

```powershell
& .\scripts\run_phase1_sampling.ps1 -Action check
& .\scripts\run_phase1_sampling.ps1 -Action generate
& .\scripts\run_phase1_sampling.ps1 -Action evaluate
```

Default output: `tmp/phase1_sampling_20261022`. Generation requires a fresh run
directory; evaluation requires a fresh `evaluation` subdirectory. Console logs
are timestamped under `tmp/phase1_console`. Use `-OutputDirectory tmp\a_new_run`
only for an explicitly justified replay, not to select favorable draws.

The user-authorized geometry-only rescore has a separate
[reference policy](../docs/reference_type_policy.md). It preserves signed type
metadata rather than interpreting or changing it, keeps all structural checks
and reads the **existing frozen clouds** for both images:

```powershell
& .\scripts\run_phase1_sampling.ps1 -Action evaluate-geometry
```

It writes a fresh `evaluation_opaque_integer` directory, including new overlays,
and preserves the strict report and its artifacts. The common SWC evaluator and
standard predicted-output validation keep their nonnegative default.

The [first results](../docs/phase1_sampling_results.md) preserve an incomplete
two-case comparison: one scored reference, one nonstandard type-code issue.
Do not rerun preparation to resolve a reference policy. To reproduce the
one-scorable-case figure from its saved report:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe .\scripts\plot_phase1_sampling.py `
  --report .\tmp\phase1_sampling_20261022\evaluation\report.json `
  --output .\tmp\phase1_sampling_20261022\comparison_figure.png
```

## Other commands

`audit_input_retention.py` performs the bounded image-only accounting specified
in the [input-retention draft](../docs/phase1_input_retention_draft.md). It reads
six explicit existing training sources and frozen image provenance, counts
modal-value exceptions in each existing box, estimates decoded payload sizes,
and verifies exact in-memory round trips. It creates no clouds, guides or model
inputs and opens no SWCs. The [completed results and exact command](../docs/input_retention_accounting_results.md)
separate storage arithmetic from encoder capacity and reconstruction quality.

`audit_training_references.py` inventories only explicitly declared training
SWCs, preserving source bytes and unknown type metadata. It reports encoded
components, junctions, cross-component exact-coordinate coincidences and
reported radii; it does not repair graphs or construct labels. The
[33-reference audit](../docs/training_reference_contract_audit.md) records
the command, checks and limits. It never reads images or development/test
references.

The [disk-backed guide](../docs/phase1_disk_guide.md) is an explicit alternate
execution backend for the same operator. `verify_disk_guide.py` compares it
against saved guides and ordered selected voxel IDs on the two original cases,
without reference access. `run_phase1_sampling.ps1` now accepts `-CasesManifest`,
`-GuideBackend disk` and `-BlockSide` (default 64); the historical defaults still
use the original two cases and whole-volume backend. The implementation document
records parity evidence, resource limits and fresh-directory commands for the
already fixed four-case comparison. No training follows these commands.

`preflight_phase1_resources.py` reads explicit image headers and completed-run
metadata to estimate resource costs without decoding images, reading references
or launching generation. The [four-context qualification plan](../docs/phase1_qualification_plan.md)
records its command and rough-projection limits. The subsequent
[complete qualification](../docs/phase1_qualification_results.md) records actual
disk-backend costs and the unchanged failed comparative gate. The preflight does
not change the filter or sampler.

`inspect_raw_images.py` starts with explicit original image files. It writes
linear/log-display projections, one native source slice, raw histograms and
byte/range provenance. No mask, reference, sampling or model is used.

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe .\scripts\inspect_raw_images.py `
  --cases .\configs\phase1_raw_review.json `
  --output-dir .\tmp\my_raw_review
```

The case list contains two explicit training raw images, chosen by decoded size
within uint8/uint16 storage for initial inspection. They are not a representative
benchmark. Output directories must be new. See the
[Phase 1 design and inspected views](../docs/pn2_phase1_design.md).

`audit_image_inputs.py` decodes strict image-only cases and inventories every
spatial tile on CPU. It reads no reference, chooses no foreground threshold and
runs no model. See its [schema, command and verification](../docs/image_input_interface.md).
Output directories must be fresh and generated outputs belong under `tmp/`.

`evaluate_reconstructions.py` compares saved SWCs using explicit benchmark and
model-output JSON manifests. It never launches training or inference. The first
prediction manifest is the paired baseline. Each declared case remains visible,
including missing/invalid/empty outputs; reference-positive failures score zero
F1 in the macro average.

From the PN2 root, after creating manifests for actual saved outputs:

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe .\scripts\evaluate_reconstructions.py `
  --benchmark .\configs\development_benchmark.json `
  --predictions .\tmp\comparison\PN2_control_outputs.json `
  --predictions .\tmp\comparison\PN2_candidate_outputs.json `
  --output .\tmp\comparison\report.json `
  --csv .\tmp\comparison\report.csv `
  --tolerances 2 6 --maximum-step 0.5 --check-convergence
```

The two prediction manifests above are paths to prepare, not existing
model results. Reports require fresh paths and are never overwritten.
The half-step convergence check reports numerical differences; the requested
step remains primary and there is no automatic convergence pass gate.

Only development/synthetic partitions are accepted at this stage. The explicit
seven-case inventory is development data. Distances use each case's declared
units; the defaults 2/6 are historical voxel settings, not universal physical
tolerances. Structural support does not establish topology accuracy.

See the [manifest and scoring contract](../docs/benchmark_contract.md) and
[verification record](../docs/evaluator_verification.md). No new training
command exists yet. Generated artifacts belong in `PN2/tmp/`.

The PN1-only `audit_reference_forward.py` has been removed from active source.
Its [historical report](../docs/reference_forward_interface.md) and pushed commit
`48519ff` preserve the detour; no PN1 rebuild is scheduled.

## Untrained PointNeXt capacity check, 23 October 2026

See [input/encoder contract](../docs/pointnext_input_contract.md). The standalone
diagnostic explicitly reads the hash-pinned historical native encoder; fresh
weights only, no checkpoint. The existing Torch runtime is used for Torch-only
work; image-filter checks continue to use the isolated Phase 1 runtime.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_pointnext_capacity.ps1 -Action check
& .\scripts\run_pointnext_capacity.ps1 -Action sweep
```

`check` is a small CPU execution/operator check. `sweep` is a separate manual GPU
command: synthetic unique voxel points, N=8192/16384/32768/65536, batch 1, FP32,
one warmup and two measured iterations per forward or forward/backward mode,
no optimizer. A fresh process per budget has a 300-second cap. Stop on first
timeout/OOM/failure; no automatic retry or budget change. At most 20 minutes
plus overhead. Outputs default to a new timestamped ignored `tmp` directory;
`-Output` can name another **new** directory. No SWC or real-image path is read.
Reports cover this encoder backend only, not Transformer/GAT/full inference or
thin/dim branch quality. The user-launched 29 October sweep is now
[reviewed](../docs/pointnext_capacity_results.md). Its 64k training-forward memory
failure led to a CPU-only saved-index-storage audit, with the unchanged historical
operator. Reproduction command and interpretation are in that report. No encoder
correction or automatic GPU retry is included in the audit.

The [compact-index correction and CPU validation](../docs/pointnext_compact_replay64.md)
are now complete. To replay only the previously failing 64k budget:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_pointnext_capacity.ps1 -Action replay64
```

This repeats the CPU exact-equivalence/storage gate before starting a fresh
five-minute-capped GPU process. The only encoder computation change is compact
index storage; the recipe, point count and model settings are fixed. It is not
a training fit. Outputs are new timestamped ignored `tmp` directories. Original
`check`/`sweep` actions retain the historical implementation for reproducibility.

The user-launched corrected 64k replay is now complete and passes both modes;
[results and memory figures](../docs/pointnext_compact_replay64.md) are saved.
No repeat is needed for this execution milestone. Real-cloud retention and
learned reconstruction are separate next questions.

## Frozen real-cloud evidence audit

The [four-box plan](../docs/real_branch_evidence_plan.md) and
[completed results](../docs/real_branch_evidence_results.md) cover two explicit
training sources. Image/reference/cloud hashes, exact counts and point attributes
are checked. No guide, sampling, GPU model or training is rerun.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\audit_real_branch_evidence.py --output .\tmp\a_new_real_evidence_run
```

The completed run is `tmp/real_branch_evidence_20261029`. Repetition is not needed
for the current milestone. It scored 60 input arms and traced twelve seed-zero
hierarchies in about six minutes on CPU. The underlying source-ID point clouds
are unchanged; trace arrays and generated reports belong under ignored `tmp`.

`validate_real_branch_counts.py <report.json>` independently reconstructs contrast
fractions from saved point attributes and adds `point_counts.json` (new file only).
`plot_real_branch_evidence.py <report.json>` renders the saved report to a new
`summary.png`; it does not rescore or decode images. The figure, visible report
and all-radius evidence summary are linked from the results document.

## PN1 versus PN2 voxel-input components

The user specifically requested the [matched comparison](../docs/pn1_input_comparison_results.md).
It uses the [fixed plan](../docs/pn1_input_comparison_plan.md) and configuration,
not the earlier closed PN1 model-integration detour. Hash-pinned PN1 decoder and
pure sampler functions are read from the original `main` checkout without edits.
No model or GPU is imported or run.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\compare_pn1_input_evidence.py --output .\tmp\a_new_pn1_input_comparison
```

The completed run is `tmp/pn1_input_comparison_20261029`; no repeat is needed.
It contains 280 fixed records, explicit empty pools and repeated PN1 slot draws,
scored once per unique source voxel. Frozen PN2 rankings supply nested lower-budget
views; no guide is regenerated. `plot_pn1_input_comparison.py <report.json>`
renders saved results only, writing a new `comparison_2048.png`. Existing output
directories/files are preserved. The runtime and original/main baseline layout
are local dependencies; no PN1 training or full reconstruction is performed.

## Inputs for the controlled learned comparison

The [protocol](../docs/sampler_learnability_protocol.md) fixes two samplers and
four training boxes. Input preparation is complete at
`tmp/sampler_learnability_inputs_20261030/inputs.json`: eight verified packets,
2048 slots each, with 195 distinct observations in the thresholded frog tile 2
control. The CPU command verifies source image/packet hashes and attributes,
applies the audited coordinate/intensity adapter and changes only point order.
It opens no reference and imports no model. It does not generate training targets.

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B `
  .\scripts\prepare_sampler_comparison.py --output .\tmp\a_new_sampler_input_preparation
```

The completed preparation does not need repeating. A new directory is required
for a repeat. Target clipping, normalization, supervision and the final paired
training recipe must pass the protocol's review before a GPU command is ready.

## CPU target and model integration check

`check_sampler_model_interface.py` uses the prepared packets, two explicit
training references and hash-pinned historical encoder/head sources. It audits
the eight-voxel grid in actual rectangular bounds, then checks eval/train
forward and synthetic backward on CPU with fresh weights. It has no optimizer,
training action or GPU path. See the [completed results](../docs/sampler_model_interface_results.md).

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B `
  .\scripts\check_sampler_model_interface.py --output .\tmp\a_new_sampler_interface_check
```

The verified run is `tmp/sampler_model_interface_verified_20261030`; no repeat is
needed. The target graph has reference edges solely for auditing representation;
it is not an oracle-assisted inference path or GAT output. The candidate target
module is under `src/pn2/supervision/`, separate from image-only preparation.

## Bounded two-sampler node fit: run timing first

The [fixed recipe, verification and stopping rules](../docs/sampler_node_fit.md)
are prepared. The actual position/used-slot loss passed backward on all eight
inputs without optimizer updates. Twenty-nine focused tests pass across the
retained Torch runtime and isolated CPU runtime. The
[user-launched paired GPU timing now passes](../docs/sampler_node_timing_results.md);
the agent has not launched GPU work. That review records the exact capped-fit command.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_sampler_node_fit.ps1 -Action timing
```

This performs eight throwaway GPU updates per sampler, in separate fresh
processes, then verifies full model/Adam/RNG serialization reload in memory.
It saves logs/predictions/reports under a fresh ignored `tmp/sampler_node_timing_*`
directory and retains no trained checkpoint. It does not launch the 2000-update
per-arm fit. Review the paired timing report before using the separate train
action, which requires that passed directory through `-TimingRun`.

The completed CPU preflight is `tmp/sampler_node_cpu_verified_20261030/report.json`.
If a future source change invalidates it, `-Action check -Output <fresh directory>`
runs the CPU loss/gradient check; the resulting report can be supplied explicitly
with `-CpuPreflight <report.json>`. No repeat is needed for current source.

## Review completed saved node fits without training or inference

Both user-launched fits completed and failed the frozen learnability gate.
[Results, geometric/normalization separation and figures](../docs/sampler_node_fit_results.md)
are saved. `review_sampler_node_fit.py` reads the two declared training references,
saved predictions and hash-checked state files on CPU. It executes no model
forward, performs no optimizer update and changes no inference policy or cutoff.
Independent dense one-to-one assignment reproduces all 80 saved mode/box/step
count comparisons. The checkpoint audit confirms all sixteen BN counters and
Adam reach 2000. Continuous-curve proximity is a separate diagnostic, not a
replacement acceptance metric.

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B `
  .\scripts\review_sampler_node_fit.py --run .\tmp\sampler_node_train_20261030 `
  --output .\tmp\a_fresh_saved_fit_review
```

The completed review is `tmp/sampler_node_review_verified_20261030` and does not
need repeating. Two terminal figures are exported under `docs/learning/`. The
earlier review and all training checkpoints/logs remain unchanged. This outcome
does not schedule a retry, extension or another sampler fit.

## Normalization-only GN8 control

The [separate GN8 control](../docs/groupnorm_node_control.md) retains the fixed
Sato clouds, targets, Transformer and node-learning recipe. CPU mode equality
and supervised gradient checks pass. Baseline scripts/configuration remain
unchanged; `groupnorm_node_experiment.py` reuses the verified common helpers and
adds an explicit GN factory and consistency checks.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_groupnorm_node_fit.ps1 -Action timing
```

One worker runs eight throwaway GPU updates, checks exact train/eval consistency
and full model/Adam/RNG reload, retaining no trained checkpoint. It does not run
the later 2000-update fit, which requires its own passed GN timing. The completed
CPU preflight is `tmp/groupnorm_node_cpu_20261030/report.json`; there is no reason
to rerun it for current source. Generated outputs remain under `tmp`.

The [passed GN GPU timing review](../docs/groupnorm_node_timing_results.md)
records zero mode differences, measured cost and the exact next training command
using `tmp/groupnorm_node_timing_20261031_141828`. No timing repeat or source change
is needed. The user has now completed the full fit; [consistency passes but node
learnability fails](../docs/groupnorm_node_fit_results.md). Do not rerun the fit
from these preserved launch instructions.

## Review the completed GN fit

`review_groupnorm_node_fit.py` reads saved predictions and CPU-loaded state,
recomputes all 40 node-count comparisons and verifies train/eval equality and
checkpoint hashes. It exports the BN/GN terminal figure and computes a
prospective FPS512 starting-location diagnostic on the frozen input XYZ.
References score the locations after selection; no model forward, optimizer,
new inference policy or changed query model is executed.

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B `
  .\scripts\review_groupnorm_node_fit.py --run .\tmp\groupnorm_node_train_20261031 `
  --output .\tmp\a_fresh_gn_fit_review
```

The completed review is `tmp/groupnorm_node_review_final_20261031`; no replay is
needed. Read the [results](../docs/groupnorm_node_fit_results.md) and
[geometric-query design rationale](../docs/geometric_query_proposal.md). The
next section records the subsequently approved implementation and timing action.

## Approved geometric-query candidate

The [implementation and CPU results](../docs/geometric_query_implementation.md)
record the new query content/offset mechanism. Baseline modules and parameter
values, GN8, frozen clouds, node loss, optimizer, cutoff and fit cap are unchanged.
`geometric_query_experiment.py` reuses the existing data/update/evaluation/state
helpers without replacing or monkey-patching the GN runner. Its completed CPU
check is `tmp/geometric_query_cpu_20261031/report.json` and needs no rerun.

Preserved timing command (now completed by the user):

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_geometric_query_fit.ps1 -Action timing
```

This runs eight throwaway GPU updates on the same four training boxes, with
exact train/eval consistency and full model/Adam/RNG reload checks, a 300-second
worker cap, and no retained trained checkpoint. It writes a fresh
`tmp/geometric_query_timing_<timestamp>` directory and does not start training.
The separate train action requires this candidate's own passed `-TimingRun`;
review its timing report before a fit. No old GN timing is compatible.

The [geometric timing review](../docs/geometric_query_timing_results.md) confirms
the current run passes. No timing repeat is needed. Next manual command:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_geometric_query_fit.ps1 -Action train `
  -TimingRun ".\tmp\geometric_query_timing_20261031_155707" `
  -Output ".\tmp\geometric_query_train_20261031"
```

The train action dispatches to `geometric_query_train_controller.py`, which
validates the unchanged worker/recipe/CPU/timing identity and invokes that same
worker with an external 90-minute wall allowance. Measured update time is about
42-62 minutes before evaluation/IO. The original 60-minute parent allowance
could stop the run prematurely. This changes only the parent wall limit:
2000 updates, exact timing-prefix replay, model/optimizer settings and gates
remain fixed. `launch.json` and the final report record effective launch policy
and controller hashes. Controller tests use mocked launches; no GPU fit was
started by the agent.

## Completed geometric-query fit: read-only review

The user completed the bounded fit and [all four training boxes pass](../docs/geometric_query_fit_results.md).
No fit repeat or extension is needed. `review_geometric_query_fit.py` independently
recomputes saved node matches, verifies mode equality, checkpoint hashes and the
CUDA prefix, and plots terminal accepted nodes against the saved GN baseline.
It executes no model forward or optimizer and reads no new data.

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B `
  .\scripts\review_geometric_query_fit.py --run .\tmp\geometric_query_train_20261031 `
  --output .\tmp\a_fresh_geometric_fit_review
```

The audit is already complete at `tmp/geometric_query_review_20261031` and its
portable evidence/figure are under `docs`. The successful training sanity check
does not schedule a new run or qualify reconstruction on unseen neurons.

## Broader node-study data and exposure planning

The [cohort preflight](../docs/node_cohort_preflight_results.md) and
[proposed protocol](../docs/node_generalization_protocol.md) are complete for
review. The audit reads only 40 explicit image headers and SWCs, under a
300-second cap; it neither decodes images nor opens reserved source files.
The planner reads metadata and creates a deterministic 9200-update schedule,
with 20 exposures per supervised training box and no development gradients.

Preserved reproduction commands, from the PN2 root, with fresh output folders:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe -B .\scripts\audit_node_cohort.py `
  --output .\tmp\a_fresh_node_cohort_preflight
& .\tmp\phase1_cpu_env\Scripts\python.exe -B .\scripts\plan_node_generalization.py `
  --output .\tmp\a_fresh_node_exposure_plan
```

Both have already completed and need no repeat. Original outputs are
`tmp/node_cohort_preflight_20261031` and
`tmp/node_generalization_plan_cache_stage_20261031`. These are planning utilities,
not image-cache or training commands. No bulk Sato generation or broader GPU
fit has been implemented/launched by this step.

## Image-only point cache

The [cache builder and verification report](../docs/node_cloud_cache_builder.md)
describe the completed synthetic check and two-image real replay. All four
original Sato model inputs match exactly, with all 48 image boxes prepared.
No replay repeat is needed. Generated cache: `tmp/node_cloud_cache_20261031`.

Completed single-image timing command, retained for reference; no repeat needed:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_cloud_cache.ps1 -Action timing
```

Neuron 41 / Image5, about 11.9 million voxels, 25 boxes, disk-backed Sato,
one-hour wall cap. The completed result is retained in the input cache. Logs
go to `tmp/node_cloud_cache_timing_<timestamp>`. No targets, references,
model, GPU, bulk-preparation continuation or training are involved.

The completed `-Action check` report is
`tmp/node_cloud_cache_cpu_20261031/report.json`; the completed `-Action replay`
report is `tmp/node_cloud_cache_replay_20261031/report.json`. These entry points
refuse stale policies/existing cases and do not automatically retry failures.

## Remaining image preparation after exact-output block comparison

The [completed comparison](../docs/node_cloud_cache_timing_results.md) found
byte-identical guides and identical point arrays with block128, using 29.3%
less observed filter time on Image5. The cache now contains 3/40 images.
The completed read-only bulk plan schedules 33 fresh filters and four verified
guide copies, roughly 30.4 filter hours plus overhead. No bulk run is launched.

Preparation command used (now complete at 40/40; do not repeat):

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_cloud_cache_bulk.ps1 -Action prepare `
  -PlanRun ".\tmp\node_cloud_cache_bulk_plan_20261031" `
  -Output ".\tmp\node_cloud_cache_bulk_prepare_20261031"
```

The launcher uses the isolated Phase 1 CPU runtime. One worker runs at a time;
verified completed cases are skipped and partial cases/failures stop without
automatic retry or deletion. There is a 72-hour outer wall cap and volume-scaled
per-case allowances. Outputs remain in the shared cache and the named run folder.
No references, targets, GPU, training or automatic downstream handoff is involved.

Completion update, 2 November: [all 40 images and 2379 boxes passed independent
review](../docs/node_cloud_cache_completion.md). The shared cache and full guides
are retained; no further preparation command is needed. Larger-budget input
views, target/loader/metric checks and full-model timing are subsequent work.

## Two-budget complete-model resource timing

The [paired-study specification](../docs/node_budget_comparison.md) records
the proposed 2048/8192 comparison and provisional compute allowance. Four-box
input preparation and complete-model CPU checks have already passed. The
subsequent [GPU timing also passes](../docs/node_budget_timing_results.md).
Command used, retained as an execution record; no repeat is needed:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_timing.ps1 -Action timing
```

Eight throwaway updates per budget, same fresh model/optimizer settings and
18/85/7/68 targets. Each fresh GPU worker allows 600 seconds, with no retry or
retained trained checkpoint. The command measures updates, inference, memory,
mode consistency and exact reload; it cannot launch training. Outputs use
`tmp/node_budget_timing_<timestamp>` and the retained Torch runtime.

## Broad paired data and integrated resource timing

The [full paired-data build and CPU integration](../docs/node_budget_data_integration.md)
are complete; no preparation repeat is needed. The subsequent integrated GPU
timing passed. Command used, retained as an execution record; do not repeat:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_integration_timing.ps1
```

Two fresh budgets, eight throwaway training updates each, then all 415
development boxes inferred/scored with no gradients. Uses the real loader,
shared targets/schedule, core metrics and budget/data/cursor-bound state reload.
Rough forward-only estimate: 15 minutes for both development passes; actual
integration overhead is being measured. Per-worker stop limits: 15/30 minutes.
Outputs: `tmp/node_budget_integration_timing_<timestamp>`. No fit or retry action.

The CPU data build is retained at `tmp/node_budget_data_build_20261102`, its
verification at `tmp/node_budget_data_verify_20261102`, and real-model CPU proof
at `tmp/node_budget_data_integration_cpu_20261102`. Neither changes the original
2048 cache or reruns Sato. Reference-empty boxes stay unknown. Thin/dim profile
diagnostics and the full fit/resume launcher are subsequent work.

## Ready bounded paired node fits

The [integrated timing, profile cache and fit launcher](../docs/node_budget_fit_ready.md)
are verified. The user has now completed this command; retain it for historical
reproducibility rather than rerunning it. [Both accuracy gates failed](../docs/node_budget_fit_results.md).

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_fit.ps1 -Action train
```

Sequential fresh 2048/8192 arms, each 20 epochs/9200 updates across 460
supervised boxes of 33 training neurons. No development gradients; all seven
development images scored at epochs 0/5/10/20; terminal training scoring covers
every image box. Unchanged loss/model/cutoff, frozen training-only profile
labels and fixed terminal verdict. Planning allowance: 10–14 hours combined;
worker stop limits 6/12 hours. Full epoch checkpoints, strict manual recovery,
no automatic retry/extension. Output: `tmp/node_budget_fit_train_<timestamp>`.
No source filtering or GAT/radius/full-SWC training is involved.

See the report for recovery examples and interpretation limits. The image and
profile preparation are already complete; the command starts the actual
broader paired training study. It was launched by the user and is complete.

## Consistent node objective: GPU timing only

The [candidate objective and CPU evidence](../docs/consistent_node_objective.md)
are separate from the completed study. Next manual execution check:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_consistent_node_objective.ps1 -Action timing
```

Two independently forked copies of the terminal 2048 model/Adam/RNG state,
baseline versus candidate objective. Four deterministic training-only count
strata, eight throwaway updates each plus an exact next-update reload probe.
Same clouds, target packets, model, optimizer recipe and cutoff. Current passed
CPU evidence is required before GPU initialization. Separate worker processes,
600-second cap each, no retained trained checkpoint and no full-fit action.
Outputs belong to `tmp/consistent_objective_timing_<timestamp>`. This checks
execution, not learnability or reconstruction improvement.

## Five-epoch objective comparison: ready manual fit

The user passed objective GPU timing and completed the
[bounded paired continuation](../docs/consistent_node_fit_ready.md). Its
[accuracy screening failed](../docs/consistent_node_fit_results.md). The command
below is retained for reproducibility, not a request to rerun it:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_consistent_node_fit.ps1 -Action train
```

Both arms fork the same terminal 2048 model **and saved Adam/RNG** at historical
update 9200. Each receives five additional epochs / 2300 updates over the same
460 supervised training boxes. Only the objective differs. Terminal scoring
covers all 415 development and 1964 training boxes. Same clouds, targets, model,
LR, clipping and cutoff; no reserved data, radius/GAT training or source filters.

Planning allowance is 3–4 hours for both arms, provisionally; a separate
four-hour worker limit includes evaluation. Immutable complete-state checkpoints
every epoch, no automatic retry/extension or best-dev selection. Outputs:
`tmp/consistent_node_fit_train_<timestamp>`. The report documents explicit
method-bound recovery into a fresh directory and interpretation limits.
