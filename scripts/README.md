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
is needed. The agent has not launched the full fit.
