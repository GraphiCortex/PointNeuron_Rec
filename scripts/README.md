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
records its command, rough-projection limits and current memory hold. It does
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
