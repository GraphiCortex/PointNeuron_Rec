# Compact interpolation indices: validated correction and 64k replay

29 October 2026. Correction implemented for the explicit execution diagnostic;
CPU validation passed. **GPU replay not launched.** No production model, training
fit, sampling change or normalization change is introduced.

The [original sweep review](pointnext_capacity_results.md) identified that native
three-neighbor interpolation saves a slice backed by a full sorted-index tensor.
The proposed correction is now implemented in
`scripts/pointnext_compact_interpolation.py`: clone only the selected index slice
before feature gathering. Stable sort, tie order, distance arithmetic, weights
and gradient paths are preserved. No parameters or loss terms are added.

For this diagnostic, the loader binds the corrected function to the historical
encoder module **in the new Python process only**. The historical operator file,
encoder file, checkpoints and original run outputs remain unchanged. This makes
reuse explicit without copying or promoting the old full pipeline into PN2.
Original commands retain their baseline interpolation binding. Both historical
files and the two correction/check scripts have explicit hash checks in
`configs/pointnext_compact_replay64.json`.

## Completed CPU evidence

Five operator fixtures passed exact neighbor, output and feature-gradient
comparisons: random clouds, tied distances with batch two, and one/two/three
available supports. Fixtures include incomplete final chunks.

In the 1,024-query/256-support fixture, saved index backing storage fell from
2,097,152 bytes to 24,576 bytes (2 MiB to 24 KiB). A whole 256-point encoder,
initialized identically in both variants, passed exact training-output,
all-parameter-gradient, state/BatchNorm-buffer and eval-output comparisons.
The baseline and corrected operator sources stayed hash-verified.

The corrected recipe also passed the standalone loader's 256-point CPU
forward/backward/operator check. Three subprocess-controller tests passed, and
the updated PowerShell launcher passed a syntax parse. No GPU execution was used
for any of these checks. These are execution/equivalence results, not neuron
reconstruction or generalization results.

Visible [CPU parity/storage evidence](evidence/pointnext_compact_cpu_20261029.json).
Original CPU outputs: `tmp/pointnext_compact_cpu_20261029/` and
`tmp/pointnext_compact_loader_cpu_20261029/`.

## Manual GPU command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_pointnext_capacity.ps1 -Action replay64
```

The command first repeats the CPU parity/storage check against the pinned recipe.
Failure prevents GPU launch. Then a fresh CUDA process attempts **65,536 points
only** on device zero (the previous run used the RTX 3070 Ti Laptop GPU).
All baseline settings remain fixed: seed-zero initial weights, same synthetic
unique grid positions/order/intensity, batch one, float32, fixed normalization,
four-stage hierarchy, radii/K and chunk size. Each mode has one warmup and two
measured iterations: eval forward, then training-mode forward/backward without
an optimizer. No learned checkpoint is loaded or retained.

GPU process cap: **300 seconds**, plus CPU preflight/process overhead. The prior
64k process lasted about 90 seconds before failing, so budget a few minutes; the
corrected runtime is unmeasured. No retry, automatic extension, mixed precision
or budget reduction. Fresh timestamped output under
`tmp/pointnext_capacity_replay64_*/`, with `cpu/report.json`, per-mode GPU files
and `gpu/report.json`. Existing output directories are rejected.

Success requires CPU exact parity/storage checks, both GPU modes completing
within the cap, finite `[1,65536,32]` descriptors and finite gradients for all
encoder parameters, with expected hierarchy shapes. Record allocated/reserved
memory and timings. The original failed 64k training mode saved no peak, so a
measured before/after peak reduction at 64k cannot be calculated from that run.
The predicted index-storage reduction is distinct from the new GPU measurement.

If another failure occurs, preserve its phase/trace and inspect the remaining
allocation issue before changing point counts or architectural settings. A pass
would establish 64k encoder execution for this backend, not whole-pipeline
capacity, learned feature usefulness or a new biological sampling budget.
