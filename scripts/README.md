# New commands

`audit_reference_forward.py` runs a fresh, untrained PN1-style model on CPU
after explicit image-only cloud preparation. It accounts for every tile and
records unsupported singleton clouds rather than padding or skipping them.
It saves raw proposals, not SWCs or trained checkpoints. See the
[control and observed limitation](../docs/reference_forward_interface.md).

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe .\scripts\audit_reference_forward.py `
  --cases .\tmp\my_input_check\images.json `
  --tiling .\configs\input_tiling.example.json `
  --clouds .\configs\reference_cloud_control.example.json `
  --output-dir .\tmp\my_input_check\reference_forward
```

The image manifest is an input to prepare. This command executes all its tiles
with CPU inference and may take substantial time on larger images; it does not
launch training. The example settings are a reference control, not a universal
PN2 foreground recipe. Output directories must be new.

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
  --predictions .\tmp\comparison\R_outputs.json `
  --predictions .\tmp\comparison\N_outputs.json `
  --output .\tmp\comparison\report.json `
  --csv .\tmp\comparison\report.csv `
  --tolerances 2 6 --maximum-step 0.5 --check-convergence
```

`R_outputs.json` and `N_outputs.json` above are paths to prepare, not existing
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
