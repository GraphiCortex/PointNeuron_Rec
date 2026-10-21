# New commands

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
