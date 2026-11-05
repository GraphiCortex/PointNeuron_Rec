# Stage2 CPU launch commands

**6 November status: these actions were run and the comparison STOPPED. Do not
launch again or continue this recipe.** C0/C1 pooled CV length1.254998x/1.740456x
both fail the unchanged1.25 gate. Read the
[completed results](stage2_trees_results_20261106.md). Commands below are preserved
for provenance, not a new launch instruction. No automatic extension is authorized.

Implementation is ready for local launch, not a measured selector result.
The [readiness receipt](evidence/stage2_trees_readiness_20261105.json) records
48 passing synthetic tests and the successful metadata preflight.
Read the [fixed protocol](stage2_trees_protocol_20261105.md) and
[bound config](../configs/stage2_trees_20261105.json).
The isolated CPU environment is already installed on this machine. Older CPU
and Torch environments were preserved. For rebuilding it or repeating the
synthetic checks only:

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\scripts\setup_stage2_cpu.ps1'
```

Run these **separately, in order, after each preceding action completes**.
None starts the next command automatically.

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\run_stage2_trees.py' --action prepare
```

CPU only;611 image-only packets. Re-decodes original image bytes for raw
intensity profiles, reuses Sato guides and all candidates; no filtering or fit.
Runtime unmeasured; allow tens of minutes, with a **two-hour hard cap**.
Output `tmp\stage2_trees_20261105\prepare\manifest.json` and `report.md`.

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\run_stage2_trees.py' --action policy
```

CPU only;fit handcrafted node tree on33 training neurons, freeze representatives,
then report the151-box development diagnostic. It selects nothing. Runtime
unmeasured; allow tens of minutes, with a **one-hour hard cap**.
Output `tmp\stage2_trees_20261105\policy\report.json` and `report.md`.

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\run_stage2_trees.py' --action compare
```

CPU only;33 conditional selector leave-neuron-out folds for C0/C1, then same
folds for PointNeXt edges under the selected contract. Handcrafted node selection
stays fixed between feature arms. Freeze both final edge models, report training
fit without gating, score handcrafted development once, and only if it passes,
score PointNeXt development once. Stop on any required failure.
Runtime unmeasured; **expect hours, up to a six-hour hard cap**. A conservative
first-fold throughput projection stops early if that cap appears insufficient.
This is not a promise the full CV fits the cap. No automatic extension follows.
Output `comparison\C0_C1_training_CV.{json,md}`, `feature_CV.{json,md}` and
`final\*_development_report.{json,md}` under `tmp\stage2_trees_20261105`.
The guard console logs/receipts live in sibling `stage2_trees_20261105_*_guard_*`
directories. Keep the laptop awake; wall caps include sleep.

All commands use at most four compute threads and disable CUDA. There is no
Torch inference, GPU training, GATv2 or automatic next stage. If STOP.json or a
partial stage appears, preserve it and share the report/console output; don't
delete outputs and retry, alter thresholds or extend the budget. Each action is
write-once and bound to the frozen source, inputs, versions and protocol.

Development segment floors59.33/74.14%,strict wins>=5/7 at both tolerances,
resampled floors52.43/59.72% and actual classical+5pp,pooled length<=1.25x.
PointNeXt must additionally gain>=2pp at2 while meeting all6-voxel floors.
Development is repeatedly inspected; a per-box pass would not certify final
whole-image reconstruction, biological joins or radius quality.
