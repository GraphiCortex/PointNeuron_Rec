# User-launched four-box recovery — 4 November 2026

The user requested a local command. The assistant prepares code/checks and
does not launch this timed GPU run. No architecture, targets, losses, optimizer,
cutoff, source points or original terminal gates change.

Restore the existing verified update-1000 model, all Adam moments/clocks, and
Python/NumPy/CPU/CUDA RNG state. Continue original updates 1001–2000, preserving
the original four-box order. This replays unsaved work; extra executed compute
is reported separately from the final model's 2000-update history.

Before any optimizer update, require exact replay of all four saved prediction
arrays. Compare every available replayed optimizer update with the original
persisted loss/gradient trace; stop on a mismatch. No silent fresh start,
optimizer reset, device substitution or automatic repair. Save new checkpoints
at 1250/1500/1750/2000, then evaluate the unchanged four-box gate.

Expected duration: approximately 10–15 minutes based on the measured active
update/evaluation costs, not a guarantee. Hard recovery wall limit: 1800 seconds,
checked using wall and monotonic clocks by the tested separate guard. A job
discovered after its deadline is budget-limited. It cannot automatically advance.
The actual host-suspension path has not been retested; clock-jump unit fixtures
cover the guard logic. Source checkpoint and the interrupted run remain intact.

Run in PowerShell:

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe' -B '.\scripts\resume_local_curve_four.py'
```

Device: the same RTX 3070 Ti Laptop GPU. The launcher checks its identity and
the existing source/data/checkpoint hashes. It prints a new timestamped output
folder under `tmp/local_curve_head_20261104/recovery4_dense_*` and a corresponding
`*_wall_guard` log folder. At completion inspect the printed `report.json` plus
the guard report. A completed execution is not an accuracy pass; inspect `gate`.
There is no automatic sixteen-box fit, even when the four-box gate passes.

Optional read-only preflight, using the same Python executable:

```powershell
& '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe' -B '.\scripts\resume_local_curve_four.py' --check
```

The preflight loads the checkpoint on CPU, verifies complete finite model/Adam
state, cursor, original optimizer settings and RNG metadata, and validates
frozen data/recipe bindings. It creates no CUDA context and performs no update.
Exact GPU prediction/optimizer replay is checked inside the user-launched run.
The original scientific driver and recipe remain byte-identical; recovery
execution sources are separately bound in `configs/local_curve_recovery.json`.
