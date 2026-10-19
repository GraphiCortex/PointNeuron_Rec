# Joint fit interruption: Windows diagnostic-file replacement

20 October 2026. The user-launched joint fit stopped after **148 completed
updates**, while attempting update 149. Its first eight updates exactly matched
the reviewed GPU timing, including continuous predictions and discrete graph
decisions. No retained fit has reached the terminal reconstruction gate.

Subsequent recovery reached 1,013 updates, then stopped on the edge-supervision
guard. See the [latest verified results](joint_saved_adam_fit_stop_results.md);
the recovery command below is historical, not a recommendation to retry that stop.

## What failed and what is recoverable

`Path.replace` raised `PermissionError: [WinError 5]` when replacing
`model/active_update.json.writing` with `model/active_update.json`. The pending
file records update 149, crop 0/0, phase `forward`, optimizer step not returned.
The preceding active file and all 148 journal entries agree on completed update
148 and Adam counters 4,148/148. This failure occurred **before the next forward
pass**, rather than in the model, loss, supervision guard or CUDA memory allocator.
It does not establish whether the longer joint fit will succeed.

The files have ordinary Archive attributes. The evidence does not identify which
Windows handle or process denied replacement, or establish that the denial was
transient. A Windows reader opened without delete sharing can produce this kind
of failure; that mechanism is reproduced in a regression test. No permissions,
antivirus settings or filesystem ownership are changed.

Only `step_0000.pt` and `step_0008.pt` exist. Update 148's model/Adam/RNG state
was in memory and was lost when Python exited. Recover from **update eight**,
not 148. Updates 9–148 must be repeated; the accepted trajectory still ends at
new-stage update 4,000. There is no cap extension or exact continuation from 148.
The original failed directory and its partial JSON remain untouched.

## Scoped repair

The fit controller now serializes each JSON document once, then retries its
atomic replacement on Windows error 5/32/33 only. Five backoffs of
0.05/0.1/0.2/0.4/0.8 seconds allow at most **1.55 seconds** of waiting. The prior
valid destination remains in place until replacement succeeds. Exhausted or
unrelated errors still stop execution and preserve the pending document.
No retry reruns an optimizer update. Model, loss, candidate graph, assignment,
normalization, data, thresholds, optimizer settings and the terminal gate are
unchanged. Historical shared helpers remain byte-for-byte unchanged.

Resume normally requires identical provenance. The sole reviewed transition
allows the controller recorded with SHA256
`2e9030270ec558846c7ec328d1912c5762b28c7e0d51264fd943212f78082c66`
to the repaired controller. Every other source hash, source-file membership,
configuration and backend field must match. The new report explicitly records
both controller hashes; future checkpoints use current provenance. This is not
a general bypass for source or experiment changes.

The repaired zero-update check uses `check_io_repair_v1/`, preserving the
historical `check/`. Resume creates a fresh `resume_step_0008/` folder. The saved
update-eight checkpoint precedes its replay proof, so recovery re-evaluates and
verifies it against the original GPU timing **before update nine**. All original
checkpoints, counters, RNG and next-crop validations remain required.

## Verification and recovery command

**22 tests pass**, including a real Windows reader lock, transient-denial
recovery, bounded permanent-denial failure, preservation of valid/pending JSON,
exact synthetic optimizer/RNG continuation across the approved source change,
and rejection of unrelated source/backend changes.

The real-model CPU check performs zero optimizer updates and reproduces weights,
Adam state and predictions exactly after disk reload. A separate
[read-only checkpoint audit](../../scripts/pointneuron2/check_joint_fit_io_recovery.py)
verifies the actual update-eight file hash, model/optimizer state and counters
4,008/8, and 119 protected source/data artifacts. It performs no forward pass,
optimizer update or CUDA RNG restoration. GPU recovery has **not** been launched
or validated by these CPU checks. Measurements and hashes are saved in
[validation.json](joint_saved_adam_fit_io_recovery_seed0/validation.json).

The CPU prerequisite is already complete. Launch recovery visibly:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_joint_saved_adam_fit.ps1 -Action resume -Checkpoint ".\tmp\pointneuron2\local_runs\joint_saved_adam_fit_20261020\model\step_0008.pt"
```

The same four training crops, RTX 3070 Ti Laptop GPU and maximum stage counter
4,000 apply. The previous timing estimate remains about 51–77 minutes of updates
plus checkpoint/evaluation overhead. Inspect the replay message and subsequent
automatic snapshots at 500/1,000/2,000/4,000. This repair addresses execution;
automatic reconstruction quality remains the experiment's unanswered question.
