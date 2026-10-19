# GPU timing command with saved proposal Adam state

19 October 2026. Prepared after the agreed [eight-update CPU joint stability
check](joint_saved_adam_stability_results.md). **GPU timing has not been launched.**
The new entry point exposes only `check` and `timing`; it cannot start or resume
a full fit. The original fresh-optimizer joint launcher remains unchanged.

## What the GPU action will do

Use the same four fixed TRAINING crops (0/0, 0/1, 92/4, 86/3), frozen Phase 1
clouds, warm-start PointNeXt/Transformer and fitted GAT weights, 512-query graph
context, candidate construction, losses, trust mask and 0.5 output cutoffs as
the tested CPU joint check. LR is 0.0001, weight decay 0.0001 and one global
gradient clip is 10. Proposal Adam moments/counters start at 4,000; GAT Adam
starts empty. After eight updates the counters must be 4,008 and eight.

1. Verify checkpoint/cache/source hashes and the completed CPU stability report.
2. Compare initial GPU proposals with the saved balanced GPU predictions at
   relative/absolute tolerance 1e-5. Save stage-zero automatic outputs and SWCs.
3. Perform exactly **eight throwaway GPU updates**, cyclic over the same four
   crops. Save incoming losses, assignments, positions/distances, edge-label
   counts, gradient norms, attempted/completed steps and optimizer cursors.
   Probe all four crops after every update.
4. Retain the existing loss guard: an active crop without either known edge class
   stops before its update. Unknown edges remain unknown. A failure saves the
   incoming prediction packet and traceback. No threshold, mask, loss, node head,
   graph budget or architecture policy changes to rescue a failure.
5. Save terminal automatic outputs at update eight. Report the same short
   stability criteria: eight updates complete, crop-0 retains six trusted matches
   after four/eight, and every crop has both edge classes at the end.
6. Measure PyTorch peak allocated/reserved GPU memory across updates/probes and
   compare it with device capacity. This is allocator memory, not total usage by
   every process on the GPU.
7. Serialize a temporary, explicitly **nonresumable timing-only** checkpoint with
   model, mixed optimizer history and RNG state. Reload it on the same backend.
   Require exact equality of weights, optimizer state and four-crop predictions
   (coordinates, radii, confidence, candidates and edge logits). Discard the
   temporary checkpoint even if this verification raises; prediction artifacts,
   hashes, metadata and error logs remain. No trained checkpoint is retained.

`timing_readiness_gate` requires the declared stability criteria and exact reload,
with completed execution and original-source hash checks. A completed action
with a false stability gate is not a passed readiness result. The strict automatic
reconstruction gate is reported separately; eight updates do not need to solve
it to measure timing, and a timing pass never replaces that learning milestone.

## Timing interpretation and limits

Each measured update includes its incoming diagnostic forward/loss description,
backward pass, clipping, optimizer step, CUDA synchronization and the associated
report writes. The all-four-crop post-update probes are excluded from those
durations. A planning range uses the second cycle's mean update duration times
4,000, with a 1.5 multiplier as its upper planning bound. It is explicitly an
update-time estimate; snapshot evaluation/checkpoint IO, repeated four-crop
probing and future changes in candidate density are excluded. No training
duration or memory-readiness result is available until the GPU action completes.

No full trainer is implemented here. A later retained stage would need explicit
mixed-history checkpoint/resume semantics and the existing 4,000-new-update cap.
Its destination remains clean automatic nodes, correct decoded graph/SWC and
at least 99% reference arclength precision and recall within two voxels on every
training crop. Broader evaluation and a matched PN 1.0 comparison remain later
requirements before claiming reconstruction improvement.

## Local verification and command

Twelve targeted CPU tests pass. They cover mixed-state disk reload and identical
next-update behavior, optimizer-counter mapping, backend mismatch and resumable
flag rejection, unchanged original supervision-stop behavior, failure prediction
capture and temporary-checkpoint cleanup even on a reload error. The new launcher
also has a zero-update real-model CPU check; its measurements and source hashes
are in [the visible readiness record](joint_saved_adam_timing_ready_seed0/validation.json).

That CPU action verifies the new device adapter against the previous CPU
stability probes, validates initial automatic SWCs and checks exact on-disk
weight/optimizer/prediction reload. It does not make GPU execution or stability
claims. The known CPU/GPU warm-prediction difference is reported on CPU; the
GPU action separately enforces same-GPU historical replay before updating.

Launch the new GPU timing action yourself:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_joint_saved_adam_timing.ps1 -Action timing
```

Expected device: NVIDIA GeForce RTX 3070 Ti Laptop GPU. There is no CPU fallback
for `timing`. Default output root:
`tmp/pointneuron2/local_runs/joint_saved_adam_timing_20261019`.
Generated reports, predictions, SWCs and console logs remain ignored; existing
action folders cannot be overwritten. The already-completed CPU check lives in
`check/`; the future GPU report will be `timing/report.json`. Review its stability
gate, reload, memory, estimate and automatic outputs before planning a retained
fit. The launcher accepts no `train` or `resume` action.

Visible implementation:
[joint_saved_adam_timing.py](../../scripts/pointneuron2/joint_saved_adam_timing.py),
[PowerShell launcher](../../scripts/pointneuron2/run_joint_saved_adam_timing.ps1),
and [timing recipe](../../configs/pointneuron2/joint_saved_adam_timing.json).
