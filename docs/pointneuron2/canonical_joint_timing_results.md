# Canonical-target GPU timing: execution passes; capped fit is next

21 October 2026. **The user-run eight-update timing check passes.** The separate
CPU saved-output audit verifies all eight prediction/SWC pairs. No full training
run has been launched. The [prospective matched-fit contract](canonical_joint_fit_ready.md)
is unchanged; no model, loss, cutoff, budget or executable source was changed.

| Check | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| Fixed TRAINING crops | 0/0, 0/1, 92/4, 86/3 |
| Canonical target counts | 6, 11, 15, 408 |
| Initial inference vs saved old-arm baseline | Exact centers, radii, logits, features, candidates, scores and selection |
| Throwaway updates completed | Eight, two exposures per crop |
| Terminal Adam clocks | Proposal 5,008; GAT 1,008 |
| Known edge supervision | Both positive and negative classes on all eight updates |
| Placement trust | All assigned targets within two voxels on incoming updates and both all-crop probes |
| Disk state/prediction reload | Runtime full model/Adam/RNG fingerprint unchanged; prediction delta zero |
| Peak PyTorch allocated memory | 0.303780 GiB; not total system/driver GPU memory |
| Trained checkpoints retained | None |
| Timing action elapsed | 49.31 seconds; user console includes additional setup time |
| Update-only extrapolation for 3,000 new updates | 75.65 minutes; excludes probe/evaluation/checkpoint costs |

The review independently checks automatic 0.5 selection, inherited candidate
edges/scores, transferred known/unknown training labels, graph and continuous
geometry metrics, original-form automatic gates, artifact hashes and SWC
serialization. Initial GPU arrays exactly match the saved old-arm arrays despite
different target assignments. Source, checkpoint, cache and reference hashes are
unchanged. The discarded eight-update model/Adam/RNG state is not available for
independent fingerprint recomputation: its exact reload is a runtime check,
preserved in the report and its recorded fingerprint.

## Quality remains an unanswered fit question

| Crop | Accepted targets + extras at eight | Correct required edges, 0 → 8 | Precision / recall at eight, within 2 voxels |
|---|---:|---:|---:|
| 0/0 | 6 + 1 | 5/5 → 5/5 | 94.05% / 100% |
| 0/1 | 11 + 19 | 5/9 → 3/9 | 47.61% / 90.48% |
| 92/4 | 15 + 6 | 13/14 → 13/14 | 81.71% / 100% |
| 86/3 | 408 + 10 | 378/394 → 379/394 | 99.84% / 99.64% |

All four automatic reconstruction gates fail. Sparse-crop precision worsens and
crop 0/1 also loses edge accuracy/recall during these eight updates, while dense
curve geometry improves. This is not broad reconstruction improvement or evidence
of convergence. Two exposures per crop do not decide the existing bounded
learnability question. Placement stays trusted; execution, supervision, replay
and reload requirements pass. No early-quality qualification gate was specified
for timing, and none is retroactively added or waived.

`learnability=None` and `paired_improvement=None` in the console are intentional
for the timing action. Those terminal verdicts belong to the capped fit. Passing
timing does not turn either verdict into a pass.

## Next command and stopping rules

The existing prepared launcher is ready for the agreed matched fit:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_canonical_joint_fit.ps1 -Action train
```

It reimports the original shared stage-1,000 weights, Adam and RNG; the throwaway
timing updates are not continued or added to the trajectory. It changes only the
target representation, retains LR 0.00001 and both 0.5 decisions, and caps at
3,000 new updates, ending at historical stage 4,000. The first eight updates
must reproduce this timing ledger and model/Adam/RNG fingerprint. Any replay,
source, finite-state or reload failure stops execution. The declared geometry
stop and terminal learnability/paired-geometry criteria remain unchanged.

Allow roughly **90–120 minutes**, with load/thermal/I/O variability potentially
making it longer. The measured update-only estimate is already 75.65 minutes.
The launcher's earlier 60–90-minute planning text predates this timing and is
superseded by the observed estimate here; executable files are kept unchanged
so the completed CPU/timing checks remain applicable. Duration never changes the
update cap. No automatic extension, LR/threshold/loss change or checkpoint
selection follows a failure.

If canonical training learnability passes while paired geometric noninferiority
fails, report that tradeoff and do not claim a reconstruction win. Even a paired
pass remains a four-crop training result, not unseen or whole-neuron performance.

Raw timing evidence:
`tmp/pointneuron2/local_runs/canonical_joint_fit_20261021/timing/`.
The full fit will create a separate `model/` directory at the same experiment root.

[CPU audit script](../../scripts/pointneuron2/review_canonical_joint_timing.py)
and [verified timing summary/provenance](canonical_joint_timing_results_seed0/summary.json).
The audit performs no model forwards, optimizer updates or CUDA initialization.
