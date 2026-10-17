# Evaluation failure repaired; original 500-update attempt preserved

17 October 2026. The user-run fit completed 500 updates and then failed while
evaluating the dense training crop. This was a bug in the runner's metric-cache
decoding. It was not a reported nonfinite loss/gradient failure. The original
run, log, partial predictions and source snapshots are preserved unchanged.

## Exact failure and bounded repair

The saved failure is:

```text
x must consist of vectors of length 3 but has shape (0,)
```

The dense crop's reference geometry contains two zero-length clipped chain
pieces, indices 49 and 50. Their probe points originally have shape `(0, 3)`;
JSON stores them as `[]`, and the old cache decoder reconstructed shape `(0,)`.
Once predictions are nonempty, the support evaluator creates a KD-tree and
queries those probe coordinates. SciPy correctly rejects the malformed shape.
When no proposal is accepted, the evaluator never calls that query, which is
why the initial CPU/GPU timing checks did not expose the bug.

The crop has **133 critical reference points**; its critical-point list was not
the source of this exception. The cache-only reproduction identifies the two
empty chain probes and reproduces the exact saved error.

The fit runner now restores coordinate dimensions on empty probe and critical
arrays when decoding cached JSON, and checks point/weight alignment. It retains
the zero-length pieces: no annotation or curve support is fabricated or removed.
All existing nonempty coordinates and scoring formulas are unchanged.

The shared historical metric/decoder files are untouched. Only the new runner
and its PowerShell launcher change at runtime. Model, optimizer update rule,
query loss, targets, field policy, learning rate, crop order, normalization,
cutoffs and 4,000-update cap remain identical to the agreed experiment.

## Checkpoint and error-handling corrections

The original runner saved its snapshot **after** evaluating all four crops.
That ordering was a mistake: the metric exception prevented `step_0500.pt`
from being written. Only `step_0000.pt` exists. The three partial prediction
files do not contain encoder/model weights, AdamW moments or RNG state and
cannot be used to reconstruct an exact continuation from update 500.

The repaired runner saves complete model/optimizer/RNG/cursor state atomically
**before evaluation**, then writes its path/hash/step into the report. An
evaluation failure therefore leaves the completed snapshot available for
explicit resume. Interrupted temporary writes are retained rather than
mistaken for complete checkpoints. This does not alter the learning objective.

The Python failure handler now writes a full `error_traceback.txt` independently
of the shell. Windows PowerShell previously treated the first redirected native
stderr line as a terminating error and hid the rest of the traceback. The
launcher now drains and logs the entire stream, then reports failure using the
native exit code. A native-command stub regression verifies both traceback
lines and exit code 7; this test starts no model or GPU work.

## Evidence and limits

**Ten targeted tests pass**, including nonempty predictions with empty JSON
probes, correct zero-length weighting, malformed-cache rejection, a deliberate
evaluation exception after checkpoint persistence, full native stderr capture,
and the existing resume/gate/normalization checks. The save-order regression
uses artificial step metadata with no training updates; it is not a 500-update fit.

On the actual dense cached crop, the original scoring exception reproduces.
The repaired decoder successfully scores a nonempty diagnostic point set.
Empty-prediction scores match exactly before/after the repair, and reference
length remains **2,481.570099 voxels**. No model forward or optimizer update is
needed for that reproduction.

The fresh restart CPU check passes in 13.21 seconds excluding imports: prediction
reload delta 0.0, model state unchanged, initial weights identical to the prior
seed-zero CPU check. Original source snapshots and all original run files retain
their recorded hashes. The restart cache is a byte-identical copy, not regenerated
clouds or targets. No replacement GPU timing/training has been launched.

The three completed partial 500-update crop evaluations are useful but incomplete:

| Training crop | Trusted matches within 2 voxels | Accepted assigned | Accepted unassigned |
| --- | ---: | ---: | ---: |
| 0 / 0 | 6 / 6 | 0 | 0 |
| 0 / 1 | 7 / 12 | 0 | 8 |
| 92 / 4 | 15 / 16 | 0 | 0 |
| 86 / 3 | Evaluation interrupted | Unknown | Unknown |

These show partial training-set geometry learning with unresolved confidence.
They do not establish the combined gate, dense-crop learnability or a
reconstruction improvement. In particular, all six trusted control-0 targets
still yield only 3/5 available reference-edge candidates: finding points alone
does not establish connectivity readiness. Keep the existing candidate criterion
and confidence cutoff; do not tune them in response to this partial snapshot.

## User-launched restart with the same experiment protocol

The original failed run remains in `branched_query_20261017`. The launcher now
defaults to the fresh folder:

```text
C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\branched_query_restart_20261017
```

Its identical cache and repaired CPU check are already prepared. Repeat the
short GPU timing action because runtime source hashes have changed; the strict
identity guard is retained rather than editing the original timing evidence.
This also gives a fresh estimate: the original 76–114 minute estimate was based
on one short timed cycle, whereas the failed run reached update 500 in about
151 seconds including initial evaluation.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_branched_query.ps1 -Action timing
```

After timing completes successfully, the separate fit command is:

```powershell
& .\scripts\pointneuron2\run_branched_query.ps1 -Action train
```

The retry starts from fresh seed-zero weights and stops at 4,000 model updates,
with the same data and gates. The failed 500-update attempt remains a separately
reported execution failure; it is not hidden in the new trajectory. An exact
GPU prefix replay is not promised. No confidence-only replay, extra data,
threshold change, GATv2 fit or architectural replacement is introduced.

- [Visible repair evidence](branched_query_repair_seed0/summary.json)
- [Original experiment and criteria](branched_query_ready.md)
- [Repaired runner](../../scripts/pointneuron2/branched_query_experiment.py)
- [Launcher](../../scripts/pointneuron2/run_branched_query.ps1)
- [Regression tests](../../tests/test_branched_query_fit.py)

The metric reproduction and reconstructed diagnostic traceback are in ignored
`tmp/pointneuron2/local_runs/branched_query_repair_20261017`. Original/full
tracebacks must not be confused: the first original console stream was truncated;
the exact error is stored in its failed report and independently reproduced.
