# Frozen four-crop confidence stage: ready for a visible CPU fit

18 October 2026. The user agreed to test staged confidence learning after the
[capacity oracle](four_crop_confidence_capacity_results.md) verified a shared
separator. Preparation, tests and an eight-update throwaway CPU check pass.
**The retained fit has not started.** No GPU work or whole-model training is
prepared. All completed runs remain preserved.

## Exact control and its interpretation

Initialize the two confidence rows from the completed balanced joint model at
source update 4,000. Freeze its PointNeXt, Transformer, hidden ReLU features,
geometry, radii and saved assignments. Only a separate 128-to-2 linear module
trains: **258 parameters in two tensors**. The geometry rows are not in this
module, so AdamW cannot decay them. No new matching or sampling occurs.

Use a **fresh AdamW optimizer** at LR .0001, weight decay .0001, clip ten,
float32 and two CPU threads. Each update averages four per-crop class-balanced
CE terms, each `.5 positive_mean + .5 negative_mean`, then applies the existing
outer coefficient two. All four cached crops contribute every update. No
standardization, new hidden layer, positive weight eight or hard-negative term
is applied. Inference decisions use logit difference >=0, equivalent to the
fixed .5 threshold without softmax rounding ties.

The new stage has a **4,000-update cap**, with snapshots at
0/100/500/1,000/2,000/4,000. Its cursor starts at zero; source joint training
remains a separate completed 4,000-update run. This is deliberately a fresh
optimizer stage, not an exact continuation of the old whole-model optimizer.
It also changes crop batching from cyclic single crops to all four per update.
Therefore a result tests staged-learning viability; it cannot isolate changing
labels as the sole cause of joint-training failure or claim matched compute.

Every stage update sees 2,048 cached feature vectors and 452 positives distributed
6/12/16/418 across neuron 0/crops 0,1, neuron 92/crop 4 and neuron 86/crop 3.
All are training data. No new images, neurons, development or test data enter.
No image, field, encoder or Transformer forward occurs in this control.

## Implementation and provenance

- [Recipe](../../configs/pointneuron2/four_crop_frozen_confidence.json)
- [Confidence-only module](../../src/pointneuron/models/four_crop_frozen_confidence.py)
- [Runner](../../scripts/pointneuron2/four_crop_frozen_confidence.py)
- [PowerShell launcher](../../scripts/pointneuron2/run_four_crop_frozen_confidence.ps1)
- [Tests](../../tests/test_four_crop_frozen_confidence.py)

The frozen cache is reconstructed from saved final decoder features through the
source checkpoint's hidden linear/ReLU layer, without upstream forwards. Its
feature bytes match the capacity oracle exactly. Centers, radii, decoder
features, labels and candidate graph arrays are cached unchanged, with source
report/checkpoint/prediction hashes protected. The initial confidence rows
reproduce the balanced source's .5 selection counts: 6+11, 12+32, 16+17 and
411+4, where each pair denotes accepted assigned queries plus extras.

Only a new confidence-stage checkpoint is written by a retained fit. It has a
separate schema and cannot be mistaken for a full PN 2.0 model checkpoint.
No fitted head is automatically installed into the reconstruction pipeline.
Full head optimizer/RNG state and source/cache identity are saved atomically
before evaluation. Explicit resume creates a separate directory, restores the
same stage and retains the original cap. Check-only and terminal checkpoints
cannot resume; weights-only restart is not presented as exact continuation.

## Verification

Thirteen targeted tests pass, covering balanced four-crop reduction and
gradients, the exact parameter scope, detached feature validation, synthetic
learnability, bitwise frozen-cache invariance, exact next-update agreement after
disk optimizer/RNG restore, recipe/cap rejection, checkpoint preservation before
metric failure, and separation of label success from geometry success. The
capacity-oracle known-case tests also pass in the same invocation.

The real CPU check reproduces initial selection, performs **eight throwaway
head updates**, verifies frozen tensor/source hashes, and reloads head/optimizer
state with zero output difference. It retains no fitted checkpoint and records
zero retained-stage updates. Its measured update-loop estimate is **36–64
seconds**, plus Python startup, checkpoint and evaluation IO. This is a planning
estimate from a short check, not a runtime guarantee or a learnability result.
See [readiness evidence](four_crop_frozen_confidence_ready_seed0/summary.json).

## Decision and stopping rules

Diagnostic success requires **all 452 fixed positives accepted and all 1,596
fixed negatives rejected**, at .5 on every crop simultaneously. All frozen
features/labels/geometry must remain bitwise unchanged and reload must match.
This stricter zero-error criterion tests recovery of the verified separator;
it does not replace the earlier proposal/reconstruction gates.

Report the full snapshot trajectory, loss, lowest positive/highest negative
scores, curve support and frozen-reference edge recall after confidence. Do not
select a convenient intermediate snapshot as an undeclared terminal success.
Counts of trusted geometry and junctions are carried over separately. Even a
perfect label classifier accepts the 14 poorly placed dense assignments and
leaves 76/77 trusted dense junctions; full reconstruction remains unsolved.

If successful, this supports a staged confidence fit on these fixed examples,
not generalization or a causal diagnosis of assignment drift. If incomplete,
the declared optimizer/objective/budget did not find a clean boundary even
though the oracle demonstrates representability. That is not evidence of an
unavoidable feature/head capacity limit. Do not automatically extend the budget,
change LR, sweep penalties, deploy the head, unfreeze the model or train GATv2.
Stop on nonfinite values, source/cache changes or checkpoint failure and preserve
the full traceback. No geometry or threshold repairs are part of this control.

## Run the retained fit

The preparation and CPU check are complete. The next user-launched command is:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_four_crop_frozen_confidence.ps1 -Action train
```

This uses **CPU**, with no additional GPU timing command. Outputs stay under
ignored `tmp/pointneuron2/local_runs/four_crop_frozen_confidence_20261018`, split
into cache/check/model directories. Existing action outputs cannot be overwritten.
After interruption only, use explicit
`-Action resume -Checkpoint <full path to this stage's model step checkpoint>`.
