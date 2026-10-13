# Learning-rate-only confidence replay: ready

14 October 2026. The user agreed to replay the fixed-feature confidence test
at LR **.001**, versus the completed **.0001** baseline. The runner, recipe,
paired checks and readable comparison output are prepared. **No real GPU
timing or replay fit has started.** The real read-only source check passed;
four tests passed in 68.68s.

## Purpose and fixed comparison

The [completed baseline](query_confidence_results.md) passes crop 1 but still
misses three crop-0 positives and accepts two unmatched queries. A numerical
linear-capacity oracle verifies that the existing fixed features admit one
shared linear separator for all 1,024 labels. The next question is whether a
larger learning rate reaches the available decision boundary within the same
2,000-update budget. The oracle does not establish that LR is the unique cause
or that this replay will pass.

| Setting | Baseline | Replay |
| --- | --- | --- |
| AdamW learning rate | .0001 | **.001** |
| Initialization | Original cached confidence weights/bias | Exactly the same |
| Feature cache and assignments | Fixed neuron-0 crops 0/1, 6/12 positive queries | Same files/tensors |
| Trainable parameters | 258, final 128-to-2 linear output | Same |
| Updates / snapshots | 2,000 / 0,100,500,1,000,2,000 | Same |
| Weight decay / clipping | .0001 / 10 | Same |
| Loss | Weight 2 × equal mean of crop weighted CEs; positive weight 8 | Same |
| Cutoff / augmentation | .5 / none | Same |
| Backend | RTX 3070 Ti Laptop GPU, float32, TF32 off | Must match saved baseline |

The recipe checker rejects every difference other than learning rate. AdamW
uses LR when applying both gradient and decay steps; the decay coefficient
remains fixed. This is a fresh replay from original cached weights with a
fresh optimizer, **not a continuation from the baseline's final head**.

PointNeXt, Transformer, geometry, radii, labels and features remain frozen.
No encoder/decoder or field forwards occur in this replay. Both cached crops
contribute to each update. The cache is reused read-only, with no copying,
resampling or new model inference to rebuild features.

## Checks and results to inspect

The runner verifies the completed baseline report, recipe, initial/final
checkpoint hashes, source code and immutable feature cache. It compares the
baseline's step-zero weights directly with the original cached initialization.
Timing and fit must use the saved baseline's backend. Step-zero crop decisions
must match and numerical metrics must agree within 1e-5.

The same final gate applies: all 6/12 fixed positives accepted, no unmatched
accepted queries, fixed distinct placement within one voxel and at least 95%
accepted curve coverage within six voxels. No gate, label, threshold or target
changes are made to obtain a pass. Cache tensors and baseline source/output
hashes are checked again after timing/fit; strict head reload must match.

The final `comparison.json` contains both crop records at all five snapshots,
including loss and accepted/missed/extra query counts, score margins, curve
support and deltas against the saved baseline. Execution success and gate
success remain separate. A pass would support faster optimization for this
fixed two-crop task, not generalization or a ready full model. A failure needs
loss/margin review; it does not negate the verified linear capacity.

## Visible commands

No preparation is needed: the original cache is already verified. First run
four throwaway confidence updates on your GPU for timing/reload checks:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_confidence_lr.ps1 -Action preflight
```

The baseline fit recorded 8.25s compute and 14s observed wall time. This replay
has the same update count and dimensions, so a similar short runtime is expected;
use the new measured estimate. Timing retains no fitted checkpoint and does
not launch training.

After reviewing that estimate, run the separate fresh fit:

```powershell
& .\scripts\pointneuron2\run_confidence_lr.ps1 -Action train
```

Default output folder:
`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\query_confidence_lr_20261014`.

It contains `timing.json`, `model/fit_report.json`, five confidence-head
checkpoints, ten score CSVs and the paired `comparison.json`. Console logs
are beside the folder. Existing and failed outputs are preserved; no automatic
retry, extension or production promotion occurs.

The optional read-only command already passed here:
`run_confidence_lr.ps1 -Action check`. It performs zero forwards/updates and
does not create a replay output directory.

## Source files and verification

- [PowerShell entry point](../../scripts/pointneuron2/run_confidence_lr.ps1)
- [Replay and paired comparison](../../scripts/pointneuron2/confidence_lr_experiment.py)
- [LR-only recipe](../../configs/pointneuron2/query_confidence_lr.json)
- [Tests](../../tests/test_confidence_lr.py)

Tests cover rejection of extra recipe changes, a full synthetic timing/fit/
paired-output workflow with preserved baseline/cache, and corrupted initial
weights failing before output creation, plus rejection of missing initial crop
records. Synthetic snapshot copies are contract
fixtures and do not claim a learned baseline. The real baseline runner,
classifier module, config, cache and trained checkpoints are unchanged.
