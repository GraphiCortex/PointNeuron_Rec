# GN8 CUDA timing: consistency verified, subsequent fit completed

Current status: the user completed the fit described below. **Mode consistency
passes; node learnability fails.** Read the [fit results](groupnorm_node_fit_results.md)
before further work. The command below is a preserved execution record, not a
new run request.

31 October 2026. The user launched
`tmp/groupnorm_node_timing_20261031_141828/`. **All eight updates completed,
train/eval prediction differences were exactly zero, and model/Adam/RNG reload
passed.** The [normalization-only control](groupnorm_node_control.md) is ready
for the agreed manual 2000-update fit. No full fit has been launched by the
agent and no model/loss/data setting was changed after this timing result.

## Verified measurements

One fresh GN8 PointNeXt/Transformer on the NVIDIA GeForce RTX 3070 Ti Laptop GPU,
Torch 2.12.1+cu130, float32, 2048-point Sato clouds, 512 queries. Four fixed training
boxes; two updates per box. Baseline input, target, loss and optimizer settings
remain fixed.

| Check | Result |
|---|---|
| Completed updates | 8 / 8 |
| Ordinary eval versus train-mode diagnostic, update 0 | Bitwise-identical centers, logits and scores in all four boxes |
| Ordinary eval versus train-mode diagnostic, update 8 | Bitwise-identical centers, logits and scores in all four boxes |
| Maximum recorded mode difference | 0.0 |
| Model / Adam / RNG serialization reload | Exact match |
| Second-cycle mean update time | 0.948 seconds |
| Peak allocated / reserved CUDA memory | 0.233 / 0.262 GiB |
| Estimated 2000-update time | 25.3-37.9 minutes, plus evaluation/checkpoint overhead |
| Retained trained checkpoints | Zero |

The range is the declared 0.8-1.2 times heuristic from four measured updates,
not a confidence interval or guaranteed completion time. The worker wall cap for
the full fit remains 3600 seconds. Peak memory applies to this small component
experiment and includes the reload probe; it does not qualify dense-input or
whole-reconstruction memory use. The user-observed timing command took 49 seconds.

The eight update-journal rows are complete/finite, both saved prediction packet
hashes match their reports, and all saved arrays are finite. The update-eight
prefix model hash matches the post-evaluation model hash. Current GN source,
recipe, upstream source and CPU preflight validators all accept the result.

## What this resolves and what it does not

The previous BN fit had a large change of geometric placement between training
statistics and ordinary inference. In the GN variant, this mode discrepancy is
absent on the actual GPU before and after eight parameter updates. This verifies
the intended functional consistency property for the tested configuration.

It does not establish learned node quality. Both evaluations accept zero nodes;
unfiltered two-voxel node recall also remains zero. Eight updates give only two
exposures per box. The model is still near initialization, and the timing's
`learnability=None` is deliberate. Consistent predictions can still be inaccurate.

The question for the next fit is whether the same node-learning objective now
learns a useful automatic proposal set under ordinary inference. The known
confidence/geometry failures are not declared solved by this engineering check.

## Next manual command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_groupnorm_node_fit.ps1 -Action train `
  -TimingRun ".\tmp\groupnorm_node_timing_20261031_141828" `
  -Output ".\tmp\groupnorm_node_train_20261031"
```

One fresh model trains on the same Sato boxes, capped at 2000 total updates
(500 per box). The completed Sato/BN fit remains the comparator. There is no
threshold-control repeat or transfer of trained BN parameters. Fresh parameter
values match the original initialization; only normalization differs.

The first eight updates must replay this GN timing's model/Adam/RNG prefix
exactly, and they are included in the 2000-update cap. Complete states are saved
before evaluations at 0/8/500/1000/2000. Each evaluation verifies exact mode
consistency and separately reports node quality. A consistency discrepancy or
execution failure stops the control; a false intermediate node gate alone does
not cause a retry or settings change.

The terminal node criterion is unchanged: >=95% one-to-one node precision AND
recall within two native voxels in each box, cutoff 0.5. At the end, the report
pairs ordinary GN predictions with the saved ordinary BN predictions by box
identity and checks equal target denominators.

If consistency remains correct but node learnability still fails, stop
normalization variants and revisit the query/assignment formulation. Do not
extend updates or tune group count, cutoff, targets or LR automatically. Even a
pass would establish only training-set learnability, not a general sampler or
whole-neuron reconstruction improvement.

The [visible timing evidence](evidence/groupnorm_node_timing_20261031.json)
preserves source identities, prefix hashes, raw file hashes, exact consistency
results and measured resources. Original timing and baseline files are unchanged.
