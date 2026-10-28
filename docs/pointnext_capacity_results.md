# PointNeXt capacity sweep: observed results and implementation finding

29 October 2026. User-launched sweep of experiment commit `82f32eb`, reviewed
against saved per-mode files and the aggregate report. No GPU retry, model edit,
optimizer or training fit was launched during this review.

## Observed execution

RTX 3070 Ti Laptop GPU, approximately 8 GiB VRAM; Torch 2.12.1+cu130, CUDA build
13.0. Batch one, float32, fresh seed-zero native encoder with 790,720 parameters.
Synthetic unique grid points only. One warmup and two measured iterations per
mode, without optimizer steps. Ordinary native BatchNorm, not the historical
per-crop normalization variant. Output descriptors have shape `[1,N,32]`.

| Points | Eval forward, mean seconds | Eval peak allocated GiB | Forward/backward, mean seconds | Forward/backward peak allocated / reserved GiB | Verdict |
|---:|---:|---:|---:|---:|---|
| 8,192 | 3.173 | 0.075 | 4.432 | 0.235 / 0.262 | Both completed |
| 16,384 | 5.609 | 0.147 | 2.391 | 0.731 / 0.895 | Both completed |
| 32,768 | 7.482 | 0.291 | 6.380 | 2.550 / 3.846 | Both completed |
| 65,536 | 16.441 | 0.580 | Not completed | Not saved | Inference completed; CUDA OOM in first training forward |

The successful budgets passed finite-output/parameter-gradient checks and actual
abstraction shape checks. The 64k inference output also passed its shape/finite
check and has a separate saved mode report. Its training warmup failed before
backward; no successful training-mode time or peak is available. The aggregate
64k failure row omits its completed inference result, but that result is preserved
in `65536/eval_forward_result.json` and included in the evidence summary below.

Total budget-process wall times: 36.8, 32.6, 54.7 and 89.7 seconds. No time cap was
hit. Two timings per mode are a coarse execution observation, not a stable speed
benchmark. In particular, the nonmonotonic training times and faster training-mode
timings than eval in some cases do not establish a performance advantage. Clock,
runtime warmup and different normalization paths have not been isolated.
Allocated/reserved figures are PyTorch CUDA allocator counters, not total device
usage. Encoder-only costs exclude the Transformer, GAT, optimizer state, real
image preparation and whole-volume inference.

The runner reports generic `failed` because this error surfaced as
`torch.AcceleratorError`, outside its narrower `torch.cuda.OutOfMemoryError`
classification. The saved error explicitly says CUDA out of memory. This is a
reporting limitation, not a different biological or numerical failure. CUDA warns
the reporting stack frame can be asynchronous, so do not identify the exact
allocation from that frame alone.

## Confirmed implementation issue: retained full-sort index storage

The pinned native interpolation contains:

```python
indices = distance.argsort(dim=-1, stable=True)[..., :min(3, support.shape[1])]
```

The slice has only three index values per query, but is a view into the entire
sorted-index tensor. Feature gathering needs these indices for backward.
Consequently autograd retains each chunk's full backing allocation, even though
coordinates, distances and index selection are calculated without gradients.
The chunk loop limits transient distance work but does not limit the sum of
backing index allocations retained until backward.

A CPU-only diagnostic on the unchanged, hash-verified historical operator used
1,024 queries, 256 supports, eight feature channels and chunks of 64. Saved-tensor
hooks recorded 16 distinct index storages. Each saved tensor had logical shape
`[1,64,3]` (1,536 bytes), but retained 131,072 bytes of full-sort storage. Across
chunks: **24,576 logical index bytes retained 2,097,152 backing bytes**. Output and
feature gradients were finite. This directly establishes the storage behavior
on CPU, without training or a GPU run.

At the last restoration level there are N fine queries and N/4 coarse supports.
The corresponding retained index storage is `N * (N/4) * 8` bytes:

| Input N | Full-sort backing storage at last restoration alone |
|---:|---:|
| 8,192 | 0.125 GiB |
| 16,384 | 0.500 GiB |
| 32,768 | 2.000 GiB |
| 65,536 | 8.000 GiB |

These large-budget storage figures are source-derived extrapolations of the
verified CPU mechanism, not measurements of the failed GPU allocation. The 8-GiB
requirement at 64k exceeds available VRAM before other live allocations. It is a
strong, specific explanation for the observed failure, and its scale agrees with
the much larger training-mode than inference memory at successful budgets.
The existing trace does not independently locate every live GPU allocation.

## Consequence and proposed next step

Do not interpret this run as establishing a 32k architectural ceiling or redesign
Phase 1 to work around it. It establishes execution through 32k in this backend
and successful 64k inference, with a confirmed avoidable storage-retention issue
in the training path. It says nothing yet about thin/dim retention or learned
reconstruction quality.

The smallest proposed correction is to materialize the selected three-neighbor
indices into their own compact allocation before gathering (for example, clone
the slice). Preserve stable sort/ties, indices, interpolation weights, point
counts, features and architecture. This is a proposed implementation correction,
not an implemented new encoder or a promotion of the historical model.

Before a GPU rerun, require CPU equality of neighbor identities and forward
values, feature-gradient parity and direct confirmation of reduced saved index
storage. Then prepare a bounded 64k forward/backward replay with the same settings
and explicit corrected-source provenance. A successful replay would establish
the correction's execution effect; full model capacity and biological quality
would still need their separate checks. Do not start a learned fit or change
sampling, radius, K, normalization or precision to make this memory gate pass.

After resolving this backend artifact, continue the agreed real-cloud
neighborhood/contrast retention measurements before choosing the trained
comparison. The current data do not select a biological point budget.

## Evidence and reproducibility

* Original run: `tmp/pointnext_capacity_sweep_20261029_005401/`; retain unchanged.
* Visible [evidence summary](evidence/pointnext_capacity_review_20261029.json)
  includes the aggregate-report hash, original configuration, completed per-mode
  results including 64k inference, and the CPU storage-audit output.
* Diagnostic: `scripts/audit_pointnext_interpolation_storage.py`; no source
  changes or accelerator operation. Original CPU output:
  `tmp/pointnext_storage_audit_20261029/report.json`.

To reproduce only that small CPU audit into a new directory:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\audit_pointnext_interpolation_storage.py `
  --output .\tmp\pointnext_storage_audit_repeat
```
