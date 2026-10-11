# Full guides: implementation and local checks

New sampling caches preserve **every extracted guide vertex**. The 512 guide
cap belonged to the early execution pilot; it was not required by the method.
The guide supplies distances and the auxiliary loss. It is independent of the
512 image points currently passed to the encoder.

The new default in `configs/pointneuron2/sampling_training.json` is:

| Setting | Value | Meaning |
|---|---:|---|
| `guide_points` | `null` | Preserve the entire extracted guide |
| `candidate_points` | 1,024 | Image candidate pool |
| `selected_points` | 512 | Encoder inputs selected from that pool |
| `distance_chunk_size` | 256 | Pairwise distance block size; no guide subsampling |

These last two point counts remain experimental settings. Full guides cannot
recover a neurite absent from the candidate pool. The extraction method, original
global mask and isotropic voxel ruler have not been replaced by an alternative
mask from the preceding comparison.

## Implementation and compatibility

`src/pointneuron/data/sampling_cache.py` preserves full guides when the cap is
`None`. Schema 2 records full-guide caches and verifies the saved count against
the extractor's count. Variable-length guides are padded and masked in batches.
Explicit positive caps retain the schema-1 path. Existing artifacts are untouched.

`src/pointneuron/models/guide_distances.py` reduces exact Euclidean distances in
blocks. During backward it recomputes blocks instead of retaining all pairwise
distances. Equal nearest neighbours share gradients; coincident points have zero
distance gradient. This implements first-order derivatives, appropriate for the
current trainer. Pairwise workspace scales with batch size times block size
squared, plus guide/input storage. The sampler's soft selection and encoder have
their own memory costs.

`adaptive_sampling.py` uses this reduction for the guide prior and symmetric
squared Chamfer. New checkpoints save the distance setting; inference rebuilds
the same full-guide pools. Old checkpoints without that setting retain their
original dense calculations and explicit saved guide caps. The optional viewer's
candidate builder also accepts an uncapped guide.

## Cache check on original preprocessing

A fresh cache was prepared for four image-defined training-neuron-0 patches.
The candidate coordinates, raw intensities and indices are identical to the
corresponding historical cache arrays. All extracted vertices were retained:

| Patch | Previous retained guide | New retained guide | Candidates |
|---|---:|---:|---:|
| 0 | 512 | 1,296 | 1,024 |
| 1 | 512 | 2,585 | 1,024 |
| 2 | 512 | 885 | 1,024 |
| 3 | 512 | 1,875 | 1,024 |

Saved validation: [full_guide_cache_validation.json](full_guide_cache_validation.json).
The fresh artifact folder is `tmp/pointneuron2/full_guide_cache_smoke`.
It contains training patches only and is not a complete training/validation
manifest. The historical cache manifest hash remains unchanged.

## Measured on this computer

The local RTX 3070 Ti Laptop GPU used PyTorch 2.12.1+cu130. Each diagnostic batch
repeated one saved training patch twice, using 1,024 candidates and selecting
512 points. There was one warm-up and three measured forward/backward passes per
mode, **zero optimizer updates**, and no saved model weights.

| Guide vertices | Dense full peak MiB | Blocked full peak MiB | Dense full median ms | Blocked full median ms |
|---:|---:|---:|---:|---:|
| 608 | 30.4 | 24.2 | 506.9 | 558.0 |
| 3,845 | 76.5 | 24.5 | 504.5 | 645.6 |
| 5,567 | 104.4 | 24.6 | 528.4 | 704.4 |
| 13,056 | 220.7 | 25.2 | 532.6 | 992.5 |

Both full-guide methods fit these cases. Blocking reduces retained distance
memory at a runtime cost; it is not evidence that dense full guides were too
large. The new default uses blocking to bound pairwise storage as guide sizes
grow. A saved `distance_chunk_size: null` selects the legacy dense calculation.

A fresh DGCNN/proposal model with the full 5,567-vertex chick guide also completed
supervised and Chamfer backward: **116.0 MiB** peak allocated, **744.9 ms** median
step, finite nonzero alpha gradient. This is an execution check with randomly
initialized model weights, not reconstruction evaluation.

Peaks are PyTorch allocated memory, not total device occupancy; driver allocations
and other applications are excluded. No optimizer-state memory is included.
Repeated identical patches and these few guide sizes do not establish full-dataset
throughput or the maximum safe training batch. The foreground alternatives here
are saved diagnostic inputs, not a promotion of their mask rules.

GPU measurements: [full_guide_profile_seed0.json](full_guide_profile_seed0.json).
Original console: `tmp/pointneuron2/full_guide_profile_seed0_console.log`.

## Correctness checks

The repository suite passed **120 tests**, including blocked values and gradients
for both coordinate sets, ties, coincident points, padding, empty guides, CUDA
sampling, full-cache preservation and checkpoint compatibility.

The profiling dense path uses the historical `torch.cdist` matrix shortcut.
Floating-point perturbations of equal voxel-distance ties can alter alpha
gradients, even when scalar losses are almost identical. A separate verification
therefore compared blocked sampling with **direct dense Euclidean distances** on
all four real profiled patches. Probabilities agreed, hard selections were
identical, losses agreed, and alpha gradients agreed within the saved tolerance.
This verifies the reduction against the intended geometry without changing old
checkpoint behavior. See
[full_guide_direct_verification.json](full_guide_direct_verification.json).

## Reproduce the short diagnostics

From the real local `pointneuron2` folder, with fresh output paths:

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u scripts/pointneuron2/build_sampling_cache.py --root C:/Users/Seywe/Documents/GitHub/PointNeuron/data/gold166 --sample-index 0 --max-patches 4 --output-dir tmp/pointneuron2/full_guide_cache_new
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u scripts/pointneuron2/profile_full_guides.py --output-dir tmp/pointneuron2/full_guide_profile_new
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u scripts/pointneuron2/verify_full_guides.py --profile tmp/pointneuron2/full_guide_profile_new/profile.json --output tmp/pointneuron2/full_guide_profile_new/direct_verification.json
```

The GPU scripts read the existing training-only foreground-comparison artifacts;
they do not launch training. Source hashes and input hashes are saved in reports.

## Where stage 1 stands

Guide truncation is removed from the new default and the local gradient/memory
path is checked. **Segmentation quality remains unresolved**, especially the
chick case: retaining its original full guide gives only 19.1% reference-curve
proximity within three voxels. The background-corrected full guide gives 75.1%,
but includes denser, fragmented structures and has not established anatomical
correctness. Those are preprocessing diagnostics, not reconstruction F1.

Next, inspect automatic mask specificity/branch continuity and candidate support
on broader training examples, then settle the cache recipe before preparing
matched train/validation runs. Long training and PointNeXt remain subsequent
steps to agree on together.
