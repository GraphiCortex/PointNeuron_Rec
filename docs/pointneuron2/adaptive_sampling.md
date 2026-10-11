# PointNeuron 2.0: adaptive sampling implementation

Stage 1 is partially complete: fixed-parameter image-guided medial-axis sampling
runs in cache generation and inference, and the learnable sampler now trains
jointly with the existing proposal model and reloads for inference. A corrected
two-epoch CUDA smoke passed; matched reconstruction evaluation remains unfinished.
See `sampling_training.md` for the current results and supervision limitations.
The clean PN 1.0 baseline is complete; see `baseline_validation.md` for its
seven-neuron results. PointNeXt, the query decoder, and GATv2 are later stages.

## Implemented paths

`src/pointneuron/data/medial_sampling.py` builds a padded XYZ foreground crop,
extracts its medial axis with Kimimaro 5.8.5, and samples a fixed number of image
voxels using a distance prior. The guide uses all foreground voxels in the patch;
neither reference SWCs nor predicted proposals supply guide coordinates.
Disconnected components remain separate. Dust removal can yield an empty guide,
in which case sampling becomes uniform. Extraction errors and oversized crops
raise errors; they are not silently relabeled as successful skeletonization.

The prior is `(1-u) * softmax(-alpha*d) + u/N`, where `d` is distance to the nearest
image skeleton vertex and `u=0.25` reserves uniform exploration probability. This
is a probability mixture, not a guaranteed quota of uniform points. Weighted
Gumbel top-k selects unique candidates without replacement; only undersized
foreground pools are padded. Outputs retain voxel XYZ and raw intensity, matching
the existing encoder interface.

`configs/pointneuron2/medial_sampling.json` records every exposed parameter.
Spacing defaults to `[1,1,1]`, explicitly using voxel units until calibrated
source-specific spacing is available. Kimimaro receives physical XYZ spacing;
its vertices are converted back to global voxel coordinates exactly once.
The padded-crop cap is 16 million voxels, not a bound on total algorithm memory.
Kimimaro's behavior and physical units follow its
[official documentation](https://github.com/seung-lab/kimimaro).

`src/pointneuron/models/adaptive_sampling.py` implements a learnable
decay parameter, masked Gumbel-softmax selection, and symmetric squared Chamfer
loss. Forward selection returns actual candidate points and unique row indices;
backward selection uses a biased straight-through surrogate. Image extraction and
hard masks are discrete. Evaluation uses seeded weighted Gumbel top-k, retaining
the uniform exploration probability and reproducibility per record. The training
relaxation and hard evaluation selection still need an ablation. The new
`scripts/pointneuron2/train_sampling.py` connects sampler, encoder and proposal
head; `sampling_pipeline.py` restores sampling settings and weights at inference.
The bounded cache retains candidates and guides, rather than final selected
points, so the sampler receives training gradients. Legacy fixed-budget cache
sampling remains preprocessing.

The relaxation requires bounded candidate and guide pools: distance storage scales
as `B*N*M` and relaxed selection storage as `B*K*N`. It must not be applied directly
to a full dense volume. Chamfer alignment to a noisy extracted skeleton alone is
not sufficient supervision; downstream losses and ordinary foreground exploration
will be retained when building the joint PN 2.0 model.

## Run

The optional project dependency extra is `pointneuron2`. The verified local
environment is `tmp/pointneuron2/sampling_env`, created with access to the existing
CUDA Torch installation and isolated Kimimaro dependencies. Its executable is
`tmp/pointneuron2/sampling_env/Scripts/python.exe`. Heavy outputs stay under ignored
`tmp/`; source code, compact configurations and results live outside it. The
new development files are not yet committed to Git.

From the `pointneuron2` worktree:

```powershell
python scripts/pointneuron2/sampling_probe.py --root "C:/Users/Seywe/Documents/GitHub/PointNeuron/data/gold166"
python scripts/build_training_cache.py --root "C:/Users/Seywe/Documents/GitHub/PointNeuron/data/gold166" --sample-index 0 --patches-per-sample 3 --patch-radius 48 --max-points 512 --min-points 256 --threshold-fraction 0.2 --center-strategy coverage --point-sample-strategy medial --medial-sampling-config configs/pointneuron2/medial_sampling.json --output-dir tmp/pointneuron2/medial_cache_smoke
```

Use a fresh output directory for new experiments. The probe refuses overwriting
its report and restricts both neuron index and identity to the training partition.
The legacy cache builder retains its existing overwrite/resume semantics; resume
now checks medial parameters before accepting cached records. The cache builder
still uses SWCs as labels and retains labeled patches, so this cache smoke is a
training diagnostic. It is not a held-out inference input protocol.

`aggregate_proposals.py` and `run_end_to_end.py` accept `--point-sample-strategy
medial` and `--medial-sampling-config`. For a sampling experiment, generate new
proposals in a fresh directory; reused proposals do not test the new sampler.
The inference path retains the existing proposal model until PointNeXt and query
decoding are implemented. A frozen checkpoint smoke checks tensor compatibility,
not the quality of a model retrained with medial sampling.

## Validation and next stages

All 93 repository tests pass in the verified environment. The synthetic tests
cover physical XYZ conversion, a thin branch, disconnected
components, dust fallback, memory caps, deterministic sampling, unique candidates,
padding, numerical stability, CPU/CUDA gradients, and downstream gradient flow.
Existing PN 1.0 tests remain part of the suite. `sampling_probe_seed0.json` stores
three image-only patch comparisons on training neuron 0, with dataset/configuration
provenance and source hashes. It records guide-relative metrics; it does not use
the reference SWC to calculate them. The guide itself can contain image noise.

In the three-patch training probe, mean sampled-point distance to the image axis
was 1.45/3.75/1.45 voxels for medial sampling versus 2.65/5.23/2.77 for random.
Image-axis coverage within three voxels was 87.7%/32.8%/96.8% versus
77.9%/23.4%/87.0%. These results are expected for a sampler biased toward that
axis and do not establish anatomical accuracy. The dense middle patch remains
poorly covered at 512 points. All sampled indices were unique foreground voxels,
and all three bounded gradient probes returned finite nonzero alpha gradients.

The cache smoke retained two labeled patches out of three requested, each with
512 four-channel points. CUDA inference on two image patches completed with 185
finite proposals and retained the medial configuration in artifact metadata.
That tiny patch subset cannot support a whole-neuron reconstruction claim.

The newer corrected joint-training smoke uses eight unfiltered image patches
(four train, four validation), with 1,024 candidates and at most 512 guide points.
Only one training patch and two validation patches contain local SWC nodes.
Unannotated patches contribute image-guide loss only, without invented negative
proposal labels. Alpha receives finite nonzero gradients in all four training
batches and changes from 0.25 to 0.249359. Reloaded inference produced zero
proposals above score 0.5 on two training-image patches; this freshly initialized
two-epoch model is not a usable reconstruction model. This is a different check
from the earlier fixed-medial smoke using a trained baseline checkpoint.

Two annotated patch projections are saved in `sampling_inspection.png`. The guide
includes many structures outside the local reference, and some thin reference
branches appear weakly represented. These projections do not prove full 3D
alignment or distinguish noise from unannotated neurites. The guide is a prior,
not ground truth. Current reference diagnostics omit edges crossing patch
boundaries and are not whole-neuron anatomical accuracy metrics.

Next finish stage 1: assess guide fidelity and candidate coverage, then evaluate
random, spatial and medial sampling with the same encoder, point budgets and
validation neurons. The new radius-48 unfiltered cache protocol differs from the
older baseline cache, so it needs a freshly trained matched random control.
Measure reference branch/segment coverage
and final reconstruction quality. Retain failures and runtime costs. Then implement
PointNeXt, a DETR-style query decoder with Hungarian matching, and multi-scale
candidate edges with GATv2. Reserve the eight test neurons for final evaluation.

Dependency packaging note: Kimimaro is distributed under GPL-3.0-or-later. Its
license and the project's distribution license must be recorded in release
packaging; no third-party implementation source is copied into this repository.
