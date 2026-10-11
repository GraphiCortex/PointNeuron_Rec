# PointNeuron 2.0 scripts

These scripts and their dependencies are local files on your computer. This is
the permanent folder for PN 2.0 entry scripts; new stage-specific runners belong
here. Generated caches, checkpoints and logs stay in ignored `tmp/pointneuron2`.
Development snapshots are kept on the `pointneuron2` branch. Generated caches,
environments, logs and checkpoints remain outside those snapshots.

Current work: [reader/range/coordinate repairs and measured evidence](../../docs/pointneuron2/root_cause_findings.md).
`audit_root_causes.py` runs bounded diagnostics. Old sampling caches are
historical data and cannot be reused for new training. The staged runner accepts
`--config configs/pointneuron2/repair_comparison.json` at preparation for the
new corrected-data comparison; no stage automatically starts the next.
`probe_repaired_foreground.py` follows a single corrected image crop through
automatic masks, full guides, candidates and 512-point inputs; it does not train
a model or score a connected reconstruction.

Latest: [corrected training-mask results and next proposed input control](../../docs/pointneuron2/repaired_foreground_comparison.md).
`compare_foreground.py` accepts `--artifact-dir` to keep raw arrays in ignored
storage while retaining reports and figures in the visible documentation folder.
`summarize_repaired_foreground.py` compares saved guides at exactly matched counts;
it performs no extraction or training.

`compare_mixed_candidates.py` compares bright, permissive and preset mixed pools
on corrected saved training crops with matched seeds and fixed budgets.
See [the measured tradeoffs](../../docs/pointneuron2/mixed_candidate_comparison.md).

| File | Purpose | Current status |
|---|---|---|
| `baseline.py` | Prepare the neuron-disjoint split and run fresh PN 1.0 baseline training | 30-epoch baseline and seven-neuron validation completed |
| `sampling_probe.py` | Compare random, spatial and image-medial sampling on three patches of a training neuron | Completed implementation probe; not a reconstruction benchmark |
| `run_local.ps1` | Launch either runner with local Python, a fresh output directory and saved console logs/status | Local launcher |
| `build_sampling_cache.py` | Preserve bounded image candidates, full medial guides, and separate SWC labels | Corrected reader/range checked on six new patches; historical caches are for read-only analysis |
| `train_sampling.py` | Jointly train sampler and proposal model; random-input control uses the same pools | Two-epoch CUDA integration smoke completed |
| `sampling_ablation.py` | Prepare one cache, train three matched arms, then reconstruct/score in separate visible stages | Three training arms and all 21 user-launched reconstructions completed |
| `summarize_ablation.py` | Validate completed metrics, audit saved proposal selection and render matched SWC projections | CPU audit and figures completed; no additional inference or training |
| `inspect_ablation_confidence.py` | Check every cached validation patch's proposal scores and changes versus initial alpha | Completed inference-only check on all three checkpoints; no reconstruction claim |
| `check_noise_guard.py` | Check bounded noise thresholds on exact saved training scores | Four mask-only regression cases passed; no new extraction or training |
| `evaluate_sampling.py` | Compare random/fixed/learned inputs against local validation-reference curves | Representation diagnostic; not whole-neuron reconstruction scoring |
| `visualize_sampling.py` | Inspect image candidates, guides and SWC references in XY/XZ | Two annotated patches inspected; broader guide assessment remains |
| `inspect_sampling_checkpoint.py` | Overlay raw images, full guides, learned/random inputs and checkpoint proposal scores | Cached-patch inference diagnostic; no training or connected reconstruction |
| `audit_sampling.py` | Check original image support, full-guide/pool coverage and boundary-crossing references | Eight existing patches audited on CPU; see input audit report |
| `compare_foreground.py` | Compare fixed, automatic raw and background-corrected masks at identical locations and budgets | CPU comparison completed on 12 patches from three training neurons; no method promoted |
| `compare_candidates.py` | Compare candidate budgets/preparation and fixed selectors on saved masks/full guides | User-run six-neuron, 48-patch experiment completed without failures; no recipe promoted |
| `inspect_foreground_slices.py` | Read saved masks and show actual slices with depth-clipped references/guides | Four training cases inspected; contrast/clipping diagnostic available |
| `prepare_noise_pilot.py` | Test noise hysteresis on exact saved background scores | Four-case pilot completed; helps noisy lattice, mixed guide support |
| `profile_full_guides.py` | Measure local GPU memory and forward/backward time for capped/dense-full/blocked-full guides | Completed on four training patches up to 13,056 guide vertices, plus DGCNN/proposal backward; zero optimizer steps |
| `verify_full_guides.py` | Compare full-guide probabilities, selections, loss and alpha gradient against direct dense distances | Diagnostic reference for voxel-distance ties; no optimizer steps |
| `review_sampling.py` + `review_sampling.html` | Inspect image slices/3D skeletons, adjust extraction and save review decisions | Optional development diagnostic; no mandatory manual step; does not paint neurite masks |

Reusable algorithm implementations belong to the Python package:

- [Image-medial extraction and fixed sampling](../../src/pointneuron/data/medial_sampling.py)
- [Learnable sampler and Chamfer loss](../../src/pointneuron/models/adaptive_sampling.py)
- [Full-guide implementation and GPU results](../../docs/pointneuron2/full_guides.md)
- [User-run checkpoint visual inspection](../../docs/pointneuron2/checkpoint_inspection_full_guides.md)
- [Expanded mask/candidate results and contrasting overlays](../../docs/pointneuron2/preprocessing_comparison_expanded_results.md)
- [Noise-aware method, slice results and reproduction command](../../docs/pointneuron2/noise_aware_preprocessing.md)
- [Completed noise-aware comparison and identified failure](../../docs/pointneuron2/noise_comparison_results.md)
- [Next experiment: three matched training arms and commands](../../docs/pointneuron2/sampling_ablation.md)
- [Completed sampler pilot results and next reconstruction command](../../docs/pointneuron2/sampling_ablation_pilot_results.md)
- [Completed reconstruction results and proposed next diagnostic](../../docs/pointneuron2/sampling_ablation_reconstruction_results.md)
- [CPU mask/candidate experiment specification and reproduction command](../../docs/pointneuron2/preprocessing_comparison_expanded.md)
- [Training integration results](../../docs/pointneuron2/sampling_training.md)
- [Source-image audit and next steps](../../docs/pointneuron2/input_audit.md)
- [Automatic foreground results and reproduction command](../../docs/pointneuron2/foreground_comparison.md)
- [Automatic preprocessing: next proposed comparison](../../docs/pointneuron2/automatic_preprocessing.md)
- [Optional viewer and restart command](../../docs/pointneuron2/manual_review.md)
- [Sampling parameters](../../configs/pointneuron2/medial_sampling.json)
- [Sampling results and limitations](../../docs/pointneuron2/adaptive_sampling.md)
- [Clean baseline results](../../docs/pointneuron2/baseline_validation.md)
- [Current pipeline status](../../docs/pointneuron2/README.md)

Stage 1 remains partially complete. Learnable-sampler training and checkpoint
inference are connected and execution-tested. The longer matched reconstruction
comparison and broader image-skeleton assessment come before PointNeXt, the query decoder
and GATv2. The two-epoch model produced no accepted proposals at score 0.5; it is
an integration check, not a usable reconstruction model.

## Run on your machine

**Current discussion:** [completed reconstruction results and proposed next diagnostic](../../docs/pointneuron2/sampling_ablation_reconstruction_results.md).
The completed noise comparison exposed a collapsed-MAD failure. A separate
bounded mask check now passes, while the three training arms keep the same
original fixed-threshold inputs. Use `ablation-prepare`, then `ablation-train`,
then `ablation-evaluate` as separate terminal commands; none starts the next stage.
The viewer and historical reproduction commands below remain optional.

New caches retain the **full extracted guide** (`guide_points: null`). Encoder
input remains 512 points selected from 1,024 image candidates; these are separate
settings. Historical caches/checkpoints retain their saved caps. The four-patch
full-guide cache is an execution diagnostic with training patches only, so it is
not the train/validation manifest for the matched pilot. The pilot's cache
preparation and three five-epoch CUDA runs have completed through user-launched
commands, as has reconstruction evaluation of all three arms. Those commands
below now serve as reproduction instructions; another full run is not needed
to inspect the saved results.

From the original `Documents/GitHub/PointNeuron` folder, the local `pointneuron2`
folder is a real local checkout. The earlier Windows directory junction has
been removed, and all files, including ignored run artifacts, were transferred
and verified. The original checkout stays on `main`; the development checkout
is on `pointneuron2`. The parent checkout excludes this nested checkout locally;
source changes are tracked by Git inside the PN 2.0 folder.

```powershell
& .\pointneuron2\scripts\pointneuron2\run_local.ps1 -Action sampling-probe -ArtifactRoot $PWD.Path
& .\pointneuron2\scripts\pointneuron2\run_local.ps1 -Action sampling-smoke -ArtifactRoot $PWD.Path
```

That command uses the already configured local CUDA Python environment. It prints
the fresh run directory and saves `console.log`, `launch.json` and the probe's
JSON report there. Use `-Python` to supply another compatible executable. Heavy
runs use the same launcher, with `-Action baseline-smoke` or `baseline-train`;
these actions rerun the baseline and are not needed to continue sampler development.
Nothing is scheduled or sent to a remote compute service.

For a visible full-guide training check, run the `sampling-smoke` command above.
It prepares four train-neuron-0 and four validation-neuron-19 patches on CPU,
then runs two epochs on CUDA (at most two training batches per epoch). It prints
each batch's losses, alpha, elapsed time and current CUDA allocated memory. Logs,
run settings and the best checkpoint are saved under the printed local run folder.
This is a small training/checkpoint check, not a reconstruction-quality benchmark.
Direct trainer runs accept `--log-every 1` for the same batch visibility.

`-Action preprocessing-compare` reproduces the completed training-only CPU mask and
candidate experiment, with saved figures and live logs. It uses full guides,
six neurons and up to eight patches each. It does not launch GPU training or
modify the main recipe. See the linked experiment note for exact settings.

`-Action noise-compare -Comparison <absolute-prior-masks-comparison.json>` runs
the noise-aware mask comparison on exactly the prior patch boxes, then
checks 512 inputs from 1,024 uniform candidates. It is CPU-only and does not
change training settings. The linked noise-aware note records the mixed small
pilot results and reproduction command. The user-run six-neuron comparison is
complete: repeated-intensity fish images expose a degenerate noise-estimate
failure and selected-input coverage worsens overall. No alternative is promoted.

`-Action sampling-cache` prepares pools for all train/validation neurons, preserving
test. `-Action sampling-train -Manifest <absolute-manifest-path>` trains the sampler
with its saved cache configuration. The trainer also accepts `--strategy random`
for the matched control. Full runs need fresh output directories and local runtime.

You can also invoke the Python entry scripts directly from any working directory.
The old root-level PN 2.0 script names remain thin compatibility entry points.
Historical result files retain the original script paths/hashes recorded when
those experiments ran.
