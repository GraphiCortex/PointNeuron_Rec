# Per-crop connectivity prototype implemented; CPU execution checks pass

17 October 2026. The user agreed to proceed from the connectivity design. The
first per-crop graph/scorer/label/decoder implementation is complete, with a
matched pairwise control. **42 targeted tests pass.** A separate fixed CPU
execution check completes in 7.02 seconds excluding imports. There are zero
optimizer updates, upstream model forwards, GPU workloads or development/test
reconstruction evaluations.

This is an implementation milestone. Neither GATv2 nor the control has been
trained; no learned reconstruction improvement or learnability is established.
The main pipeline still needs usable upstream proposals and reference-free
whole-neuron integration. Phase 1 and existing encoder/proposal/checkpoint
policies are unchanged.

## Implemented interfaces

| Component | Actual behaviour |
| --- | --- |
| Proposal adapter | Retains every query's XYZ, radius, logits, local centers and 128 Transformer features; checks explicit provenance and feature digest |
| Candidate graph | Reference-free symmetrized kNN-16 union radius-eight pairs, with stable IDs settling exact-distance ties; no spatial merging or silent node cap |
| Edge supervision | Transfers existing assignments; trusts placements within two voxels; distinguishes full-reference fragmentation from crop-induced disconnection |
| Native GATv2 | Two residual LayerNorm blocks, four 32-channel heads; no BatchNorm/dropout; geometric edge attributes |
| Edge classifier/control | Symmetric 264 -> 128 -> 1 head; pairwise control shares the initial stem/head weights and has no graph message passing |
| Loss | Known-positive/negative class means, equal supervised-crop averaging; unknown edges masked; all-unknown result explicitly signals no optimizer step |
| Decoder | Existing node cutoff .5, edge cutoff .5, cycle rejection, isolated-node preservation and deterministic forest orientation |
| SWC export | Parent-before-child, no forced bridges, checked geometry/topology round-trip; generic node type/root convention |
| Checkpoints | Separate strict GATv2/pairwise schemas; optional optimizer/scheduler/RNG snapshots for a future trainer |

The scorer does not import training-target code and its learned inputs do not
contain GT roles, assignments, fragment IDs or explicit query IDs. The adapter
records an upstream normalization policy; it does not load/convert the upstream
model. Checkpoint/cloud hashes must additionally be verified by a future real
feature-cache writer. Stable node IDs are only indexing/tie-breaking metadata.

## Synthetic execution results

Both scorers receive identical hashed inputs: **512 synthetic proposals** and
**4,807 undirected candidates**. The labels alternate only to exercise loss and
backward computation. They are not a neuron learning task. No optimizer exists
in the execution runner and no weights are updated.

| Scorer | Parameters | Finite parameter gradients | Train/eval max logit delta | Disk-reload max delta |
| --- | ---: | --- | ---: | ---: |
| Native GATv2 + edge head | 119,041 | Yes | 0.0 | 0.0 |
| Pairwise edge-head control | 51,201 | Yes | 0.0 | 0.0 |

The control matches input data, stem/head initial weights and interface, not
total parameter count. No conclusion about quality, optimization or comparative
speed follows from this check. Recorded times include reload/verification and
are not GPU training estimates.

The native attention equation and input gradients agree with an independent
float64 loop reference (declared forward tolerance 1e-10, gradient tolerance
1e-9). A constructed graph verifies that attention ranking can depend on the
receiving query. This tests the declared GATv2 formulation, not official
compiled-kernel parity, training-recipe reproduction or published throughput.

## Fixed reference-score decoder results

For these cases, both nodes and edge scores are deliberately reference-defined.
The new GATv2 model is NOT run on the reference cases. Scores of +12 on known
edges and -12 on other candidates test the decoder with ideal inputs. Those
oracle scores do not turn unknown edges into negative training labels.

| Training-reference case | Nodes | Known edges retained | Components exported | Export/topology |
| --- | ---: | ---: | ---: | --- |
| Neuron 92, crop 4 | 16 | 15 / 15 | 1 | Pass |
| Neuron 86, crop 3 | 418 | 404 / 404 | 14 | Pass |
| Neuron 92, crops 1+2, already consolidated by oracle | 73 | 64 / 64 | 9 | Pass |
| Neuron 66, full reference | 1,541 | 1,341 / 1,341 | 200 | Pass |

All known edges and branch incidence survive the new score decoder and SWC
round-trip. The last case stress-tests a forest larger than one crop's query
capacity; it is not a 1,541-query model fit or a proposed global cap.
The joining case receives the previous oracle-consolidated node graph; it does
not implement or evaluate automatic cross-crop matching.

In the fragmented-reference label check, **10,015 candidates remain unknown**,
while 1,341 are positive and 3,098 are known same-tree nonadjacencies. Thus the
new label policy does not silently treat every inter-fragment candidate as
background. Known same-tree negatives still assume the annotated path topology
is correct, and exact target adjacency remains the chosen representation.

## Tests and preservation

The 42 tests cover the new modules plus the existing reference-target/oracle
checks. They include:

- Native attention values/gradients and receiver-conditioned rankings.
- Forward/backward to proposal geometry, confidence and Transformer features.
- Permutation, translation, endpoint symmetry and metadata independence.
- Exact-distance ties, coincident nodes, empty/single-node graphs and malformed inputs.
- Trusted/untrusted assignments, missing positive candidates, unmatched queries,
  unknown fragments, crop-induced disconnection and class/crop loss normalization.
- .5 cutoff inclusion, low-confidence rejection, cycles, isolated nodes,
  deterministic root/ties, empty reconstruction failure and SWC overwrite refusal.
- Unchanged shared GATv2/control initialization, train/eval consistency, strict
  disk reload/schema separation and independent checkpoint-state snapshots.

CPU tests use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because an unrelated installed
napari plugin is incompatible with the environment's NumPy. No package or
environment changes were made. Checkpoint snapshot tests instantiate an
optimizer/scheduler but perform no optimizer update; exact training continuation
has not been tested. The saved execution checkpoints contain untrained weights
and explicitly have no resumable optimizer state.

All protected upstream sources/configs/partition/audit and the three selected
training SWCs match the preceding oracle hashes. New code matches its execution
report and saved source snapshots. Inputs to both synthetic scorer modes match
exactly by hash. No source volumes or development/test SWCs are read.

## Remaining research decisions

The existing confidence gate can still reject correctly placed proposals.
Scoring all queries does not repair that gate. Unknown/unmatched proposals can
also affect message passing and inference, despite being masked from edge loss.

The two-voxel label-reliability rule must be reported on actual upstream outputs.
A Hungarian match alone does not mean a target was found. Poor geometry can
leave too little supervised connectivity; low edge loss in that situation would
not establish learnability. The two previously fitted crops contain no junctions.

Exact adjacency is an initial baseline: the edge loss does not teach arbitrary
longer connections to replace missing intermediate targets. Replacing that
representation or allowing connectivity to rescue low-confidence nodes would
need a separately discussed comparison.

Next, define one bounded real learning comparison against the pairwise control,
with upstream geometry/feature availability made explicit before fitting. Agree
the dataset, normalization policy, frozen/joint settings, update budget, success
criteria and local GPU timing before preparing the visible training command.
No optimizer recipe, joint-loss weight or actual fit is queued here.

Automatic crop correspondence, full-image coverage, cross-crop candidates,
biological soma/root direction, radii accuracy and a matched end-to-end PN1
comparison are still required for the intended system. There is no new
production selector, reconstruction model or default promoted by these checks.

## Files and reproduction

- [Agreed design](gatv2_connectivity_design.md)
- [Graph adapter/candidates](../../src/pointneuron/graph/proposal_graph.py)
- [Training-only edge labels/loss](../../src/pointneuron/data/connectivity_targets.py)
- [GATv2 and matched control](../../src/pointneuron/models/gatv2_connectivity.py)
- [Forest decoder/export](../../src/pointneuron/graph/connectivity_decoder.py)
- [Visible CPU runner](../../scripts/pointneuron2/check_connectivity.py)
- [Fixed execution-check config](../../configs/pointneuron2/connectivity_check.json)
- [Complete visible execution summary](connectivity_execution_seed0/summary.json)
- [Preservation/test review](connectivity_execution_seed0/review.json)

Generated untrained checkpoints, oracle SWCs and source snapshots remain in
ignored `tmp/pointneuron2/local_runs/connectivity_execution_20261017`. Test XML
is in ignored `tmp/pointneuron2/connectivity_execution_tests_20261017.xml`.
The runner refuses an existing output directory. For a reproduction, use a
fresh directory in this visible command:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\check_connectivity.py `
  --config .\configs\pointneuron2\connectivity_check.json `
  --data-root "C:\Users\Seywe\Documents\GitHub\PointNeuron\data\gold166" `
  --output-dir .\tmp\pointneuron2\local_runs\connectivity_execution_repeat
```

This command runs CPU execution checks, not training or image reconstruction.
