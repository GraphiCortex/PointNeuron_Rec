# Stage1 preparation and frozen-encoder export — execution contract

Stage0 passed its original >=80% union geometric-oracle gate at 90.4232% F1
at2 (95.6720% at6). This authorizes the already agreed candidate-selection
arm's CPU preparation, not another dense-head run or a reconstruction claim.
The Stage0 sources, protocol, settings, outputs and historical failures stay fixed.

## Candidate construction

All460 reference-present training boxes/33 neurons use the exact Stage0 pool:
pruned classical graph + Sato-top512 + frozen broad-head nodes, greedy radius1
real-coordinate representatives, OR source flags and membership counts,
k6/k10 plus inherited classical/Sato/learned forest edges. No raw-input arm is
added. All611 pool packets are saved before training references are opened.
The151 development pools reuse Stage0 image-only packets, never oracle scores.

Reuse six training classical extractions (aliases22/32/41/75/85/126); extract
only missing27 from retained Sato guides using the preserved Otsu256/Lee3D/
26-neighbor/one-pass<2 spur recipe. No image filtering is repeated. New Sato
sources use the same score-ranked radius1 suppression, flat-index ties and cap512,
and the same fixed k6 image-cost forest as the preserved development source.
All saved broad graphs are reused. Reserved neurons remain inaccessible.

## Training-only trace-agreement labels

Node distance to the FULL trace <=2 is positive, farther is negative. A near-tie
alternative projection (within .25 voxel of nearest distance) whose trace-path
separation exceeds .5 voxel makes ownership unknown; disconnected coincident
components cannot share incident directions. Isolated reference vertices are
included. Empty reference geometry yields unknown labels. No crop-edge target
attractors are introduced. Only reference-present TRAINING boxes are prepared;
the label API rejects development. No development label packets are created.

Positive edges require consecutive projected positive groups along one
unambiguous reference chain, with path length <=32 native voxels (the bound
declared before Stage0). Group arclengths within .5 of the group's first member
nontransitively; co-group pairs are unknown. Adjacent groups allow their
cross-product edges; skipped groups or path length >32 are negative. An edge
touching any known-negative node is negative, including when its other endpoint
is unknown. Distinct components and unanchored junction transitions are unknown.
Unique within-fragment geometry can supply labels; across-fragment relations
cannot. Same-component nonincident chains are negative. This supervises supplied
trace agreement, not exhaustive biological foreground or geometric-oracle labels.

Radius1 merging can remove an exact junction coordinate. For TRAINING LABELS
only, assign a degree>=3 reference junction one distinct nearest unambiguous
positive candidate within2 Euclidean AND2 trace-path voxels (XYZ distance then
candidate index ties; junction external ID order). Its label membership spans
the incident chain endpoints. No candidate coordinate, inference edge, root or
selection is changed. Unrepresented junction transitions remain unknown rather
than teaching shortcuts. Report anchored and unanchored instances. These counts
test full-reference junctions per box; repeated/outside-box junction instances
are not unique cropped branch counts. Counts/masks/reasons are reported per
training neuron; overlapping boxes repeat candidate instances.

## Inference-only encoder export

After completed Stage1, optionally launch the separate export command. The
broad terminal checkpoint is immutable; load only its model weights, never
resume its optimizer/RNG or update weights. Eval/inference mode returns the32D
PointNeXt-restored feature at each2048 input point. Export source features and
candidate queries using deterministic 3NN inverse-Euclidean-distance weights;
copy an exact source at distance zero, source-index ties. Save neighbors,
weights and distances, including sparse/halo extrapolation. This interpolation
is image-only and is not retraining or a learned geometric refinement.

Export uses image-only data access and never opens labels, SWCs or oracle
probabilities. Check unchanged model weights/buffers and RNG, no gradients,
zero optimizer updates and checkpoint hash. CPU is default; an explicit user
`--device cuda:0` selects GPU inference. The assistant launches neither.

Both scripts are source/input bound, write under ignored PN2/tmp, print
progress, have one-hour wall caps, preserve partial outputs and stop without
automatic retry or model training. Explicit --resume reuses matching completed
records after interruption; it is not an automatic budget extension.

## What follows, and what does not

Preparation/tests establish construction and labeling execution only. Inspect
label counts/masking and feature distances before preparing selector training.
No Stage2 selector/model hyperparameters, CV sweep or training is implemented
or launched by this preparation. Handcrafted features (including actual
straight-edge raw intensity profiles and geodesic Sato costs), GBT training and
GATv2 remain subsequent work; cached input intensities alone must not be called
full edge intensity profiles. GAT still needs >=2pp F1_2 over frozen trees.

Selector-only leave-neuron-out CV is CONDITIONAL on frozen encoder AND learned
positions already trained on all33 training neurons, as the user explicitly
chose. This is not strict full-pipeline CV. Development is repeatedly inspected;
each frozen selector receives one scoring with no subsequent tuning. Final
classical+5pp macro floors at2/6 and >=5/7 strict neuron wins each remain fixed;
training metrics are reported, not gated. Stop on any failed criterion, no
automatic extension. Wrong joins, breaks, train/dev parity and oracle-gap capture
remain required reporting; no new numeric gate is invented here.

The Transformer departure is approved for this arm only, professor discussion
pending. Whole-image assembly, production graph stages and supervised radius
are still unbuilt; no per-box result establishes final full-neuron quality.
