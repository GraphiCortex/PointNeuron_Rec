# Stage2 CPU tree experiment — approved implementation, not yet run

This protocol implements the user's Stage2 approval and subsequent answers:
keep the handcrafted node selector and representatives identical in both feature
arms; add PointNeXt to **edge features only**. Require its two-percentage-point
development segment-F1 gain at **two voxels**, alongside every six-voxel floor.
No GPU, GATv2, automatic extension, new development labels or reserved-neuron access.
Timed preparation, diagnostics and fitting are user-local launches.

The experimental arm departs from the PDF's Transformer node proposal stage,
as explicitly approved earlier. Professor discussion remains pending. This is
a tree selector experiment on the strong image-derived candidate pool; it does
not promote an alternative complete product pipeline. DraftB remains rejected;
DraftA remains held. Earlier checkpoints, failures, label packets and audits stay
unchanged. C1 is a new representative-conditional comparison, not DraftB.

## Frozen inputs and representation

Use the saved Stage1 union and edges: classical, Sato-top-512 and broad-run learned
candidates merged within one voxel, with source flags. Retain the existing k6/k10
and inherited classical/forest edges. Raw input points are not added as candidates.
No node moves, new neighbors, rewiring or reference-based inference roots.
The broad head remains a failed historical model retained as one frozen source.

Fit a handcrafted node trace-support classifier on training nodes only. Node labels
remain the original Stage1 labels: support within two native voxels; farther known
nodes negative; ambiguous/coincident and reference-empty cases unknown. This is
trace agreement with the supplied annotation, not biological foreground or unique
representative probability. Imbalance-weighted scores are not calibrated posteriors.

Eligibility is p>=.5. Greedy inclusive radius-two NMS is ordered by source priority
**classical > Sato > learned**, then descending Sato response, descending selector
probability, XYZ and index. Combined flags use the highest source priority present.
This is the user's approved image-only ordering, different from the historical
oracle's probability-first ordering. Remove incident edges of suppressed nodes;
never regenerate an edge. Both feature arms share the same retained node set.

## Actions and order

1. **prepare:** image-only node features on 611 reference-present packets
   (460 training/33 neurons; 151 development/seven neurons). Decode original raw
   image bytes only for raw-intensity profiles, verifying source/decoded hashes
   and retained sampled intensities. Reuse the retained Sato guides; no filtering
   or Phase1 restart. Save crop halo12, with image bounds applied. Reject any
   original decoded image exceeding 2,000,000,000 bytes; this is an input cap,
   not an estimate of peak process memory. No SWC geometry, fit or label creation.
2. **policy:** fit the single final handcrafted node model on all33 training
   neurons, freeze its model and all611 selections before evaluation. On151 dev
   boxes report segment F1_2/F1_6, resampled F1_2/F1_6 and length for (a) the
   image-only Sato-cost forest, and (b) the reference-assisted edge oracle on
   these frozen image-only node representatives. The latter uses saved Stage0
   edge scores only, never oracle node eligibility/order. This diagnostic selects
   nothing and has no quality gate. Compare with the compact union oracle's
   **88.52% segment F1_2,64.80% resampled F1_2,1.18475x length**. Its ordering is
   a different diagnostic policy, so these are context, not an achievable guarantee.
3. **compare:** perform the training-neuron C0/C1 comparison, then the feature
   comparison, then freeze both final edge models before final development scoring.
   The handcrafted model is scored first; failure stops before PointNeXt development
   scoring. A passing handcrafted model permits the one frozen PointNeXt scoring.

## Image-only features and fixed CPU model

The21 node channels are Sato, intensity, source flags/counts, nearest raw-input
distance/intensity, full-pool spacing/density, local covariance geometry/tangent,
raw-vote support within2, crop-boundary distance and outside-image flag.
The63 handcrafted edge channels include length/absolute direction, straight Sato
cost, Sato/intensity profiles at .5-voxel midpoint intervals, actual bounded Sato
geodesic cost, endpoint tangent alignment, mean/difference node channels, and
intervening-candidate counts and local spacing on both full and retained sets.
Intervening means another candidate within1 of the segment with strictly interior
orthogonal projection; **all candidates**, not reference-positive witnesses.
PointNeXt adds64 mean/absolute-difference endpoint channels from the completed
frozen32-channel export, for127 edge channels. It never changes node selection.

Geodesic:26-neighbor MCP_Geometric on the rounded-endpoint bounding box plus two
voxels, native unit spacing, cost1+4*(1-normalized Sato), maximum32,768 cells.
Outside or oversized searches yield NaN plus a missing flag; no straight-cost
substitution. Report unavailable counts. Raw profile interpolation uses float
values, not integer-rounded interpolation. No reference distance, projection,
branch, component or label is an inference feature. No selector probability is
an edge feature; node probability enters the decoder cost only.

Use scikit-learn1.6.1 HistGradientBoostingClassifier, learning_rate=.05,
max_iter=128,max_leaf_nodes=15,min_samples_leaf=32,l2_regularization=1,
max_bins=127,early_stopping=False,random_state=0,categorical_features=None.
No sweep, threshold selection, calibration or early stopping. Fixed .5 node/edge
thresholds. Four CPU threads; CUDA disabled; Torch is not imported. Use the
separate locked `tmp/stage2_cpu_env`, preserving both older environments.
[Official tree API](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
and [MCP API](https://scikit-image.org/docs/0.25.x/api/skimage.graph.html).

Training is capped deterministically per neuron/class:6,000 node examples and
10,000 edge examples, SHA256(seed0,case,row identity) ordering. Unknowns excluded.
Each neuron receives equal total weight, split equally between its available
classes. A neuron with one known class contributes that class; the combined fit
must contain both classes. Counts and class weights are disclosed, not hidden
as a change in label semantics. No classification accuracy is reported.

## C0/C1 and conditional selector CV

For each of33 leave-neuron-out folds, fit the node model on32 neurons and apply
that model to all460 training packets. Edge-training representatives therefore
use in-sample node predictions for those32 neurons; the held-out neuron's node
predictions are outer-held-out. Fit edges on the same32 only. Never reuse the
global33 node model in a CV fold. Report this training/held-out representation
shift. PointNeXt and handcrafted edges use exactly the same fold representatives.

- **C0:** original Stage1 full-pool edge labels, sliced onto retained endpoints.
- **C1:** call the original training-only adjacency labeler on the same retained
  induced graph. Consecutive retained projection groups may become positive when
  an intervening candidate was suppressed. Node tolerance2,ambiguity distance.25,
  projection separation/group threshold.5,path bound32,junction anchor<=2 remain
  unchanged. Original unknown edges remain unknown, including fragmentation and
  junction-transition uncertainty. No reference adds or moves any inference edge.

Save new C1 labels separately under this experiment, training partitions only.
The original460 label packets remain unchanged. Compare paired decoded segment
F1,resampled F1 and length per held-out training neuron, then neuron-macro summaries.
Choose among contracts with pooled CV length<=1.25 by segment F1_2, then F1_6,
then resampled F1_2, then stable contract ID. If neither meets the length limit,
stop before final development scoring. Do not compare AP across contracts.

Under that training-selected contract run PointNeXt-edge CV. Report AP per neuron
for **the same chosen contract** in the two feature arms, with unknowns masked,
alongside decoded metrics. No development edge-label packet or AP is generated.
The upstream encoder AND learned candidate positions have seen all33 training
neurons. As explicitly accepted by the user, CV is **selector-only conditional on
that frozen upstream system**, not strict full-pipeline CV or unseen-neuron evidence.

## Decoding and measurement

Reuse the probability-only decoder unchanged: eligible-component spanning forest,
edge cost -log(edge p)+.5*(-log(node p_u)-log(node p_v)), then one-pass leaf-to-
junction spur pruning for paths shorter than2. Isolated eligible nodes are retained.
Spanning eligible components is explicit; unconstrained positive-cost optimization
would choose an empty graph. Root each SWC component from image-guide response.
Radius1/type0 are placeholders, not supervised biological radii/types.

Reuse exactly the matched classical segment scorer: clipped box geometry,
.5-voxel midpoint quadrature,2/6-voxel tolerances,native XYZ spacing(1,1,1).
Uniformly resample maximal skeleton chains every2 voxels, include structural
endpoints/junctions/isolates once by topology, then maximum-cardinality bipartite
matching at2/6. Pool counts/length within neuron, then neuron macro. Length gate
uses pooled predicted/reference length; also report per-neuron and macro ratios.

Report final training fit (ungated), conditional CV and development, their gaps,
and fraction of compact-oracle headroom above classical captured (diagnostic only).
Reference-only risks after frozen graph export: different-component edges,
same/adjacent/nonincident reference-chain pairs, conditional shortcut flag
reference path>2*chord+4, plus uncovered intervals/length/longest at2/6. Ambiguous
and fragment relations stay unknown. Cross-chain transitions can be valid
junctions; these rates do not certify biological wrong joins. Uncovered intervals
measure geometric breaks, not full connectivity correctness.

## Final gates, fixed before fitting

| Development requirement | Two voxels | Six voxels |
|---|---:|---:|
| Segment neuron-macro F1 | >=59.33% | >=74.14% |
| Strict neuron wins over matched classical | >=5/7 | >=5/7 |
| Resampled neuron-macro F1 | >=52.43%,and actual classical+5pp | >=59.72%,and actual classical+5pp |

Pooled length ratio<=1.25x is required. Matched classical is54.33/69.14 segment,
47.43/54.72 resampled. Freeze both final models on all33 before any final dev
score. Handcrafted failure stops without PointNeXt dev scoring. PointNeXt must
pass all those gates AND add>=2pp segment F1_2 over handcrafted to stay; no gain
requirement at6 beyond clearing every6-voxel floor. Training fit is not gated.
The7 development neurons are **repeatedly inspected**. This is not a new untouched
validation set. A pass remains per-box evidence, not final whole-image reconstruction.

Stop on any required failure, execution cap or resource failure; no tuning after
development, no automatic retry/extension and no automatic next stage even if all
pass. STOP.json blocks further actions. Partial folders and hashes are preserved;
an interrupted action requires review, not deleting its lock to resume blindly.

Resource ceilings:prepare7200s,policy3600s,compare21600s. Actual runtime is unknown.
After the first C0/C1 fold project remaining cost as
50*max(0,elapsed-cold_image_feature_time)+3*cold_image_feature_time+1800 seconds.
If projected cost exceeds21600s, stop before further fitting. This is a conservative
resource diagnostic, not an accuracy selector. The outer watchdog enforces the
wall cap regardless. No training has been launched while writing this protocol.
