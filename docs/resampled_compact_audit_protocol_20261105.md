# Fixed resampled-skeleton metrics and compact-oracle CPU audit

User direction: DraftB REJECTED, DraftA HELD. No Stage2 feature/model or label
contract implementation. Explicitly authorized CPU diagnostics only, selecting
no architecture, threshold, label contract or inference policy. New compact
graphs are diagnostic outputs; saved candidates/labels/graphs stay unchanged.
The original Stage0 gate PASSED at90.42% geometric F1_2 with2.39785x pooled
reference length (approximately2.40x). Keep that pass and its limitation on record.

## Resampling and one-to-one metric, fixed before results

- Native XYZ units, effective spacing(1,1,1). Clip both graphs to the original
  closed voxel-cell crop BEFORE resampling. Keep original in-box vertex IDs;
  clipping creates boundary IDs specific to the source edge. Never weld two
  disconnected/coincident branches by XYZ. Evaluation resamples are not new
  training labels; development samples remain ephemeral in scoring memory.
- Resample maximal UNDIRECTED degree-two chains every2 native voxels in
  arclength, retaining critical endpoints/junctions ONCE per topological identity.
  Include the terminal endpoint even for a remainder shorter than2; isolates
  count once. This stated endpoint convention is part of the metric. Branch
  junctions are not duplicated per incident chain. Only connected exact-zero-
  length edges collapse for sampling, not disconnected coincidences.
- Deterministic chain orientation: lexicographic endpoint XYZ then vertex ID.
  Pure cycles start at lexicographic XYZ/ID with deterministic neighbor order,
  sample every2 and do not duplicate the closure. Arbitrary degree-two node
  subdivision does not create new sampling phases. No geometric NMS is applied
  by the metric itself: duplicated paths must retain their multiplicity penalty.
- At tolerance2 AND6 use maximum-cardinality bipartite matching of reference
  and predicted samples. Precision=matched/predicted,recall=matched/reference,
  F1=2*matched/(reference+predicted). Count matches within each box, pool counts
  within neuron, then macro across neurons. No matching across crop boundaries.
  This is geometric one-to-one agreement, not anatomical branch correspondence.
- Add length ratio=clipped predicted arclength/clipped reference arclength,
  both pooled and neuron-macro. Reuse existing segment scores only when graph/
  reference crop lengths match. No original metric report is overwritten.

## Historical scope

5361 saved graph entries,4853 distinct graph-hash/case identities before the
611 new compact results: all four Stage0 oracle arms; training label/geometric
oracles; old held-A/rejected-B diagnostic SWCs (METRICS ONLY); four-fit original,
supported variant/raw/Sato forests; development transfer and broad forests;
cached classical raw/pruned graphs on all611 reference-present boxes; cached
reference-edge eight-voxel representation oracle. Training classical source
graphs are scored as additional baselines without new extraction/filtering.

Raw/pruned classical matched DEVELOPMENT comparisons use exactly the151 known
boxes/seven neurons. Annotation-unknown regions are not relabeled/scored as
new negatives. Old full-grid length accounting remains in its prior ledger.
Candidate-edge unions are opportunity diagnostics,not skeleton outputs; their
length ratios remain available but they are not silently called decoded graphs.
Isolated proposal-only experiments are not converted into invented skeletons.
Every actual graph output in the inventory gets the new metrics, with exact
artifact-hash/crop cache reuse for identical geometry. The inventory and per-box
CSV make coverage and scope reviewable.

## Fixed duplicate handling BEFORE spanning

Run ONLY on Stage0 UNION development oracle151 boxes and original saved-label
training oracle460 boxes. Reuse the original diagnostic oracle probabilities
(or saved training labels) and verify an exact replay of each original oracle
SWC before adding NMS. These probabilities NEVER enter real model inference.

Eligible nodes have oracle probability>=.5. Greedy nontransitive NMS within
Euclidean distance<=2 native voxels, ordered by decreasing node probability,
then source priority classical>Sato>learned (lowest source bit for multisource
nodes), then XYZ/index ties. Keep real coordinates. Oracle node probabilities
are binary in these arms, so source/XYZ ties determine their eligible order.
Suppress incident edges of removed nodes. NO edge rewiring, endpoint remapping,
new kNN edges, coordinate averaging or relabeling. This tests the SAME fixed
candidate/label graph under duplicate handling,not a newly repaired graph.

Then use unchanged .5 edge eligibility,node-confidence -log costs,eligible-
component spanning forest,one-pass leaf-to-junction spur<2,image-only roots,
placeholder radius/type. Report original/compact segment F1_2/F1_6,resampled
F1_2/F1_6,length ratios and NMS/edge/decoded counts. Isolates remain visible to
the resampled metric even when they supply no segment coverage.

Apply the ORIGINAL branch rules to COMPACT UNION macro segment F1_2:
>=.80 proceed;>=.65 and<.80 only alongside candidate-recall work;<.65 stop.
No new resampled-F1 or length threshold is invented after results. A branch
decision is not adoption of NMS or a label contract. No automatic training or
candidate repair follows. The old uncompact Stage0 pass stays on record.

## Draft and comparison boundary

After results, draft Stage2 supervision with image-only candidate-geometry
features (intervening-candidate count/local spacing), WITHOUT implementing it.
Define a training-only leave-neuron-out comparison on decoded F1 and length
ratio, keeping PointNeXt/upstream conditioning disclosed. No trained fold
comparison exists without implementing/fitting those models; do not call
per-neuron reference-assisted oracle results trained selector CV. DraftB is
excluded from prospective contracts; DraftA remains held until user steers.

## Execution

CPU only,NumPy/SciPy runtime,no Torch/model/GPU. Output
`tmp/resampled_compact_audit_20261105`,3600-second cap,progress/cached records,
no automatic retry/extension. Estimated several to tens of minutes,unmeasured
for this new matching workload. Explicit resume can reuse completed matching/
compact records after an interruption; it does not authorize an automatic cap
extension. A scoped awake request covers the active CPU job and is released.

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\audit_resampled_compact.py'
```
