# Classical-anchored editing contract — DRAFT ONLY

Drafting is authorized by the user's fixed headroom rule:(b) achieved61.7407%
development neuron-macro resampled F1_2 with the declared length constraint;
(a) achieved58.5846%. This document adopts no architecture,labels,features,
thresholds,training run or acceptance floor. Stage2 remains stopped,unchanged.
Professor discussion of the experimental Transformer departure is still pending.

## What justifies this draft,and its practical limit

The measured feasible oracle solution retains exact classical geometry and adds
only connected branches drawn from existing image candidates. It achieves segment
F1_2/F1_6=68.56/80.43%,resampled61.74/68.54%,pooled length.55105x. This passes
the drafting gate by1.74pp,not a learned-accuracy gate. Native2-voxel segment
recall is55.49% and resampled recall51.53%;adult zebrafish alias94 still has
resampled F1_2=27.29%. Shorter output is not complete reconstruction.

The oracle uses reference edge agreement to filter/order edges and reference-
assisted pruning. Its forest/path units are NOT an inference candidate set and
must never be copied into a training manifest as image-only candidates. The
bounded search is not a globally optimal ceiling. It shows a feasible editing
opportunity exists;it does not show a model can identify those edits.

## Proposed representation:select edits over a strong geometric scaffold

Preserve the original classical skeleton as an explicit base graph,with real
positions,edge identity and provenance. Propose keep/prune operations on its
maximal degree-two chains and attach optional image-derived branch paths.
Classical anchors and branch ownership replace the failed dense-pool point
selector as the proposed unit of responsibility. No network invents positions.

Candidate units must be constructed **from images alone**,before labels:

- Base chains come from deterministic classical graph geometry. Preserve bends;
  degree-two contraction may change storage,not the embedded polyline.
- Addition candidates use retained Sato/learned source geometry and existing
  neighbor edges. Association to a classical anchor must have explicit provenance,
  not coordinate coincidence with a different component or a clipping intersection.
- One anchored addition component must not silently join two existing classical
  components. New joins/gap closure would require a separate explicit proposal.
- Path enumeration and any forest/compactness policy must use image costs,not
  oracle edge probabilities,reference roots or the supplied gold branch IDs.
- Any long-chain subdivision or alternative branch enumeration must be fixed
  before its results. Their constants are not settled by this draft.

This changes the representation and learning task;it is not a restart of C0/C1
with a different threshold. It remains a departure from PDF Transformer node
proposals in the already approved experimental arm. The diagram's graph stage
would select scaffold edits;GATv2 is an optional later comparison,not assumed useful.

## Proposed supervision contract for discussion

Training neurons only. Gold provides polyline/path agreement and explicit
uncertainty masks. Trace agreement is not biological foreground or neuron-instance
identity certainty. WED's texture flood and the bright-soma examples show why
foreground intensity alone cannot define centerline edit labels.

Propose branch-unit supervision that separately describes:

1. Reference-supported length along an existing unit,with the actual curve scorer's
   quadrature;retain continuous overlap information,not just a dense-pool adjacency bit.
2. Marginal coverage versus redundant represented length,computed for labels only
   against a declared reference representation. Avoid labeling every nearby parallel
   unit positive merely because all are within tolerance.
3. Anchor/path consistency and ambiguous reference component/branch relationships.
   Fragmented,coincident or nonunique projection relations remain unknown.

A specific loss/positive-negative rule requires agreement;this draft does not
promote rejected geometric DraftB,release held DraftA,mask somata automatically,
or reuse the oracle's greedy edit sequence as ground truth. Different near-optimal
edit sets may be valid;an arbitrary tie winner must not become false biological
supervision. No reference-empty box becomes background by default.

## Proposed learner,decoder and controls

Start with a handcrafted branch-edit baseline,with image/Sato path profiles,
length,curvature,source flags,local spacing and anchor context. Preserve an
image-only no-learning editing rule as a matched control. If discussed and
authorized,compare frozen PointNeXt features at the edit/edge level;its value
must be demonstrated,not inferred from the encoder's presence. No new encoder
training,GATv2 or Transformer role is authorized by this draft.

Decoder responsibilities should be explicit:select a compatible subset of base
chains/additions,avoid duplicate paths,maintain anchor connectivity,and manage
length/coverage tradeoffs. A minimum spanning forest alone does not define a
useful subgraph objective. Empty graphs and heavy undercoverage must fail useful
reconstruction criteria even when their length ratios are small. No decoder may
access reference length,labels,edges or roots during real inference.

Controls must share the same image-only candidate/edit units and decoder:
preserved classical;fixed no-learning edits;handcrafted learned edits;optional
PointNeXt and graph-model comparisons. The benchmark algorithm comparison is
still pending identity recovery and should inform the discussion before freezing
another learned-success bar. No lower acceptance floor is invented from this pass.

## Validation work that the draft still needs

First describe and verify the exact image-only edit-unit pool and its source
provenance. Re-measure reference-assisted opportunity on that same pool before
claiming the61.74% result transfers to it. This is a proposed pretraining check,
not a newly launched job or adopted numerical gate. Candidate recall and known
weak cases must remain visible;do not exclude alias94 or other failures.

Then agree on budget,model/loss,training-only selection rules and final acceptance
criteria before any training. Conditional selector CV remains conditional if
the frozen encoder and learned candidate positions saw all33 training neurons.
Development is repeatedly inspected,not independent confirmation. Reserved
neurons remain untouched. The old59.33/74.14 segment floors and52.43/59.72
resampled floors are historical declared learned criteria,not automatically
relaxed or satisfied by the reference-assisted drafting result.

Report execution,fit learnability,conditional transfer,development accuracy and
whole-image reconstruction quality separately. The latter still lacks crop
assembly,qualified topology,supervised radius and independent confirmation.
No implementation or launch follows this draft without the requested discussion.
