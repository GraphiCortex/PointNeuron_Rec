# The PN1 reference for the fresh PN2 comparison

21 October 2026. Milestone A specification, supporting the
[revised proposal](revised_proposal.md). This defines the comparison's lineage
and necessary repairs. It is **not a ready training recipe** or evidence that
PN2 already improves PN1. No model is trained or evaluated by this document.

## Three records with different purposes

| Record | Meaning | Permitted interpretation |
|---|---|---|
| H: historical PN1 | Frozen guarded30 DGCNN/pointwise proposal model and promoted adaptive geodesic reconstruction | The working historical system, with its actual training exposure and original preprocessing |
| R: repaired PN1-style reference | Fresh DGCNN/pointwise proposal training and PN1-style geodesic reconstruction under verified input/coordinate handling and shared neuron restrictions | The controlled learning reference for N |
| N: PN2 candidate | PointNeXt, Transformer proposals, automatic crop consolidation, candidate graph, GATv2 and explicit forest/SWC decoding | A candidate whose advantage must appear in matched whole-neuron outputs |

The local PN1 implementation is not an exact reproduction of every module in
the published PointNeuron paper. In particular, the learned GAE was not the
promoted connectivity path. Replacing R's geodesic path with that failed GAE
would weaken the reference artificially.

## Preserve H as history

The frozen [PN1 inventory](../../pointneuron2/configs/pointneuron2/frozen_pointneuron1.json)
records original source revision `1719a56e63328782dda7e160a880f16ec1f49937` and
the checkpoint at
`../../tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt`
relative to this document's folder. Its SHA256 is
`378b1b5f42a893dc5e33396546a29f61c91105d8829f8cdc018fea96027831e0`.
Its promoted graph configuration is geodesic initialization,
`adaptive_connected_coverage_nms`, confidence 0.85, distance NMS 18 and at most
128 graph nodes. These are recorded historical settings, not universal PN2
defaults.

The historical split shuffled patches. Every validation/test neuron also
appeared in its training partition. All 48 identities in the current PN2
33/7/8 split were exposed to H during training. The historical 55 evaluated
neurons absent from training still provide useful evidence of reconstruction
ability, but their repeatedly inspected results are not a new independent
test. The [evidence audit](research_reset_evidence.md) records the distinctions.

Keep H's source, checkpoint and saved outputs unchanged. Loading H's weights
through a corrected reader is a compatibility experiment; it does not create
R. Recomputing H outputs with a new evaluator also does not remove training
exposure or repair its historical images.

## Minimum repairs that define R

R retains the DGCNN point encoder, pointwise center/radius/confidence proposal
task and promoted style of geodesic reconstruction. The following confirmed
repairs are part of R's identity and must be versioned together:

1. **Certified image decoding.** Reuse the corrected PBD8 packed-difference
   order and verified PBD16 handling, with independent format fixtures. An
   unsupported encoding extension is an explicit case failure, never a reason
   to silently use the old decoder or omit the neuron.
2. **Signal range from the image.** Storage dtype alone cannot determine the
   usable signal range. The historical repair's image-max-fraction policy is
   a documented starting control, with known outlier/faint-signal limitations.
   Choose and record one image-only policy shared by preparation and inference;
   do not choose a cutoff separately for each neuron using its reference.
3. **Local proposal features.** Reuse the translation-equivariant local XYZ
   head. Preserve the declared prediction units for offsets and radii. This
   changes a learned input and therefore requires fresh training, rather than
   interpreting historical raw-coordinate weights as repaired weights.
4. **Fresh caches with provenance.** Rebuild required training inputs from
   corrected images after the preparation policy is fixed. The inherited
   patch cache contains intensities from the defective decoder and is not a
   certified R training source. Record image/reference hashes, decoder version,
   transforms and preparation configuration in the new cache manifest.
5. **Shared inference behavior.** Explicitly version normalization, full-volume
   tiling, crop aggregation, selection and export. Handle empty/short clouds,
   capacity limits and failures automatically, without reference access. A
   budget that prevents full declared coverage is disclosed as unsupported or
   failed, not hidden by a capped crop cache. Ordinary proposal/anchor pruning
   is instead part of the algorithm: disclose its counts and score the resulting
   reconstruction, including any branches lost to that selection.

The [input investigation](../../pointneuron2/docs/pointneuron2/root_cause_findings.md)
contains the independent evidence for the first three repairs. Reuse verified
code deliberately; the fresh folder is not a reason to reimplement operators
that already have meaningful checks.

## Shared data restrictions

Provisionally retain the existing sample-ID partition: 33 training neurons,
seven development neurons and eight PN2-reserved internal neurons. Before
fitting, check known duplicate/related-acquisition grouping from permitted
metadata; if this changes a partition, version that prospective change.

R and N start fresh using only the same training identities. Neither inherits
H's learned weights, a broader-data field model, normalization statistics or
an auxiliary model trained on development/reserved neurons. Any pretrained
component needs explicit training-exposure provenance and equal disclosure.

Development labels may support declared model selection or a global operating
policy. They must not become gradient-training examples while the same cases
are described as held out. The seven development neurons have already been
inspected repeatedly. The eight reserved identities are internal PN2 holdouts,
not globally untouched project data; all were in H training, and prior output
metadata exists for some. New external/source-held-out confirmation remains
necessary for broad generalization claims.

Reference-empty crops are unknown. A reference-containing crop does not prove
that all remaining bright structure is background. The exact reference-
agreement supervision assumption, positive target construction and treatment
of unknown regions must be written into both recipes. A common final-SWC
evaluator does not itself settle the two architectures' training labels.

## What must be specified before either fresh fit

- The executable data/preparation path, normalization and checkpoint reload
  behavior, with the same policy in training and ordinary inference.
- R's retained target/loss convention and N's node/edge label contract, with
  any departure from historical behavior explicitly named.
- Automatic whole-volume inference and final SWC outputs for each arm before
  spending a substantial training budget.
- Architecture-specific initialization, optimizer, schedule, data order,
  augmentation, update budget and checkpoint-selection rule. Record them;
  no LR or reliable duration is inferred from a warm-start four-crop fit.
- Full-volume coverage, actual point/proposal/graph counts and time/memory
  accounting. System comparisons may use different internal budgets if the
  quality/resource tradeoff is visible. A component ablation fixes the
  relevant upstream inputs and downstream control.
- The common [benchmark contract](benchmark_contract.md), structural review
  procedure and prospective advancement margins, before N's decisive results.

The [source inspection and integration draft](baseline_integration_plan.md)
identifies reusable operations, reference-coupled runners, the vertex-only crop
label issue and the next bounded port. The [explicit training inventory](../configs/training_inventory.json)
records the provisional 33 training identities without reading source contents.

The existing five-epoch repaired control is historical evidence, not the fresh
R recipe or a sufficient training budget by default. Both arms need meaningful
learning curves and comparable opportunity under the declared resource policy;
an undertrained R must not be made to lose by construction.

Milestone A now has an explicit development inventory and a common evaluator
starting point. It remains in progress while recipe, structural assessment and
advancement decisions are completed. No training command is authorized merely
by this specification. The next evidence sought is paired whole-neuron
reconstruction, not another attempt to perfect four training crops.
