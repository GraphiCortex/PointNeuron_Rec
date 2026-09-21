# PointNeuron1.0 diagnostic table

Run `python scripts/build_sample_diagnostics.py` from the repository. Only the
output directory (`tmp/diagnostics` by default) is written. No checkpoint is
loaded, no inference/training runs, and no source experiment is changed.

## Scope and provenance

The CSV has one row per discovered evaluated/requested sample. The JSON also
retains source-qualified observations from historical runs, oracle experiments,
selection audits, paper-style evaluation and GAE evaluations. It does not average
different checkpoints, node sets or thresholds into a synthetic reconstruction.

Discovery covers JSON/JSONL summaries and failures, topology and selection
reports, paper-style reports, per-edge reports, standalone per-sample CSVs, and
proposal/graph NPZ metadata. Training caches and dataset inventories do not by
themselves count as reconstruction evaluations. Data inventories supply identity
and volume dimensions only; their GT `swc_valid` is never used for output validity.
GT alignment and out-of-bounds counts are retained as distinct data-quality
fields. Empty sample directories beside run summaries retain unfinished requests. A
requested count without identifiable sample indices cannot recover missing IDs.
CSV copies beside equivalent JSON reports are not counted twice. JSONL events
remain historical observations; they are not added to summary means.

Primary run preference is the promoted configuration, verified using saved graph
metadata (checkpoint, initializer, score cutoff, selection, NMS and node budget).
A failed request can be recognized from its saved checkpoint command and the
verified configuration of successful siblings. Within the same priority, source
paths sort lexically; this is deterministic, not selection by best metric or file
mtime. Historical runs are explicitly labeled. Auxiliary-only rows have no
canonical reconstruction metrics. They remain visible and require review.

Topology joins by experiment directory and exact comparison-artifact path
(separating initializers evaluated within the same directory), selection by exact graph path, and
paper-style geometry by exact exported SWC path. Oracle and promoted GAE values
have separate prefixes. Each populated source metric has a `field_sources`
entry in JSON. Conflicting values are retained in `conflicts` and cause review;
the first value is not silently overwritten. The source manifest lists exact
paths, byte sizes and observation counts. `observations` retains original metric
names and available evaluator context.

## Metric mapping and missingness

- `foreground_threshold` is the **actual graph** threshold;
  `requested_foreground_threshold` is its requested value. `foreground_voxels`
  belongs to that graph threshold, not proposal extraction.
- `proposal_threshold` and `proposal_threshold_fraction` come from proposal
  metadata. They are separate from graph preprocessing and `min_proposal_score`.
- `patches` is saved inference patch count. `points_per_patch` is null when the
  archive did not record it, even though current code has a default.
- `proposal_candidates` is `len(all_centers)`: concatenated retained local
  proposals **after local filtering**, before whole-volume NMS. It is not the
  number of raw network outputs. `proposal_nodes` is saved final proposal count;
  `nodes` is selected graph-node count.
- `proposal_coverage`, `proposal_precision`, terminal and branch metrics map to
  proposal metadata `metrics.coverage`, `precision`, `terminal_coverage` and
  `branch_coverage`. This precision is GT-node proximity, not segment precision.
  Audit `proposal_skeleton_precision` and `selected_skeleton_precision` retain
  their distinct names. Audit coverage is measured at 8 voxels with segment
  sampling step 4 in the current script; saved context is preserved where present.
  The audit first applies its saved score threshold: `audit_proposal_nodes` and
  `audit_proposal_branch_coverage` therefore remain separate from aggregation
  counts/coverage. Its empty-branch coverage convention also differs from the
  aggregation evaluator. `audit_proposal_score_threshold` records this filter.
- `point_distance_precision/recall/f1` map to the paper evaluator's
  `precision/recall/f1` at saved distance threshold 6, sampling step 1.
  `approx_esa/dsa/pds` remain local approximations, not official Vaa3D outputs.
- `bridge_edges` is a count. The existing topology evaluator returns
  `bridge_hit_rate=1.0` for an empty bridge set; that is a convention, not measured
  perfect bridge accuracy. Saved geodesic distances on unreachable edges can be
  penalized fallback costs, not lengths of observed foreground paths.

Unrecorded values are JSON null / CSV `NA`, never zero. `incomplete` means at
least one of primary source, topology F1, output validity, proposal coverage or
paper-style F1 is absent. `missing_fields` identifies which. Optional missing
paper-style geometry alone does not trigger diagnostic FAIL. Missing proposal or
graph evidence does. Missingness is not evidence of a particular biological or
model failure.

## Conservative triage rules

`execution_status` describes actual run completion. `run_status` is diagnostic
triage: PASS means no trigger in recorded evidence; FAIL means review is required
because of a trigger, failed execution or insufficient core evidence. Neither is
a validated biological-quality acceptance test. All FAIL rows require review.

The following explicit **review heuristics** are not fitted scientific cutoffs:

| Evidence | Attribution |
|---|---|
| Recorded foreground cap unsatisfied | data_preprocessing |
| Saved GT alignment failure, aggregation command and return code 2 | data_preprocessing; consistent with the pre-inference alignment guard, without claiming a recovered subprocess trace |
| Oracle reachability < 0.50 and > 5 bridges | connectivity_graph evidence despite GT-derived nodes |
| Candidate segment coverage (or GT-node coverage if no segment audit) < 0.50 | proposal-side evidence |
| Candidate segment coverage >= 0.70, selected coverage loss >= 0.35 | selection-side evidence |
| Invalid/non-single-root SWC | output_geometry |
| Edge F1 >= 0.80 and spatial F1 < 0.50 | output_geometry discrepancy, causal mechanism unresolved |
| More than one independently supported stage | mixed; contributing stages listed |
| Low F1/reachability or many bridges without stage-isolating evidence | unknown |

Graph review indicators F1 < 0.65, reachability < 0.90 and bridges > 5 reuse the
existing eligible-connectivity manifest's thresholds as triage flags, not a
claim that GAE training eligibility defines reconstruction success. The selection
coverage/drop cutoffs follow the existing audit; the stricter <0.50 proposal and
oracle thresholds deliberately reserve attribution for substantial deficits.
No diagnosis is copied blindly from an older audit's `bottleneck` field.
Subprocess return code 2 alone does not identify foreground, decoder, memory or
model failure. For samples 3, 4 and 6, separate saved alignment evidence (17, 25
and 24 out-of-bounds nodes) corroborates a specific early guard in aggregation.
Threshold adaptation alone is not a failure either.

For another dataset, keep the schema and provide `--dataset-id` and an artifact
root with stable sample IDs and compatible source adapters. Indices are local to
that dataset; cross-dataset joins should use `(dataset_id, sample_id)`. No Allen
ingestion, parameter changes or PointNeuron2.0 work is included.
