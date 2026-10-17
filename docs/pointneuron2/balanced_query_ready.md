# Matched confidence normalization comparison

18 October 2026. Prepared after agreement to examine class normalization and a
candidate distance margin. **GPU timing and the bounded fit are now complete.**
See the [paired results](balanced_query_results.md); all combined gates fail and
no further fit is prepared. The protocol below records the original comparison. The
completed four-crop baseline remains preserved. This is a training-set
learnability comparison, not validation of reconstruction quality.

## Question and controlled change

Can equal positive/negative contributions per crop improve unique-proposal
selection across sparse and dense crops, without sacrificing placement?
The baseline weighted classification objective has very different constant-score
optima at 6 versus 418 positives among 512 queries. This motivates a comparison;
it does not establish that weighting explains all observed failures.

The new classification term is

`0.5 * mean(-log p_used over matched queries) + 0.5 * mean(-log p_unused over unmatched queries)`.

Its outer loss coefficient remains 2. The center/radius terms, their coefficients,
Hungarian cost, optimizer, architecture and normalization policy are unchanged.
The inherited `positive_class_weight: 8` remains available to the original
assignment validator and reference-loss evaluator; it is **not applied to the
new classification term**. If every query is matched, the single available class
contributes its full mean. Missing annotation is skipped, never treated as a
background-only crop.

The assignment algorithm is the same; assignments need not stay identical as the
models learn different parameters. All model parameters train jointly. These
scores concern membership in the matched proposal set, not a calibrated
probability that a location contains neuronal foreground. Unmatched proposals
can lie close to a real neurite.

The isolated implementation is
[balanced_query_loss.py](../../src/pointneuron/models/balanced_query_loss.py),
with a separate [runner](../../scripts/pointneuron2/balanced_query_experiment.py)
and [comparison specification](../../configs/pointneuron2/balanced_query_comparison.json).
Baseline source files and production defaults are unchanged.

## Matched protocol

| Training neuron/crop | Distinct frozen input points | Targets | Visible junctions |
|---|---:|---:|---:|
| 0/0 | 848 | 6 | 0 |
| 0/1 | 2,048 | 12 | 0 |
| 92/4 | 1,219 | 16 | 2 |
| 86/3 | 2,048 | 418 | 77 |

The prepared cache is byte-identical to the completed baseline, and every fresh
model state tensor has the same hash. Phase 1 is frozen. Native PointNeXt and
the Transformer retain 512 queries, explicit per-crop normalization, batch one,
seed zero, no augmentation/dropout, AdamW LR .0001, weight decay .0001 and clip
10. Four crops cycle in the same order for 4,000 total updates, with evaluations
at 0/500/1,000/2,000/4,000. No development or test data enter this experiment.

Saved checkpoints include optimizer, RNG, crop cursor, loss policy and source/cache
identity. They use the existing fit-state container with an additional comparison
identity and recipe policy. A baseline checkpoint cannot be resumed as this
comparison. Checkpoints precede evaluation. Explicit resume preserves the original
4,000-update cap; the terminal checkpoint cannot be extended.

## Separate candidate-margin result

The CPU comparison used all 20 saved baseline crop/snapshot predictions, with
zero model forwards or optimizer updates. Both arms retain the same 512 proposals,
scores, assignments and symmetrized kNN16, changing only the union radius.
The final snapshot gives:

| Crop | Required edges available, R8 → R12 | Candidate edges, R8 → R12 | Edge availability after node cutoff, both radii |
|---|---:|---:|---:|
| 0/0 | 4/5 → 5/5 | 31,998 → 55,934 | 0% |
| 0/1 | 9/10 → 10/10 | 6,415 → 14,658 | 50% |
| 92/4 | 13/15 → 15/15 | 19,499 → 42,085 | 26.67% |
| 86/3 | 353/404 → 353/404 | 4,877 → 4,877 | 87.38% |

Thus radius twelve fixes the observed candidate omissions between trusted nodes
in the smaller crops. It does not fix selection, or the 51 dense-crop edges whose
endpoints lack trusted placement. Extra candidates are mainly unknown labels:
crop 0/0 still has no trusted negative edge. Candidate counts grow by about
1.75–2.28 times in the smaller crops. These are edge/message costs; this CPU
diagnostic does not measure GATv2 GPU memory or learned edge accuracy.

The nominal margin is spacing eight plus two endpoint errors of two. Cached
float32 target chords include a maximum of 8.00001934, so radius twelve is not
claimed as an exact universal floating-point guarantee. The observed trusted
edges in these snapshots are covered. No reference edges are inserted.

Both radii will be reported for the new model as well. **The original radius-eight
gates remain primary.** Radius twelve stays a separate interface diagnostic;
neither a production policy change nor a way to relabel a failed primary gate.
See [saved comparison](balanced_query_ready_seed0/candidate_margin.json).

## Checks and decision rules

Eighteen targeted tests pass. They check assignments and geometry-loss/output
gradients against the baseline, positive/negative normalization, unknown crops,
single-class cases, a crowded-kNN radius counterexample, checkpoint identity and
one discarded synthetic full-model CPU update with finite gradients for all 123
parameter tensors. The real four-crop CPU check makes zero optimizer updates,
matches initialization, verifies source/cache identities and reloads with maximum
output difference 0.0. It completes in 50.7 seconds. These establish execution
and comparison integrity, not learnability. Evidence is in
[readiness summary](balanced_query_ready_seed0/summary.json).

At update 4,000, assess the predeclared selection criteria on **every** crop:

- At least 95% of assigned positives accepted at the fixed .5 threshold: all
  6/12/16 in the smaller crops and at least 398/418 in the dense crop.
- At most 0/0/0/20 accepted unmatched queries, respectively.
- Accepted-curve support within six voxels at least 95%.

Also require at least 95% of targets placed within two voxels and all visible
junctions trusted; confidence improvements with worse placement do not establish
a solved proposal interface. Report all five snapshots, counts, original gates
and both candidate radii. Weighted reference classification loss and balanced
optimization loss are named separately; their raw magnitudes are not equivalent
quality metrics.

Passing selection supports this normalization on these four training crops only.
Improvement short of those criteria is partial evidence. Failure across the fixed
cap weakens the claim that normalization alone resolves confidence; it does not
prove the representation incapable. Review assignment stability, the meaning of
unmatched negatives and shared-feature optimization before choosing another fit.

The original combined gate also requires at least 95% required-edge availability
and positive/negative edge labels per crop. Its conservative per-crop negative
requirement is unresolved for the smallest crop and must be discussed before a
connectivity-training decision. It is retained and reported, not silently changed.
No outcome automatically authorizes GATv2 training, a new cutoff, more updates,
radius promotion or a broader dataset run.

Stop on nonfinite values, provenance/backend mismatch or checkpoint failure.
Preserve errors and outputs; no automatic retries. Stop training at 4,000 even
if curves are still improving. The question at that point is whether the paired
evidence justifies this objective, not whether more compute can eventually fit it.

## Visible commands

Preparation, CPU checks and the user-launched GPU timing are complete. The eight
throwaway updates passed with zero reload difference. Source hashes, cache,
initialization and backend match the baseline. Torch peak allocated memory was
93.1 MiB and peak reserved memory 118 MiB; these exclude driver/context and other
process memory. No trained timing checkpoint was retained. The estimate is
22.6–33.8 minutes plus checkpoint IO/system variability. Fresh early evaluation
failures are expected and do not measure learnability. See the
[timing evidence](balanced_query_ready_seed0/timing.json).

The completed timing command was:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_balanced_query.ps1 -Action timing
```

The device is the NVIDIA GeForce RTX 3070 Ti Laptop GPU. The matched baseline
actually took about 18 minutes. The timing checks support proceeding with the
agreed comparison; they do not establish reconstruction quality. The separate
bounded fit command is:

```powershell
& .\scripts\pointneuron2\run_balanced_query.ps1 -Action train
```

All generated data go to ignored
`tmp/pointneuron2/local_runs/balanced_query_20261018/{cache,cpu_check,timing,model}`.
The runner refuses to overwrite existing action outputs. For an interrupted fit,
explicit `-Action resume -Checkpoint <full path to a comparison step checkpoint>`
creates a separate resume folder, restores optimizer/RNG and retains the total cap.
Do not resume baseline weights into a fresh optimizer under this comparison name.
