# Saved-label diagnostic: fixed execution and interpretation

User accepted Stage1 as preparation only and requested this CPU diagnostic
before ANY selector training. Frozen candidates, labels, decoder and scorer
remain unchanged. Training data only; the151 development boxes are not scored
again. No candidate/source choice, label repair, tuning or training is authorized
by a diagnostic result. Professor discussion of the experimental departure is
still pending.

1. Decode all460 training packets using node/edge label1 as probability1 and
   label0 as probability0. Unknowns are ineligible (probability0 solely in the
   diagnostic decoder), not rewritten as saved negative training labels. Use
   the same .5 eligibility thresholds, probability-only minimum spanning forest,
   node costs, fixed one-pass spur<2, image-guide roots and unsupervised radius1.
   Score clipped segments with the same .5-voxel arclength scorer at2/6.
   Pool lengths within neuron, then macro across33 neurons. Also score the
   union of all label-positive edges before decoding to expose the available
   label geometry versus forest/pruning loss.
2. Apply the user's new diagnostic stopping rule: label-oracle F1_2 must be at
   least saved Stage0 development union oracle F1_2 minus.03, exactly
   .8742320394860465. More than3 percentage points below stops before selector
   training. Equality passes. This compares different cohorts and targets;
   therefore also run the unchanged geometric-oracle recipe on THESE SAME460
   training pools. This new matched comparator is diagnostic only and does not
   substitute a more permissive gate. Neither result certifies topology or
   asymptotic learnability; binary-label MSF ties can matter.
3. Reuse all per-neuron Stage1 positive/masked counts and name all zero-positive
   boxes. Audit all both-unambiguous-positive candidate-edge endpoints against
   full-reference component identities, replaying the original projection. Report
   cross-component unknown/negative/positive counts; any known label on such a
   cross-component pair stops training. Also report masking conditional on BOTH
   endpoints positive. Known fragmented aliases23/66/75 are flagged explicitly.
   Edges touching a known-negative endpoint remain negative by the agreed rule;
   they are not certified cross-fragment relations. Overall masking is diluted
   by these numerous negative-endpoint edges, so the overall fraction alone
   cannot validate fragmentation handling.
4. Fixed image-only edge ranking controls, scored separately on each held-out
   training neuron (all its known edges), with no fitted parameters or selected
   thresholds/signs. Sato score = NEGATIVE preserved straight-segment path cost
   `length*(1+4*(1-mean_normalized_Sato))`, .5-voxel midpoint samples. This is
   the existing scaffold cost, explicitly NOT geodesic shortest-path cost;
   the later Stage2 geodesic feature is still unbuilt. Distance score = NEGATIVE
   maximum of the two endpoint distances to the nearest of that box's2048
   input points. No reference is accepted by the feature function. AP is pooled
   across boxes within the held-out neuron, never mixed with that neuron's
   training-fold labels to fit anything. The fixed ranking has no training
   step; this is a leave-neuron-out scoring control, not a fitted CV model.
   Report per-neuron prevalence and tie-grouped noninterpolated average
   precision (stepwise AUPRC); trapezoidal PR area is also saved. No accuracy
   metric. No-positive folds are undefined, not fabricated zero. Baselines are
   edge rankings only; no probability calibration/decoder threshold is fitted
   to manufacture a baseline reconstruction. Decoded F1 is reported for the
   label and geometric oracles; future trained selectors require both held-out
   AP and decoded F1. All upstream conditioning/leakage remains disclosed.

The current user message explicitly authorizes agent CPU diagnostic execution
and CPU inference-only feature export now, superseding the earlier local-launch
preference for this bounded work. An existing user-started CPU feature export
was found running; monitor/verify it rather than duplicate it. CPU export is
independent of the label-oracle gate, zero updates, existing3600-second cap.
AC power was verified connected; a scoped SetThreadExecutionState system-awake
request covers active CPU work, released afterward without changing global
power-plan settings. Neither job executes GPU work.

Diagnostic command (executed by assistant as explicitly requested):

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\audit_candidate_labels.py'
```

Output `tmp/candidate_label_audit_20261105`; source/input bindings in
`configs/candidate_label_audit_20261105.json`;3600-second wall cap, no automatic
retry/extension/training. Cached per-box diagnostic records, fixed feature
scores, clearly diagnostic-only SWCs and readable/JSON per-neuron reports are
preserved. Completion of this diagnostic is distinct from passing its criterion.
