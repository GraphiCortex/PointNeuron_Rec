# Stage1 completed: candidates and training-only labels

User-local CPU run completed successfully. Worker time442.43 seconds (user
console7m34s including wrapper/preflight);3600-second cap, exit0. No retry,
model execution, GPU, training, development labels or reserved-neuron access.

Verified all611 candidate packets,460 training label packets and33 classical
graph packets against their recorded SHA256, plus manifests, training-label
identities and candidate-to-label bindings. Six historical training classical
graphs were reused;27 missing graphs were extracted from retained guides with
the frozen recipe. No guide filtering or completed audit was repeated. The151
development union packets are reused unchanged from Stage0 image-only outputs.

| Training candidate instances | Positive | Negative | Unknown | Masked | Positive among known |
|---|---:|---:|---:|---:|---:|
| Nodes:673,235 | 118,681 | 549,054 | 5,500 | 0.82% | 17.77% |
| Edges:4,172,031 | 163,536 | 3,899,191 | 109,304 | 2.62% | 4.03% |

These are per-box instances, including repeated halo candidates (5,936 training halo
nodes), not unique whole-neuron objects. Node labels describe supplied trace
agreement, not exhaustive biological foreground. Distant real untraced neurites
can be counted negative under the agreed restricted task.

All33 training neurons supply both positive node and edge labels. Known-positive
fractions vary widely: nodes1.24–42.02%, edges0.25–11.35%. The agreed balanced
loss/weights are necessary; raw accuracy would be a misleading objective. Alias20
(WED_SLP_PN_neuron) alone contributes161,482 node and958,527 edge instances,
about24%/23% of the training pool, despite only9/460 boxes. Training-neuron
weighting/resource choices need to be explicit in the later selector protocol;
no new sampling or weighting rule is adopted from development here.

Unknown edges comprise51,244 coincident projected-group pairs,41,159 unknown-
endpoint cases,13,961 junction transitions and2,940 cross-component relations.
They remain masked. Most negative edges touch a known-negative endpoint
(3,718,914);180,269 are nonconsecutive and8 exceed the32-voxel bound. This is
the stricter training contract, not the Stage0 oracle's geometric-overlap labels.
The90.42% geometric oracle therefore does not guarantee achievable performance
with these supervision and topology constraints.

Three training boxes have no positive candidate node at the two-voxel tolerance:
alias0/tile37, alias7/tile11 and alias95/tile8. All their nodes are farther than2,
not ambiguity-masked. Seven boxes have no positive edge labels: those three,
alias0/tile46, alias1/tile22, alias23/tiles71 and76. The first two additional boxes
have just one positive node; alias23/tile71 has6 positive nodes but its14
positive-endpoint edges are coincident-group unknown; tile76 has2 positive
nodes with no positive edge. Do not drop these boxes or silently repair the
candidate pool. A segment between distant endpoints can still cross a trace;
zero positive endpoint labels do not prove zero candidate-edge geometric
coverage. They do expose local gaps under the frozen learning contract.

The highest neuron edge-mask fraction is alias85 at10.70%, followed by149 at
9.71%,108 at9.33%,158 at9.03% and53 at7.92%. Per-neuron counts/reasons are in
[portable evidence](evidence/candidate_stage1_completed_20261105.json) and the
[generated report](../tmp/candidate_stage1_20261105/report.md). Junction counters
test every full-reference junction per box; their unanchored fraction is NOT a
cropped-branch retention estimate.

This establishes successful construction and usable supervision counts. It
does not establish separable features, optimization learnability, development
selector accuracy, correct joins or final reconstruction quality. No scientific
criterion, candidate recipe, label constant or stopping rule changed.

## Next user-local launch: frozen PointNeXt features

Feature-export metadata preflight passed in the retained runtime; no forward
pass was run. The immutable broad checkpoint supplies32D features at each input
point; deterministic3NN interpolation queries the fixed candidates, retaining
neighbor distances/weights to expose extrapolation. Both training and development
image inputs are included, without reading their labels/SWCs or oracle scores.

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe' -B '.\scripts\export_candidate_pointnext.py' --device cpu
```

CPU inference only, zero updates; no optimizer or automatic selector training.
Output `tmp/candidate_pointnext_20261105/complete.json` and611 feature packets;
progress every10 boxes, guard logs in timestamped sibling directory. Runtime
unmeasured: allow several to tens of minutes, hard3600-second cap; stop and
report on failure/cap. No automatic extension. Explicit user `--device cuda:0`
would be GPU inference only; CPU is the supplied/default command.

After inspecting feature completeness/distances, prepare remaining image-derived
features and the gradient-boosted-tree baseline before GATv2. Conditional
selector-only LONO CV discloses the frozen encoder AND learned candidate positions
trained on all33 training neurons; it is not full-pipeline CV. Seven development
neurons are repeatedly inspected and receive one scoring per frozen selector.
Final classical+5pp floors/wins, GAT>=2pp over trees, topology/break reporting
and stopping rules remain as agreed. No model is selected or trained now.
