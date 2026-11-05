# Stage2 CPU tree comparison — stopped by the fixed length rule

All33 C0/C1 conditional selector leave-neuron-out folds completed. The comparison
worker exited normally after8,454.36s (2h20m54s); this was a measured criterion
failure, not a crash, time cap or failure to fit the tree backend. Neither contract
met the predeclared pooled CV length<=1.25 rule. **No contract was selected.**
C0=1.254998x,C1=1.740456x. Keep the gate unchanged, STOP.json and all artifacts.
No extension, threshold change, exclusion or new run is adopted.

The user launched prepare,policy and compare locally using implementation
commit `c53cd11`. Actual preparation526.00s,policy76.56s,comparison8,454.36s.
First-fold resource projection17,660.85s was below the21,600s cap, so the run
continued as declared. Preparation reused guides/candidates and decoded raw
images for intensity only; no Phase1 filtering was repeated.

## Paired training-neuron CV

Pool box statistics within neuron, then macro across33 neurons. Ratios below
are pooled predicted/reference length. All scores use native voxel units and
the same frozen segment/resampled scorers. The classical row is the already
measured control on these SAME460 training boxes, not its seven-neuron
development score. Its large training length ratio is also disclosed.

| Arm | Segment F1_2 / F1_6 (%) | Resampled F1_2 / F1_6 (%) | Pooled length |
|---|---|---|---:|
| Matched classical,training boxes | 55.97 / 77.13 | 48.96 / 66.13 | 3.43198x |
| C0: original dense-pool adjacency | 51.32 / 82.47 | 39.41 / 55.88 | 1.25500x |
| C1: retained-representative adjacency | 58.49 / 83.72 | 40.85 / 55.06 | 1.74046x |

C1 improves segment F1_2 over C0 by7.16pp, winning31/33 neurons (27/33 at6).
That consistent paired effect supports the relevance of the representation/label
interaction. It does not establish a useful reconstruction recipe: resampled
F1_2 improves only1.45pp, resampled F1_6 falls0.82pp and predicted length rises
38.68%. Segment overlap and one-to-one reconstruction agreement are telling
different parts of the story. C1 exceeds matched classical segment F1_2 by2.51pp
but wins only16/33 neurons and trails its resampled F1_2 by8.10pp; at6 it wins22/33.
C0 beats classical on11/33 at2 and21/33 at6. No cross-contract AP is compared.

CV is conditional selector-only: outer node and edge fits exclude the held-out
training neuron, but frozen upstream PointNeXt and learned candidate positions
already saw all33 training neurons, as explicitly accepted by the user. These
numbers are not strict full-pipeline CV, a training-fit measurement or development
accuracy. No in-sample fit-vs-CV parity or asymptotic learnability claim is available.

## A concentrated but genuine failure under the declared trace contract

WED_SLP_PN_neuron accounts for28.56% of C0 predicted length and37.32% of C1.
Its predicted/reference ratios are20.19x and36.59x. The held-out handcrafted
node policy retains53,394 of161,482 candidates across nine boxes:
830 trace-positive,52,493 trace-negative and71 unknown under the original labels.
Thus only1.56% of its known retained nodes agree with the supplied trace within2.
This is a major node-selection failure under the accepted trace-agreement
contract, not evidence that every negative is biological background. Real but
untraced neurites may exist; reference completeness needs image/reference review.

For concentration diagnosis ONLY, subtracting this neuron from saved length
totals givesC0=.91280x,C1=1.11063x. This does not remove it, change a gate or qualify
either recipe. Seven C0 neurons and ten C1 neurons individually exceed1.25x.
Conversely, severe undercoverage in some neurons can offset overgrowth elsewhere;
an aggregate ratio close to1 is not a topology or completeness certificate.

## The earlier policy diagnostic matters

These151-box development diagnostics were authorized to select nothing. They
used the frozen all33-training-neuron handcrafted node policy. They are separate
from final edge-selector development scoring, which was NOT reached.

| Policy diagnostic | Segment F1_2 / F1_6 (%) | Resampled F1_2 / F1_6 (%) | Pooled length |
|---|---|---|---:|
| Image-only nodes + Sato-cost forest | 58.22 / 79.62 | 43.83 / 55.73 | 1.56156x |
| Same image-selected nodes + reference-assisted edge oracle | 74.74 / 88.27 | 46.78 / 58.26 | 1.05603x |
| Historical compact union oracle,reference-assisted node ordering | 88.52 / 95.10 | 64.80 / 72.54 | 1.18475x |
| Matched classical,development boxes | 54.33 / 69.14 | 47.43 / 54.72 | .91754x |

The conditional edge oracle loses13.77pp segment F1_2 and18.01pp resampled F1_2
relative to the earlier compact union oracle. Representative selection and its
induced edges therefore remain unresolved before adding learned edge features.
This is evidence about the declared policy/decoder interaction, not a mathematical
upper bound for every selector or an inference result using oracle edges.
Its apparently reasonable length1.056x coexists with only65.77% two-voxel
segment recall and46.78% resampled F1: compact length does not guarantee coverage
or a clean representative set. Eligible isolated nodes also count in resampled
scoring while contributing zero segment length; no causal isolate breakdown
has been newly run here.

The actual image-only policy scaffold improves segment overlap over classical
on development, but fails length and two-voxel resampled agreement. It is not
a qualifying learned-edge model result. Development has been repeatedly inspected.

## What stopped, and what remains unknown

- C0/C1 handcrafted node/edge CV completed,99 fold model pickles and920 decoded
  SWCs retained. SHA checks passed for all99 models and920 SWCs; original460
  training label packets still match their original hashes. Stored-row aggregate
  replay is exact for both contracts; source/runtime/input metadata checks pass.
- No contract selected, no PointNeXt-edge CV, no final edge model fit/freeze,
  no final development edge-selector scoring, no GATv2 and no GPU execution.
  Same-contract AP and final train-vs-CV-vs-development parity were downstream of
  selection and consequently were not produced. This is an intentional stop,
  not missing successful results.
- No new development training labels were created by this analysis. No new
  inference, scoring, model fit or metric sweep was launched while reading results.
  Original packets/protocols and the failed broad head remain preserved.
- Whole-image assembly, qualified graph learning, biological wrong joins, topology
  losses, supervised radius and final reconstruction remain unestablished.
  No newly measured wrong-join rate or final quality claim follows from these F1s.

## Discussion proposal, not an adopted revision

Do not proceed to PointNeXt or GATv2 on this recipe. The next formulation
hypothesis is that **image-local trace-support selection plus greedy representatives
does not reliably identify a compact instance of the annotated neuron**. C1 repairs
some adjacency supervision after suppression, but cannot fix an unsuitable node
set and can connect more of its excess structure. The concentrated WED failure
also makes annotation completeness/instance identity a plausible issue, not a
proven cause. That requires reviewing saved image/reference/prediction context
and node/decoder responsibilities before proposing any new training contract.

No alternative source priority, NMS radius, threshold, label mask, root rule,
neuron exclusion, architecture or budget is implemented by this report.
There is **no next launch command** for the stopped recipe. Any further timed
diagnostic needs a separately visible local command; a substantive strategy
revision remains a discussion with the user. Professor discussion of the
Transformer departure is still pending.

Portable [evidence](evidence/stage2_trees_stopped_20261106.json) and
[paired per-neuron table](evidence/stage2_trees_paired_neurons_20261106.csv).
Full local outputs: `tmp/stage2_trees_20261105/{policy,comparison,STOP.json}`.
