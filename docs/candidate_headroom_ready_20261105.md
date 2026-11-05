# Stage0 headroom — ready for user-local CPU launch

The user approved this candidate-selection arm only: Transformer no longer
proposes nodes, professor discussion pending. The failed broad head stays
failed; its saved nodes are a candidate source and its encoder a frozen feature
provider. No head retraining, GPU launch or learned selector run occurred here.
See [active staged protocol](candidate_selection_protocol_20261105.md) and
[current implementation table](proposal_implementation_audit_20261104.md).

## Run Stage0 first

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\measure_candidate_headroom.py'
```

CPU-only, no new dependencies. Uses the retained NumPy/SciPy runtime, cached
guides/classical graphs/Sato candidates and broad terminal nodes. No model
execution, feature extraction, training, filtering or reserved-neuron access.
Runtime is UNMEASURED; allow several to tens of minutes, with a hard one-hour
cap. Source/graph sizes vary across151 boxes; this is a bound, not a completion
promise. Each box/arm prints progress and checkpoints its completed artifacts.

Fresh `tmp/candidate_headroom_<UTC timestamp>` contains:

- candidate packets and source flags/membership/edge provenance, saved before
  reference geometry is opened for oracle evaluation;
- oracle probabilities in explicitly diagnostic-only files, oracle SWCs with
  image-only roots and unsupervised type/radius placeholders;
- graph counts, halo nodes, k6/k10 candidate coverage at2/6, node-eligible
  coverage, per-box/per-neuron oracle scores, pruning loss, summary and gate;
- `report.json` and `report.md`; sibling timestamped guard console/receipt.

Ten short CPU fixtures passed: no transitive chain collapse, exact kNN parity
with ties, inherited-edge remapping, clipped halos, per-edge scorer agreement,
probability-forest semantics, pruning, gate boundaries, SWC roots and the actual
oracle packet/JSON/cache loop. Metadata preflight validates original inputs and
saved graph hashes. These are EXECUTION checks only. The real151-box oracle
has NOT run and has no headroom verdict yet.

## The fixed oracle/gate

Sato means top512. Classical means preserved pruned graph. Primary edges are
k10+inherited classical and forest edges; k6 coverage selects nothing. Greedy
merge radius1 keeps real centers/source flags, without transitive clustering.

The reference-assisted geometric oracle uses two-voxel node agreement and
per-edge in-box arclength agreement, then the fixed .5-threshold eligible-
component spanning forest with -log p/node-confidence costs and one-pass<2
leaf-to-junction pruning. The SAME selected graph is scored at2 and6. This
oracle directly uses development traces for a diagnostic, never for inference.
It does not prove correct topology, global optimality or selector learnability.

Union oracle MACRO F1 at2: >=80% proceed;65–<80% proceed ONLY alongside candidate-
recall work;<65% stop/report. No threshold/source/k choice or new model is
selected from these results. Candidate-union recall is the separate geometric
ceiling for selecting those fixed segments. Stages1–3 await this gate; no
premature full-cohort candidate extraction or selector training.

If interrupted, preserve the output. Completed construction/oracle records are
reusable with explicit `--resume '<that output directory>'`, after checking the
guard and agreeing any further launch. There is no automatic retry/extension.

## Later stages already constrained

The user explicitly chose conditional selector-only LONO CV, given the frozen
upstream broad-trained encoder AND candidate generator. Those components saw
all33 training-neuron labels; disclose that limit. This is not strict pipeline
CV. Each final frozen model receives one development scoring; the development
set is repeatedly inspected, not an untouched final test.

Node/edge labels are training-only, with trace-agreement/fragment/ambiguity masks
and per-neuron count disclosure. The proposed explicit32-voxel path bound and
consecutiveness rules are written in the active protocol before label results.
Trees precede GATv2; GAT requires>=2pp F1_2 over trees and the same final
reconstruction floors/wins. A failed final tree criterion stops progression
under the stated stop rule. No development tuning or automatic extension.

Terminal broad-run packets contain positions/scores/graph geometry, not latent
PointNeXt point-feature tensors. The checkpoint preserves encoder weights.
An inference-only export command will be prepared after candidate positions
exist if no compatible saved feature cache is available. The assistant will
not launch it on GPU. CPU sklearn/boosted-tree/GAT dependencies are not yet
present in this runtime; no packages are installed before the Stage0 gate.

Whole-image merging, learned connectivity, biological roots/radii, topology
qualification and reserved-neuron validation remain unbuilt/unqualified.
