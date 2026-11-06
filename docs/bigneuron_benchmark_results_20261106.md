# BigNeuron comparison — completed under frozen rules, interpretation provisional

The user-local recovery completed in 2, 227.08s (37m07s), guard exit 0 within the
two-hour cap. It reused the archive, identified all seven approved cases and
scored 44 implementation/mode variants. No redownload, model fitting, new development
training label, GPU work, Stage2 continuation or criterion change occurred.

237 extracted outputs cover the cases in counts 36/42/30/20/37/32/40. All 96 earlier
named-case files were corroborated. Human Allen was resolved by exact source-
group/case identity, not by substituting the shared Cell1 basename or another
annotation. No duplicate case/variant replicas occurred. These are 44 variants,
not 44 independent algorithms; ordinals/dataset IDs are removed from method labels,
while method parameters and seed-encoded mode remain distinct.

## Main comparison on the same 151 development boxes

Metrics pool within neuron then macro over seven. Native XYZ identity, 2/6-voxel
segment tolerance and 2-voxel-arclength skeleton sampling/maximum bipartite matching are
unchanged. Oracle rows use reference information; archive rows are external
tracing outputs. They are not an equal learning contest or independence claim.

| Method/context | Segment F1_2/F1_6 (%) | Resampled F1_2/F1_6 (%) | Pooled length |
|---|---|---|---:|
| Preserved Otsu/Lee classical | 54.33/69.14 | 47.43/54.72 | .91754x |
| Best fixed archive variant: neutube_updated | **64.00/73.12** | **58.72/68.14** | 1.25017x |
| Median 44 archive variants, with failure zeros | 30.61/39.73 | 27.29/35.51 | not a single output |
| Reference-assisted prune-only oracle | 64.77/77.01 | 58.58/64.74 | .49075x |
| Reference-assisted prune+add oracle | 68.56/80.43 | 61.74/68.54 | .55105x |

`neutube_updated` is the archive label; its SWC headers identify the generating
software as NeuTu. It leads all four primary metrics and all four separate whole-
image macro metrics. Whole-image scores for it are segment 63.22/72.20%, resampled
58.24/67.88%, length 1.28823x. Keep those separate from the 151-box comparison.

The fixed archive leader improves segment F1_2 by 9.67pp and resampled F1_2 by
11.30pp over the preserved classical control. It is stronger context than the
control used for earlier learned floors, but is not adopted as a candidate stage
or certified as a reference-free independent algorithm. The archive is parameter-
optimized; some variants encode seeds whose origin is unknown here. No radius,
topology or whole-image PN2 quality claim follows. Its length 1.25017x is slightly
ABOVE 1.25; rounding must not turn that into a pass. The external comparison was
not required to pass PN2's learned gate and is not excluded for this ratio.

## Failures, availability and coordinate-screen limitation

Of 44x7=308 declared method/case slots, 208 pass structural/native-bounds checks,
71 have no output for that exact variant, and 29 supplied files score zero:
21 cyclic SWCs, two nonpositive-node-ID files, four missing-parent(-2) files and
two native-bounds failures. No file was repaired or removed from the denominator.
Only 10 variants have valid outputs on all seven; 33 have missing slots; four score
zero at both tolerances throughout. The median is a COMPLETE-GRID median with
missing/incompatible-file zeros, not a typical successful tracing's accuracy.
Archive absence does not establish that a method tried and failed.

The best variant has six accepted cases. Its Series017 file has 16 nodes; one
node at Z 46.7044 is .2044 voxel beyond the upper cell bound 46.5 of a 47-slice image.
Thus 15/16=93.75% inside fails the frozen 99% screen and the ENTIRE case scores
zero. This is a protocol zero, not a measured zero geometric overlap. The other
bounds failure, mouse neurostalker, is different: 35/649 vertices outside, maximum
overshoot 568.5 voxels.

The per-vertex 99% screen I introduced is too coarse to distinguish small endpoint
overshoot from frame mismatch, and depends on node sampling density. Passing it
does not prove subvoxel origin either. Preserve the frozen table and zero; do
not silently relax the rule, clip/rescore the file or fit a translation after
seeing the leader. A separately discussed coordinate audit would be needed
before using this as a decisive acceptance bar. No such revision or new scoring
run is implemented here.

## Per-neuron comparison with editing

| Neuron | Archive leader resampled F1_2 (%) | Editing(b) F1_2 (%) |
|---|---:|---:|
| SLP | 70.15 | 78.36 |
| Frog | 63.99 | 44.59 |
| Human Allen | 65.25 | 68.03 |
| Series017 | 0.00 (bounds-screen zero) | 71.43 |
| Mouse RGC | 76.88 | 61.37 |
| Adult zebrafish | 55.27 | 27.29 |
| Janelia | 79.52 | 81.12 |

Oracle(b) is 3.02pp above the leader in resampled F1_2, and only .40pp above at 6.
It wins 4/7 neurons at 2 and 3/7 at 6; segment wins are 3/7 and 4/7. Its apparent macro
advantage depends on the Series017 protocol zero. This arithmetic retains all
seven cases; it does not exclude Series017 or modify either gate.

The fixed drafting gate still PASSES at 61.74>=60, and prune-only still misses it.
The benchmark does not revoke that authorization, but weakens a claim of a large
useful learned margin. No learned model has demonstrated an advantage over this
stronger context.

## Bounded oracle search and the draft

Saved metadata shows prune-only hit 128 edits in 0/151 boxes and used no budget
component drops. Prune+add hit 128 edits in 106/151 boxes; 97 then used whole-
component drops to meet budget, 767 drops total. The budget remains honored and
is not extended. This helps explain the achieved solution's sparsity without
proving that its candidate pool lacks further coverage.

61.74% is a feasible achieved solution, NOT an upper bound or solved ceiling.
Its small margin does not prove all editing formulations impossible; a hypothetical
better search is not measured learnability either. The actual image-only edit
pool still differs from the reference-filtered/ordered oracle pool.

The [editing contract](classical_anchored_editing_contract_DRAFT_20261106.md)
remains a discussion draft. Before training, resolve coordinate-screen and external
source/seed interpretation, then discuss the image-only representation and a
prospective comparator/acceptance contract. Old floors and failed gates stay
unchanged; no new floor is adopted from these inspected development results.
No new timed job is prepared or launched.

## Full records

- [44-variant summary](evidence/bigneuron_variants_20261106.csv).
- [All 308 method/neuron rows](evidence/bigneuron_per_neuron_20261106.csv).
- [Completed evidence](evidence/bigneuron_benchmark_completed_20261106.json).
- Full report:`tmp/bigneuron_benchmark_recovery_20261106/scoring/report.json`.
- Inventory:`tmp/bigneuron_benchmark_recovery_20261106/inventory/manifest.json`.

Per-neuron best methods are diagnostics, not an adopted best-of composite.
All 237 SWC hashes match; primary/whole-image aggregates replay exactly from saved
primitive rows. Original Stage2 STOP SHA remains
`0edf832a39d52bd5cd48296dbb5d613b1af51c0bdd5ea762f1af797425de6eae`.
