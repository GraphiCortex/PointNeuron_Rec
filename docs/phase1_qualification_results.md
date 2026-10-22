# Four-context Phase 1 qualification

23 October 2026. **Complete: the predefined qualification criteria were not
met.** All four images and 120 clouds were generated and scored successfully;
87 of 96 comparative checks passed. Four checks were undefined because the
mouse reference has no encoded junctions; five comparisons failed numerically.
No sampling setting, exclusion or acceptance check changed. No model was trained.

Structural guidance helps strongly on frog, but this comparison does not
justify making the current Sato selector mandatory for every PN2 input.

## Scope and provenance

This executes the fixed [four-context plan](phase1_qualification_plan.md) using
the [disk backend](phase1_disk_guide.md) at block side 64. Source commit:
`115a8d9`. The original provided Gold166 image files are read unchanged; their
upstream acquisition/processing histories remain unverified. They should not be
described as guaranteed unprocessed acquisitions.

The four training identities are frog 22, mouse 75, silkmoth 85 and Janelia 126.
They were fixed before image/reference inspection for this qualification.
Three count-matched controls (uniform, raw intensity, Sato), two densities and
five fixed seeds give 120 clouds and 51,082,815 point records. No development
or reserved image is used. Historical exposure means this is engineering and
training-input evidence, not independent generalization evidence.
Physical calibration remains unverified. The same native-voxel tolerance or
filter scale need not represent the same physical distance across sources;
folder context labels do not establish independent acquisition domains.

The [recipe](../configs/phase1_sampling_probe.json) is unchanged: whole-image
affine normalization; Sato scales 1/2/4 native voxels; 10% uniform probability
mass in weighted draws; exact distinct-point quotas in disjoint clipped
128-cube selection boxes. Cell sides 4 and 8 name dense and lean budgets.
They are selection-count settings, not a new image resolution or voxel spacing.
Input-cloud counts remain separate from encoder, proposal and graph budgets.
For example, a full 128-cube box has 32,768 dense or 4,096 lean source points.
Those counts are not an instruction to reuse the old 2,048-point encoder input
cap. Encoder aggregation may preserve useful evidence while reducing point
locations, but its effective retention must be assessed rather than assumed
from the sampler's score.

All clouds were frozen before SWC access. The manifest SHA-256 is
`0106b61fc81875af140af0e8f835e80328e993f85725729ff703fb645e4e751f`.
All 120 cloud and four guide hashes were checked after generation. Scoring
independently verifies distinct counts, source indices, XYZ and raw intensity.
References use the explicit `opaque_integer` metadata policy without filtering
nodes, welding components, fitting alignment or inferring anatomical types.

## Measured support and local losses

Mean reference-curve arclength within **2 native voxels** of selected points,
over all five fixed seeds. These are spatial-support percentages, not
reconstruction accuracy or verified foreground recall.

| Source | Budget | Points per cloud | Uniform | Intensity | Sato |
|---|---|---:|---:|---:|---:|
| Frog | Dense | 331,776 | 39.81% | 63.84% | 97.95% |
| Mouse | Dense | 310,000 | 40.10% | 96.38% | 97.86% |
| Silkmoth | Dense | 671,744 | 39.67% | 100.00% | 100.00% |
| Janelia | Dense | 1,713,607 | 39.60% | 100.00% | 100.00% |
| Frog | Lean | 41,472 | 4.93% | 11.98% | 66.53% |
| Mouse | Lean | 38,750 | 6.22% | 55.63% | 63.15% |
| Silkmoth | Lean | 83,968 | 5.50% | 99.54% | 99.32% |
| Janelia | Lean | 214,204 | 5.82% | 99.78% | 99.81% |

Dense local results below average each seed's minimum encoded-path support and
longest unsupported run. A high global mean can coexist with local losses.

| Source | Worst path: intensity | Worst path: Sato | Longest gap: intensity | Longest gap: Sato |
|---|---:|---:|---:|---:|
| Frog | 37.34% | 85.68% | 14.20 vox | 4.39 vox |
| Mouse | 61.34% | 22.61% | 9.15 vox | 7.55 vox |
| Silkmoth | 100.00% | 100.00% | 0.00 vox | 0.00 vox |
| Janelia | 100.00% | 100.00% | 0.00 vox | 0.00 vox |

The lean frog and mouse Sato clouds still have mean longest gaps of 23.31 and
39.46 voxels and mean minimum-path support of 25.24% and 0%, respectively.
Their relative gains do not establish absolute adequacy of the lean budget.

| Source | Checks passed | Unchanged failures |
|---|---:|---|
| Frog | 24/24 | None |
| Mouse | 18/24 | Four undefined junction checks; worst-path regression versus intensity at both budgets |
| Silkmoth | 22/24 | Lean global gain below 5 percentage points; lean worst-path regression versus intensity |
| Janelia | 23/24 | Lean global gain below 5 percentage points versus intensity |

Silkmoth's lean worst-path mean is 63.99% for Sato versus 71.60% for intensity.
Janelia's is 89.85% versus 85.85%; its longest-gap mean is 1.22 versus 1.41
voxels. Janelia's global Sato gain is only 0.027 percentage points. A five-point
gain cannot be achieved when a control already exceeds 95%. That failure
answers the preset incremental-benefit question; it does not imply poor
absolute global support. The original checks remain unchanged.

All 120 rows pass source-coordinate/intensity/count verification. All four
references were scorable with no outside-image quadrature samples. Maximum
0.5-to-0.25-voxel quadrature refinement change was 0.516 percentage points,
below the predefined one-point bound. No declared case was excluded.

Frog benefits strongly from structural weighting. Mouse improves in global
support but fails the predefined worst-path comparison against intensity.
Silkmoth and Janelia have 100% dense traced-curve support with either weighted
method. At the lean budget silkmoth Sato is slightly worse globally and worse
on the minimum encoded-path metric. The evidence supports conditional benefit from Sato;
it does not support promoting it to an unconditional product default.

Support within 2 or 6 native voxels measures the proximity of selected image
evidence to the reference curve. It is a retention proxy, not a mathematical
ceiling on a model's center predictions: a learned model can infer a medial
position from surrounding image points. It also does not measure image
foreground precision, connectivity, radius accuracy or final SWC quality.
The proximity calculation treats every selected location equally, including
zero-valued pixels. Nearby zero points can therefore count as spatial support
without retaining a particular bright voxel. Conversely, their spatial density
may encode the guide, and surrounding points may support medial inference.
Neither effect has been tested with the fresh encoder. Spatial support alone
cannot certify preservation of image contrast or downstream learnability.

The mouse reference encodes 80 components, 160 endpoints and no junctions.
Four junction checks are undefined and remain failed. Two additional failures
are real worst-path regressions under the declared 2-voxel definition.
A read-only examination identifies the seed-wise minimum Sato paths as a
0.1597-voxel path in three seeds and a 10.9913-voxel path in two seeds, in the
same central region. Those paths have full support at the separately
predeclared 6-voxel tolerance for both weighted methods. The 2-voxel failure
remains; it does not by itself prove the disappearance of a whole biological
branch. An encoded degree-defined SWC path is not necessarily an anatomical
branch.

The image/reference projections reviewed for all four sources show no
gross coordinate mismatch. Projection review cannot establish 3D alignment at
subvoxel precision, anatomy or exhaustive annotation. Mouse has visible bright
structures outside the provided trace. Their identity and annotation status
are unknown; do not label them background just because the SWC omits them.

The [33-training-reference audit](training_reference_contract_audit.md) finds
three files with multiple components, shared coordinates across components
and no encoded junctions. This needs a source annotation contract before
graph labels are constructed. It does not explain all historical fitting
failures, and no automatic repair is proposed by this report.

## Input regimes and budget use

Post-generation counts describe numerical values, not biological foreground.

| Source | Raw zero voxels | All-minimum selection boxes | Voxels above observed minimum | Dense cloud points |
|---|---:|---:|---:|---:|
| Frog | 0% | 0/16 | 21,233,663 | 331,776 |
| Mouse | 7.1856% | 0/49 | 18,414,384 | 310,000 |
| Silkmoth | 99.9755% | 27/32 | 10,543 | 671,744 |
| Janelia | 98.7636% | 47/64 | 1,355,922 | 1,713,607 |

All silkmoth boxes could retain every above-minimum voxel within their dense
quota, yet random selection need not do so. Mean dense Sato clouds contain
98.5194% zero-valued points there. Janelia has fewer nonzero voxels than its
global dense budget, but nine local boxes exceed their dense quota; 85.3779%
of its dense Sato points are zero-valued. Both percentages aggregate all five
seeds. The noisy frog image has no comparable sparse default-value structure.

This exposes a limitation of fixed volume-proportional quotas as a product
policy. It does not establish that zeros are background, prove how the source
was processed, or authorize intensity thresholding. Equal-count controls remain
valid for their declared question. A future evidence-retention policy must
handle both sparse and noisy inputs without neuron-ID rules or manual masks.
No such policy is implemented here.

Representing repeated pixel values compactly is a data-encoding choice; labeling
those locations non-neuronal is a supervision claim. Keep them separate. A
future sparse representation must preserve the default value, domain and
spatial context that the encoder needs; simply deleting zero points is not
established as an adequate PointNeXt input. Dark structures, weak contrast,
wide cell regions and source processing remain relevant limitations.

## Resource observations

Generation took 169.75 minutes on this Windows laptop's CPU; no GPU was used.
Guide computation alone took 13.56 / 14.36 / 36.32 / 94.00 minutes for frog /
mouse / silkmoth / Janelia, respectively. These are measured execution times
for this implementation and host load, not inherent method lower bounds.

The generation process's peak working set was 961.62 MiB (0.939 GiB); its peak
pagefile/commit counter was 2.22 GiB. Working set excludes paged allocations;
the commit counter is not measured paging I/O. These are process-wide cumulative
peaks, not independently isolated per-case requirements. Decoding and initial
normalization still use full-volume arrays.

The earlier 19.42-GiB whole-backend estimate for Janelia was a planning
projection, not an observed allocation or failed run. The disk backend did
complete on the 15.71-GiB installed-RAM machine. This does not establish minimum
software hardware requirements: learned-model training and inference are not
included, and full-source decode/normalization remains a scaling concern.

The research run constructs six clouds per seed and separately evaluates
reference distances. That comparison cost is not a product inference timing.
Even so, the current CPU guide latency is a material integration cost that
must be reported alongside support gains.
Reference scoring took a further 91.95 minutes. Total generation plus scoring
time was 261.70 minutes (about 4 hours 22 minutes). No extra images, seeds or
model fits were added during that time.

## Recommended decision for discussion

Keep the verified decoding, coordinate/provenance checks, fixed controls and
disk backend. Do not freeze this Sato selector as the universal Phase 1 policy.
Do not tune its weights or scales merely to make this comparison pass. Its
demonstrated benefit on some inputs remains useful evidence.

The next design decision should specify how image evidence and spatial context
are retained under a budget across sparse and noisy inputs, including how
PointNeXt consumes or reduces that representation. Exact repeated values can
be represented compactly; uncertain low-contrast structure cannot be called
background just to reduce the count. Preserve original intensities and explicit
loss/provenance. A guide can remain a fallible cue where compression is needed.
This is a proposed direction to discuss, not an implemented replacement sampler.

Resolve the bounded reference graph-encoding issue before deriving connectivity
labels. Keep geometric agreement, incomplete annotation and verified negatives
distinct. Report radius supervision according to its actual provenance.

These decisions preserve PointNeXt, Transformer proposals and GATv2. The
destination remains a runnable automatic image-to-SWC candidate with measured
whole-neuron reconstruction. Another isolated confidence fit or a guide-support
score would not establish that result. No training is authorized by this report.

## Artifacts and fixed stopping point

Generated guides, clouds, overlays, raw audits and evaluation outputs remain
under ignored `tmp/phase1_qualification_disk_20261022/`. Launch metadata and
bounded resource monitors are under
`tmp/phase1_qualification_launch_20261022/`. Original two-case results remain
separate and unchanged. No training, parameter grid, additional seed search or
automatic extension follows a failed qualification.
The retained run files total about 1.14 GiB of logical file sizes after guide
scratch removal; peak temporary disk use was not measured. No checkpoints
were created.

The [compact complete evidence](evidence/phase1_qualification_20261023_summary.json)
retains every seed row at both tolerances, all summary ranges, the unchanged
96 checks, code/source hashes, resource records and the raw-report hash.
Per-path arrays remain in the raw report. Its SHA-256 is
`9bb431e2d0c0aa29fde46765f794270f4058d1f0c694547f3cd3fd91eff06bac`.

Visible figures show every seed, alongside means. Overlays are seed-zero lean
projections and do not establish 3D connectivity:

- Frog: [support and gaps](phase1/qualification_frog22_support.png), [overlay](phase1/qualification_frog22_overlay.png).
- Mouse: [support and gaps](phase1/qualification_mouse75_support.png), [overlay](phase1/qualification_mouse75_overlay.png).
- Silkmoth: [support and gaps](phase1/qualification_silkmoth85_support.png), [overlay](phase1/qualification_silkmoth85_overlay.png).
- Janelia: [support and gaps](phase1/qualification_janelia126_support.png), [overlay](phase1/qualification_janelia126_overlay.png).

Additional byte-preserved audits record [source sparsity](evidence/qualification_20261023_input_sparsity_audit.json),
[reference structure](evidence/qualification_20261023_reference_structure_audit.json),
[mouse coordinate coincidences](evidence/qualification_20261023_mouse_reference_endpoint_audit.json),
[the failed mouse local paths](evidence/qualification_20261023_mouse_local_support_audit.json)
and [pre-reference generation integrity](evidence/qualification_20261023_generation_integrity.json).

The existing backend's regression/parity checks remain the implementation
evidence recorded in its own report. Current verification checks frozen hashes,
all 120 scored cloud records and unchanged source files; the new reference-audit
script has a successful synthetic smoke check and a complete 33-file run.
No reconstruction or generalization test was performed.
