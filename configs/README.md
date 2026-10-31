# New configurations

Fresh benchmark manifests and reviewed experiment configurations belong here.
The first bounded node-learning recipe and user-launched fits are complete.
Both fits failed the frozen learnability gate; see the
[saved-fit review](../docs/sampler_node_fit_results.md) before scheduling more training.

[geometric_query_fit.json](geometric_query_fit.json) declares the user-approved
geometric-query candidate against the completed GN fit. It derives the same
data, loss, optimizer and cap, pinning the GN baseline and candidate code hashes.
Only query content initialization and position parameterization change. The
[implementation report](../docs/geometric_query_implementation.md) records the
passed CPU check and next manual GPU timing command. This does not promote an
untrained model or change the earlier failed gates.

The [user-launched geometric timing passes](../docs/geometric_query_timing_results.md).
The experiment recipe stays byte-identical for the capped fit. The external
train controller allows 90 minutes of wall time because measured updates alone
can exceed the legacy parent's 60-minute allowance. Its launch manifest records
that process policy separately; the 2000-update cap and all learning settings
are unchanged.

[groupnorm_node_fit.json](groupnorm_node_fit.json) now declares the separately
approved normalization-only control against the saved Sato/BN fit. It derives
unchanged learning/data settings from the original recipe and pins its baseline
and two variant source files. The [GN8 control report](../docs/groupnorm_node_control.md)
records CPU equality/gradient checks, fixed eight-group policy and the next manual
GPU timing command. No new optimizer/LR/query/target policy is included.

The [user-launched GN timing now passes](../docs/groupnorm_node_timing_results.md),
including exact mode consistency and state reload. The same frozen config is
used for the completed single-model bounded fit; no setting changed after timing.
The [fit review](../docs/groupnorm_node_fit_results.md) confirms mode consistency
but failed node learnability. The configuration remains frozen; no new query,
loss, threshold or optimizer configuration is adopted by that result.

[sampler_learnability_inputs.json](sampler_learnability_inputs.json) fixes the
two-arm, four-training-box input preparation for the
[learned comparison protocol](../docs/sampler_learnability_protocol.md).
It preserves existing 2048-slot packets and discloses repeated control points.
It is not an executable model/optimizer recipe and does not authorize training.

[sampler_interface_check.json](sampler_interface_check.json) pins the prepared
input manifest, references' prior evidence and encoder/head source for the
[CPU interface audit](../docs/sampler_model_interface_results.md). It selects no
supervised training loss or optimizer and does not expose a GPU action.

[sampler_node_fit.json](sampler_node_fit.json) fixes the user-approved node-only
comparison following that audit. The [fit instructions](../docs/sampler_node_fit.md)
declare the loss, ordinary-BN inference, diagnostic policy, optimizer, 2000-update
per-arm cap and fixed acceptance gate. The CPU preflight and
[user-launched paired timing](../docs/sampler_node_timing_results.md) pass.
The separate train action requires passed paired timing, preserving exact
model/optimizer/RNG state and the four declared boxes throughout.

[phase1_sampling_probe.json](phase1_sampling_probe.json) freezes the three-arm
CPU input diagnostic, not a training/product policy. Its
[protocol](../docs/phase1_sampling_protocol.md) explains the scales, count budgets,
controls, measurements and stopping rules.
[phase1_cpu_requirements.txt](phase1_cpu_requirements.txt) declares direct runtime
pins; [phase1_cpu_lock.txt](phase1_cpu_lock.txt) records the complete installed
dependency set for Python 3.11 on Windows. Both belong to the isolated new runtime.

[phase1_qualification_cases.json](phase1_qualification_cases.json) fixes four
additional image-only training cases, aliases 22/75/85/126, using the unchanged
sampling recipe. The [qualification plan and resource preflight](../docs/phase1_qualification_plan.md)
record why whole-volume generation is currently held. No clouds or training are
created by this configuration.

[development_benchmark.json](development_benchmark.json) lists seven existing
development identities, explicit image/reference paths, native voxel units and
training-exposure metadata. Identity joins use stable sample IDs. Its source
inventory has 163 records and must not be indexed as the filtered 155-record
inventory. File existence was checked; image decoding, calibration and source
content provenance remain separate work.

[training_inventory.json](training_inventory.json) lists the provisional 33
training identities and explicit source paths. It is source metadata, not a
cache or executable training recipe. The development-only evaluator rejects its
training partition. No image/reference contents were read to create it.

[input_tiling.example.json](input_tiling.example.json) declares a complete
voxel-grid example for input audits; it is not a frozen model-preparation
recipe. [reused_components.json](reused_components.json) pins the reused decoder
source and destination hashes, including explicit LF normalization.

[phase1_raw_review.json](phase1_raw_review.json) lists two original training
images without reference fields for the raw inspection script. No segmentation
or sampling policy is selected by this file.

The PN1 cloud-control config is removed from active source, preserved in commit
`48519ff` with its [historical result](../docs/reference_forward_interface.md).
Its fixed cutoff and point budget are not PN2 defaults.

The retained [historical neuron partition](../../pointneuron2/configs/pointneuron2/partition_seed0.json)
is reference metadata for the next benchmark specification. Its 33/7/8 groups
are disjoint within PN2, but all 48 neurons appeared in historical PN1 training.
Old cache paths must not be assumed to be valid fresh-workspace inputs.
