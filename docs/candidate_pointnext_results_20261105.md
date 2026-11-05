# Frozen PointNeXt export completed

The existing user-started CPU export was found running and monitored to completion,
not launched twice. Guard completed exit0 in390.42 seconds. All611 feature
packets were verified against recorded hashes,32-channel shapes, finite source
and queried features, and normalized interpolation weights. Training673,235
and development160,634 candidate instances are covered. No labels/SWCs were
opened by the export; model weights/buffers and RNG unchanged, optimizer absent,
updates0, CUDA not initialized. No GPU job was launched.

Source `tmp/candidate_pointnext_20261105/complete.json` binds the immutable broad
terminal checkpoint and Stage1 candidate manifest. [Portable evidence](evidence/candidate_pointnext_completed_20261105.json)
records completion/hash/validation and interpolation-distance summaries.

| Candidate nearest-input distance | Training | Development |
|---|---:|---:|
| Median | 1.092 | 0.534 |
| 90th percentile | 4.472 | 1.935 |
| 95th percentile | 5.745 | 2.337 |
| 99th percentile | 7.874 | 3.742 |
| Maximum | 17.205 | 11.832 |
| More than2 voxels | 28.01% | 8.78% |
| More than6 voxels | 4.13% | 0.15% |

These native-voxel distances describe feature interpolation support, not feature
error or reference coverage. Classical halo/unsampled candidate positions can
be far from input points. The distributions pool candidate instances, with
large source/count differences between neurons; they are not a controlled
encoder-generalization comparison. No distance cutoff/source/policy is selected
or feature discarded from these results.

Successful export establishes execution and feature availability, not label
separability or reconstruction improvement. Frozen encoder AND learned candidate
positions were trained on all33 training neurons. The user's chosen selector-only
CV remains conditional on that upstream exposure; the handcrafted-only arm and
CV/development gap remain required checks. Development is repeatedly inspected;
this export uses image inputs only and does not spend a frozen model's development
selector scoring. No selector has been trained. The saved-label diagnostic gate
must be reviewed before any selector training.
