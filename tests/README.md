# New verification

The fresh project's `pyproject.toml` configures local source/test discovery.
Synthetic checks cover analytic curve distances, arclength/subdivision behavior,
spacing, SWC forests and validation, explicit manifests, failure denominators,
paired reporting, CLI output and preservation of existing reports.
Image-input checks additionally cover independent raw/PBD format fixtures,
byte order/axes, explicit image-only manifests, native/physical metadata,
whole-volume tile boundaries, small images and failed/unsupported cases.
The PN1-only source/tests were removed after the 22 October scope clarification;
their historical 150-test run remains documented at commit `48519ff`. Current
tests cover the retained evaluator and image interface, without a model
dependency. Raw inspection is checked by running its two explicit inputs and
visually reviewing its saved figures, not by adding tests for plotting layout.

Phase 1 adds analytic 3D Gaussian tubes, scale/polarity/axis/boundary and junction
checks; unique count/source recovery and sampling distribution controls; analytic
curve support and branch gaps; and a two-command integration fixture in which the
reference does not exist until clouds are saved. A rounded spatial-index boundary
regression was found and fixed before real cloud generation. Candidate search is
expanded conservatively; measured distances and scoring tolerances are unchanged.

From the PN2 root, use the isolated CPU runtime for the complete suite:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe -m pytest -q
```

The retained historical environment has an incompatible napari/skimage plugin;
do not use it for the new image-filter tests or repair it in place. The isolated
runtime and lock are documented in scripts/README.md. No real dataset is opened
by the tests.

The explicit geometry-only reference mode adds signed-type preservation,
unchanged strict defaults, rejection of malformed geometry under either mode,
and a CLI control proving that changing only type metadata cannot change curve
support. The integration check also preserves the original strict report and
cloud packets during an opt-in rescore. The complete suite passes **149 tests**
as of the 22 October reference-policy change; no real dataset is opened.

Retained historical tests remain useful evidence for reused components. No broad
test or training run is started by creating this folder.

30 October node-learning preparation adds twelve Torch-dependent tests in
`test_node_learning.py`. Run these in the retained Torch runtime, disabling
unrelated third-party plugin auto-loading for that command:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -B -m pytest tests\test_node_learning.py --basetemp=tmp\node_tests_torch_20261030 -q
```

The test module skips explicitly when Torch is unavailable in the isolated image
runtime. Its optimizer updates use only a tiny synthetic fixture to verify exact
checkpoint continuation; no real neuron is fitted. Five separate controller
tests mock all workers, checking timeout/report failures without Torch or GPU.
Together with twelve rectangular target tests, the focused preparation checks
pass **29 tests**. The separate real-input CPU preflight checks actual supervised
loss/backward without an optimizer; it is execution evidence, not learnability.

The GN8 control adds fourteen Torch tests in `test_groupnorm_node.py`, including
fresh-layer replacement, rejection of trained BN, same-input batch independence,
exact mode behavior, recipe-bound checkpoints, explicit consistency failure and
box/target pairing for the saved baseline. Together with the twelve shared node
tests, **26 pass** in the retained runtime. The real-input GN CPU preflight has
no optimizer and is recorded separately from those synthetic tests.
