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

From the PN2 root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe -m pytest -q
Remove-Item Env:\PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

The retained environment contains an unrelated incompatible napari/skimage
pytest plugin. Disabling automatic third-party plugin loading isolates these
tests; it does not change project dependencies. No dataset is opened by them.

Retained historical tests remain useful evidence for reused components. No broad
test or training run is started by creating this folder.
