# New verification

The fresh project's `pyproject.toml` configures local source/test discovery.
Synthetic checks cover analytic curve distances, arclength/subdivision behavior,
spacing, SWC forests and validation, explicit manifests, failure denominators,
paired reporting, CLI output and preservation of existing reports.

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
