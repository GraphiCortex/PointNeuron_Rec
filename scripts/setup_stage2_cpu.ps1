$ErrorActionPreference = 'Stop'
$pn2Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $pn2Root
$pn2Environment = Join-Path $pn2Root 'tmp\stage2_cpu_env'
$pn2Python = Join-Path $pn2Environment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pn2Python)) {
    & '.\tmp\phase1_cpu_env\Scripts\python.exe' -m venv $pn2Environment
    if ($LASTEXITCODE -ne 0) { throw 'Stage2 environment creation failed' }
}
& $pn2Python -m pip install --only-binary=:all: -r '.\requirements-stage2-cpu.txt'
if ($LASTEXITCODE -ne 0) { throw 'Stage2 CPU dependency installation failed' }
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $pn2Python -B -m pytest '.\tests\test_stage2_trees.py' -q --basetemp=tmp/stage2_setup_tests
if ($LASTEXITCODE -ne 0) { throw 'Stage2 synthetic CPU checks failed; do not launch data runs' }
Write-Output 'STAGE2 CPU ENVIRONMENT READY; no neuron model or data experiment launched'
