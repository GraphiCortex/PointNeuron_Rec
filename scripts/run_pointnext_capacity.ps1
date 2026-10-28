param(
    [ValidateSet('check', 'sweep', 'replay64')][string]$Action = 'check',
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) {
    $Output = Join-Path $Root ("tmp\pointnext_capacity_{0}_{1}" -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if ($Action -eq 'replay64') {
    $Config = Join-Path $Root 'configs\pointnext_compact_replay64.json'
    Write-Host '64k untrained replay with compact three-neighbor index storage only.'
    Write-Host 'Same seed-zero weights, synthetic grid, batch 1, FP32, radii/K and normalization.'
    Write-Host 'CPU exact-output/gradient/neighbor and storage gate runs first; failure prevents GPU execution.'
    Write-Host 'GPU: 65536 points only; 1 warmup + 2 measured iterations for each of eval and forward/backward.'
    Write-Host 'No optimizer or checkpoint. Fresh GPU process; 300-second cap; no automatic retry.'
    Write-Host "Output: $Output"
    if (Test-Path -LiteralPath $Output) { throw "Output already exists; preserve it and choose a new directory: $Output" }
    $null = New-Item -ItemType Directory -Path $Output
    & $Python (Join-Path $PSScriptRoot 'check_pointnext_compact_interpolation.py') --config $Config --output (Join-Path $Output 'cpu')
    if ($LASTEXITCODE -ne 0) { throw 'CPU parity/storage check failed. GPU replay was not launched.' }
    & $Python (Join-Path $PSScriptRoot 'probe_pointnext_capacity.py') --action sweep --config $Config --output (Join-Path $Output 'gpu')
    if ($LASTEXITCODE -ne 0) { throw '64k replay stopped. Inspect preserved logs/results; no automatic retry.' }
    return
}
Write-Host 'Pinned historical native PointNeXt encoder, fresh seed-zero weights, batch 1, FP32.'
Write-Host 'Synthetic image-grid points only. No optimizer, checkpoint, real cloud or reference SWC.'
if ($Action -eq 'sweep') {
    Write-Host 'GPU: 8192/16384/32768/65536 points; 1 warmup + 2 measured iterations per mode.'
    Write-Host 'Modes: eval forward; train forward/backward without parameter updates.'
    Write-Host 'Each budget has a 300-second process cap. Stop on first timeout/OOM/failure; no retry.'
    Write-Host 'At most 20 minutes plus process overhead; this measures this native backend only.'
} else {
    Write-Host 'CPU only: 256-point adapter/operator/forward/backward check.'
}
Write-Host "Output: $Output"
& $Python (Join-Path $PSScriptRoot 'probe_pointnext_capacity.py') --action $Action --output $Output
if ($LASTEXITCODE -ne 0) { throw "Capacity $Action stopped with exit code $LASTEXITCODE. Inspect saved evidence; no automatic retry." }
