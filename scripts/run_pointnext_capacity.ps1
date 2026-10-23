param(
    [ValidateSet('check', 'sweep')][string]$Action = 'check',
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) {
    $Output = Join-Path $Root ("tmp\pointnext_capacity_{0}_{1}" -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'))
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
