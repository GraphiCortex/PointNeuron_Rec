param(
    [ValidateSet('check', 'timing', 'train')][string]$Action = 'check',
    [string]$Output,
    [string]$TimingRun,
    [string]$CpuPreflight
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) {
    if ($Action -eq 'check') {
        $Output = Join-Path $Root 'tmp\geometric_query_cpu_20261031'
    } else {
        $Output = Join-Path $Root ("tmp\geometric_query_{0}_{1}" -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
}
if (Test-Path -LiteralPath $Output) { throw "Output exists; preserve it and choose a fresh directory: $Output" }
Write-Host 'Geometric queries: image-derived FPS starting locations + encoded features; position = start + learned unbounded offset.'
Write-Host 'Four frozen Sato TRAINING boxes: fruitfly 32/27,32/20; frog 22/2,22/5. Targets 18/85/7/68.'
Write-Host 'Same fresh GN8 parameters, all 2048 points in memory, 512 queries, node loss, AdamW LR .0001 and cutoff .5.'
Write-Host 'Reference: saved GN learned-token fit. No source image/filter/sampler rerun or learned checkpoint transfer.'
$Mode = if ($Action -eq 'check') { 'cpucheck' } else { $Action }
if ($Action -eq 'check') {
    Write-Host 'CPU: real-data backward without optimizer; mode consistency; raw starts vs actual untrained predictions.'
    Write-Host 'Synthetic 256/4096-point interfaces and one throwaway synthetic Adam update/reload. No GPU.'
} elseif ($Action -eq 'timing') {
    Write-Host 'One GPU worker: eight throwaway updates, exact train/eval consistency and model/Adam/RNG reload.'
    Write-Host 'RTX 3070 Ti Laptop GPU; 300-second hard cap. No trained checkpoint retained. Training is a separate action.'
} else {
    if (-not $TimingRun) { throw 'Training needs -TimingRun from this geometric candidate; prior GN timing is incompatible.' }
    Write-Host 'One GPU fit: maximum 2000 updates, 500 per box. No extension, search or automatic resume.'
    Write-Host 'First eight updates must replay candidate timing. State saved before evaluations. Same 95% node P/R gate.'
    Write-Host 'Measured timing: about 42-62 minutes of updates plus IO/evaluation. External wall allowance: 90 minutes; update cap unchanged.'
}
Write-Host "Output: $Output"
$EntryPoint = if ($Action -eq 'train') { 'geometric_query_train_controller.py' } else { 'geometric_query_experiment.py' }
$Arguments = @('-B', (Join-Path $PSScriptRoot $EntryPoint), '--action', $Mode,
    '--config', (Join-Path $Root 'configs\geometric_query_fit.json'), '--output', $Output)
if ($TimingRun) { $Arguments += @('--timing-run', (Resolve-Path -LiteralPath $TimingRun).Path) }
if ($CpuPreflight) { $Arguments += @('--cpu-preflight', (Resolve-Path -LiteralPath $CpuPreflight).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Geometric query $Action stopped with exit code $LASTEXITCODE. Preserve results; no automatic retry." }
