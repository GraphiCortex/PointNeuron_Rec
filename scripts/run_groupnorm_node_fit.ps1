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
        $Output = Join-Path $Root 'tmp\groupnorm_node_cpu_20261030'
    } else {
        $Output = Join-Path $Root ("tmp\groupnorm_node_{0}_{1}" -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
}
if (Test-Path -LiteralPath $Output) { throw "Output exists; preserve it and choose a fresh directory: $Output" }
Write-Host 'Normalization-only control: fresh PointNeXt/Transformer with sixteen GroupNorm layers, eight groups each.'
Write-Host 'Four frozen Sato TRAINING boxes: fruitfly 32/27,32/20; frog 22/2,22/5. Targets 18/85/7/68.'
Write-Host 'Same 2048 points, 512 queries, initialization values, node loss, AdamW LR .0001 and cutoff .5.'
Write-Host 'Reference: completed Sato/BatchNorm fit. No threshold-control rerun or old-checkpoint conversion.'
$Mode = if ($Action -eq 'check') { 'cpucheck' } else { $Action }
if ($Action -eq 'check') {
    Write-Host 'CPU checks: exact train/eval outputs on four real clouds and synthetic 256/4096-point grids; node-loss backward only.'
    Write-Host 'No optimizer or learned update on real data.'
} elseif ($Action -eq 'timing') {
    Write-Host 'One GPU timing worker: eight throwaway updates, exact train/eval consistency and model/Adam/RNG reload.'
    Write-Host '300-second hard cap. No trained checkpoint retained; full training is a separate action.'
} else {
    if (-not $TimingRun) { throw 'Training needs -TimingRun from this GN control; the old BN timing is incompatible.' }
    Write-Host 'One GPU fit: maximum 2000 updates, 500 per box. No extension, search or automatic resume.'
    Write-Host 'Train/eval consistency and the unchanged 95% node precision/recall gate are separate verdicts.'
    Write-Host 'The first eight updates must replay the passed GN timing. State saved before every evaluation.'
}
Write-Host "Output: $Output"
$Arguments = @('-B', (Join-Path $PSScriptRoot 'groupnorm_node_experiment.py'), '--action', $Mode,
    '--config', (Join-Path $Root 'configs\groupnorm_node_fit.json'), '--output', $Output)
if ($TimingRun) { $Arguments += @('--timing-run', (Resolve-Path -LiteralPath $TimingRun).Path) }
if ($CpuPreflight) { $Arguments += @('--cpu-preflight', (Resolve-Path -LiteralPath $CpuPreflight).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "GN node $Action stopped with exit code $LASTEXITCODE. Preserve results; no automatic retry." }
