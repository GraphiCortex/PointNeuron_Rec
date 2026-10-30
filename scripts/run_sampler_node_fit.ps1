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
        $Output = Join-Path $Root 'tmp\sampler_node_cpu_verified_20261030'
    } else {
        $Output = Join-Path $Root ("tmp\sampler_node_{0}_{1}" -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
}
if (Test-Path -LiteralPath $Output) { throw "Output exists; preserve it and choose a fresh directory: $Output" }
Write-Host 'Two samplers, four fixed TRAINING boxes: fruitfly 32/27,32/20; frog 22/2,22/5.'
Write-Host 'Same fresh PointNeXt/Transformer, 2048 slots, targets 18/85/7/68, 512 queries.'
Write-Host 'Node position + balanced used-slot loss only. Radius unsupervised. Ordinary saved-BN inference scored at .5.'
Write-Host 'Thresholded frog 22/2 has 195 distinct voxels in 2048 slots; no added/jittered locations.'
$Mode = if ($Action -eq 'check') { 'cpucheck' } else { $Action }
if ($Action -eq 'check') {
    Write-Host 'CPU loss/gradient checks only; no optimizer or real-data fit.'
} elseif ($Action -eq 'timing') {
    Write-Host 'GPU throwaway timing: 8 updates per sampler; fresh process each; 300-second cap per worker.'
    Write-Host 'Exact model/Adam/RNG serialization reload in memory. No trained checkpoint retained.'
    Write-Host 'This does not start the capped training comparison.'
} else {
    if (-not $TimingRun) { throw 'Training requires -TimingRun pointing to a passed current-recipe paired timing run.' }
    Write-Host 'GPU fit: 2000 updates per sampler (500 per box); no extension, retry or automatic resume.'
    Write-Host 'Fresh AdamW LR .0001, weight decay zero, clip10. First eight updates must exactly replay timing.'
    Write-Host 'Model/optimizer/RNG states saved before evaluations. 3600-second hard cap per worker.'
}
Write-Host "Output: $Output"
$Arguments = @('-B', (Join-Path $PSScriptRoot 'sampler_node_experiment.py'), '--action', $Mode,
    '--config', (Join-Path $Root 'configs\sampler_node_fit.json'), '--output', $Output)
if ($TimingRun) { $Arguments += @('--timing-run', (Resolve-Path -LiteralPath $TimingRun).Path) }
if ($CpuPreflight) { $Arguments += @('--cpu-preflight', (Resolve-Path -LiteralPath $CpuPreflight).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Sampler node $Action stopped with exit code $LASTEXITCODE. Preserve logs/states; no automatic retry." }
