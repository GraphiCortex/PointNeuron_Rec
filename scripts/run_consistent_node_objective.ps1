param(
    [ValidateSet('timing')][string]$Action = 'timing',
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) { $Output = Join-Path $Root ("tmp\consistent_objective_timing_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Preserve existing output: $Output" }
Write-Host 'Objective-only timing: frozen 2048-point model and saved Adam/RNG at update 9200, independently forked for both arms.'
Write-Host 'Same PointNeXt/GN8/geometric Transformer, four training-only target-count strata, 512 queries, clouds, targets, LR .0001 and clip 10.'
Write-Host 'Candidate: per-query CE and normalized geometry; assignment minimizes that same objective. No new heads, labels or count penalty.'
Write-Host 'Eight throwaway updates per arm plus two executions of one next-update reload probe. No trained checkpoint retained.'
Write-Host 'RTX 3070 Ti Laptop GPU; separate processes; 600-second limit each. Passed current CPU audit required before GPU access.'
Write-Host 'Timing/reload only. No full-training action or reconstruction-quality claim.'
Write-Host "Output: $Output"
& $Python -B (Join-Path $PSScriptRoot 'check_consistent_node_objective.py') --action $Action --output $Output
if ($LASTEXITCODE -ne 0) { throw "Objective timing failed with exit code $LASTEXITCODE. Preserve logs; no automatic retry." }
