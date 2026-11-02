param([string]$Output)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
$Python=Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) { $Output=Join-Path $Root ("tmp\node_budget_integration_timing_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Preserve existing output: $Output" }
Write-Host 'Integrated resource check: 2048 and 8192 points, same fresh GN8 PointNeXt/geometric queries, 512 proposals.'
Write-Host 'Eight throwaway updates per budget from the real shared TRAINING schedule; all 415 boxes of seven DEVELOPMENT neurons inferred/scored without gradients.'
Write-Host 'Includes packet loading, transfer, core node/curve metrics and exact model/Adam/RNG/data-cursor reload. No training action or retained trained checkpoint.'
Write-Host 'Previous four-box projections suggest roughly 15 minutes for the two development passes; actual loader/scoring cost is unmeasured.'
Write-Host 'Worker stop limits: 15 minutes at 2048, 30 minutes at 8192. Failure stops without retry. No source image filtering.'
Write-Host "Output: $Output"
& $Python -B (Join-Path $PSScriptRoot 'node_budget_integration_timing.py') --action timing --output $Output
if ($LASTEXITCODE -ne 0) { throw "Integrated timing failed with exit code $LASTEXITCODE. Preserve logs; no automatic retry." }
