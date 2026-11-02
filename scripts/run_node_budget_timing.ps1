param(
    [ValidateSet('timing')][string]$Action = 'timing',
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if (-not $Output) { $Output = Join-Path $Root ("tmp\node_budget_timing_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Preserve existing output: $Output" }
Write-Host 'Two point budgets: 2048 and 8192. Four fixed TRAINING boxes: fruitfly 32/27,32/20; frog 22/2,22/5.'
Write-Host 'Same fresh GN8 PointNeXt/geometric-query Transformer, 512 queries, targets 18/85/7/68, AdamW LR .0001.'
Write-Host 'Eight throwaway updates per budget, inference timing and exact model/Adam/RNG reload. No trained checkpoint retained.'
Write-Host 'Fresh GPU process per arm; 600-second cap each; failure stops without retry. No training action exists.'
Write-Host 'This measures complete-model resource cost on exposed boxes; it does not score generalization or settle the input budget.'
Write-Host "Output: $Output"
& $Python -B (Join-Path $PSScriptRoot 'node_budget_timing.py') --action $Action --output $Output
if ($LASTEXITCODE -ne 0) { throw "Two-budget timing failed with exit code $LASTEXITCODE. Preserve reports; no automatic retry." }
