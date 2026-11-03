param(
    [ValidateSet('train','resume')][string]$Action = 'train',
    [ValidateSet('baseline','consistent')][string]$Method,
    [string]$Checkpoint,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if ($Action -eq 'resume' -and (-not $Method -or -not $Checkpoint)) { throw 'Explicit recovery requires Method and Checkpoint.' }
if ($Action -eq 'train' -and ($Method -or $Checkpoint)) { throw 'Fresh paired continuation forks the agreed origin; use resume only for recovery.' }
if (-not $Output) { $Output = Join-Path $Root ("tmp\consistent_node_fit_{0}_{1}" -f $Action,(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Preserve existing run: $Output" }
Write-Host 'PN2 paired objective continuation: SAME saved 2048-point model, Adam and RNG at update 9200; objective is the only arm difference.'
Write-Host '33 TRAINING neurons, 460 supervised boxes per epoch. Five NEW epochs / 2300 updates per arm. No automatic extension.'
Write-Host 'Same GN8 PointNeXt/geometric Transformer, 512 queries, clouds, targets, AdamW LR .0001, clip 10 and cutoff .5.'
Write-Host 'Terminal evaluation: all 415 DEVELOPMENT boxes and 1964 TRAINING boxes, including unknown boxes; no reserved data.'
Write-Host 'Passed objective GPU timing and current CPU state proof required. Immutable full-state checkpoint every 460 updates, before terminal evaluation.'
Write-Host 'RTX 3070 Ti Laptop GPU. Planning allowance: 3-4 hours for BOTH arms; provisional, not an upper bound. Worker limit: four hours per arm including evaluation.'
Write-Host 'A scientific gate failure preserves results and allows the other arm; an execution failure stops without retry.'
Write-Host "Action: $Action | Output: $Output"
$Arguments = @('-B',(Join-Path $PSScriptRoot 'fit_consistent_node_objective.py'),'--action',$Action,'--output',$Output)
if ($Action -eq 'resume') { $Arguments += @('--method',$Method,'--checkpoint',$Checkpoint) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Paired objective $Action failed with exit code $LASTEXITCODE. Preserve logs/checkpoints; no automatic retry or cap change." }
