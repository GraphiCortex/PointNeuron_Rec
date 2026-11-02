param(
    [ValidateSet('train','resume')][string]$Action='train',
    [ValidateSet('both','2048','8192')][string]$Budget='both',
    [string]$Checkpoint,
    [string]$Output
)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
$Python=Join-Path $Root '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Retained Torch runtime missing: $Python" }
if ($Action -eq 'resume' -and (-not $Checkpoint -or $Budget -eq 'both')) { throw 'Resume requires one explicit -Budget and a complete -Checkpoint file.' }
if ($Action -eq 'train' -and $Checkpoint) { throw 'Use -Action resume for a saved checkpoint.' }
if (-not $Output) { $Output=Join-Path $Root ("tmp\node_budget_fit_{0}_{1}" -f $Action,(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Preserve existing output: $Output" }
Write-Host 'PN2 matched point-budget study: 33 TRAINING neurons / seven exposed DEVELOPMENT neurons. No reserved test files.'
Write-Host 'Fresh GN8 PointNeXt/geometric-query Transformer; 512 proposals; 2048 versus nominal 8192 source points; clipped 7650-point box retained without padding.'
Write-Host '460 supervised training boxes; 20 epochs / maximum 9200 total updates per arm. Unknown boxes receive no negative-only training.'
Write-Host 'Same AdamW LR .0001, balanced node loss, clipping and cutoff .5. No radius/GAT training or graph/SWC quality claim.'
Write-Host 'All 415 development boxes scored at 0/5/10/20 epochs; all 1964 training boxes scored at the terminal state. Profiles use fixed training-only cutoffs.'
Write-Host 'Full immutable model/Adam/RNG/data-cursor checkpoint each epoch. First eight fit updates must replay the passed integrated timing.'
Write-Host 'Planning allowance for BOTH arms: 10-14 hours including margin; not a measured upper bound. Device: RTX 3070 Ti Laptop GPU.'
Write-Host 'Independent worker stop limits: 6 hours at 2048 / 12 hours at 8192. Failure stops without automatic retry or cap extension.'
Write-Host "Action: $Action | Budget: $Budget | Output: $Output"
$Arguments=@('-B',(Join-Path $PSScriptRoot 'node_budget_fit.py'),'--action',$Action,'--output',$Output)
if ($Budget -ne 'both') { $Arguments+=@('--budget',$Budget) }
if ($Checkpoint) { $Arguments+=@('--checkpoint',(Resolve-Path -LiteralPath $Checkpoint).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Bounded paired fit failed with exit code $LASTEXITCODE. Preserve logs/checkpoints; no automatic retry." }
