param(
    [ValidateSet('plan','prepare')][string]$Action = 'plan',
    [string]$Output,
    [string]$PlanRun,
    [string]$CacheRoot
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root 'tmp\phase1_cpu_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Pinned CPU runtime missing: $Python" }
if (-not $CacheRoot) { $CacheRoot = Join-Path $Root 'tmp\node_cloud_cache_20261031' }
if (-not $Output) { $Output = Join-Path $Root ("tmp\node_cloud_cache_bulk_{0}_{1}" -f $Action,(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw "Run output exists; preserve it: $Output" }
Write-Host 'PN2 CPU image preparation: 33 training + 7 development images. No references, targets, model or GPU.'
Write-Host 'Same Sato kernels/scales and point sampling. Processing blocks 128; model crops 128 and point count 2048 remain unchanged.'
if ($Action -eq 'plan') {
    Write-Host 'Metadata/complete-cache verification and remaining-work estimate only. No image filtering.'
} else {
    if (-not $PlanRun) { throw 'Preparation requires -PlanRun from the qualified bulk plan.' }
    Write-Host 'One image worker at a time. Completed valid cases skipped; partial cases/failures stop without retry or overwrite.'
    Write-Host 'Fresh-case time limits scale from measured preparation (1-6 hours); verified guide copies allow 10 minutes.'
    Write-Host 'Overall wall allowance: 72 hours. This is data preparation, not training; inspect the plan estimate before launching.'
}
Write-Host "Cache root: $CacheRoot"
Write-Host "Run output: $Output"
$Arguments = @('-B',(Join-Path $PSScriptRoot 'node_cloud_cache_bulk.py'),'--action',$Action,
    '--config',(Join-Path $Root 'configs\node_cloud_cache_bulk.json'),'--cache-root',$CacheRoot,'--output',$Output)
if ($PlanRun) { $Arguments += @('--plan-run',(Resolve-Path -LiteralPath $PlanRun).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "CPU cohort preparation $Action failed with exit code $LASTEXITCODE. Preserve completed/partial cache evidence; no automatic retry." }
