param(
    [ValidateSet('check','replay','timing')][string]$Action = 'check',
    [string]$Output,
    [string]$CacheRoot,
    [string]$CpuPreflight
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root 'tmp\phase1_cpu_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Pinned Phase 1 CPU runtime missing: $Python" }
if (-not $CacheRoot) { $CacheRoot = Join-Path $Root 'tmp\node_cloud_cache_20261031' }
if (-not $Output) {
    if ($Action -eq 'check') { $Output = Join-Path $Root 'tmp\node_cloud_cache_cpu_20261031' }
    else { $Output = Join-Path $Root ("tmp\node_cloud_cache_{0}_{1}" -f $Action,(Get-Date -Format 'yyyyMMdd_HHmmss')) }
}
if (Test-Path -LiteralPath $Output) { throw "Run output exists; preserve it: $Output" }
Write-Host 'PN2 image-only point-cache builder. Original raw images; no SWCs, targets, model or GPU.'
Write-Host 'Same 128-voxel clipped boxes, Sato scales 1/2/4, uniform probability mass .1, 2048 distinct source voxels per box.'
if ($Action -eq 'check') {
    Write-Host 'Synthetic uint16 source only: exact whole/disk/reused guide and point-input parity. No real images.'
} elseif ($Action -eq 'replay') {
    Write-Host 'Two previously inspected TRAINING images, fruitfly 32 and frog 22. Verified existing guides copied; no filter rerun.'
    Write-Host 'All 48 image boxes prepared; four original Sato input packets must replay exactly. 300-second cap per image.'
} else {
    Write-Host 'One uncached TRAINING image: cultured human neuron 41 / Image5; XYZ 640x640x29, 25 boxes.'
    Write-Host 'CPU disk-backed Sato preparation and point caching, 3600-second wall cap. Valid outputs are retained for later use.'
    Write-Host 'Historical similar-size disk filters took about 8-10 minutes; this new case has not been timed.'
    Write-Host 'This does not start the remaining cohort or any model training.'
}
Write-Host "Cache root: $CacheRoot"
Write-Host "Run output: $Output"
$Arguments = @('-B',(Join-Path $PSScriptRoot 'node_cloud_cache.py'),'--action',$Action,
    '--config',(Join-Path $Root 'configs\node_cloud_cache.json'),'--output',$Output,'--cache-root',$CacheRoot)
if ($CpuPreflight) { $Arguments += @('--cpu-preflight',(Resolve-Path -LiteralPath $CpuPreflight).Path) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Image cache $Action failed with exit code $LASTEXITCODE. Preserve logs/partial files; no automatic retry." }
