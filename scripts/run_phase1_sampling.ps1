param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('check','generate','evaluate','evaluate-geometry')][string]$Action,
    [string]$OutputDirectory = 'tmp\phase1_sampling_20261022'
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = Join-Path $Root 'tmp\phase1_cpu_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Prepare the isolated CPU runtime as described in scripts/README.md.' }
$Run = [IO.Path]::GetFullPath((Join-Path $Root $OutputDirectory))
$LogFolder = Join-Path $Root 'tmp\phase1_console'
New-Item -ItemType Directory -Path $LogFolder -Force | Out-Null
$Log = Join-Path $LogFolder ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + $Action + '.log')
Write-Host 'CPU input diagnostic only. No GPU, model training or PN1 inference.'
Write-Host 'Two original TRAINING images; three controls; two densities; five fixed seeds.'
if ($Action -eq 'check') {
    & $Python -m pytest -q 2>&1 | Tee-Object -FilePath $Log
} elseif ($Action -eq 'generate') {
    Write-Host 'Image-only cloud generation; references are not read. Output directory must be fresh.'
    & $Python scripts\probe_phase1_sampling.py --cases configs\phase1_raw_review.json --recipe configs\phase1_sampling_probe.json --output-dir $Run 2>&1 | Tee-Object -FilePath $Log
} elseif ($Action -eq 'evaluate') {
    Write-Host 'Score frozen clouds against the two explicit training references; no cloud changes.'
    & $Python scripts\evaluate_phase1_sampling.py --clouds (Join-Path $Run 'clouds.json') --training-inventory configs\training_inventory.json 2>&1 | Tee-Object -FilePath $Log
} else {
    Write-Host 'Explicit geometry-only reference mode: preserve all signed type metadata and nodes.'
    Write-Host 'No anatomical type classification, source edits, filtering rerun or training. Original strict report is preserved.'
    & $Python scripts\evaluate_phase1_sampling.py --clouds (Join-Path $Run 'clouds.json') --training-inventory configs\training_inventory.json --reference-type-policy opaque_integer --output-dir (Join-Path $Run 'evaluation_opaque_integer') 2>&1 | Tee-Object -FilePath $Log
}
$Result = $LASTEXITCODE
if ($Result -ne 0) { throw "Phase 1 $Action failed with exit code $Result. Preserve outputs; no automatic retry or recipe change." }
Write-Host "Completed $Action. Console log: $Log"
