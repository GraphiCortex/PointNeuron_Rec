param(
    [Parameter(Mandatory=$true)][ValidateSet('check','compare')][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$SourceExperiment,
    [string]$NativeExperiment,
    [string]$ExperimentDirectory,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $SourceExperiment) { $SourceExperiment = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_attention_20261013' }
if (-not $NativeExperiment) { $NativeExperiment = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_pilot_20261012' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_image_reservation_20261013' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
if ($Action -eq 'compare' -and (Test-Path -LiteralPath $ExperimentDirectory)) { throw 'Preserve outputs; choose a fresh ExperimentDirectory.' }
$cliArguments = @('-u',(Join-Path $PSScriptRoot 'compare_field_image_reservation.py'),'--action',$Action,
    '--manifest',(Join-Path $NativeExperiment 'cache/manifest.json'),
    '--models',(Join-Path $SourceExperiment 'baseline/model'),
    '--source-comparison',(Join-Path $SourceExperiment 'baseline/comparison'),
    '--data-root',(Join-Path $ArtifactRoot 'data/gold166'),'--output-dir',$ExperimentDirectory)
Write-Output 'Phase 1 allocation control only: cached fresh-baseline epoch-30 fields; no GPU inference, no optimizer updates.'
Write-Output 'Seven development neurons, 56 crops; matched N=min(budget,foreground), budgets 512/2048, seeds 0-4.'
Write-Output 'Reserve floor(N/2) uniform foreground points; uniform expanded-pool fill excludes reserved points. Compare saved expanded random at the same N.'
if ($Action -eq 'check') {
    Write-Output 'Read-only source/cache/field checks; no comparison starts automatically.'
    & $Python @cliArguments
} else {
    Write-Output "Outputs: $ExperimentDirectory"
    $parentDirectory = Split-Path -Parent $ExperimentDirectory
    if (-not (Test-Path -LiteralPath $parentDirectory)) { New-Item -ItemType Directory -Path $parentDirectory | Out-Null }
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath "$ExperimentDirectory.console.log"
}
if ($LASTEXITCODE -ne 0) { throw "Image reservation $Action failed with exit code $LASTEXITCODE. Preserve logs; no automatic restart." }
