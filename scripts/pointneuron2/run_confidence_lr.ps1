param(
    [Parameter(Mandatory=$true)][ValidateSet('check','preflight','train')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$BaselineDirectory,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/query_confidence_lr_20261014' }
if (-not $BaselineDirectory) { $BaselineDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/query_confidence_20261013' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing environment with -Python.' }
$cliArguments = @('-u',(Join-Path $PSScriptRoot 'confidence_lr_experiment.py'),'--action',$Action,
    '--experiment-directory',$ExperimentDirectory,'--baseline-directory',$BaselineDirectory,'--device',$Device)
Write-Output 'LR-only confidence replay: .001 versus saved .0001; same initialization, fixed features/labels/geometry, 2000 paired updates.'
Write-Output 'Only 258 parameters train; no PointNeXt/Transformer inference; baseline outputs preserved.'
if ($Action -eq 'check') {
    Write-Output 'Read-only baseline, cache, recipe and initial-weight checks; no forward or optimizer.'
    & $Python @cliArguments
} else {
    $parentDirectory = Split-Path -Parent $ExperimentDirectory
    if (-not (Test-Path -LiteralPath $parentDirectory)) { New-Item -ItemType Directory -Path $parentDirectory | Out-Null }
    if ($Action -eq 'preflight') { Write-Output 'Four throwaway timing updates; no fitted checkpoint retained; training stays separate.' }
    else { Write-Output 'Fresh LR .001 fit from the original cached weights; checks 0/100/500/1000/2000 and paired comparison against the saved baseline.' }
    Write-Output "Experiment: $ExperimentDirectory"
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath "$ExperimentDirectory.$Action.console.log"
}
if ($LASTEXITCODE -ne 0) { throw "LR replay $Action failed with exit code $LASTEXITCODE. Outputs preserved; no automatic retry." }
