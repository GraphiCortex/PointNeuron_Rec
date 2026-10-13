param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','preflight','train')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$SourceDirectory,
    [string]$ControlDirectory,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/query_confidence_20261013' }
if (-not $SourceDirectory) { $SourceDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/pointnext_query_20261013' }
if (-not $ControlDirectory) { $ControlDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing environment with -Python.' }
$cliArguments = @('-u',(Join-Path $PSScriptRoot 'query_confidence_experiment.py'),'--action',$Action,
    '--experiment-directory',$ExperimentDirectory,'--source-directory',$SourceDirectory,'--control-directory',$ControlDirectory,'--device',$Device)
Write-Output 'Confidence-only diagnostic: fixed neuron 0 crops 0/1, fixed features/geometry/assignments; 258 trainable parameters.'
if ($Action -eq 'prepare') { Write-Output 'CPU only: two frozen model forwards; no optimizer. Freeze final-layer inputs and existing 6/12 assignments.' }
elseif ($Action -eq 'preflight') { Write-Output 'Four throwaway confidence-only updates; runtime/reload check; no fitted checkpoint retained.' }
else { Write-Output '2000 cached-feature updates on both crops; LR .0001, unchanged weighted confidence loss, cutoff .5; no encoder/Transformer inference.' }
$parentDirectory = Split-Path -Parent $ExperimentDirectory
if (-not (Test-Path -LiteralPath $parentDirectory)) { New-Item -ItemType Directory -Path $parentDirectory | Out-Null }
Write-Output "Experiment: $ExperimentDirectory"
& $Python @cliArguments 2>&1 | Tee-Object -FilePath "$ExperimentDirectory.$Action.console.log"
if ($LASTEXITCODE -ne 0) { throw "Confidence $Action failed with exit code $LASTEXITCODE. Outputs preserved; no automatic retry." }
