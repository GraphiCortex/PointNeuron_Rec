param(
    [Parameter(Mandatory=$true)][ValidateSet('check','prepare','preflight','train')][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$ExperimentDirectory,
    [string]$FieldExperiment,
    [string]$NativeExperiment,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/pointnext_query_20261013' }
if (-not $FieldExperiment) { $FieldExperiment = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_attention_20261013' }
if (-not $NativeExperiment) { $NativeExperiment = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_pilot_20261012' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$cliArguments = @('-u',(Join-Path $PSScriptRoot 'pointnext_query_experiment.py'),'--action',$Action,
    '--experiment-dir',$ExperimentDirectory,'--native-manifest',(Join-Path $NativeExperiment 'cache/manifest.json'),
    '--field-models',(Join-Path $FieldExperiment 'baseline/model'),'--data-root',(Join-Path $ArtifactRoot 'data/gold166'),'--device',$Device)
Write-Output 'Native PointNeXt-S prototype -> existing Transformer query body; GATv2 remains a later stage.'
Write-Output 'Exactly training neuron 0 crops 0/1; frozen fresh-baseline field, expanded uniform up to 2048 distinct points; targets 6/12; no reserved tests.'
if ($Action -eq 'check') {
    Write-Output 'Read-only source/cache contracts: no model inference or optimizer.'
    & $Python @cliArguments
} else {
    $parentDirectory = Split-Path -Parent $ExperimentDirectory
    if (-not (Test-Path -LiteralPath $parentDirectory)) { New-Item -ItemType Directory -Path $parentDirectory | Out-Null }
    if ($Action -eq 'prepare') { Write-Output 'Two frozen field forward passes only; freeze clouds then targets; no PointNeXt inference/training.' }
    elseif ($Action -eq 'preflight') { Write-Output 'Four throwaway PointNeXt/query updates, reload check and timing estimate; no trained checkpoint retained.' }
    else { Write-Output 'Explicit bounded fit: 2000 updates, batch 1, dropout zero, no augmentation; checks 0/500/1000/2000. Review timing first.' }
    Write-Output "Experiment: $ExperimentDirectory"
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath "$ExperimentDirectory.$Action.console.log"
}
if ($LASTEXITCODE -ne 0) { throw "PointNeXt $Action failed with exit code $LASTEXITCODE. Preserve logs; no automatic retry." }
