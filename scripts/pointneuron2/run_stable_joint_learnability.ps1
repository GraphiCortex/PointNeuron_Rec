param(
    [Parameter(Mandatory=$true)][ValidateSet('check','train','resume')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python,
    [string]$Checkpoint
)
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python=Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory=Join-Path $projectRoot 'tmp/pointneuron2/local_runs/stable_joint_learnability_20261020' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments=@('-u',(Join-Path $PSScriptRoot 'stable_joint_learnability.py'),'--action',$Action,'--directory',$ExperimentDirectory)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Explicit intermediate checkpoint from this stable-LR fit required.' }
    $arguments+=@('--checkpoint',$Checkpoint)
} elseif ($Checkpoint) { throw '-Checkpoint is only accepted for resume.' }
Write-Output 'Four TRAINING crops: 0/0, 0/1, 92/4, 86/3; frozen Phase 1; existing PointNeXt/Transformer/GAT; 512 queries.'
Write-Output 'Existing partial-supervision loss; LR .00001; exact saved joint stage 1000 weights/Adam. No new head, loss, threshold or architecture.'
if ($Action -eq 'check') { Write-Output 'CPU zero-update import/probe/checkpoint check only. No CUDA initialization or training.' }
else {
    Write-Output 'RTX 3070 Ti Laptop GPU: at most 3000 NEW updates, ending at historical stage 4000. No automatic extension.'
    Write-Output 'First 64 updates must reproduce the reduced-LR control, including complete model/Adam/RNG state. Original automatic reconstruction gates remain required.'
    Write-Output 'Plan roughly an hour or longer depending on GPU load and IO; sustained duration is not yet measured. Console reports elapsed time.'
    Write-Output 'Geometry probes every four updates through replay, then every twenty. Full state saved before scheduled graph/SWC evaluation.'
}
Write-Output "Output: $ExperimentDirectory"
$taskParent=Split-Path -Parent $ExperimentDirectory
if (-not (Test-Path -LiteralPath $taskParent)) { New-Item -ItemType Directory -Path $taskParent | Out-Null }
$logPath="$ExperimentDirectory.$Action.$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').console.log"
$env:PYTHONIOENCODING='utf-8';$env:OMP_NUM_THREADS='2';$env:MKL_NUM_THREADS='2';$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$taskPreviousErrorAction=$ErrorActionPreference
try {
    $ErrorActionPreference='Continue'
    & $Python @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
    $taskNativeExitCode=$LASTEXITCODE
} finally { $ErrorActionPreference=$taskPreviousErrorAction }
if ($taskNativeExitCode -ne 0) { throw "Stable joint $Action failed with exit code $taskNativeExitCode. Preserve evidence; no automatic retry or cap change." }
