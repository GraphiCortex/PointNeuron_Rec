param(
    [Parameter(Mandatory=$true)][ValidateSet('check','timing','train','resume')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python,
    [string]$Checkpoint
)
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python=Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory=Join-Path $projectRoot 'tmp/pointneuron2/local_runs/canonical_joint_fit_20261021' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing sampling environment with -Python.' }
$arguments=@('-u',(Join-Path $PSScriptRoot 'canonical_joint_fit.py'),'--action',$Action,'--directory',$ExperimentDirectory)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Explicit intermediate checkpoint from this canonical fit required.' }
    $arguments+=@('--checkpoint',$Checkpoint)
} elseif ($Checkpoint) { throw '-Checkpoint is only accepted for resume.' }
Write-Output 'Four TRAINING crops: 0/0, 0/1, 92/4, 86/3. Exact frozen image clouds; canonical targets 6/11/15/408; 512 queries.'
Write-Output 'Only targets change. Same stage-1000 weights/Adam/RNG, LR .00001, partial loss, PointNeXt/Transformer/GAT and .5 decisions.'
Write-Output 'Learnability and paired full-SWC geometry are separate verdicts. Historical failed gates remain unchanged.'
if ($Action -eq 'check') { Write-Output 'CPU import/probe/checkpoint check; zero updates and no CUDA initialization.' }
elseif ($Action -eq 'timing') { Write-Output 'Eight throwaway GPU updates, exact initial baseline predictions and mixed-clock reload. No trained checkpoint retained.' }
else {
    Write-Output 'RTX 3070 Ti Laptop GPU: cap 3000 new updates, historical stage 4000. First eight updates must match the new timing state exactly.'
    Write-Output 'Prior matched baseline took 71 minutes; plan roughly 60-90 minutes plus load/IO variability. No automatic extension.'
    Write-Output 'Matching completed CPU check and GPU timing required. Checkpoints precede fallible metric/SWC exports.'
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
if ($taskNativeExitCode -ne 0) { throw "Canonical joint $Action failed with exit code $taskNativeExitCode. Preserve evidence; no automatic retry or cap change." }
