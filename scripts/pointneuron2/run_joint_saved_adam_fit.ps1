param(
    [Parameter(Mandatory=$true)][ValidateSet('check','train','resume')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/joint_saved_adam_fit_20261020' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'joint_saved_adam_fit.py'),'--action',$Action,'--directory',$ExperimentDirectory)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Resume requires -Checkpoint from this fit. The total stage cap remains 4000.' }
    $arguments += @('--checkpoint',$Checkpoint)
} elseif ($Checkpoint) { throw '-Checkpoint is only supported with -Action resume.' }
Write-Output 'Four fixed TRAINING crops: 0/0, 0/1, 92/4, 86/3. Phase 1 frozen; all 512 queries reach GAT before final .5 filtering.'
Write-Output 'Saved proposal Adam at 4000; fresh GAT Adam at zero. Joint LR .0001; same losses, trust masks and automatic reconstruction gates.'
if ($Action -eq 'check') { Write-Output 'CPU zero-update fit checkpoint/reload check. No GPU training.' }
else {
    Write-Output 'RTX 3070 Ti Laptop GPU: maximum 4000 NEW stage updates; no automatic extension.'
    Write-Output 'Passed timing estimated 51-77 minutes of updates plus evaluation/checkpoint overhead. First eight updates must replay passed GPU timing.'
    Write-Output 'Automatic metrics at 0/500/1000/2000/4000; state saved before evaluation. A terminal failure stops this control.'
}
Write-Output "Output: $ExperimentDirectory"
$taskParent = Split-Path -Parent $ExperimentDirectory
if (-not (Test-Path -LiteralPath $taskParent)) { New-Item -ItemType Directory -Path $taskParent | Out-Null }
$logPath = "$ExperimentDirectory.$Action.$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').console.log"
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$taskPreviousErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
    $taskNativeExitCode = $LASTEXITCODE
} finally { $ErrorActionPreference = $taskPreviousErrorAction }
if ($taskNativeExitCode -ne 0) { throw "Saved-Adam joint fit $Action failed with exit code $taskNativeExitCode. Preserve logs/checkpoints; no automatic retry or cap change." }
