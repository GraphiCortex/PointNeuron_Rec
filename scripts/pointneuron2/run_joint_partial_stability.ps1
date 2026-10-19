param(
    [Parameter(Mandatory=$true)][ValidateSet('check','compare')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/joint_partial_stability_20261020' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'joint_partial_stability.py'),'--action',$Action,'--directory',$ExperimentDirectory)
Write-Output 'Four fixed TRAINING crops: 0/0, 0/1, 92/4, 86/3. Frozen Phase 1; PointNeXt/Transformer/GAT; all 512 queries before final .5 filtering.'
Write-Output 'Import exact joint checkpoint 1000: proposal Adam 5000, GAT Adam 1000. Known edge classes only; no known edges means proposal-only, GAT state unchanged.'
if ($Action -eq 'check') { Write-Output 'CPU zero-update preparation: both fork initializations, probe noninterference and diagnostic disk reload. No CUDA initialization or fitting.' }
else {
    Write-Output 'RTX 3070 Ti Laptop GPU: A LR .0001 and B LR .00001, at most 64 new updates EACH (128 total). Both restart from identical saved weights/moments/RNG.'
    Write-Output 'A must reproduce historical steps 1001-1014 before comparison. All-crop geometry probes every four updates; automatic graph/SWC at start and endpoint.'
    Write-Output 'Planning estimate: several minutes plus setup/probe/export overhead; this new control has not been GPU-timed. Hard update caps; no extension or long-fit action.'
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
if ($taskNativeExitCode -ne 0) { throw "Partial-supervision $Action failed with exit code $taskNativeExitCode. Preserve outputs; no automatic retry or policy change." }
