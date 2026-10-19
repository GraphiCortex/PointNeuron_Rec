param(
    [Parameter(Mandatory=$true)][ValidateSet('check','timing')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/joint_saved_adam_timing_20261019' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'joint_saved_adam_timing.py'),'--action',$Action,'--directory',$ExperimentDirectory)
Write-Output 'Four fixed TRAINING crops: 0/0, 0/1, 92/4, 86/3; Phase 1 frozen; all 512 queries reach GAT before final .5 filtering.'
Write-Output 'Saved proposal Adam state at 4000; fresh GAT Adam state at zero. Same joint loss, LR .0001, clipping and trust masks.'
if ($Action -eq 'check') { Write-Output 'CPU zero-update adapter/checkpoint/reload check only.' }
else { Write-Output 'RTX 3070 Ti Laptop GPU: eight throwaway updates, per-step diagnostics, peak memory and exact mixed-state reload. No full training or resume action.' }
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
if ($taskNativeExitCode -ne 0) { throw "Saved-Adam joint $Action failed with exit code $taskNativeExitCode. Preserve logs; no automatic retry." }
