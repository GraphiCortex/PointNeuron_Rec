param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','check','timing','train','resume')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python,
    [ValidateSet('cpu','cuda')][string]$Device,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/joint_query_graph_ready_20261019' }
if (-not $Device) { $Device = if ($Action -in @('prepare','check')) { 'cpu' } else { 'cuda' } }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'joint_query_graph_experiment.py'),'--action',$Action,
    '--directory',$ExperimentDirectory,'--device',$Device)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Resume requires -Checkpoint; maximum remains 4000 total new-stage updates.' }
    $arguments += @('--checkpoint',$Checkpoint)
}
Write-Output 'Four fixed TRAINING crops: 0/0, 0/1, 92/4, 86/3. Phase 1 frozen; all 512 queries reach GAT before final .5 filtering.'
Write-Output 'Warm-start balanced PointNeXt/Transformer and fitted GAT; NEW joint AdamW optimizer LR .0001; no new heads or topology loss.'
if ($Action -eq 'prepare') { Write-Output 'CPU source/cache/checkpoint identity verification; no forward or optimizer updates.' }
elseif ($Action -eq 'check') { Write-Output 'CPU gradients, automatic reconstruction, SWC and reload checks; zero optimizer updates.' }
elseif ($Action -eq 'timing') { Write-Output 'Eight throwaway GPU updates only; no trained checkpoint retained. Training is a separate command.' }
else { Write-Output 'Bounded joint fit: maximum 4000 new-stage updates, no automatic extension. Review timing first.' }
Write-Output "Device: $Device | Output: $ExperimentDirectory"
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
} finally {
    $ErrorActionPreference = $taskPreviousErrorAction
}
if ($taskNativeExitCode -ne 0) { throw "Joint query/graph $Action failed with exit code $taskNativeExitCode. Preserve logs; no automatic retry." }
