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
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/frozen_proposal_gat_20261018' }
if (-not $Device) { $Device = if ($Action -in @('prepare','check')) { 'cpu' } else { 'cuda' } }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'frozen_proposal_gat.py'),'--action',$Action,
    '--directory',$ExperimentDirectory,'--device',$Device)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Resume requires -Checkpoint; the cap remains 4000 total stage updates.' }
    $arguments += @('--checkpoint',$Checkpoint)
}
Write-Output 'Four TRAINING crops: 0/0, 0/1, 92/4, 86/3. Oracle-selected 6/12/16/418 frozen balanced proposals.'
Write-Output 'Fresh native GATv2: two layers, four heads, 128 channels, seed zero. No upstream training or inference.'
if ($Action -eq 'prepare') { Write-Output 'CPU cache/label preparation and reference-connected feasibility only; zero optimizer updates.' }
elseif ($Action -eq 'check') { Write-Output 'CPU gradients, graph metrics, SWC and checkpoint reload; zero optimizer updates.' }
elseif ($Action -eq 'timing') { Write-Output 'Eight throwaway updates only; no trained checkpoint retained. Training is a separate command.' }
else { Write-Output 'Connectivity-only fit: cyclic four crops, AdamW LR .001, maximum 4000 updates. No automatic extension.' }
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
if ($taskNativeExitCode -ne 0) { throw "Frozen proposal GAT $Action failed with exit code $taskNativeExitCode. Preserve logs; no automatic retry." }
