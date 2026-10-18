param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','check','train','resume')][string]$Action,
    [string]$ExperimentDirectory,
    [string]$Python,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/four_crop_frozen_confidence_20261018' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local environment with -Python.' }
$arguments = @('-u',(Join-Path $PSScriptRoot 'four_crop_frozen_confidence.py'),'--action',$Action,
    '--directory',$ExperimentDirectory)
if ($Action -eq 'resume') {
    if (-not $Checkpoint) { throw 'Resume requires -Checkpoint; the cap stays at 4000 total updates.' }
    $arguments += @('--checkpoint',$Checkpoint)
}
Write-Output 'Four fixed TRAINING crops: neuron 0/crops 0,1; neuron 92/crop 4; neuron 86/crop 3.'
Write-Output 'CPU confidence-only stage; 258 parameters; frozen balanced geometry/features/labels; all four cached crops per update.'
if ($Action -eq 'prepare') { Write-Output 'CPU cache preparation from saved decoder features; no encoder/Transformer forwards or optimizer updates.' }
elseif ($Action -eq 'check') { Write-Output 'CPU replay plus eight throwaway head updates; no retained fitted checkpoint.' }
else { Write-Output 'CPU confidence-only fit; 4000 new-stage updates maximum; fresh optimizer, LR .0001; no automatic extension.' }
Write-Output "Output: $ExperimentDirectory"
$taskParent = Split-Path -Parent $ExperimentDirectory
if (-not (Test-Path -LiteralPath $taskParent)) { New-Item -ItemType Directory -Path $taskParent | Out-Null }
$logPath = "$ExperimentDirectory.$Action.$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').console.log"
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$taskPreviousErrorAction = $ErrorActionPreference
try {
    # Windows PowerShell wraps native stderr as ErrorRecord. Let the full stream
    # drain, then decide failure from the native exit code instead of its first line.
    $ErrorActionPreference = 'Continue'
    & $Python @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
    $taskNativeExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $taskPreviousErrorAction
}
if ($taskNativeExitCode -ne 0) { throw "Frozen confidence $Action failed with exit code $taskNativeExitCode. Preserve logs; no automatic retry." }
