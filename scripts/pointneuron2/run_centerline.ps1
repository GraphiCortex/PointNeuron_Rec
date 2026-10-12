param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','preflight','fit','loss-control','loss-control-preflight')][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$Manifest,
    [string]$OutputDirectory,
    [string]$Python,
    [string]$BaselineModels,
    [string]$LossConfig,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply -Python with the local PyTorch environment.' }
$lossControl = $Action -in @('loss-control','loss-control-preflight')
if ($lossControl) {
    if (-not $BaselineModels) { $BaselineModels = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/centerline_fit_20261012_144121_14488/model' }
    if (-not $LossConfig) { $LossConfig = Join-Path $projectRoot 'configs/pointneuron2/centerline_loss_control.json' }
    $BaselineModels = (Resolve-Path -LiteralPath $BaselineModels).Path
    $LossConfig = (Resolve-Path -LiteralPath $LossConfig).Path
} elseif ($BaselineModels -or $LossConfig) { throw 'BaselineModels and LossConfig apply only to loss-control actions.' }
if (-not $Manifest) {
    if ($Action -eq 'prepare') {
        $Manifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/query_implementation_20261011/cache/manifest.json'
    } else {
        $Manifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/centerline_prepare_20261012/cache/manifest.json'
    }
}
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot ('tmp/pointneuron2/local_runs/centerline_{0}_{1}_{2}' -f ($Action -replace '-', '_'), (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a fresh output directory to preserve earlier runs.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$runDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
if ($Action -eq 'prepare') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'prepare_centerline_fit.py'), '--manifest', $Manifest,
        '--data-root', (Join-Path $ArtifactRoot 'data/gold166'), '--output-dir', (Join-Path $runDirectory 'cache'))
    Write-Output 'CPU native-image preparation only: 8 training crops from 4 neurons; visible overlays; no training.'
} else {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'train_centerline_fit.py'), '--manifest', $Manifest,
        '--output-dir', (Join-Path $runDirectory 'model'), '--device', $Device)
    if ($lossControl) {
        $cliArguments += @('--baseline-models', $BaselineModels, '--loss-config', $LossConfig)
        Write-Output 'Loss-only paired control: top min(N,3P) known-negative BCE; reuse completed baseline; fresh matched initialization; other settings fixed.'
    }
    if ($Action -in @('preflight','loss-control-preflight')) {
        $cliArguments += '--preflight'
        Write-Output 'Timing only: 4 throwaway updates, no trained checkpoint retained.'
    } else {
        Write-Output 'Fixed training-crop learnability: 400 updates, batch 1; evaluations 0/100/200/400; no validation/test or SWC reconstruction.'
    }
}
Write-Output "Local run directory: $runDirectory"
& $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'console.log')
if ($LASTEXITCODE -ne 0) { throw "Centerline $Action failed with exit code $LASTEXITCODE" }
