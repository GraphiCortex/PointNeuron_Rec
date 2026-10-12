param(
    [ValidateSet('compare','preflight')][string]$Action = 'compare',
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$Models,
    [string]$Manifest,
    [string]$OutputDirectory,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $Models) { $Models = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/centerline_duration_20261012_220038_14488/model' }
if (-not $Manifest) { $Manifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/centerline_prepare_20261012/cache/manifest.json' }
foreach ($path in @($Python,$Models,$Manifest)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing local input: $path" } }
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot ('tmp/pointneuron2/local_runs/field_sampling_{0}_{1}_{2}' -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a fresh output directory.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$runDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$cliArguments = @('-u', (Join-Path $PSScriptRoot 'compare_field_sampling.py'), '--models', $Models,
    '--manifest', $Manifest, '--data-root', (Join-Path $ArtifactRoot 'data/gold166'),
    '--output-dir', (Join-Path $runDirectory 'comparison'), '--device', $Device)
if ($Action -eq 'preflight') {
    $cliArguments += '--preflight'
    Write-Output 'One training crop (96/3), nominal 2048, seed 0: timing and execution checks only.'
} else {
    Write-Output 'Eight fitted training crops; four controls; budgets 512/2048; seeds 0-4; matched unique-point counts.'
}
Write-Output 'Frozen CUDA field inference followed by CPU point selection and reference scoring. No training or later architecture stages.'
Write-Output "Local run directory: $runDirectory"
& $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'console.log')
if ($LASTEXITCODE -ne 0) { throw "Field sampling $Action failed with exit code $LASTEXITCODE" }
