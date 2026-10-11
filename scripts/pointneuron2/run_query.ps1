param(
    [ValidateSet('prepare', 'smoke', 'train')]
    [Parameter(Mandatory = $true)][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$Manifest,
    [string]$OutputDirectory,
    [string]$Python,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [int]$Steps = 100
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply -Python with the local PyTorch environment.' }
if (-not $Manifest) {
    if ($Action -eq 'prepare') {
        $Manifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/repair_20261011_130158/cache/manifest.json'
    } else { throw 'Supply -Manifest pointing to the prepared query cache manifest.json.' }
}
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot ('tmp/pointneuron2/local_runs/query_{0}_{1}_{2}' -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a fresh output directory to preserve earlier runs.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$runDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
if ($Action -eq 'prepare') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'build_query_cache.py'), '--manifest', $Manifest,
        '--data-root', (Join-Path $ArtifactRoot 'data/gold166'), '--output-dir', (Join-Path $runDirectory 'cache'))
    Write-Output 'CPU target preparation: corrected image candidates unchanged; common SWC curve targets; no training.'
} else {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'train_query_comparison.py'), '--manifest', $Manifest,
        '--output-dir', (Join-Path $runDirectory 'models'), '--device', $Device, '--log-every', '5')
    if ($Action -eq 'smoke') {
        $cliArguments += @('--smoke', '--steps', "$Steps")
        Write-Output "Fixed two-training-crop check: both arms, $Steps updates each; device=$Device. This is not validation."
    } else {
        Write-Output "Matched pilot: 33 train / 7 validation neurons, 30 epochs per arm; device=$Device. Whole-neuron evaluation is separate."
    }
}
$launch = [ordered]@{ status = 'running'; action = $Action; manifest = $Manifest; device = $Device; arguments = $cliArguments }
$launchPath = Join-Path $runDirectory 'launch.json'
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Write-Output "Local run directory: $runDirectory"
try {
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'console.log')
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) { throw "Run failed with exit code $runExitCode" }
    $launch.status = 'completed'
} catch {
    $launch.status = 'failed_or_interrupted'
    throw
} finally {
    $launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
}
